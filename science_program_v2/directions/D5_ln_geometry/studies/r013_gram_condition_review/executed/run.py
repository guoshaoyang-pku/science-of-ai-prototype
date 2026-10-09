from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
PREREG = STUDY / 'preregistration.json'
RESULTS = STUDY / 'results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def dataset(name, cfg):
    generator = torch.Generator().manual_seed(cfg['data_seed'] + cfg['functions'].index(name))
    dimension = 12 if name.endswith('12') else 8
    count = cfg['train_n'] + cfg['test_n']
    x = (torch.randn(count, dimension, generator=generator) if dimension == 12
         else 2 * torch.rand(count, dimension, generator=generator) - 1)
    if name == 'trigonometric8':
        y = torch.sin(2*x[:, 0]) + .7*torch.cos(3*x[:, 1]) + .3*x[:, 2]*x[:, 3]
    elif name == 'quadratic8':
        y = x[:, 0]**2 + .5*x[:, 1]**2 - .6*x[:, 2]**2 + .2*x[:, 3]
    elif name == 'interaction12':
        y = torch.sin(x[:, 0]*x[:, 1]) + .5*torch.tanh(x[:, 2]+x[:, 3]) + .2*x[:, 4]*x[:, 5]
    elif name == 'radial12':
        y = torch.exp(-.25*x[:, :6].square().sum(1)) + .1*x[:, 6]
    else:
        raise ValueError(name)
    count = cfg['train_n']
    center, scale = y[:count].double().mean(), y[:count].double().std(unbiased=False)
    y = ((y.double()-center)/scale).float().unsqueeze(1)
    return x[:count], y[:count], x[count:], y[count:]


class Block(nn.Module):
    def __init__(self, width, use_ln):
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_ln else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(self.norm(x)))


class Model(nn.Module):
    def __init__(self, dimension, recipe):
        super().__init__()
        width = recipe['width']
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.GELU(),
                                 *[Block(width, flag) for flag in recipe['layer_norm']],
                                 nn.Linear(width, 1))

    def hidden(self, x):
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        return self.net(x)


def committed_gate(commit, cfg):
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    pins = cfg['source_sha256']
    for relative in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / relative
        saved = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if saved != path.read_bytes():
            raise RuntimeError('current source differs from preregistration commit: ' + relative)
        if relative in pins and sha(path) != pins[relative]:
            raise RuntimeError('pinned source hash mismatch: ' + relative)
    receipt_path = STUDY / 'executed/receipt.json'
    expected = {'preregistration_commit': commit, 'preregistration_sha256': sha(PREREG),
                'source_sha256': pins}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt[key] != value for key, value in expected.items()):
            raise RuntimeError('receipt contract changed')
    else:
        receipt = {**expected, 'execution_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'commit_epoch': int(subprocess.check_output(
                ['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT, text=True)),
            'gate_passed_at_epoch': time.time(), 'python': os.sys.executable,
            'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'threads': 1}
        save_json(receipt_path, receipt)
    return receipt


def request(function, label, mean, seed, cfg, receipt):
    tx, ty, vx, vy = dataset(function, cfg)
    arrays = {'train_x': tx.numpy(), 'train_y': (ty+mean).numpy(),
              'test_x': vx.numpy(), 'test_y': (vy+mean).numpy()}
    req = {'function': function, 'label': label, 'mean': mean, 'seed': seed,
           'recipe': cfg['recipes'][label], 'intervention': cfg['intervention'],
           'preregistration_sha256': sha(PREREG), 'source_sha256': cfg['source_sha256'],
           'preregistration_commit': receipt['preregistration_commit'],
           'inputs': {name: array_pin(value) for name, value in arrays.items()}}
    return req, arrays


def validate(path, req):
    row = json.loads(path.read_text())
    array_path = path.with_suffix('.npz')
    if row['contract'] != req or row['contract_sha256'] != digest(req):
        raise RuntimeError('existing result contract mismatch: ' + path.name)
    if row['status'] != 'completed' or not array_path.exists() or sha(array_path) != row['arrays_sha256']:
        raise RuntimeError('existing result array/status mismatch: ' + path.name)
    return row


def run_cell(path, req, arrays, cfg):
    if path.exists() or path.with_suffix('.npz').exists():
        raise RuntimeError('refusing to overwrite existing result')
    started = time.time()
    timer = time.monotonic()
    tx, ty, vx, vy = [torch.from_numpy(arrays[name]) for name in ('train_x', 'train_y', 'test_x', 'test_y')]
    torch.manual_seed(req['seed'])
    model = Model(tx.shape[1], req['recipe'])
    initial_linear = {name: array_pin(parameter.detach().numpy())
                      for name, parameter in model.named_parameters() if '.norm.' not in name}
    head = model.net[-1]
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    arrays['head_weight_initial'] = head.weight.detach().numpy().copy()
    arrays['head_bias_initial'] = head.bias.detach().numpy().copy()
    recipe = req['recipe']
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe['lr'],
                                momentum=recipe['momentum'], weight_decay=recipe['weight_decay'])
    spectra, kappas = {}, {}

    def checkpoint(step):
        with torch.inference_mode():
            hidden = model.hidden(tx).double()
            centered = hidden-hidden.mean(0, keepdim=True)
            gram = (centered.T@centered/tx.shape[0]).numpy()
            eigenvalues = np.linalg.eigvalsh(gram)[::-1]
            arrays[f'gram_{step}'] = gram
            arrays[f'predictions_{step}'] = model(vx).numpy().copy()
            if step in (0, recipe['steps']):
                arrays[f'hidden_{step}'] = hidden.numpy().copy()
                arrays[f'train_predictions_{step}'] = model(tx).numpy().copy()
        if not np.isfinite(gram).all() or not np.isfinite(arrays[f'predictions_{step}']).all():
            raise RuntimeError('nonfinite checkpoint')
        spectra[str(step)] = eigenvalues.tolist()
        floor = cfg['mediator']['eigenvalue_floor']
        kappas[str(step)] = float(np.log10(max(eigenvalues[0], floor)/max(eigenvalues[-1], floor)))

    checkpoint(0)
    batches = []
    for step in range(1, recipe['steps']+1):
        indices = torch.randint(0, tx.shape[0], (recipe['batch_size'],))
        batches.append(indices.numpy().copy())
        loss = ((model(tx[indices])-ty[indices])**2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in cfg['checkpoints']:
            checkpoint(step)
    arrays['batch_indices'] = np.stack(batches)
    arrays['head_weight_final'] = head.weight.detach().numpy().copy()
    arrays['head_bias_final'] = head.bias.detach().numpy().copy()
    if not all(np.array_equal(arrays[f'head_{role}_initial'], arrays[f'head_{role}_final'])
               for role in ('weight', 'bias')):
        raise RuntimeError('frozen head moved')
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path.with_suffix('.npz'))
    row = {'contract': req, 'contract_sha256': digest(req),
           'arrays_sha256': sha(path.with_suffix('.npz')), 'initial_linear': initial_linear,
           'batch_sha256': array_pin(arrays['batch_indices'])['sha256'],
           'spectrum': spectra, 'log10_condition': kappas, 'status': 'completed',
           'started_at_epoch': started, 'saved_at_epoch': time.time(),
           'seconds': time.monotonic()-timer}
    save_json(path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads(PREREG.read_text())
    receipt = committed_gate(args.commit, cfg)
    cells = []
    for function in cfg['functions']:
        for label in cfg['recipes']:
            for mean in cfg['means']:
                for seed in cfg['seeds']:
                    req, arrays = request(function, label, mean, seed, cfg, receipt)
                    path = RESULTS / f'{function}_{label}_{mean}_{seed}.json'
                    cells.append((path, req, arrays))
    completed = {}
    for path, req, _ in cells:
        if path.exists():
            completed[path] = validate(path, req)
        elif path.with_suffix('.npz').exists():
            raise RuntimeError('orphan arrays: ' + path.name)
    expected = {path for path, _, _ in cells}
    if any(path not in expected for path in RESULTS.glob('*.json')):
        raise RuntimeError('unexpected result metadata')
    used_seconds = sum(row['seconds'] for row in completed.values())
    new = 0
    for path, req, arrays in cells:
        if path in completed:
            continue
        if args.max_new_cells is not None and new >= args.max_new_cells:
            break
        if used_seconds >= cfg['execution']['max_seconds']:
            raise RuntimeError('compute budget exhausted before next cell')
        row = run_cell(path, req, arrays, cfg)
        completed[path] = row
        used_seconds += row['seconds']
        new += 1
        save_json(STUDY/'current.json', {'status': 'running', 'completed': len(completed),
                  'total': len(cells), 'new_cells': new, 'experiment_seconds': used_seconds,
                  'last_cell': path.stem})
        print(f'{len(completed)}/{len(cells)} {path.stem} {row["seconds"]:.3f}s', flush=True)
    state = {'status': 'measurements_complete' if len(completed) == len(cells) else 'partial',
             'completed': len(completed), 'total': len(cells), 'new_cells': new,
             'reused_cells': len(completed)-new, 'experiment_seconds': used_seconds}
    save_json(STUDY/'current.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
