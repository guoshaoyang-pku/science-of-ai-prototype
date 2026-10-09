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
OLD = STUDY.parent / 'r013_gram_condition_review'
COUPLING = STUDY.parent / 'r045_mean_centered_coupling'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def save(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


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

    def forward(self, x):
        return self.net(x)


def selected_parameters(model):
    head_ids = {id(p) for p in model.net[-1].parameters()}
    return [(name, p) for name, p in model.named_parameters()
            if id(p) not in head_ids and '.norm.' not in name]


def gate(commit, cfg):
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for relative in ('preregistration.json', 'executed/input_manifest.json',
                     'executed/run.py', 'analysis.py', 'executed/verify.py'):
        path = STUDY / relative
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'],
                                       cwd=ROOT) == path.read_bytes(), relative
        if relative in cfg['source_sha256']:
            assert sha(path) == cfg['source_sha256'][relative], relative
    manifest = json.loads((STUDY / 'executed/input_manifest.json').read_text())
    for relative, item in manifest['files'].items():
        path = ROOT / relative
        assert sha(path) == item['sha256'], relative
        assert path.stat().st_mtime_ns == item['mtime_ns'], relative
        assert subprocess.check_output(['git', 'show', f"{manifest['snapshot_commit']}:{relative}"],
                                       cwd=ROOT) == path.read_bytes(), relative
    receipt_path = STUDY / 'executed/receipt.json'
    fixed = {'preregistration_commit': commit, 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
             'source_sha256': cfg['source_sha256'], 'manifest_sha256': sha(STUDY / 'executed/input_manifest.json')}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert all(receipt[k] == v for k, v in fixed.items())
    else:
        receipt = {**fixed, 'commit_epoch': int(subprocess.check_output(
            ['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT, text=True)),
            'gate_epoch': time.time(), 'python': os.sys.executable,
            'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'threads': 1}
        save(receipt_path, receipt)
    return receipt


def request(function, label, seed, cfg, receipt):
    sources = [OLD / 'results' / f'{function}_{label}_{offset}_{seed}.json'
               for offset in cfg['offsets']]
    rows = [json.loads(path.read_text()) for path in sources]
    assert all(row['status'] == 'completed' for row in rows)
    assert all(row['initial_linear'] == rows[0]['initial_linear'] for row in rows)
    base = OLD / 'results' / f'{function}_{label}_0_{seed}.npz'
    with np.load(base) as data:
        arrays = {name: data[name].copy() for name in ('train_x', 'train_y',
                  'train_predictions_0', 'head_weight_initial', 'head_bias_initial')}
    for row, path in zip(rows, sources):
        assert sha(path.with_suffix('.npz')) == row['arrays_sha256']
        with np.load(path.with_suffix('.npz')) as data:
            assert np.array_equal(data['train_x'], arrays['train_x'])
            assert np.array_equal(data['train_predictions_0'], arrays['train_predictions_0'])
            target = data['train_y'].astype(np.float64).ravel()
            base_target = arrays['train_y'].astype(np.float64).ravel()
            assert np.max(np.abs((target-target.mean())-(base_target-base_target.mean()))) < 3e-7
    for offset, source in zip(cfg['offsets'], sources):
        with np.load(source.with_suffix('.npz')) as data:
            for name in ('train_y', 'train_predictions_256'):
                arrays[f'{name}__{offset}'] = data[name].copy()
    req = {'function': function, 'label': label, 'seed': seed, 'offset_representative': 0,
           'recipe': rows[0]['contract']['recipe'], 'initial_linear': rows[0]['initial_linear'],
           'input_pins': {name: pin(value) for name, value in arrays.items()},
           'preregistration_commit': receipt['preregistration_commit'],
           'preregistration_sha256': receipt['preregistration_sha256'],
           'source_sha256': cfg['source_sha256'], 'manifest_sha256': receipt['manifest_sha256']}
    return req, arrays


def validate(path, req):
    row = json.loads(path.read_text())
    assert row['status'] == 'completed' and row['contract'] == req
    assert sha(path.with_suffix('.npz')) == row['arrays_sha256']
    return row


def measure(path, req, arrays):
    assert not path.exists() and not path.with_suffix('.npz').exists()
    started, timer = time.time(), time.monotonic()
    torch.manual_seed(req['seed'])
    model = Model(arrays['train_x'].shape[1], req['recipe'])
    initial = {name: pin(p.detach().numpy()) for name, p in model.named_parameters()
               if '.norm.' not in name}
    assert initial == req['initial_linear']
    selected = selected_parameters(model)
    selected_ids = {id(p) for _, p in selected}
    for p in model.parameters():
        p.requires_grad_(id(p) in selected_ids)
    prediction = model(torch.from_numpy(arrays['train_x'])).ravel()
    assert np.array_equal(prediction.detach().numpy(), arrays['train_predictions_0'].ravel())
    parameters = [p for _, p in selected]
    jacobian = np.stack([np.concatenate([g.detach().numpy().ravel() for g in
        torch.autograd.grad(prediction[i], parameters, retain_graph=True)])
        for i in range(prediction.numel())]).astype(np.float64)
    n = prediction.numel()
    kernel = np.einsum('ip,jp->ij', jacobian, jacobian, optimize=False)/n
    eigenvalues = np.linalg.eigvalsh(kernel)
    assert eigenvalues[0] >= -1e-10 and .001*eigenvalues[-1] < 1
    vectors = {0: np.ones(n, dtype=np.float64)}
    vector = vectors[0].copy()
    for step in range(1, 257):
        vector = vector-.002*np.einsum('ij,j->i', kernel, vector, optimize=False)
        if step in (1, 256):
            vectors[step] = vector.copy()
    chords = {}
    for step, vector in vectors.items():
        centered = vector-vector.mean()
        arrays[f'propagated_ones_{step}'] = vector
        arrays[f'centered_propagated_ones_{step}'] = centered
        chords[str(step)] = float(9*np.mean(centered**2))
    arrays['kernel'] = kernel
    arrays['kernel_eigenvalues'] = eigenvalues
    arrays['prediction_initial'] = prediction.detach().numpy().copy()
    assert all(np.isfinite(a).all() for a in arrays.values())
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path.with_suffix('.npz'))
    row = {'status': 'completed', 'contract': req, 'arrays_sha256': sha(path.with_suffix('.npz')),
           'parameter_names': [name for name, _ in selected],
           'parameter_count': sum(p.numel() for _, p in selected),
           'fixed_kernel_train_chord': chords,
           'kernel_pin': pin(kernel), 'kernel_eigenvalue_min': float(eigenvalues[0]),
           'kernel_eigenvalue_max': float(eigenvalues[-1]),
           'started_epoch': started, 'saved_epoch': time.time(), 'seconds': time.monotonic()-timer}
    save(path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = gate(args.commit, cfg)
    cells = []
    for function in cfg['functions']:
        for label in cfg['recipes']:
            for seed in cfg['seeds']:
                req, arrays = request(function, label, seed, cfg, receipt)
                path = STUDY / 'results' / f'{function}_{label}_{seed}.json'
                cells.append((path, req, arrays))
    completed = {}
    for path, req, _ in cells:
        if path.exists():
            completed[path] = validate(path, req)
        else:
            assert not path.with_suffix('.npz').exists(), 'orphan NPZ'
    assert set((STUDY / 'results').glob('*.json')) <= {p for p, _, _ in cells}
    used = sum(r['seconds'] for r in completed.values())
    new = 0
    for path, req, arrays in cells:
        if path in completed:
            continue
        if args.max_new_cells is not None and new >= args.max_new_cells:
            break
        assert used < cfg['execution']['max_seconds']
        row = measure(path, req, arrays)
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        completed[path] = row
        used += row['seconds']
        new += 1
        print(f'{len(completed)}/40 {path.stem} {row["seconds"]:.4f}s', flush=True)
    state = {'status': 'complete' if len(completed) == 40 else 'partial',
             'saved_measurement_cells': len(completed), 'new_measurement_cells': new,
             'new_training_cells': 0, 'reused_training_cells': 120, 'measurement_seconds': used}
    save(STUDY / 'current.json', state)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
