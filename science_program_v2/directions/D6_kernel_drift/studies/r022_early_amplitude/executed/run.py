from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
PREREG = STUDY / 'preregistration.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def committed_contract(config):
    for name, expected in config['source_sha256'].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f'Pinned source differs: {name}')
    for name, expected in config['input_sha256'].items():
        if sha(REPO / name) != expected:
            raise RuntimeError(f'Old input differs: {name}')
    relative = str(PREREG.relative_to(REPO))
    history = subprocess.run(['git', 'log', '--format=%H', '--', relative],
                             cwd=REPO, capture_output=True, text=True, check=True)
    for commit in history.stdout.splitlines():
        blob = subprocess.run(['git', 'show', f'{commit}:{relative}'],
                              cwd=REPO, capture_output=True, check=True).stdout
        if blob != PREREG.read_bytes():
            continue
        for name, expected in config['source_sha256'].items():
            path = str((STUDY / name).relative_to(REPO))
            archived = subprocess.run(['git', 'show', f'{commit}:{path}'],
                                      cwd=REPO, capture_output=True, check=True).stdout
            if hashlib.sha256(archived).hexdigest() != expected:
                raise RuntimeError('Preregistration commit does not pin sources')
        return commit
    raise RuntimeError('No matching preregistration commit; training is forbidden')


def make_data(config, function):
    data = config['data_contract']
    x = np.random.default_rng(data['data_seed']).normal(size=(data['train_n'], data['input_dim']))
    raw = (np.tanh(x[:, 0] * x[:, 1]) if function == 'product' else
           np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2]))
    y = (raw - raw.mean()) / raw.std(ddof=0)
    return x, y, np.array([raw.mean(), raw.std(ddof=0)])


def jacobian(model, inputs):
    parameters = list(model.parameters())
    rows = []
    for value in model(inputs).reshape(-1):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in gradients]).detach())
    return torch.stack(rows).numpy()


def ratio(v, k, initial, scale):
    numerator = np.einsum('i,ij,j->', v, k, v, optimize=False)
    denominator = scale * np.einsum('i,ij,j->', v, initial, v, optimize=False)
    return float(numerator / denominator)


def predict(config, early, target, width):
    features = {'E': [1., early], 'W': [1., early, float(width == 8)],
                'T': [1., early, target]}
    return {name: float(np.dot(features[name], coefficients))
            for name, coefficients in config['frozen_coefficients'].items()}


def execute(config, function, width, seed, data, early_path, request, commit):
    x, y, normalization = data
    inputs, targets = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)).double()
    recipe = config['recipe']
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe['lr'], momentum=0., weight_decay=0.)
    js, kernels, parameters, residuals, losses = [], [], [], [], []
    for step in range(recipe['steps'] + 1):
        prediction = model(inputs).reshape(-1)
        residual = prediction - targets
        loss = residual.square().mean() / 2
        losses.append(loss.item())
        if step in config['checkpoints']:
            j = jacobian(model, inputs)
            k = np.einsum('ip,jp->ij', j, j, optimize=False) / len(x)
            js.append(j)
            kernels.append(k)
            parameters.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy())
            residuals.append(residual.detach().numpy())
            if step == config['early_step']:
                scale = np.trace(k) / np.trace(kernels[0])
                e = float(np.log(ratio(residuals[0], k, kernels[0], scale)))
                q = float(np.log(ratio(y, k, kernels[0], scale)))
                early_record = {'request': request, 'preregistration_commit': commit,
                                'step': step, 'E': e, 'Q': q,
                                'predictions': predict(config, e, q, width),
                                'time_ns': time.time_ns(), 'late_step_executed': False}
                if early_path.exists():
                    previous = json.loads(early_path.read_text())
                    for name in ['request', 'preregistration_commit', 'step', 'E', 'Q', 'predictions']:
                        if previous[name] != early_record[name]:
                            raise RuntimeError('Existing early record differs')
                else:
                    save(early_path, early_record)
        if step == recipe['steps']:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    kernels, residuals = np.stack(kernels), np.stack(residuals)
    scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
    fixed = [ratio(residuals[0], k, kernels[0], a) for k, a in zip(kernels, scales)]
    target = [ratio(y, k, kernels[0], a) for k, a in zip(kernels, scales)]
    return {'train_x': x, 'train_y': y, 'target_normalization': normalization,
            'checkpoints': np.array(config['checkpoints']), 'jacobians': np.stack(js),
            'kernels': kernels, 'parameters': np.stack(parameters), 'residuals': residuals,
            'nonlinear_loss': np.array(losses), 'trace_scales': scales,
            'fixed_r0_ratios': np.array(fixed), 'fixed_target_ratios': np.array(target)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int, default=12)
    args = parser.parse_args()
    config = json.loads(PREREG.read_text())
    commit = committed_contract(config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    new_cells = 0
    for function in config['functions']:
        data = make_data(config, function)
        inputs_hash = {name: hashlib.sha256(array.tobytes()).hexdigest()
                       for name, array in zip(['train_x', 'train_y'], data[:2])}
        for width in config['widths']:
            for seed in config['seeds']:
                cell = f'{function}_w{width}_s{seed}'
                path = STUDY / 'results' / f'{cell}.json'
                early_path = path.with_name(f'{cell}.early.json')
                request = {'function': function, 'width': width, 'seed': seed,
                           'preregistration_sha256': sha(PREREG),
                           'source_sha256': config['source_sha256'], 'inputs_sha256': inputs_hash}
                if path.exists():
                    previous = json.loads(path.read_text())
                    if (previous['request'] != request or previous['status'] != 'success'
                            or previous['contract_sha256'] != digest(request)
                            or previous['preregistration_commit'] != commit
                            or previous['arrays_sha256'] != sha(path.with_suffix('.npz'))
                            or previous['early_sha256'] != sha(early_path)):
                        raise RuntimeError(f'Saved cell hash/contract differs: {cell}')
                    print(json.dumps({'cell': cell, 'resumed': True}), flush=True)
                    continue
                if new_cells >= args.max_new_cells:
                    return
                if time.monotonic() - started >= config['execution']['max_seconds']:
                    raise RuntimeError('Round compute budget exhausted')
                if path.with_suffix('.npz').exists():
                    raise RuntimeError(f'Orphan arrays require audit: {cell}')
                cell_started, start_ns = time.monotonic(), time.time_ns()
                try:
                    arrays = execute(config, function, width, seed, data, early_path, request, commit)
                    if not all(np.isfinite(value).all() for value in arrays.values()):
                        raise RuntimeError('Nonfinite cell result')
                except Exception as error:
                    save(path.with_name(f'{cell}.failure.json'),
                         {'request': request, 'status': 'failed', 'error': repr(error)})
                    raise
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                temporary = path.with_suffix('.npz.tmp')
                temporary.write_bytes(stream.getvalue())
                temporary.replace(path.with_suffix('.npz'))
                save(path, {'cell_id': cell, 'request': request, 'status': 'success',
                            'contract_sha256': digest(request), 'preregistration_commit': commit,
                            'arrays_sha256': sha(path.with_suffix('.npz')), 'early_sha256': sha(early_path),
                            'seconds': time.monotonic() - cell_started,
                            'start_time_ns': start_ns, 'finish_time_ns': time.time_ns(),
                            'environment': {'python': sys.version, 'numpy': np.__version__,
                                            'torch': torch.__version__, 'device': 'cpu', 'threads': 1}})
                new_cells += 1
                print(json.dumps({'cell': cell, 'success': True}), flush=True)


if __name__ == '__main__':
    main()
