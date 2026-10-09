#!/usr/bin/env python3
"""仅减半已保存population标签条件的噪声方差，执行冻结特征 GD。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r019_population_label_normalization'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def committed_contract():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    if config['study'] != STUDY.name or config['round'] != 51:
        raise RuntimeError('Only the newly registered round 51 study may train')
    if config.get('execution_status') != 'authorized_after_valid_preregistration_commit':
        raise RuntimeError('Training requires a valid, matching preregistration commit')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pins = {}
    for relative in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / relative
        stored = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if stored != path.read_bytes():
            raise RuntimeError(f'Uncommitted contract: {relative}')
        pins[relative] = sha(path)
    if config['execution_source_sha256'] != pins['executed/run.py'] or config['analysis_source_sha256'] != pins['analysis.py']:
        raise RuntimeError('Source pin mismatch')
    for relative, expected in config['input_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f'Old evidence changed: {relative}')
    return config, {'git_commit': commit, 'sha256': pins}


def saved_data(n, seed):
    source = OLD / 'executed' / f'data_n{n}_seed{seed}.npz'
    manifest = json.loads(source.with_suffix('.json').read_text())
    if sha(source) != manifest['file_sha256']:
        raise RuntimeError('Original data hash mismatch')
    with np.load(source) as raw:
        arrays = {k: raw[k].copy() for k in raw.files}
    for k, v in arrays.items():
        if hashlib.sha256(v.tobytes()).hexdigest() != manifest['arrays'][k]['sha256']:
            raise RuntimeError(f'Original array hash mismatch: {k}')
    if not np.array_equal(arrays['target_mean_rms'], np.array([0., np.sqrt(1.25)])):
        raise RuntimeError('Population normalization mismatch')
    for split in ['train', 'audit']:
        x = arrays[f'{split}_x']
        raw_y = x[:, 0] + .5 * x[:, 0] * x[:, 1]
        if not np.array_equal(raw_y / np.sqrt(1.25), arrays[f'{split}_y']):
            raise RuntimeError('Population target reconstruction mismatch')
    path = STUDY / 'executed' / source.name
    metadata = path.with_suffix('.json')
    array_pins = {k: {'sha256': hashlib.sha256(v.tobytes()).hexdigest(), 'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise RuntimeError('Partial data; refusing overwrite')
        previous = json.loads(metadata.read_text())
        if previous['file_sha256'] != sha(path) or previous['arrays'] != array_pins:
            raise RuntimeError('New saved data mismatch')
    else:
        np.savez_compressed(path, **arrays)
        save(metadata, {'n': n, 'seed': seed, 'source_file': source.relative_to(ROOT).as_posix(),
                        'source_sha256': sha(source), 'target_mean_rms': arrays['target_mean_rms'].tolist(),
                        'arrays': array_pins, 'file_sha256': sha(path)})
    return arrays, path


def measure(config, data, variance):
    x, tx = data['train_features'], data['audit_features']
    y, ty, eps = data['train_y'], data['audit_y'], data['epsilon']
    n, lr, steps = len(x), config['optimizer']['lr'], config['optimizer']['steps']
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum('ik,jk->ij', x, x) / n)
    q = 1 - 2 * lr * eigenvalues
    if q.min() < -1e-12 or q.max() > 1 + 1e-12:
        raise RuntimeError('Positive stable GD factors violated')
    gram = np.einsum('mi,mj->ij', tx, tx) / len(tx)
    projection = np.einsum('mi,m->i', tx, ty) / len(tx)
    square = float(np.mean(ty ** 2))
    spectral_map = np.einsum('ni,nj->ij', x, eigenvectors) / n
    basis_gram = np.einsum('ij,ik,kl->jl', spectral_map, gram, spectral_map, optimize=False)
    basis_target = np.einsum('i,ij->j', projection, spectral_map)
    signal_projection = np.einsum('ij,i->j', eigenvectors, y)
    targets = np.column_stack([y + variance ** .5 * eps, y, variance ** .5 * eps])
    weights = np.zeros((x.shape[1], 3))
    train_mse, bias, noise, spectral_bias = [np.empty(steps + 1) for _ in range(4)]
    factors = np.zeros(n)
    for step in range(steps + 1):
        residual = np.einsum('ni,ij->nj', x, weights) - targets
        train_mse[step] = np.mean(residual[:, 0] ** 2)
        w = weights[:, 1]
        bias[step] = np.einsum('i,ij,j->', w, gram, w) - 2 * np.dot(projection, w) + square
        coefficients = factors * signal_projection
        spectral_bias[step] = np.einsum('i,ij,j->', coefficients, basis_gram, coefficients) - 2 * np.dot(basis_target, coefficients) + square
        noise[step] = np.dot(np.diag(basis_gram), factors ** 2)
        if step < steps:
            weights -= 2 * lr * np.einsum('ni,nj->ij', x, residual) / n
            factors = q * factors + 2 * lr
    direct_final = float(np.mean((np.einsum('mi,i->m', tx, weights[:, 1]) - ty) ** 2))
    arrays = {'train_mse': train_mse, 'signal_bias': bias, 'variance_unit': noise,
              'expected_risk': bias + variance * noise, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors,
              'basis_gram': basis_gram, 'basis_target': basis_target, 'signal_projection': signal_projection,
              'audit_gram': gram, 'audit_target_projection': projection, 'audit_target_square': np.array(square),
              'final_heads': weights}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError('Nonfinite result; not saving cell')
    return arrays, {'max_train_increase': float(np.diff(train_mse).max()),
                    'max_spectral_signal_error': float(np.abs(bias - spectral_bias).max()),
                    'final_direct_bias_error': abs(direct_final - bias[-1]), 'q_min': float(q.min()), 'q_max': float(q.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error('Positive cell limit required')
    config, pins = committed_contract()
    results = STUDY / 'results'
    results.mkdir(exist_ok=True)
    receipt_path = results / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    started, previous, new = time.monotonic(), receipt['compute_seconds'], 0
    for recipe in config['recipes']:
        n, variance = recipe['n'], recipe['noise_variance']
        for seed in config['seeds']:
            data, data_path = saved_data(n, seed)
            cell = {'n': n, 'seed': seed, 'noise_variance': variance}
            label = f'n{n}_seed{seed}_var{variance:g}'
            path = results / f'{label}.json'
            contract = hashlib.sha256(json.dumps({'cell': cell, 'pins': pins['sha256'], 'data_sha256': sha(data_path)}, sort_keys=True).encode()).hexdigest()
            if path.exists() or path.with_suffix('.npz').exists() or label in receipt['cells']:
                if not path.exists() or not path.with_suffix('.npz').exists() or label not in receipt['cells']:
                    raise RuntimeError('Partial cell; refusing overwrite')
                row = json.loads(path.read_text())
                if sha(path) != receipt['cells'][label] or row['contract_sha256'] != contract or row['arrays_sha256'] != sha(path.with_suffix('.npz')):
                    raise RuntimeError(f'Saved cell mismatch: {label}')
                continue
            if previous + time.monotonic() - started >= config['compute_budget_seconds']:
                raise RuntimeError('Compute budget exhausted')
            cell_started = time.monotonic()
            start_time = datetime.now(timezone.utc).isoformat()
            arrays, audit = measure(config, data, variance)
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            save(path, {'cell': cell, 'status': 'completed', 'pins': pins, 'contract_sha256': contract,
                        'data_file': data_path.relative_to(STUDY).as_posix(), 'data_sha256': sha(data_path),
                        'arrays_sha256': sha(path.with_suffix('.npz')), 'audit': audit,
                        'started_at': start_time, 'finished_at': datetime.now(timezone.utc).isoformat(),
                        'seconds': time.monotonic() - cell_started, 'numpy': np.__version__})
            receipt['cells'][label] = sha(path)
            receipt['compute_seconds'] = previous + time.monotonic() - started
            save(receipt_path, receipt)
            new += 1
            print(json.dumps({'saved': label, 'completed': len(receipt['cells']), 'seconds': receipt['compute_seconds']}), flush=True)
            if args.max_new_cells is not None and new >= args.max_new_cells:
                return


if __name__ == '__main__':
    main()
