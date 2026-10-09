import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def contract():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pins = {}
    for name in ['preregistration.json', *cfg['source_sha256']]:
        path = STUDY / name
        committed = subprocess.check_output(['git', 'show', head + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if committed != path.read_bytes():
            raise RuntimeError('Uncommitted contract: ' + name)
        pins[name] = sha(path)
    for name, digest in cfg['source_sha256'].items():
        if pins[name] != digest:
            raise RuntimeError('Source pin: ' + name)
    for name, digest in cfg['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Input pin: ' + name)
    return cfg, {'git_commit': head, 'sha256': pins}


def product(left, right):
    return np.einsum('ij,jk->ik', left, right, optimize=False)


def moments(q, qa, labels, audit_y, eta, batch, steps):
    n, rank = q.shape
    rate = 2 * eta
    gamma = (n - batch) / (batch * (n - 1))
    h = product(q.T, q) / n
    b = np.einsum('ij,i->j', q, labels) / n
    a = np.eye(rank) - rate * h
    audit_h = product(qa.T, qa) / len(qa)
    mean = np.zeros(rank)
    covariance = np.zeros((rank, rank))
    means = np.empty((steps + 1, rank))
    covariances = np.empty((steps + 1, rank, rank))
    mean_risk = np.empty(steps + 1)
    covariance_risk = np.empty(steps + 1)
    minimum_eigenvalue = np.empty(steps + 1)
    for t in range(steps + 1):
        means[t], covariances[t] = mean, covariance
        residual_audit = np.einsum('ij,j->i', qa, mean) - audit_y
        mean_risk[t] = np.mean(residual_audit ** 2)
        covariance_risk[t] = np.einsum('ij,ji->', audit_h, covariance)
        minimum_eigenvalue[t] = np.linalg.eigvalsh(covariance)[0]
        if t == steps:
            break
        residual = np.einsum('ij,j->i', q, mean) - labels
        diagonal = np.einsum('ij,jk,ik->i', q, covariance, q) + residual ** 2
        gradient = np.einsum('ij,j->i', h, mean) - b
        gradient_variance = product(q.T * diagonal, q) / n
        gradient_variance -= product(product(h, covariance), h) + np.outer(gradient, gradient)
        covariance = product(product(a, covariance), a.T) + rate ** 2 * gamma * gradient_variance
        covariance = (covariance + covariance.T) / 2
        mean = np.einsum('ij,j->i', a, mean) + rate * b
    return {'mean_coordinates': means, 'covariance_coordinates': covariances,
            'mean_risk': mean_risk, 'covariance_risk': covariance_risk,
            'expected_risk': mean_risk + covariance_risk, 'minimum_covariance_eigenvalue': minimum_eigenvalue}


def main():
    cfg, pins = contract()
    receipt_path = STUDY / 'results/receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    manifest = STUDY / 'executed/source_manifest.json'
    if not manifest.exists():
        save(manifest, {**pins, 'numpy': np.__version__, 'dtype': 'float64', 'device': 'CPU', 'new_training': 0})
    started = time.monotonic()
    previous = receipt['compute_seconds']
    for condition in cfg['cells']:
        seed = condition['seed']
        source = ROOT / condition['data']
        local = STUDY / 'executed' / source.name
        if local.exists() and sha(local) != sha(source):
            raise RuntimeError('Local data mismatch')
        if not local.exists():
            local.write_bytes(source.read_bytes())
        label = f'n64_seed{seed}_eta0.1_B8_var1_exact'
        path = STUDY / 'results' / (label + '.json')
        npz = path.with_suffix('.npz')
        if path.exists() or npz.exists() or label in receipt['cells']:
            if not (path.exists() and npz.exists() and label in receipt['cells']):
                raise RuntimeError('Partial cell; refusing overwrite')
            metadata = json.loads(path.read_text())
            if (sha(path) != receipt['cells'][label] or sha(npz) != metadata['arrays_sha256']
                    or metadata['data_sha256'] != sha(local) or metadata['pins']['sha256'] != pins['sha256']):
                raise RuntimeError('Saved cell hash mismatch')
            print(json.dumps({'skipped_verified': label}), flush=True)
            continue
        if previous + time.monotonic() - started >= cfg['compute_budget_seconds']:
            raise RuntimeError('Compute budget reached')
        cell_started = time.monotonic()
        with np.load(local) as data:
            x, xa = data['train_features'], data['audit_features']
            labels, audit_y = data['train_y'] + data['epsilon'], data['audit_y']
        _, singular, vh = np.linalg.svd(x, full_matrices=False)
        tolerance = np.finfo(float).eps * max(x.shape) * singular[0]
        rank = int(np.sum(singular > tolerance))
        basis = vh[:rank].T
        q, qa = product(x, basis), product(xa, basis)
        arrays = moments(q, qa, labels, audit_y, cfg['eta'], cfg['batch'], cfg['steps'])
        full = moments(q, qa, labels, audit_y, cfg['eta'], len(x), cfg['steps'])
        arrays.update({'basis': basis, 'singular_values': singular, 'full_risk': full['expected_risk'],
                       'full_mean_coordinates': full['mean_coordinates'],
                       'full_covariance_maxabs': np.array(np.max(np.abs(full['covariance_coordinates'])))})
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Nonfinite result')
        np.savez_compressed(npz, **arrays)
        save(path, {'seed': seed, 'status': 'completed', 'method': '独立无放回batch精确一二阶矩；float64训练span',
                    'rank': rank, 'svd_tolerance': float(tolerance), 'new_training': 0, 'steps': cfg['steps'],
                    'pins': pins, 'data_sha256': sha(local), 'arrays_sha256': sha(npz),
                    'seconds': time.monotonic() - cell_started})
        receipt['cells'][label] = sha(path)
        receipt['compute_seconds'] = previous + time.monotonic() - started
        save(receipt_path, receipt)
        print(json.dumps({'saved': label, 'rank': rank, 'seconds': receipt['compute_seconds']}), flush=True)


if __name__ == '__main__':
    main()
