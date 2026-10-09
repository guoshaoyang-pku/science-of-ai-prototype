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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))


def contract():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    relative = (STUDY / 'preregistration.json').relative_to(ROOT).as_posix()
    changes = subprocess.check_output(['git', 'log', '--format=%H', '--', relative], cwd=ROOT, text=True).splitlines()
    if len(changes) != 1:
        raise RuntimeError('Require one unique preregistration commit')
    freeze = changes[0]
    subprocess.run(['git', 'merge-base', '--is-ancestor', freeze, 'HEAD'], cwd=ROOT, check=True)
    pins = {}
    for name in ['preregistration.json', *cfg['source_sha256']]:
        path = STUDY / name
        committed = subprocess.check_output(['git', 'show', freeze + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if committed != path.read_bytes():
            raise RuntimeError('Frozen contract changed: ' + name)
        pins[name] = sha(path)
    for name, digest in cfg['source_sha256'].items():
        if pins[name] != digest:
            raise RuntimeError('Source pin: ' + name)
    for name, digest in cfg['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Input pin: ' + name)
    audit = json.loads((STUDY / 'executed/input_audit.json').read_text())
    for name, value in audit['old_files'].items():
        path = ROOT / name
        if sha(path) != value['sha256'] or path.stat().st_mtime_ns != value['mtime_ns']:
            raise RuntimeError('Historical artifact changed: ' + name)
    return cfg, {'git_commit': freeze, 'sha256': pins}


def cell_label(seed):
    return f'n64_seed{seed}_eta0.1_B2_var1_exact'


def scan_results(cfg, pins):
    result_dir = STUDY / 'results'
    receipt_path = result_dir / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    allowed = {cell_label(c['seed']) for c in cfg['cells']}
    if not set(receipt['cells']) <= allowed:
        raise RuntimeError('Unexpected receipt cell')
    expected_files = {'receipt.json'} if receipt_path.exists() else set()
    if receipt['cells'] and not receipt_path.exists():
        raise RuntimeError('Missing receipt')
    manifest_path = STUDY / 'executed/source_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['pins'] != pins:
            raise RuntimeError('Manifest freeze mismatch')
    elif receipt['cells']:
        raise RuntimeError('Saved cells without manifest')
    for condition in cfg['cells']:
        label = cell_label(condition['seed'])
        path = result_dir / (label + '.json')
        npz = path.with_suffix('.npz')
        present = (path.exists(), npz.exists(), label in receipt['cells'])
        if any(present):
            if not all(present):
                raise RuntimeError('Orphan or partial cell: ' + label)
            metadata = json.loads(path.read_text())
            if (sha(path) != receipt['cells'][label] or sha(npz) != metadata['arrays_sha256']
                    or metadata['pins'] != pins or metadata['seed'] != condition['seed']
                    or metadata['data_sha256'] != sha(ROOT / condition['data'])
                    or metadata['baseline_sha256'] != sha(ROOT / condition['baseline'])):
                raise RuntimeError('Saved cell contract mismatch: ' + label)
            expected_files.update((path.name, npz.name))
        local = STUDY / 'executed' / Path(condition['data']).name
        if local.exists() and sha(local) != sha(ROOT / condition['data']):
            raise RuntimeError('Local data mismatch')
    if {p.name for p in result_dir.iterdir()} != expected_files:
        raise RuntimeError('Unexpected result files; refusing all computation')
    return receipt


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
    receipt = scan_results(cfg, pins)
    manifest = STUDY / 'executed/source_manifest.json'
    if not manifest.exists():
        save(manifest, {'pins': pins, 'numpy': np.__version__, 'dtype': 'float64', 'device': 'CPU',
                        'new_training': 0, 'source_basis_and_full': 'read saved B4 arrays; basis/q/qa C contiguous; do not recompute B4/B8/full',
                        'started_at_unix_ns': time.time_ns()})
    started = time.monotonic()
    previous = receipt['compute_seconds']
    for condition in cfg['cells']:
        seed = condition['seed']
        label = cell_label(seed)
        if label in receipt['cells']:
            print(json.dumps({'skipped_verified': label}), flush=True)
            continue
        if previous + time.monotonic() - started >= cfg['compute_budget_seconds']:
            raise RuntimeError('Compute budget reached')
        source = ROOT / condition['data']
        local = STUDY / 'executed' / source.name
        if not local.exists():
            local.write_bytes(source.read_bytes())
        cell_started = time.monotonic()
        with np.load(local) as data:
            x, xa = data['train_features'], data['audit_features']
            labels, audit_y = data['train_y'] + data['epsilon'], data['audit_y']
        with np.load(ROOT / condition['baseline']) as baseline:
            basis = np.array(baseline['basis'], dtype=np.float64, order='C', copy=True)
            singular = baseline['singular_values'].copy()
            full_risk = baseline['full_risk'].copy()
            full_mean = baseline['full_mean_coordinates'].copy()
            full_covariance = baseline['full_covariance_maxabs'].copy()
        q = np.ascontiguousarray(product(x, basis))
        qa = np.ascontiguousarray(product(xa, basis))
        if not (basis.flags.c_contiguous and q.flags.c_contiguous and qa.flags.c_contiguous):
            raise RuntimeError('Registered C layout not met')
        arrays = moments(q, qa, labels, audit_y, cfg['eta'], cfg['batch'], cfg['steps'])
        arrays.update({'basis': basis, 'singular_values': singular, 'full_risk': full_risk,
                       'full_mean_coordinates': full_mean, 'full_covariance_maxabs': full_covariance})
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Nonfinite result')
        path = STUDY / 'results' / (label + '.json')
        npz = path.with_suffix('.npz')
        np.savez_compressed(npz, **arrays)
        save(path, {'seed': seed, 'status': 'completed', 'batch': cfg['batch'], 'steps': cfg['steps'],
                    'method': '独立无放回batch条件矩；复用B4保存basis与full，不重算旧成功cell',
                    'rank': basis.shape[1], 'new_training': 0, 'layout': {'basis': 'C', 'q': 'C', 'qa': 'C'}, 'pins': pins, 'data_sha256': sha(local),
                    'baseline_sha256': sha(ROOT / condition['baseline']), 'arrays_sha256': sha(npz),
                    'started_at_unix_ns': time.time_ns() - int((time.monotonic()-cell_started)*1e9),
                    'completed_at_unix_ns': time.time_ns(), 'seconds': time.monotonic() - cell_started})
        receipt['cells'][label] = sha(path)
        receipt['compute_seconds'] = previous + time.monotonic() - started
        save(STUDY / 'results/receipt.json', receipt)
        print(json.dumps({'saved': label, 'rank': basis.shape[1], 'seconds': receipt['compute_seconds']}), flush=True)


if __name__ == '__main__':
    main()
