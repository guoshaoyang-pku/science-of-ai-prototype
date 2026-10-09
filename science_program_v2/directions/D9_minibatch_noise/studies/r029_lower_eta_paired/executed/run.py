import argparse
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
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def contract():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pins = {}
    for name in ['preregistration.json', *cfg['source_sha256']]:
        path = STUDY / name
        stored = subprocess.check_output(['git', 'show', head + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if stored != path.read_bytes():
            raise RuntimeError('Uncommitted contract: ' + name)
        pins[name] = sha(path)
    for name, digest in cfg['source_sha256'].items():
        if pins[name] != digest:
            raise RuntimeError('Source pin: ' + name)
    for name, digest in cfg['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Input pin: ' + name)
    return cfg, {'git_commit': head, 'sha256': pins}


def measure(data, batch, variance, lr, steps, cfg, seed):
    x, y = data['train_features'], data['train_y']
    xa, ya = data['audit_features'], data['audit_y']
    n, p = x.shape
    target = y + np.sqrt(variance) * data['epsilon']
    reps = 1 if batch == n else cfg['replicates']
    out = np.empty((reps, steps + 1))
    heads, batch_seeds = [], []
    for r in range(reps):
        batch_seed = cfg['batch_seed_base'] + 100000 * seed + 1000 * batch + 10 * int(variance * 100) + r
        batch_seeds.append(batch_seed)
        rng = np.random.default_rng(batch_seed)
        w = np.zeros(p)
        for t in range(steps + 1):
            residual = xa @ w - ya
            out[r, t] = np.mean(residual * residual)
            if t < steps:
                idx = np.arange(n) if batch == n else rng.choice(n, size=batch, replace=False)
                w -= 2 * lr * (x[idx].T @ (x[idx] @ w - target[idx])) / batch
        heads.append(w)
    arrays = {'replicate_risk': out, 'mean_risk': out.mean(0),
              'standard_error': np.zeros(steps + 1) if reps == 1 else out.std(0, ddof=1) / np.sqrt(reps),
              'final_heads': np.asarray(heads), 'batch_seeds': np.asarray(batch_seeds)}
    if batch == n:
        eigenvalues, vectors = np.linalg.eigh(x @ x.T / n)
        q = 1 - 2 * lr * eigenvalues
        spectral_map = x.T @ vectors / n
        audit_gram = xa.T @ xa / len(xa)
        basis_gram = spectral_map.T @ audit_gram @ spectral_map
        factors = np.zeros(n)
        signal_w = np.zeros(p)
        signal_bias = np.empty(steps + 1)
        variance_unit = np.empty(steps + 1)
        for t in range(steps + 1):
            residual = xa @ signal_w - ya
            signal_bias[t] = np.mean(residual * residual)
            variance_unit[t] = np.diag(basis_gram) @ (factors * factors)
            if t < steps:
                signal_w -= 2 * lr * x.T @ (x @ signal_w - y) / n
                factors = q * factors + 2 * lr
        arrays.update(signal_bias=signal_bias, variance_unit=variance_unit,
                      expected_risk=signal_bias + variance * variance_unit,
                      eigenvalues=eigenvalues, final_signal_head=signal_w)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError('Nonfinite cell')
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    cfg, pins = contract()
    receipt_path = STUDY / 'results/receipt.json'
    rec = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    started, previous, new = time.monotonic(), rec['compute_seconds'], 0
    manifest_path = STUDY / 'executed/source_manifest.json'
    if not manifest_path.exists():
        save(manifest_path, {**pins, 'numpy': np.__version__, 'device': 'CPU', 'dtype': 'float64'})
    for seed in cfg['seeds']:
        src = ROOT / cfg['data_files'][str(seed)]
        local = STUDY / 'executed' / src.name
        if local.exists() and sha(local) != sha(src):
            raise RuntimeError('Local data mismatch')
        if not local.exists():
            local.write_bytes(src.read_bytes())
        with np.load(local) as z:
            data = {k: z[k] for k in z.files}
        for condition in cfg['conditions']:
            lr, steps = condition['lr'], condition['steps']
            for batch in condition['batches']:
                for variance in cfg['variances']:
                    label = f'n64_seed{seed}_eta{lr:g}_B{batch}_var{variance:g}'
                    path = STUDY / 'results' / (label + '.json')
                    npz = path.with_suffix('.npz')
                    if path.exists() or npz.exists() or label in rec['cells']:
                        if not (path.exists() and npz.exists() and label in rec['cells']):
                            raise RuntimeError('Partial cell, refusing overwrite: ' + label)
                        row = json.loads(path.read_text())
                        if (sha(path) != rec['cells'][label] or sha(npz) != row['arrays_sha256']
                                or row['data_sha256'] != sha(local) or row['pins']['sha256'] != pins['sha256']):
                            raise RuntimeError('Saved cell hash mismatch: ' + label)
                        continue
                    if previous + time.monotonic() - started >= cfg['compute_budget_seconds']:
                        raise RuntimeError('Compute budget reached')
                    cell_start = time.monotonic()
                    arrays = measure(data, batch, variance, lr, steps, cfg, seed)
                    np.savez_compressed(npz, **arrays)
                    save(path, {'cell': {'n': 64, 'seed': seed, 'lr': lr, 'batch': batch, 'noise_variance': variance},
                                'status': 'completed', 'pins': pins, 'steps': steps,
                                'replicates': len(arrays['replicate_risk']), 'data_sha256': sha(local),
                                'arrays_sha256': sha(npz), 'seconds': time.monotonic() - cell_start})
                    rec['cells'][label] = sha(path)
                    rec['compute_seconds'] = previous + time.monotonic() - started
                    save(receipt_path, rec)
                    new += 1
                    print(json.dumps({'saved': label, 'completed': len(rec['cells']), 'seconds': rec['compute_seconds']}), flush=True)
                    if args.max_new_cells and new >= args.max_new_cells:
                        return


if __name__ == '__main__':
    main()
