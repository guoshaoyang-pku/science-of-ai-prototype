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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    cfg, pins = contract()
    receipt_path = STUDY / 'results/receipt.json'
    rec = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    started, previous, new = time.monotonic(), rec['compute_seconds'], 0
    manifest = STUDY / 'executed/source_manifest.json'
    if not manifest.exists():
        save(manifest, {**pins, 'numpy': np.__version__, 'device': 'CPU', 'dtype': 'float64'})
    for condition in cfg['cells']:
        seed = condition['seed']
        src = ROOT / condition['data']
        local = STUDY / 'executed' / src.name
        if local.exists() and sha(local) != sha(src):
            raise RuntimeError('Local data mismatch')
        if not local.exists():
            local.write_bytes(src.read_bytes())
        label = f'n64_seed{seed}_eta0.1_B8_var1_r64_127_M64'
        path = STUDY / 'results' / (label + '.json')
        npz = path.with_suffix('.npz')
        if path.exists() or npz.exists() or label in rec['cells']:
            if not (path.exists() and npz.exists() and label in rec['cells']):
                raise RuntimeError('Partial cell, refusing overwrite: ' + label)
            row = json.loads(path.read_text())
            if (sha(path) != rec['cells'][label] or sha(npz) != row['arrays_sha256']
                    or row['data_sha256'] != sha(local) or row['pins']['sha256'] != pins['sha256']):
                raise RuntimeError('Saved cell hash mismatch: ' + label)
            print(json.dumps({'skipped_verified': label}), flush=True)
            continue
        if previous + time.monotonic() - started >= cfg['compute_budget_seconds']:
            raise RuntimeError('Compute budget reached')
        cell_start = time.monotonic()
        with np.load(local) as data:
            x, y = data['train_features'], data['train_y'] + data['epsilon']
            xa, ya = data['audit_features'], data['audit_y']
        risks = np.empty((64, cfg['steps'] + 1))
        heads = np.empty((64, x.shape[1]))
        indices = np.asarray(cfg['new_replicate_indices'])
        seeds = cfg['batch_seed_base'] + 100000 * seed + 9000 + indices
        with np.load(ROOT / condition['old_block']) as old:
            if np.intersect1d(seeds, old['batch_seeds']).size:
                raise RuntimeError('Batch streams overlap')
        for r in range(64):
            if previous + time.monotonic() - started >= cfg['compute_budget_seconds']:
                raise RuntimeError('Compute budget reached')
            rng = np.random.default_rng(int(seeds[r]))
            w = np.zeros(x.shape[1])
            for t in range(cfg['steps'] + 1):
                residual = xa @ w - ya
                risks[r, t] = np.mean(residual * residual)
                if t < cfg['steps']:
                    idx = rng.choice(len(x), size=cfg['batch'], replace=False)
                    w -= 2 * cfg['eta'] * (x[idx].T @ (x[idx] @ w - y[idx])) / cfg['batch']
            heads[r] = w
        arrays = {'replicate_risk': risks, 'mean_risk': risks.mean(0),
                  'standard_error': risks.std(0, ddof=1) / 8,
                  'final_heads': heads, 'batch_seeds': seeds, 'replicate_indices': indices}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError('Nonfinite cell')
        np.savez_compressed(npz, **arrays)
        save(path, {'cell': {'n': 64, 'seed': seed, 'lr': cfg['eta'], 'batch': cfg['batch'], 'noise_variance': 1},
                    'status': 'completed', 'steps': cfg['steps'], 'replicates': 64,
                    'new_replicates': 64, 'retained_replicates': 0, 'pins': pins,
                    'data_sha256': sha(local), 'old_arrays_sha256': sha(ROOT / condition['old_block']),
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
