#!/usr/bin/env python3
"""Independent direct-gradient checks for selected saved D2 curves."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

SOURCE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d2_run', SOURCE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
STUDY = runner.STUDY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    hash_count = 0
    for receipt in (STUDY / 'results').glob('*.json'):
        row = json.loads(receipt.read_text())
        for relative, pin in row['files'].items():
            if digest(STUDY / relative) != pin:
                raise RuntimeError(f'Changed result: {relative}')
            hash_count += 1
    selected = [r for r in summary['rows'] if r['n'] == 64 and r['seed'] == 6101 and r['sigma2'] == 1 and r['task'] in ('interaction', 'vector', 'quadratic', 'linear')]
    records = []
    for row in selected:
        with np.load(STUDY / row['data']) as saved:
            data = {k: saved[k] for k in saved.files}
        with np.load(STUDY / row['curve']) as saved:
            curve = {k: saved[k] for k in saved.files}
        phi, val_phi = data['phi'], data['val_phi']
        y = runner.targets(data['train_x'], row['task'])
        vy = runner.targets(data['val_x'], row['task'])
        k = y.shape[1]
        noisy = y + data['epsilon'][:, :k]
        noisy_val_y = vy + data['val_epsilon'][:, :k]
        w = np.zeros((phi.shape[1], k))
        clean_w = np.zeros_like(w)
        response = np.zeros((phi.shape[1], len(phi)))
        eye = np.eye(len(phi))
        checks = []
        for step in range(257):
            if step in (0, 1, 16, 64, 256):
                idx = int(np.where(curve['t'] == step)[0][0])
                tr = float(np.mean((np.dot(phi, w) - noisy) ** 2))
                cv = float(np.mean((np.dot(val_phi, w) - vy) ** 2))
                nv = float(np.mean((np.dot(val_phi, w) - noisy_val_y) ** 2))
                bias = float(np.mean((np.dot(val_phi, clean_w) - vy) ** 2))
                unit_noise = float(np.sum(np.dot(val_phi, response) ** 2) / len(val_phi))
                train_bias = float(np.mean((np.dot(phi, clean_w) - y) ** 2))
                expected_train = train_bias + float(np.sum((np.dot(phi, response) - eye) ** 2) / len(phi))
                expected = bias + unit_noise
                values = {'realized_noisy_train': tr, 'realized_clean_val': cv, 'realized_noisy_val': nv, 'signal_bias': bias, 'variance_unit': unit_noise, 'expected_noisy_train': expected_train, 'expected_clean_val': expected}
                errors = {key: abs(value - float(curve[key][idx])) for key, value in values.items()}
                if max(errors.values()) > 1e-8:
                    raise RuntimeError(f'Direct gate failed: {row["protocol"]}/{row["task"]}/{step}: {errors}')
                checks.append({'step': step, 'errors': errors})
            if step < 256:
                w -= .6 * np.dot(phi.T, np.dot(phi, w) - noisy) / len(phi)
                clean_w -= .6 * np.dot(phi.T, np.dot(phi, clean_w) - y) / len(phi)
                response -= .6 * np.dot(phi.T, np.dot(phi, response) - eye) / len(phi)
        records.append({'protocol': row['protocol'], 'task': row['task'], 'outputs': k, 'n': row['n'], 'checks': checks})
    errors = [e for rec in records for check in rec['checks'] for e in check['errors'].values()]
    for row in summary['rows']:
        if not row['interior'] and row['linear_factor_error'] is not None:
            raise RuntimeError('Boundary counted as an interior scaling point')
    total_expanded = sum(t['models']['linear']['expanded']['total'] for t in summary['tasks'])
    if total_expanded != 2700:
        raise RuntimeError('Expanded denominator mismatch')
    result = {'status': 'verified', 'hashes_checked': hash_count, 'direct_gd_selected_cells': len(records), 'direct_steps_per_cell': 256, 'max_direct_error': max(errors), 'expanded_denominator': total_expanded, 'boundary_cells_retained': summary['counts']['boundary_minima'], 'checks': records, 'known_approximation': 'Eigenvalues <= max(lambda)*1e-12 treated as null; minimum located on registered dense grid then locally refined. No claim of all continuous-time extrema.'}
    (STUDY / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
