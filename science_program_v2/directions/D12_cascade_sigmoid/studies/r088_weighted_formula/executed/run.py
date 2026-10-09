import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write(chr(10))


def check_frozen(commit):
    if subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip() != commit:
        raise RuntimeError('Full preregistration commit required')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    pins = {str(prereg_path.relative_to(ROOT)): sha(prereg_path), **prereg['source_hashes']}
    for relative, expected in pins.items():
        blob = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected or sha(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen source/registration mismatch: {relative}')
    for pin in prereg['input_manifest']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise RuntimeError(f'Input hash mismatch: {pin["path"]}')
    audit = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    for pin in audit['previous_study_file_manifest']:
        path = ROOT / pin['path']
        if sha(path) != pin['sha256'] or path.stat().st_mtime_ns != pin['mtime_ns']:
            raise RuntimeError(f'Historical file changed: {pin["path"]}')
    return prereg


def contract_for(prereg, request, commit):
    return {'request': request, 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
            'preregistration_commit': commit, 'source_hashes': prereg['source_hashes']}


def scan_results(prereg, commit):
    requests = {request['cell_id']: request for request in prereg['cells']}
    receipt_paths = sorted((STUDY / 'executed').glob('run_receipt_*.json'))
    if [p.name for p in receipt_paths] != [f'run_receipt_{i:03d}.json' for i in range(1, len(receipt_paths) + 1)]:
        raise RuntimeError('Nonconsecutive receipt names')
    pins = {}
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        if receipt['status'] != 'success' or receipt['preregistration_commit'] != commit:
            raise RuntimeError('Mixed commit or failed receipt')
        items = receipt['new_successes'] + receipt['resumed_successes']
        if len(items) != len({item['cell_id'] for item in items}):
            raise RuntimeError('Duplicate cell in receipt')
        for item in items:
            cell_id = item['cell_id']
            if cell_id not in requests or (cell_id in pins and pins[cell_id] != item):
                raise RuntimeError('Unknown cell or conflicting receipt')
            pins[cell_id] = item
    results = STUDY / 'results'
    directories = list(results.iterdir()) if results.exists() else []
    if any(not path.is_dir() or path.name not in requests for path in directories):
        raise RuntimeError('Extra results file/directory')
    if {path.name for path in directories} != set(pins):
        raise RuntimeError('Orphaned result or receipt; stop without overwrite')
    for path in directories:
        if {child.name for child in path.iterdir()} != {'arrays.npz', 'evaluation.json'}:
            raise RuntimeError('Incomplete/extra cell files')
        pin = pins[path.name]
        if sha(path / 'arrays.npz') != pin['arrays_sha256'] or sha(path / 'evaluation.json') != pin['evaluation_sha256']:
            raise RuntimeError('Result hash mismatch')
        row = json.loads((path / 'evaluation.json').read_text())
        if (row['status'] != 'success' or row['cell_id'] != path.name
                or row['arrays_sha256'] != pin['arrays_sha256']
                or row['contract'] != contract_for(prereg, requests[path.name], commit)):
            raise RuntimeError('Saved evaluation contract mismatch')
    return pins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    prereg = check_frozen(args.preregistration_commit)
    prior_pins = scan_results(prereg, args.preregistration_commit)
    original_summary = json.loads((ROOT / prereg['previous_summary']).read_text())
    old_rows = {r['cell_id']: r for r in original_summary['cells']}
    grid = np.array(prereg['evaluation']['grid'], dtype=np.int64)
    if not np.array_equal(grid, original_summary['grid']) or len(grid) != 150 or grid[-1] != 4997:
        raise RuntimeError('Original grid changed')
    new_successes, resumed_successes = [], []
    for request in prereg['cells']:
        if time.monotonic() - started > prereg['compute_budget_seconds']:
            raise RuntimeError('Compute budget exhausted')
        cell_id = request['cell_id']
        if cell_id in prior_pins:
            resumed_successes.append(prior_pins[cell_id])
            continue
        metadata = json.loads((ROOT / request['metadata_path']).read_text())
        wanted = {'ratio': 100, 'alpha': .25, 'seed': request['seed'], 'steps': 10000,
                  'learning_rate': .1, 'slow_eigenvalue': .01}
        if (metadata['status'] != 'success' or metadata['cell_id'] != cell_id
                or any(metadata['request'][key] != value for key, value in wanted.items())
                or metadata['arrays_sha256'] != sha(ROOT / request['arrays_path'])):
            raise RuntimeError('Saved training contract mismatch')
        with np.load(ROOT / request['arrays_path'], allow_pickle=False) as saved:
            eigenvalues = saved['eigenvalues'].copy()
            weights = saved['mode_weights'].copy()
            loss = saved['normalized_loss'][grid].copy()
            if (not np.array_equal(saved['step'], np.arange(10001))
                    or not np.array_equal(eigenvalues, [.01, 1.])
                    or not np.allclose(weights, [.25, .75], rtol=0, atol=1e-14)
                    or abs(float(weights.sum()) - 1) > 1e-14
                    or not np.isfinite(loss).all() or abs(float(loss[0]) - 1) > 1e-14):
                raise RuntimeError('Saved spectrum/weight/loss mismatch')
        rates = -2 * np.log1p(-prereg['learning_rate'] * eigenvalues)
        half_times = np.log(2) / rates
        slopes = np.full(2, 2 * np.log(2))
        components = 1 / (1 + (grid[:, None] / half_times[None, :]) ** slopes[None, :])
        prediction = (components * weights[None, :]).sum(axis=1)
        residual = prediction - loss
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        maxabs = float(np.max(np.abs(residual)))
        destination = STUDY / 'results' / cell_id
        destination.mkdir(parents=True)
        arrays_path, result_path = destination / 'arrays.npz', destination / 'evaluation.json'
        with arrays_path.open('xb') as handle:
            np.savez_compressed(handle, grid=grid, observed=loss, eigenvalues=eigenvalues,
                                weights=weights, rates=rates, half_times=half_times, slopes=slopes,
                                components=components, prediction=prediction, residual=residual)
        control = old_rows[cell_id]['best_errors']
        result = {'status': 'success', 'cell_id': cell_id, 'ratio': 100, 'alpha': .25,
                  'seed': request['seed'], 'contract': contract_for(prereg, request, args.preregistration_commit),
                  'completed_at': datetime.now().astimezone().isoformat(), 'arrays_sha256': sha(arrays_path),
                  'rmse': rmse, 'maxabs': maxabs, 'adequate': bool(rmse <= .03 and maxabs <= .05),
                  'maxabs_step': int(grid[np.argmax(np.abs(residual))]),
                  'half_times_slow_fast': half_times.tolist(), 'slopes': slopes.tolist(),
                  'amplitudes_slow_fast': weights.tolist(), 'fitted_control': control,
                  'paired_formula_minus_fitted': {m: value - control[m] for m, value in [('rmse', rmse), ('maxabs', maxabs)]},
                  'new_training_cells': 0, 'new_fits': 0}
        write_json(result_path, result)
        new_successes.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path),
                              'evaluation_sha256': sha(result_path)})
        print(json.dumps({k: result[k] for k in ['cell_id', 'rmse', 'maxabs', 'adequate']}), flush=True)
    number = len(list((STUDY / 'executed').glob('run_receipt_*.json'))) + 1
    write_json(STUDY / 'executed' / f'run_receipt_{number:03d}.json',
               {'status': 'success', 'preregistration_commit': args.preregistration_commit,
                'elapsed_seconds': time.monotonic() - started, 'new_successes': new_successes,
                'resumed_successes': resumed_successes, 'new_training_cells': 0, 'new_fits': 0})


if __name__ == '__main__':
    main()
