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
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    subprocess.run(['git', 'merge-base', '--is-ancestor', args.preregistration_commit, 'HEAD'],
                   cwd=ROOT, check=True)
    frozen = {str(prereg_path.relative_to(ROOT)): sha(prereg_path), **prereg['source_hashes']}
    for relative, expected in frozen.items():
        committed = subprocess.check_output(
            ['git', 'show', f'{args.preregistration_commit}:{relative}'], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected or sha(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen source/registration mismatch: {relative}')
    for pin in prereg['input_manifest']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise RuntimeError(f'Input hash mismatch: {pin["path"]}')
    old_summary = json.loads((ROOT / prereg['previous_summary']).read_text())
    grid = np.array(prereg['evaluation']['grid'], dtype=np.int64)
    if not np.array_equal(grid, old_summary['grid']) or len(grid) != 150 or grid[-1] != 4997:
        raise RuntimeError('Original grid changed')
    old_rows = {r['cell_id']: r for r in old_summary['cells']}
    prior_pins = {}
    for receipt_path in sorted((STUDY / 'executed').glob('run_receipt_*.json')):
        receipt = json.loads(receipt_path.read_text())
        for item in receipt['new_successes'] + receipt['resumed_successes']:
            if item['cell_id'] in prior_pins and prior_pins[item['cell_id']] != item:
                raise RuntimeError('Conflicting result receipt')
            prior_pins[item['cell_id']] = item
    new_successes, resumed_successes = [], []
    for request in prereg['cells']:
        if time.monotonic() - started > prereg['compute_budget_seconds']:
            raise RuntimeError('Compute budget exhausted')
        cell_id = request['cell_id']
        destination = STUDY / 'results' / cell_id
        result_path, arrays_path = destination / 'evaluation.json', destination / 'arrays.npz'
        contract = {'request': request, 'preregistration_sha256': sha(prereg_path),
                    'preregistration_commit': args.preregistration_commit,
                    'source_hashes': prereg['source_hashes']}
        if destination.exists():
            pin = prior_pins.get(cell_id)
            if (pin is None or not result_path.is_file() or not arrays_path.is_file()
                    or sha(result_path) != pin['evaluation_sha256']
                    or sha(arrays_path) != pin['arrays_sha256']
                    or json.loads(result_path.read_text())['contract'] != contract):
                raise RuntimeError(f'Unverified existing cell; do not overwrite: {cell_id}')
            resumed_successes.append(pin)
            continue
        metadata = json.loads((ROOT / request['metadata_path']).read_text())
        if (metadata['status'] != 'success' or metadata['request']['ratio'] != request['ratio']
                or metadata['request']['seed'] != request['seed']
                or metadata['arrays_sha256'] != sha(ROOT / request['arrays_path'])):
            raise RuntimeError('Saved training contract mismatch')
        with np.load(ROOT / request['arrays_path'], allow_pickle=False) as saved:
            eigenvalues = saved['eigenvalues'].copy()
            loss = saved['normalized_loss'][grid].copy()
            if (not np.array_equal(eigenvalues, .01 * np.array([1., request['ratio']]))
                    or not np.isfinite(loss).all()):
                raise RuntimeError('Unexpected saved spectrum/loss')
        rates = -2 * np.log1p(-prereg['learning_rate'] * eigenvalues)
        half_times = np.log(2) / rates
        slopes = np.full(2, 2 * np.log(2))
        components = 1 / (1 + (grid[:, None] / half_times[None, :]) ** slopes[None, :])
        prediction = .5 * components.sum(axis=1)
        residual = prediction - loss
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        maxabs = float(np.max(np.abs(residual)))
        exact_components = np.exp(-grid[:, None] * rates[None, :])
        destination.mkdir(parents=True)
        with arrays_path.open('xb') as handle:
            np.savez_compressed(handle, grid=grid, observed=loss, eigenvalues=eigenvalues,
                                rates=rates, half_times=half_times, slopes=slopes,
                                components=components, exact_components=exact_components,
                                prediction=prediction, residual=residual)
        result = {'status': 'success', 'cell_id': cell_id, 'ratio': request['ratio'],
                  'seed': request['seed'], 'contract': contract,
                  'completed_at': datetime.now().astimezone().isoformat(),
                  'arrays_sha256': sha(arrays_path), 'rmse': rmse, 'maxabs': maxabs,
                  'adequate': bool(rmse <= .03 and maxabs <= .05),
                  'maxabs_step': int(grid[np.argmax(np.abs(residual))]),
                  'half_times_slow_fast': half_times.tolist(), 'slopes': slopes.tolist(),
                  'fitted_control': old_rows[cell_id]['best_errors'],
                  'paired_formula_minus_fitted': {m: value - old_rows[cell_id]['best_errors'][m]
                                                   for m, value in [('rmse', rmse), ('maxabs', maxabs)]},
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
