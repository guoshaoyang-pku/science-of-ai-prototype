import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def curve(log_half, beta, grid):
    out = np.ones(len(grid))
    positive = grid > 0
    z = beta * (np.log(grid[positive]) - log_half)
    out[positive] = np.exp(-np.logaddexp(0.0, z))
    return out


def feasible(epsilon, grid, values, bounds):
    if np.max(np.abs(values[grid == 0] - 1)) > epsilon:
        return None, {'reason': 'time_zero'}
    positive = grid > 0
    x, y = np.log(grid[positive]), values[positive]
    lower, upper = y - epsilon, y + epsilon
    if np.any(lower >= 1) or np.any(upper <= 0):
        return None, {'reason': 'platform_range'}
    lower_mask, upper_mask = lower > 0, upper < 1
    lower_slopes = np.r_[bounds['log_half'][0], x[lower_mask]]
    lower_intercepts = np.r_[0., np.log(lower[lower_mask]) - np.log1p(-lower[lower_mask])]
    upper_slopes = np.r_[bounds['log_half'][1], x[upper_mask]]
    upper_intercepts = np.r_[0., np.log(upper[upper_mask]) - np.log1p(-upper[upper_mask])]
    coefficients = lower_slopes[:, None] - upper_slopes[None, :]
    right = upper_intercepts[None, :] - lower_intercepts[:, None]
    zero = coefficients == 0
    if np.any(zero & (right < 0)):
        return None, {'reason': 'parallel_constraints'}
    beta_low, beta_high = bounds['beta']
    positive_coefficient, negative_coefficient = coefficients > 0, coefficients < 0
    if np.any(positive_coefficient):
        beta_high = min(beta_high, float(np.min(right[positive_coefficient] / coefficients[positive_coefficient])))
    if np.any(negative_coefficient):
        beta_low = max(beta_low, float(np.max(right[negative_coefficient] / coefficients[negative_coefficient])))
    audit = {'beta_low': beta_low, 'beta_high': beta_high}
    if beta_low > beta_high:
        return None, {**audit, 'reason': 'empty_beta_interval'}
    beta = (beta_low + beta_high) / 2
    c_low = float(np.max(lower_slopes * beta + lower_intercepts))
    c_high = float(np.min(upper_slopes * beta + upper_intercepts))
    if c_low > c_high:
        return None, {**audit, 'reason': 'roundoff_empty_c_interval', 'c_low': c_low, 'c_high': c_high}
    return ((c_low + c_high) / (2 * beta), beta), {**audit, 'c_low': c_low, 'c_high': c_high}


def minimax(grid, values, bounds, tolerance):
    low, high = 0., 1.
    witness, high_audit = feasible(high, grid, values, bounds)
    if witness is None:
        raise RuntimeError('Initial upper bound infeasible')
    trace = []
    while high - low > tolerance:
        mid = (low + high) / 2
        candidate, audit = feasible(mid, grid, values, bounds)
        trace.append({'epsilon': mid, 'feasible': candidate is not None, **audit})
        if candidate is None:
            low = mid
        else:
            high, witness, high_audit = mid, candidate, audit
    return witness, {'lower_bound': low, 'feasible_upper_bound': high,
                     'gap': high - low, 'upper_audit': high_audit, 'trace': trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int, default=3)
    parser.add_argument('--preregistration-commit', required=True)
    args = parser.parse_args()
    if args.max_new_cells < 0:
        raise ValueError('max-new-cells must be nonnegative')
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    frozen = subprocess.check_output(['git', 'rev-parse', args.preregistration_commit], cwd=REPO, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', frozen, head], cwd=REPO, check=True)
    pinned = {prereg_path: sha(prereg_path), Path(__file__): prereg['execution_source_sha256'],
              STUDY / 'analysis.py': prereg['analysis_source_sha256'],
              STUDY / 'executed/verify_saved.py': prereg['verification_source_sha256']}
    for path, expected in pinned.items():
        committed = subprocess.check_output(['git', 'show', f'{frozen}:{path.relative_to(REPO)}'], cwd=REPO)
        if sha(path) != expected or hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f'Uncommitted or altered registered source: {path}')
    for relative, expected in prereg['input_hashes'].items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f'Input hash mismatch: {relative}')
    previous = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    if sha(STUDY / 'executed/previous_closeout_audit.json') != prereg['previous_audit_sha256']:
        raise RuntimeError('Historical audit changed')
    for item in previous['files']:
        path = REPO / item['file']
        if sha(path) != item['sha256'] or path.stat().st_mtime_ns != item['mtime_ns']:
            raise RuntimeError('Historical file changed: ' + item['file'])
    prior = json.loads((REPO / prereg['prior_summary']).read_text())
    started = time.monotonic()
    import numpy as np_module
    global np
    np = np_module
    grid = np.array(prereg['grid'], dtype=int)
    bounds = {'log_half': np.log(prereg['parameter_bounds']['t50']).tolist(),
              'beta': prereg['parameter_bounds']['beta']}
    receipts = sorted((STUDY / 'executed').glob('run_receipt_*.json'))
    metadata_pins = {}
    for path in receipts:
        receipt = json.loads(path.read_text())
        if (receipt['preregistration_commit'] != frozen
                or receipt['preregistration_sha256'] != sha(prereg_path)
                or receipt['execution_source_sha256'] != sha(Path(__file__))):
            raise RuntimeError('Mixed receipt contract')
        for row in receipt['new_cells'] + receipt['resumed_cells']:
            if row['cell_id'] in metadata_pins and metadata_pins[row['cell_id']] != row['metadata_sha256']:
                raise RuntimeError('Conflicting receipt metadata hashes')
            metadata_pins[row['cell_id']] = row['metadata_sha256']
    (STUDY / 'results').mkdir(exist_ok=True)
    expected_ids = {item['cell_id'] for item in prereg['inputs']}
    present_ids = {p.name for p in (STUDY / 'results').iterdir()}
    if present_ids - expected_ids or set(metadata_pins) != present_ids:
        raise RuntimeError('Unexpected or unreceipted result; refuse any new fit')
    for item in prereg['inputs']:
        output = STUDY / 'results' / item['cell_id']
        if not output.exists():
            continue
        if {p.name for p in output.iterdir()} != {'arrays.npz', 'metadata.json'}:
            raise RuntimeError('Partial or extra result files; refuse any new fit')
        metadata = json.loads((output / 'metadata.json').read_text())
        expected_request = {'input': item, 'preregistration_sha256': sha(prereg_path),
                            'source_sha256': sha(Path(__file__))}
        if (metadata['request'] != expected_request or metadata['status'] != 'success'
                or metadata['preregistration_commit'] != frozen
                or sha(output / 'arrays.npz') != metadata['arrays_sha256']
                or sha(output / 'metadata.json') != metadata_pins[item['cell_id']]):
            raise RuntimeError('Result preflight mismatch; refuse any new fit')
    new, resumed = [], []
    for item in prereg['inputs']:
        cell_id = item['cell_id']
        output = STUDY / 'results' / cell_id
        request = {'input': item, 'preregistration_sha256': sha(prereg_path),
                   'source_sha256': sha(Path(__file__))}
        if output.exists():
            metadata = json.loads((output / 'metadata.json').read_text())
            if (metadata['request'] != request or metadata['status'] != 'success'
                    or sha(output / 'arrays.npz') != metadata['arrays_sha256']
                    or sha(output / 'metadata.json') != metadata_pins.get(cell_id)):
                raise RuntimeError(f'Resume mismatch: {cell_id}; do not overwrite')
            resumed.append({'cell_id': cell_id, 'arrays_sha256': metadata['arrays_sha256'],
                            'metadata_sha256': sha(output / 'metadata.json')})
            continue
        if len(new) >= args.max_new_cells:
            continue
        if time.monotonic() - started > prereg['compute_budget_seconds']:
            raise RuntimeError('Compute budget exhausted')
        cell_started = time.monotonic()
        with np.load(REPO / item['arrays']) as arrays:
            values = arrays['normalized_loss'][grid].copy()
        old = next(row for row in prior['cells'] if row['cell_id'] == cell_id)
        ls_log_half, ls_beta = np.log(old['half_time_fit']), old['beta_fit']
        ls_prediction = curve(ls_log_half, ls_beta, grid)
        mm_parameters, certificate = minimax(grid, values, bounds, prereg['bisection_tolerance'])
        mm_prediction = curve(*mm_parameters, grid)
        if not np.all(np.isfinite(mm_prediction)):
            raise RuntimeError('Nonfinite prediction')
        actual_maxabs = float(np.max(np.abs(mm_prediction - values)))
        if actual_maxabs > certificate['feasible_upper_bound'] + 1e-12:
            raise RuntimeError('Feasibility witness exceeds upper error bound')
        output.mkdir()
        np.savez_compressed(output / 'arrays.npz', grid=grid, observed=values,
                            least_squares_prediction=ls_prediction, minimax_prediction=mm_prediction,
                            least_squares_parameters=np.array([ls_log_half, ls_beta]),
                            minimax_parameters=np.array(mm_parameters))
        metadata = {'status': 'success', 'cell_id': cell_id, 'request': request,
                    'execution_head': head, 'preregistration_commit': frozen,
                    'arrays_sha256': sha(output / 'arrays.npz'), 'certificate': certificate,
                    'old_errors': old['best_errors'], 'elapsed_seconds': time.monotonic() - cell_started,
                    'completed_at': datetime.now().astimezone().isoformat()}
        write_new(output / 'metadata.json', metadata)
        new.append({'cell_id': cell_id, 'arrays_sha256': metadata['arrays_sha256'],
                    'metadata_sha256': sha(output / 'metadata.json')})
        print(json.dumps({'cell': cell_id, 'status': 'saved', 'maxabs': actual_maxabs}), flush=True)
    write_new(STUDY / 'executed' / f'run_receipt_{len(receipts) + 1:03d}.json',
              {'execution_head': head, 'preregistration_commit': frozen,
               'preregistration_sha256': sha(prereg_path), 'execution_source_sha256': sha(Path(__file__)),
               'new_cells': new, 'resumed_cells': resumed,
               'elapsed_seconds': time.monotonic() - started, 'training_cells': 0,
               'argv': sys.argv, 'python': sys.version, 'numpy': np.__version__,
               'completed_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({'new': len(new), 'resumed': len(resumed)}))


if __name__ == '__main__':
    main()
