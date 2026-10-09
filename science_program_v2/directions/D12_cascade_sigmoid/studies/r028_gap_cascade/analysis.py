import hashlib
import json
import os
from pathlib import Path
import time

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    if (sha(Path(__file__)) != prereg['analysis_source_sha256']
            or sha(STUDY / 'executed' / 'run.py') != prereg['execution_source_sha256']):
        raise RuntimeError('Registered source hash mismatch')
    import numpy as np
    from scipy.optimize import brentq, least_squares
    from scipy.special import expit

    started = time.monotonic()
    slow_rate = -2 * np.log1p(-.1 * .01)
    fastest_reference_rate = -2 * np.log1p(-.1 * 1)
    grid = np.unique(np.r_[0, np.unique(np.rint(np.geomspace(
        max(1, 1 / fastest_reference_rate / 100), 10 / slow_rate, 201)).astype(int))])
    if grid[-1] > prereg['steps']:
        raise RuntimeError('Grid exceeds saved training')
    pins = {}
    receipts = sorted((STUDY / 'executed').glob('run_receipt_*.json'))
    training_seconds = 0.0
    for path in receipts:
        receipt = json.loads(path.read_text())
        training_seconds += receipt['elapsed_seconds']
        for item in receipt['new_successes'] + receipt['resumed_successes']:
            old = pins.get(item['cell_id'])
            if old is not None and old != item:
                raise RuntimeError('Receipt pins disagree')
            pins[item['cell_id']] = item
    expected = {f'ratio_{r:03d}_seed_{s}' for r in prereg['fast_to_slow_eigenvalue_ratios']
                for s in prereg['seeds']}
    if set(pins) != expected:
        raise RuntimeError('Incomplete or unexpected saved cells')

    def curve(parameters):
        prediction = np.ones(len(grid))
        positive = grid > 0
        log_step = np.log(grid[positive])
        if len(parameters) == 2:
            prediction[positive] = expit(-np.exp(parameters[1]) * (log_step - parameters[0]))
        else:
            prediction[positive] = .5 * (
                expit(-np.exp(parameters[1]) * (log_step - parameters[0]))
                + expit(-np.exp(parameters[3]) * (log_step - parameters[2])))
        return prediction

    rows = []
    for cell_id in sorted(expected):
        cell_dir = STUDY / 'results' / cell_id
        arrays_path, metadata_path = cell_dir / 'arrays.npz', cell_dir / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        request = metadata['request']
        if (sha(arrays_path) != pins[cell_id]['arrays_sha256']
                or sha(metadata_path) != pins[cell_id]['metadata_sha256']
                or metadata['arrays_sha256'] != sha(arrays_path)
                or request['preregistration_sha256'] != sha(prereg_path)
                or request['source_sha256'] != prereg['execution_source_sha256']
                or hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                != metadata['request_sha256']):
            raise RuntimeError('Saved cell hash/contract mismatch')
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
                raise RuntimeError('Nonfinite saved arrays')
            loss = arrays['normalized_loss'].copy()
            initial_loss = float(arrays['loss'][0])
            eigenvalues = arrays['eigenvalues'].copy()
            if not np.array_equal(eigenvalues, .01 * np.array([1., request['ratio']])):
                raise RuntimeError('Unexpected spectrum')
        rates = -2 * np.log1p(-.1 * eigenvalues)
        exact = .5 * np.exp(-rates[0] * np.arange(len(loss))) + .5 * np.exp(-rates[1] * np.arange(len(loss)))
        half_time = brentq(lambda t: float(.5 * np.exp(-rates * t).sum() - .5),
                          0, prereg['steps'], xtol=1e-12)
        values = loss[grid]
        fit_path = cell_dir / 'fits.json'
        fit_contract = {'arrays_sha256': sha(arrays_path), 'metadata_sha256': sha(metadata_path),
                        'preregistration_sha256': sha(prereg_path),
                        'analysis_source_sha256': sha(Path(__file__)), 'grid': grid.tolist()}
        if fit_path.exists():
            fits = json.loads(fit_path.read_text())
            recorded_hash = fits.pop('content_sha256')
            if (fits['contract'] != fit_contract or recorded_hash != hashlib.sha256(
                    json.dumps(fits, sort_keys=True, ensure_ascii=False).encode()).hexdigest()):
                raise RuntimeError('Saved fit hash mismatch; do not overwrite')
        else:
            if training_seconds + time.monotonic() - started >= 1200:
                raise RuntimeError('Compute budget exhausted')
            single_starts = [np.log([half_time * factor, beta])
                             for factor in [.5, 1, 2] for beta in [1, 2]]
            component_halves = np.log(2) / rates
            double_starts = [np.log([component_halves[1] * f, bfast,
                                    component_halves[0] * s, bslow])
                             for f in [.5, 1, 2] for s in [.5, 1, 2]
                             for bfast in [1, 2] for bslow in [1, 2]]
            fits = {'contract': fit_contract}
            for name, starts in [('single', single_starts), ('double', double_starts)]:
                attempts = []
                for start in starts:
                    count = len(start) // 2
                    fit = least_squares(lambda p: curve(p) - values, start,
                                        bounds=(np.tile(np.log([.01, .1]), count),
                                                np.tile(np.log([100000, 10]), count)),
                                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
                    attempts.append({'success': bool(fit.success), 'status': int(fit.status),
                                     'nfev': int(fit.nfev), 'start': start.tolist(),
                                     'parameters': fit.x.tolist(),
                                     'sse': float(np.sum(fit.fun ** 2))})
                fits[name] = attempts
            fits['elapsed_seconds'] = time.monotonic() - started
            fits['content_sha256'] = hashlib.sha256(json.dumps(
                fits, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            with fit_path.open('x') as handle:
                json.dump(fits, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')
        row = {'cell_id': cell_id, 'ratio': request['ratio'], 'seed': request['seed'],
               'exact_time_scale_ratio': float(rates[1] / rates[0]),
               'execution_maxabs': float(np.max(np.abs(loss - exact))),
               'initial_loss_error': abs(initial_loss - 1),
               'half_time_exact': half_time, 'fit_file_sha256': sha(fit_path)}
        for name in ['single', 'double']:
            valid = [item for item in fits[name] if item['success']]
            if not valid:
                row[name] = {'evaluable': False, 'adequate': None}
                continue
            best = min(valid, key=lambda item: item['sse'])
            residual = curve(best['parameters']) - values
            rmse, maxabs = float(np.sqrt(np.mean(residual ** 2))), float(np.max(np.abs(residual)))
            parameters = np.exp(best['parameters']).reshape(-1, 2)
            parameters = parameters[np.argsort(parameters[:, 0])]
            row[name] = {'evaluable': True, 'successful_starts': len(valid),
                         'rmse': rmse, 'maxabs': maxabs,
                         'adequate': bool(rmse <= .03 and maxabs <= .05),
                         'parameters': parameters.tolist(),
                         'maxabs_step': int(grid[np.argmax(np.abs(residual))])}
        if row['single']['evaluable'] and row['double']['evaluable']:
            row['paired_difference_double_minus_single'] = {
                m: row['double'][m] - row['single'][m] for m in ['rmse', 'maxabs']}
        rows.append(row)
        print(json.dumps({'cell': cell_id, 'single': row['single'], 'double': row['double']},
                         ensure_ascii=False), flush=True)

    predictions = {'P1': {'status': 'supported' if all(
        r['execution_maxabs'] <= 1e-10 and r['initial_loss_error'] <= 1e-12
        for r in rows) else 'refuted', 'new_discovery': False}}
    for name, selected, model, desired in [
            ('P2', [r for r in rows if r['ratio'] in [12, 15]], 'single', True),
            ('P3', [r for r in rows if r['ratio'] in [20, 25]], 'single', False),
            ('P4', rows, 'double', True)]:
        failed = [r['cell_id'] for r in selected if r[model]['evaluable']
                  and r[model]['adequate'] != desired]
        predictions[name] = {'status': 'refuted' if failed else ('not_evaluable' if any(
            not r[model]['evaluable'] for r in selected) else 'supported'), 'counterexamples': failed}
    aggregate = []
    for ratio in prereg['fast_to_slow_eigenvalue_ratios']:
        group = [r for r in rows if r['ratio'] == ratio]
        item = {'ratio': ratio, 'exact_time_scale_ratio': group[0]['exact_time_scale_ratio']}
        for model in ['single', 'double']:
            valid = [r for r in group if r[model]['evaluable']]
            item[model] = {'evaluable_seeds': len(valid),
                           'adequate_seeds': sum(r[model]['adequate'] for r in valid),
                           'error_seed_ranges': {m: [min(r[model][m] for r in valid),
                                                     max(r[model][m] for r in valid)]
                                                 for m in ['rmse', 'maxabs']} if valid else None}
        aggregate.append(item)
    passed = [item['ratio'] for item in aggregate if item['single']['adequate_seeds'] == 3]
    failed = [item['ratio'] for item in aggregate if item['single']['adequate_seeds'] == 0]
    summary = {'study': prereg['study'], 'round': 28, 'direction_round': 1,
               'direction': prereg['direction'], 'domain': 'development',
               'status': 'analyzed' if all(r[m]['evaluable'] for r in rows
                                         for m in ['single', 'double']) else 'partial_fit',
               'counts': {'planned_cells': 12, 'saved_cells': len(rows),
                          'condition_recipe_units': 4, 'coordinate_seeds_per_unit': 3},
               'preregistration_sha256': sha(prereg_path),
               'preregistration_commits': sorted({json.loads(p.read_text())['preregistration_commit']
                                                  for p in receipts}),
               'execution_source_sha256': prereg['execution_source_sha256'],
               'analysis_source_sha256': sha(Path(__file__)), 'grid': grid.tolist(),
               'training_receipt_seconds': training_seconds,
               'analysis_invocation_seconds': time.monotonic() - started,
               'predictions': predictions, 'aggregate': aggregate, 'cells': rows,
               'single_pass_ratios': passed, 'single_fail_ratios': failed,
               'boundary_note': '仅离散已注册格点，不估计连续临界谱比；最小二乘多起点无全局保证。',
               'boundaries': prereg['boundaries']}
    (STUDY / 'summary.json').write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': predictions, 'aggregate': aggregate},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
