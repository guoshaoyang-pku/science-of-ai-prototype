import hashlib
import json
import os
import time
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_summary(summary):
    (STUDY / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    if (sha(Path(__file__)) != prereg['analysis_source_sha256']
            or sha(STUDY / 'executed' / 'run.py') != prereg['execution_source_sha256']):
        raise RuntimeError('Registered source hash mismatch')
    metadata_paths = sorted((STUDY / 'results').glob('*/metadata.json'))
    base = {'study': prereg['study'], 'round': prereg['round'],
            'direction': prereg['direction'], 'domain': prereg['domain'],
            'preregistration_sha256': sha(prereg_path),
            'execution_source_sha256': prereg['execution_source_sha256'],
            'analysis_source_sha256': sha(Path(__file__)),
            'counts': {'planned_cells': prereg['cell_count'],
                       'saved_cells': len(metadata_paths), 'condition_recipe_units': len(prereg['slow_energy_fractions']),
                       'coordinate_seeds_per_unit': 3},
            'boundaries': prereg['boundaries']}
    if not metadata_paths:
        attempts = sorted((STUDY / 'executed').glob('*commit_attempt*.json'))
        if not attempts:
            raise RuntimeError('No cells and no recorded commit failure')
        evidence = [json.loads(path.read_text()) for path in attempts]
        if not any(item['phase'] == 'preregistration' and item['exit_code'] != 0
                   and not item['training_started'] for item in evidence):
            raise RuntimeError('No verified preregistration block')
        summary = {**base, 'status': 'blocked_before_training',
                   'round_result': 'failed', 'training_started': False,
                   'fit_started': False, 'preregistration_commit': None,
                   'training_seconds': 0, 'measured_claims': [],
                   'predictions': {p['id']: {'status': 'not_evaluated',
                                            'reason': 'No preregistration commit; no experiment ran'}
                                   for p in prereg['predictions']},
                   'commit_attempt_evidence': [str(path.relative_to(STUDY)) for path in attempts],
                   'next_action': 'Commit the reviewed preregistration and pinned sources in this repo before running executed/run.py.'}
        write_summary(summary)
        print(json.dumps({'status': summary['status'], 'saved_cells': 0,
                          'predictions': 'not_evaluated'}, ensure_ascii=False))
        return
    if len(metadata_paths) != prereg['cell_count']:
        raise RuntimeError('Incomplete experiment; keep saved cells and resume first')

    import numpy as np
    from scipy.optimize import brentq, least_squares
    from scipy.special import expit

    slow_rate = -2 * np.log1p(-prereg['learning_rate'] * prereg['slow_eigenvalue'])
    fastest_rate = -2 * np.log1p(-prereg['learning_rate'] * prereg['slow_eigenvalue']
                               * prereg['fast_to_slow_eigenvalue_ratio'])
    grid = np.unique(np.rint(np.geomspace(max(1, 1 / fastest_rate / 100),
                                         10 / slow_rate, 201)).astype(int))
    grid = np.unique(np.r_[0, grid])
    if grid[-1] > prereg['steps']:
        raise RuntimeError('Evaluation grid exceeds saved training budget')
    receipt_paths = sorted((STUDY / 'executed').glob('run_receipt_*.json'))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    metadata_pins = {}
    for receipt in receipts:
        for cell in receipt['new_successes'] + receipt['resumed_successes']:
            if sha(STUDY / 'results' / cell['cell_id'] / 'arrays.npz') != cell['arrays_sha256']:
                raise RuntimeError('Receipt arrays hash mismatch')
            if cell['cell_id'] in metadata_pins and metadata_pins[cell['cell_id']] != cell['metadata_sha256']:
                raise RuntimeError('Receipt metadata pins disagree')
            metadata_pins[cell['cell_id']] = cell['metadata_sha256']
    expected = {(a, s) for a in prereg['slow_energy_fractions'] for s in prereg['seeds']}
    observed, rows = set(), []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        request = metadata['request']
        key = (request['alpha'], request['seed'])
        if key not in expected or key in observed:
            raise RuntimeError('Unexpected or duplicate cell')
        observed.add(key)
        arrays_path = metadata_path.parent / 'arrays.npz'
        request_sha = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if (metadata['status'] != 'success' or sha(arrays_path) != metadata['arrays_sha256']
                or metadata_pins.get(metadata['cell_id']) != sha(metadata_path)
                or request_sha != metadata['request_sha256']
                or request['preregistration_sha256'] != sha(prereg_path)
                or request['source_sha256'] != prereg['execution_source_sha256']
                or request['steps'] != prereg['steps']
                or request['learning_rate'] != prereg['learning_rate']
                or request['slow_eigenvalue'] != prereg['slow_eigenvalue']
                or request['ratio'] != prereg['fast_to_slow_eigenvalue_ratio']):
            raise RuntimeError('Saved cell contract or hash mismatch')
        with np.load(arrays_path) as arrays:
            loss = arrays['normalized_loss'].copy()
            initial_loss = float(arrays['loss'][0])
            eigenvalues = prereg['slow_eigenvalue'] * np.array([1.0, prereg['fast_to_slow_eigenvalue_ratio']])
            if not np.array_equal(arrays['eigenvalues'], eigenvalues):
                raise RuntimeError('Saved spectrum differs from contract')
        rates = -2 * np.log1p(-prereg['learning_rate'] * eigenvalues)
        weights = np.array([request['alpha'], 1 - request['alpha']])
        exact = weights[0] * np.exp(-rates[0] * np.arange(len(loss))) + weights[1] * np.exp(-rates[1] * np.arange(len(loss)))
        half_time = brentq(lambda t: float((weights * np.exp(-rates * t)).sum() - .5),
                           0, prereg['steps'], xtol=1e-12)
        beta_local = float(4 * half_time * (weights * rates * np.exp(-rates * half_time)).sum())
        values = loss[grid]

        def curve(parameters):
            log_half_time, log_beta = parameters
            prediction = np.ones(len(grid))
            positive = grid > 0
            prediction[positive] = expit(-np.exp(log_beta) * (np.log(grid[positive]) - log_half_time))
            return prediction

        def errors(prediction):
            residual = prediction - values
            return {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                    'maxabs': float(np.max(np.abs(residual)))}

        fits = []
        for factor in [.5, 1, 2]:
            for beta in [1, 2]:
                fit = least_squares(lambda p: curve(p) - values, np.log([half_time * factor, beta]),
                                    bounds=(np.log([.01, .1]), np.log([100000, 10])),
                                    ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
                fits.append({'success': bool(fit.success), 'status': int(fit.status),
                             'nfev': int(fit.nfev), 'parameters': fit.x.tolist(),
                             'sse': float(np.sum(fit.fun ** 2))})
        valid = [f for f in fits if f['success']]
        best = min(valid, key=lambda f: f['sse']) if valid else None
        row = {'cell_id': metadata['cell_id'], 'alpha': request['alpha'], 'seed': request['seed'],
               'exact_time_scale_ratio': float(rates[1] / rates[0]),
               'execution_maxabs': float(np.max(np.abs(loss - exact))),
               'initial_loss_error': abs(initial_loss - 1), 'half_time_exact': half_time,
               'beta_local': beta_local, 'local_formula_errors': errors(curve(np.log([half_time, beta_local]))),
               'fit_attempts': fits, 'fit_evaluable': bool(valid)}
        if best:
            half_fit, beta_fit = np.exp(best['parameters'])
            row.update({'best_errors': errors(curve(best['parameters'])),
                        'half_time_fit': float(half_fit), 'beta_fit': float(beta_fit),
                        'half_time_relative_error': float(half_fit / half_time - 1),
                        'beta_relative_error': float(beta_fit / beta_local - 1)})
            row['adequate'] = row['best_errors']['rmse'] <= .03 and row['best_errors']['maxabs'] <= .05
        else:
            row['adequate'] = None
        rows.append(row)
    if observed != expected:
        raise RuntimeError('Missing registered condition')
    predictions = {'P1': {'status': 'supported' if all(r['execution_maxabs'] <= 1e-10
                          and r['initial_loss_error'] <= 1e-12 for r in rows) else 'refuted'}}
    for name, alphas, desired in [('P2', [.01, .99], True), ('P3', [.25, .5, .75], False)]:
        selected = [r for r in rows if r['alpha'] in alphas]
        failed = [r['cell_id'] for r in selected if r['fit_evaluable'] and r['adequate'] != desired]
        predictions[name] = {'status': 'refuted' if failed else
                              ('not_evaluable' if any(not r['fit_evaluable'] for r in selected) else 'supported'),
                             'counterexamples': failed}
    paired, aggregate = [], []
    for row in rows:
        reference = next(r for r in rows if r['alpha'] == .5 and r['seed'] == row['seed'])
        if row['fit_evaluable'] and reference['fit_evaluable']:
            paired.append({'alpha': row['alpha'], 'seed': row['seed'],
                           **{metric + '_difference_from_alpha0_5': row['best_errors'][metric] - reference['best_errors'][metric]
                              for metric in ['rmse', 'maxabs']}})
    for alpha in prereg['slow_energy_fractions']:
        group = [r for r in rows if r['alpha'] == alpha and r['fit_evaluable']]
        aggregate.append({'alpha': alpha, 'evaluable_seeds': len(group),
                          'adequate_seeds': sum(r['adequate'] for r in group),
                          'error_seed_ranges': {m: [min(r['best_errors'][m] for r in group), max(r['best_errors'][m] for r in group)]
                                                for m in ['rmse', 'maxabs']} if group else None})
    previous_study = STUDY.parent / 'r002_two_mode_logistic'
    previous_summary = json.loads((previous_study / 'summary.json').read_text())
    cross_round = []
    for row in rows:
        if row['alpha'] != .5:
            continue
        old = next(r for r in previous_summary['cells'] if r['ratio'] == 100 and r['seed'] == row['seed'])
        with np.load(STUDY / 'results' / row['cell_id'] / 'arrays.npz') as current, np.load(
                previous_study / 'results' / old['cell_id'] / 'arrays.npz') as prior:
            curve_difference = float(np.max(np.abs(current['normalized_loss'] - prior['normalized_loss'])))
        differences = {m: row['best_errors'][m] - old['best_errors'][m] for m in ['rmse', 'maxabs']}
        cross_round.append({'seed': row['seed'], 'curve_maxabs_difference': curve_difference,
                            'error_differences': differences,
                            'verified': curve_difference <= 1e-10 and all(abs(v) <= 1e-8 for v in differences.values())})
    write_summary({**base, 'status': 'analyzed' if all(r['fit_evaluable'] for r in rows) else 'partial_fit',
                   'training_started': True, 'fit_started': True,
                   'preregistration_commits': sorted({r['preregistration_commit'] for r in receipts}),
                   'analysis_seconds': time.monotonic() - started,
                   'successful_execution_call_seconds': sum(r['elapsed_seconds'] for r in receipts),
                   'cross_round_reproduction': cross_round,
                   'prior_summary_sha256': sha(previous_study / 'summary.json'),
                   'training_seconds': sum(json.loads(p.read_text())['elapsed_seconds'] for p in metadata_paths),
                   'resume_verified': True, 'grid': grid.tolist(), 'predictions': predictions,
                   'aggregate': aggregate, 'paired_errors': paired, 'cells': rows})
    print(json.dumps({'predictions': predictions, 'aggregate': aggregate}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
