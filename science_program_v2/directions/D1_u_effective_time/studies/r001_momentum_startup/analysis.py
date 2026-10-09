import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hitting_times(loss, threshold):
    hits = np.flatnonzero(loss <= threshold)
    first = int(hits[0]) if hits.size else None
    exceed = np.flatnonzero(loss > threshold)
    sustained = int(exceed[-1]) + 1 if exceed.size else 0
    return {'first': first, 'sustained_to_T': sustained if sustained < len(loss) else None}


def recurrence(h, momentum, steps):
    residual = np.empty(steps + 1)
    residual[0], residual[1] = -1.0, -(1 - h)
    for t in range(1, steps):
        residual[t + 1] = (1 + momentum - h) * residual[t] - momentum * residual[t - 1]
    return residual


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    if sha(Path(__file__)) != prereg['analysis_source_sha256']:
        raise RuntimeError('Analysis source differs from preregistration')
    cells, traces, audit = {}, {}, []
    metadata_paths = sorted((STUDY / 'results').glob('*/metadata.json'))
    if len(metadata_paths) != prereg['cell_count']:
        raise RuntimeError('Incomplete cell set')
    expected = {(c['id'], seed, m) for c in prereg['conditions'] for seed in prereg['seeds'] for m in prereg['momenta']}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        request = metadata['request']
        arrays_path = metadata_path.parent / 'arrays.npz'
        if metadata['status'] != 'success' or sha(arrays_path) != metadata['arrays_sha256']:
            raise RuntimeError(f'Invalid saved evidence: {metadata_path}')
        if request['preregistration_sha256'] != sha(STUDY / 'preregistration.json') or request['source_sha256'] != prereg['execution_source_sha256']:
            raise RuntimeError('Saved source/contract pin mismatch')
        with np.load(arrays_path) as data:
            loss = data['normalized_loss'].copy()
            residual = data['target_mode_residual'].copy()
            features, rotation = data['features'], data['rotation']
            h = prereg['learning_rate'] * prereg['spectrum'][request['target_index']]
            exact = recurrence(h, request['momentum'], prereg['steps'])
            audit.append({'cell_id': metadata['cell_id'],
                          'max_residual_error': float(np.max(np.abs(exact - residual))),
                          'max_loss_error': float(np.max(np.abs(exact ** 2 - loss))),
                          'hessian_error': float(np.max(np.abs(features.T @ features / len(prereg['spectrum']) - rotation @ np.diag(prereg['spectrum']) @ rotation.T))),
                          'initial_gradient_error': abs(metadata['measured_initial_gradient_norm'] - request['initial_gradient_norm'])})
        key = (request['id'], request['seed'], request['momentum'])
        if key not in expected or key in cells:
            raise RuntimeError('Unexpected or duplicate condition/seed/recipe')
        cells[key] = {'cell_id': metadata['cell_id'], 'hits': {str(q): hitting_times(loss, q) for q in prereg['thresholds']},
                      'exact_hits': {str(q): hitting_times(exact ** 2, q) for q in prereg['thresholds']}}
        traces[key] = loss
    if set(cells) != expected:
        raise RuntimeError('Missing conditions')
    t = np.arange(prereg['steps'] + 1, dtype=np.float64)
    mu = 0.9
    nominal = t / (1 - mu)
    startup = t / (1 - mu) - mu * (1 - mu ** t) / (1 - mu) ** 2
    pairs = []
    for condition in prereg['conditions']:
        for seed in prereg['seeds']:
            plain, momentum = cells[(condition['id'], seed, 0.0)], cells[(condition['id'], seed, 0.9)]
            for q in prereg['thresholds']:
                for endpoint in ['first', 'sustained_to_T']:
                    t0 = plain['hits'][str(q)][endpoint]
                    tm = momentum['hits'][str(q)][endpoint]
                    if t0 is None or tm is None:
                        raise RuntimeError('Censored endpoint; retain cells and report')
                    nominal_prediction = int(np.flatnonzero(nominal >= t0)[0])
                    startup_prediction = int(np.flatnonzero(startup >= t0)[0])
                    pairs.append({'condition': condition['id'], 'h': prereg['learning_rate'] * prereg['spectrum'][condition['target_index']],
                                  'g0': condition['initial_gradient_norm'], 'seed': seed, 'threshold': q, 'endpoint': endpoint,
                                  'plain_cell': plain['cell_id'], 'momentum_cell': momentum['cell_id'],
                                  't_plain': t0, 't_momentum': tm, 'observed_speedup': t0 / tm,
                                  'nominal_prediction': nominal_prediction, 'startup_prediction': startup_prediction,
                                  'nominal_signed_step_error': nominal_prediction - tm,
                                  'startup_signed_step_error': startup_prediction - tm,
                                  'nominal_relative_abs_error': abs(nominal_prediction - tm) / tm,
                                  'startup_relative_abs_error': abs(startup_prediction - tm) / tm})
    amplitude_comparisons = []
    for index in prereg['target_indices']:
        small, large = f'lambda_{index}_g003', f'lambda_{index}_g010'
        for seed in prereg['seeds']:
            for m in prereg['momenta']:
                amplitude_comparisons.append({'target_index': index, 'seed': seed, 'momentum': m,
                                             'max_normalized_loss_difference': float(np.max(np.abs(traces[(small, seed, m)] - traces[(large, seed, m)]))),
                                             'all_hitting_times_equal': cells[(small, seed, m)]['hits'] == cells[(large, seed, m)]['hits']})
    early_weak = [p for p in pairs if p['threshold'] == 0.9 and p['h'] <= 0.001]
    nominal_failure = [p for p in early_weak if p['h'] == 0.001]
    late = [p for p in pairs if p['threshold'] == 0.01 and p['endpoint'] == 'sustained_to_T']
    regressions = [p for p in late if abs(p['startup_signed_step_error']) > abs(p['nominal_signed_step_error'])]
    predictions = {
        'P1_exact_recurrence_audit': {'passed': max(a['max_loss_error'] for a in audit) <= 1e-10 and all(c['hits'] == c['exact_hits'] for c in cells.values()),
                                    'max_loss_error': max(a['max_loss_error'] for a in audit),
                                    'max_residual_error': max(a['max_residual_error'] for a in audit)},
        'P2_early_weak_startup': {'passed': max(abs(p['startup_signed_step_error']) for p in early_weak) <= 1 and all(p['nominal_signed_step_error'] / p['t_momentum'] <= -0.2 for p in nominal_failure),
                                'max_startup_abs_step_error': max(abs(p['startup_signed_step_error']) for p in early_weak),
                                'h001_nominal_signed_relative_error_range': [min(p['nominal_signed_step_error'] / p['t_momentum'] for p in nominal_failure), max(p['nominal_signed_step_error'] / p['t_momentum'] for p in nominal_failure)]},
        'P3_gradient_amplitude_control': {'passed': all(a['all_hitting_times_equal'] and a['max_normalized_loss_difference'] <= 1e-10 for a in amplitude_comparisons),
                                          'max_normalized_loss_difference': max(a['max_normalized_loss_difference'] for a in amplitude_comparisons),
                                          'initial_gradient_proxy_small_over_large': 0.3},
        'P4_startup_never_worse_late': {'passed': not regressions, 'counterexamples': regressions}}
    aggregate = []
    for condition in prereg['conditions']:
        for q in prereg['thresholds']:
            for endpoint in ['first', 'sustained_to_T']:
                group = [p for p in pairs if p['condition'] == condition['id'] and p['threshold'] == q and p['endpoint'] == endpoint]
                aggregate.append({'condition': condition['id'], 'h': group[0]['h'], 'g0': group[0]['g0'], 'threshold': q, 'endpoint': endpoint,
                                  **{key + '_seed_range': [min(p[key] for p in group), max(p[key] for p in group)] for key in
                                     ['t_plain', 't_momentum', 'observed_speedup', 'nominal_prediction', 'startup_prediction', 'nominal_relative_abs_error', 'startup_relative_abs_error']}})
    receipts = [json.loads(p.read_text()) for p in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    resume_verified = all(sha(STUDY / 'results' / c['cell_id'] / 'arrays.npz') == c['arrays_sha256'] and
                          sha(STUDY / 'results' / c['cell_id'] / 'metadata.json') == c['metadata_sha256']
                          for r in receipts for c in r['resumed_successes'])
    summary = {'study': prereg['study'], 'domain': prereg['domain'], 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
               'analysis_source_sha256': sha(Path(__file__)), 'preregistration_commit': receipts[0]['preregistration_commit'],
               'counts': {'saved_cells': len(cells), 'condition_recipe_units': 20, 'paired_condition_units': 10, 'coordinate_seeds_per_unit': 3},
               'training_cell_seconds_sum': sum(json.loads(p.read_text())['elapsed_seconds'] for p in metadata_paths),
               'run_wall_seconds_sum': sum(r['elapsed_seconds'] for r in receipts),
               'resume': {'verified': resume_verified, 'resumed_success_count': sum(len(r['resumed_successes']) for r in receipts)},
               'predictions': predictions, 'aggregate': aggregate, 'paired_endpoints': pairs,
               'amplitude_controls': amplitude_comparisons, 'execution_audits': audit,
               'boundaries': prereg['boundaries']}
    (STUDY / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': predictions, 'seconds': summary['training_cell_seconds_sum'], 'resume': summary['resume']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
