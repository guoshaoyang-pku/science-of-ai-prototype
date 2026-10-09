import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(loss, q):
    under = np.flatnonzero(loss <= q)
    over = np.flatnonzero(loss > q)
    sustained = int(over[-1] + 1) if len(over) else 0
    return {'first': int(under[0]) if len(under) else None,
            'sustained_to_T': sustained if sustained < len(loss) else None}


def main():
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    if sha(Path(__file__)) != prereg['analysis_source_sha256']:
        raise RuntimeError('Analysis source changed after preregistration')
    receipts = [json.loads(p.read_text()) for p in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    commit = receipts[0]['preregistration_commit']
    epoch = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=PROGRAM))
    for name in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / name
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(PROGRAM)}'], cwd=PROGRAM) == path.read_bytes()
    expected = {(c['id'], seed, mu) for c in prereg['conditions'] for seed in prereg['seeds'] for mu in prereg['momenta']}
    cells, traces, audits = {}, {}, []
    elapsed = 0.0
    for path in sorted((STUDY / 'results').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        request = metadata['request']
        cell_id = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        assert path.parent.name == cell_id == metadata['cell_id']
        assert metadata['status'] == 'success' and metadata['arrays_sha256'] == sha(path.parent / 'arrays.npz')
        assert request['preregistration_commit'] == commit and request['preregistration_sha256'] == sha(prereg_path)
        assert request['source_sha256'] == prereg['execution_source_sha256']
        assert epoch < metadata['started_at_epoch'] <= metadata['finished_at_epoch']
        condition = next(c for c in prereg['conditions'] if c['id'] == request['id'])
        assert all(request[key] == value for key, value in condition.items())
        assert all(request[key] == prereg[key] for key in ['spectrum', 'steps', 'learning_rate', 'initial_loss', 'domain'])
        key = (request['id'], request['seed'], request['momentum'])
        assert key in expected and key not in cells
        with np.load(path.parent / 'arrays.npz', allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
            loss = arrays['normalized_loss'].copy()
            eigenvalues = arrays['eigenvalues']
            assert len(loss) == prereg['steps'] + 1
            modal = np.ones((len(loss), len(eigenvalues)), dtype=np.float64)
            modal[1] = 1 - request['learning_rate'] * eigenvalues
            for t in range(1, len(loss) - 1):
                modal[t + 1] = (1 + request['momentum'] - request['learning_rate'] * eigenvalues) * modal[t] - request['momentum'] * modal[t - 1]
            exact = np.sum(modal ** 2 * arrays['energy_weights'], axis=1)
            loss_error = float(np.max(np.abs(exact - loss)))
            modal_energy_error = float(np.max(np.abs(np.sum(arrays['normalized_modal_energies'], axis=1) - loss)))
            initial_loss_error = abs(float(arrays['loss'][0]) - prereg['initial_loss'])
            fraction_error = abs(float(arrays['normalized_modal_energies'][0, 0]) - request['slow_energy_fraction'])
            cells[key] = {'cell_id': cell_id, 'hits': hits(loss, prereg['threshold']),
                          'exact_hits': hits(exact, prereg['threshold']),
                          'initial_gradient_norm': float(arrays['gradient_norm'][0])}
        traces[key] = loss
        audits.append({'cell_id': cell_id, 'arrays_sha256': metadata['arrays_sha256'],
                       'metadata_sha256': sha(path), 'max_loss_recurrence_error': loss_error,
                       'max_modal_energy_error': modal_energy_error, 'initial_loss_error': initial_loss_error,
                       'initial_slow_fraction_error': fraction_error})
        elapsed += metadata['elapsed_seconds']
    assert set(cells) == expected
    for receipt in receipts:
        assert receipt['preregistration_commit'] == commit
        for row in receipt['resumed_successes']:
            cell = STUDY / 'results' / row['cell_id']
            assert sha(cell / 'arrays.npz') == row['arrays_sha256'] and sha(cell / 'metadata.json') == row['metadata_sha256']
    pairs = []
    for condition in prereg['conditions']:
        for seed in prereg['seeds']:
            plain, momentum = cells[(condition['id'], seed, 0.0)], cells[(condition['id'], seed, 0.9)]
            for endpoint in ['first', 'sustained_to_T']:
                t0, tm = plain['hits'][endpoint], momentum['hits'][endpoint]
                if t0 is None or tm is None:
                    raise RuntimeError('Censored endpoint: preserve data and report without extending budget')
                nominal = math.ceil(t0 / 10)
                startup = next(t for t in range(prereg['steps'] + 1)
                               if t / .1 - .9 * (1 - .9 ** t) / .1 ** 2 >= t0)
                pairs.append({**condition, 'seed': seed, 'endpoint': endpoint,
                              'plain_cell': plain['cell_id'], 'momentum_cell': momentum['cell_id'],
                              't_plain': t0, 't_momentum': tm, 'observed_speedup': t0 / tm,
                              'nominal_prediction': nominal, 'startup_prediction': startup,
                              'nominal_signed_error': nominal - tm, 'startup_signed_error': startup - tm,
                              'nominal_relative_abs_error': abs(nominal - tm) / tm,
                              'startup_relative_abs_error': abs(startup - tm) / tm})
    primary = [p for p in pairs if p['endpoint'] == 'sustained_to_T']
    low = [p for p in primary if p['slow_energy_fraction'] <= .009]
    high = [p for p in primary if p['slow_energy_fraction'] >= .02]
    proxies = ['nominal_relative_abs_error', 'startup_relative_abs_error']
    coordinate_error = max(float(np.max(np.abs(traces[(c['id'], seed, mu)] - traces[(c['id'], prereg['seeds'][0], mu)])))
                           for c in prereg['conditions'] for seed in prereg['seeds'] for mu in prereg['momenta'])
    endpoints_match = all(cells[(c['id'], seed, mu)]['hits'] == cells[(c['id'], prereg['seeds'][0], mu)]['hits']
                          for c in prereg['conditions'] for seed in prereg['seeds'] for mu in prereg['momenta'])
    audit_passed = (max(a['max_loss_recurrence_error'] for a in audits) <= 1e-10
                    and max(a['max_modal_energy_error'] for a in audits) <= 1e-10
                    and max(a['initial_loss_error'] for a in audits) <= 1e-10
                    and max(a['initial_slow_fraction_error'] for a in audits) <= 1e-10
                    and all(c['hits'] == c['exact_hits'] for c in cells.values())
                    and coordinate_error <= 1e-10 and endpoints_match)
    predictions = {
        'A1_known_theory_and_coordinate_audit': {'passed': audit_passed, 'coordinate_max_loss_difference': coordinate_error,
                                                'coordinate_endpoints_identical': endpoints_match,
                                                'max_loss_recurrence_error': max(a['max_loss_recurrence_error'] for a in audits)},
        'P1_low_slow_energy_failure': {'passed': all(p[key] >= .5 for p in low for key in proxies),
                                     'paired_condition_count': len({p['id'] for p in low}),
                                     'violations': [p for p in low if any(p[key] < .5 for key in proxies)]},
        'P2_high_slow_energy_accuracy': {'passed': all(p[key] <= .05 for p in high for key in proxies),
                                       'paired_condition_count': len({p['id'] for p in high}),
                                       'violations': [p for p in high if any(p[key] > .05 for key in proxies)]}}
    aggregate = []
    for condition in prereg['conditions']:
        for endpoint in ['first', 'sustained_to_T']:
            group = [p for p in pairs if p['id'] == condition['id'] and p['endpoint'] == endpoint]
            aggregate.append({**condition, 'endpoint': endpoint,
                              **{key + '_seed_range': [min(p[key] for p in group), max(p[key] for p in group)]
                                 for key in ['t_plain', 't_momentum', 'observed_speedup', 'nominal_prediction',
                                             'startup_prediction', 'nominal_signed_error', 'startup_signed_error', *proxies]}})
    output = {'study': prereg['study'], 'round': prereg['round'], 'domain': prereg['domain'],
              'preregistration_commit': commit, 'preregistration_sha256': sha(prereg_path),
              'analysis_source_sha256': sha(Path(__file__)),
              'counts': {'saved_cells': len(cells), 'paired_condition_units': len(prereg['conditions']),
                         'condition_recipe_units': len(prereg['conditions']) * 2, 'coordinate_seeds_per_unit': len(prereg['seeds'])},
              'training_cell_seconds_sum': elapsed, 'predictions': predictions, 'aggregate': aggregate,
              'paired_endpoints': pairs, 'execution_audits': audits,
              'resume': {'verified': True, 'resumed_success_count': sum(len(r['resumed_successes']) for r in receipts)},
              'boundaries': prereg['boundaries']}
    target = STUDY / 'summary.json'
    if target.exists():
        assert json.loads(target.read_text()) == output, 'Saved summary differs; inspect without overwriting'
    else:
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': predictions, 'counts': output['counts'], 'training_seconds': elapsed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
