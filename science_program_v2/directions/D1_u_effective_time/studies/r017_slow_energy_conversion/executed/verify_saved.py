import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoints(loss, q):
    first = next((t for t, value in enumerate(loss) if value <= q), None)
    last_over = next((t for t in range(len(loss) - 1, -1, -1) if loss[t] > q), -1)
    sustained = last_over + 1
    return {'first': first, 'sustained_to_T': sustained if sustained < len(loss) else None}


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit = summary['preregistration_commit']
    epoch = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=PROGRAM))
    for name in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / name
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(PROGRAM)}'], cwd=PROGRAM) == path.read_bytes()
    assert sha(STUDY / 'executed/run.py') == prereg['execution_source_sha256']
    assert sha(STUDY / 'analysis.py') == prereg['analysis_source_sha256']
    cells, rows = {}, []
    expected = {(c['id'], seed, mu) for c in prereg['conditions'] for seed in prereg['seeds'] for mu in prereg['momenta']}
    for path in sorted((STUDY / 'results').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        request = metadata['request']
        key = (request['id'], request['seed'], request['momentum'])
        assert key in expected and key not in cells
        assert metadata['status'] == 'success' and metadata['arrays_sha256'] == sha(path.parent / 'arrays.npz')
        assert epoch < metadata['started_at_epoch'] <= metadata['finished_at_epoch']
        cell_id = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        assert path.parent.name == metadata['cell_id'] == cell_id
        with np.load(path.parent / 'arrays.npz', allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
            loss = arrays['normalized_loss']
            assert len(loss) == prereg['steps'] + 1
            fractions = [request['slow_energy_fraction'], 1 - request['slow_energy_fraction']]
            curves = []
            for eigenvalue in request['spectrum']:
                h, mu = request['learning_rate'] * eigenvalue, request['momentum']
                previous, current = 1.0, 1 - h
                values = [previous, current]
                for t in range(1, request['steps']):
                    following = (1 + mu - h) * current - mu * previous
                    values.append(following)
                    previous, current = current, following
                curves.append(np.array(values))
            exact = fractions[0] * curves[0] ** 2 + fractions[1] * curves[1] ** 2
            recurrence_error = float(np.max(np.abs(exact - loss)))
            assert recurrence_error <= 1e-10
            hits = endpoints(loss, prereg['threshold'])
            assert hits == endpoints(exact, prereg['threshold'])
            x, rotation = arrays['features'], arrays['rotation']
            n = len(request['spectrum'])
            gram_error = max(abs(math.fsum(float(x[k, i]) * float(x[k, j]) for k in range(n)) / n
                                 - math.fsum(float(rotation[i, k]) * request['spectrum'][k] * float(rotation[j, k]) for k in range(n)))
                             for i in range(n) for j in range(n))
            initial_loss = math.fsum(float(y) ** 2 for y in arrays['targets']) / (2 * n)
            gradient_expected = math.sqrt(2 * initial_loss * math.fsum(a * lam for a, lam in zip(fractions, request['spectrum'])))
            initial_fraction_error = abs(float(arrays['normalized_modal_energies'][0, 0]) - fractions[0])
            assert gram_error <= 1e-14 and abs(initial_loss - prereg['initial_loss']) <= 1e-10
            assert abs(gradient_expected - metadata['initial_gradient_norm']) <= 1e-10 and initial_fraction_error <= 1e-10
            cells[key] = {'cell_id': cell_id, 'endpoints': hits}
            modal_at_sustained = arrays['normalized_modal_energies'][hits['sustained_to_T']].tolist()
        rows.append({'cell_id': cell_id, 'condition': request['id'], 'slow_energy_fraction': fractions[0],
                     'seed': request['seed'], 'momentum': request['momentum'],
                     'arrays_sha256': sha(path.parent / 'arrays.npz'), 'metadata_sha256': sha(path),
                     'max_scalar_recurrence_error': recurrence_error, 'scalar_gram_error': gram_error,
                     'initial_loss': initial_loss, 'initial_gradient_norm': gradient_expected,
                     'initial_slow_fraction_error': initial_fraction_error,
                     'normalized_modal_energies_at_sustained': modal_at_sustained})
    assert set(cells) == expected and len(rows) == 48
    primary = []
    for pair in summary['paired_endpoints']:
        plain = cells[(pair['id'], pair['seed'], 0.0)]
        momentum = cells[(pair['id'], pair['seed'], 0.9)]
        assert pair['plain_cell'] == plain['cell_id'] and pair['momentum_cell'] == momentum['cell_id']
        t0, tm = plain['endpoints'][pair['endpoint']], momentum['endpoints'][pair['endpoint']]
        assert pair['t_plain'] == t0 and pair['t_momentum'] == tm
        nominal = math.ceil(t0 / 10)
        startup = next(t for t in range(prereg['steps'] + 1) if 10 * t - 90 * (1 - .9 ** t) >= t0)
        assert pair['nominal_prediction'] == nominal and pair['startup_prediction'] == startup
        for name, predicted in [('nominal', nominal), ('startup', startup)]:
            assert pair[name + '_signed_error'] == predicted - tm
            assert pair[name + '_relative_abs_error'] == abs(predicted - tm) / tm
        if pair['endpoint'] == 'sustained_to_T':
            primary.append(pair)
    low = [p for p in primary if p['slow_energy_fraction'] <= .009]
    high = [p for p in primary if p['slow_energy_fraction'] >= .02]
    proxy_names = ['nominal_relative_abs_error', 'startup_relative_abs_error']
    assert all(p[name] >= .5 for p in low for name in proxy_names)
    assert all(p[name] <= .05 for p in high for name in proxy_names)
    receipts = [json.loads(path.read_text()) for path in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    assert len([cell for receipt in receipts for cell in receipt['new_cells']]) == 48
    assert len({cell for receipt in receipts for cell in receipt['new_cells']}) == 48
    for receipt in receipts:
        for row in receipt['resumed_successes']:
            cell = STUDY / 'results' / row['cell_id']
            assert sha(cell / 'arrays.npz') == row['arrays_sha256'] and sha(cell / 'metadata.json') == row['metadata_sha256']
    first_pair = json.loads((STUDY / 'executed/first_pair_verification.json').read_text())
    for row in first_pair['cells']:
        cell = STUDY / 'results' / row['cell_id']
        assert sha(cell / 'arrays.npz') == row['arrays_sha256'] and sha(cell / 'metadata.json') == row['metadata_sha256']
    output = {'status': 'passed', 'scope': '只复核保存结果；不跑训练、不改冻结预测或源码',
              'preregistration_commit': commit, 'verified_cells': len(rows),
              'commit_precedes_all_training': True, 'all_arrays_finite': True,
              'first_pair_unchanged_after_resume': True, 'summary_sha256': sha(STUDY / 'summary.json'),
              'verification_source_sha256': sha(Path(__file__)),
              'max_scalar_recurrence_error': max(r['max_scalar_recurrence_error'] for r in rows),
              'max_scalar_gram_error': max(r['scalar_gram_error'] for r in rows),
              'P1_low_condition_count': len({p['id'] for p in low}),
              'P2_high_condition_count': len({p['id'] for p in high}),
              'low_error_ranges': {name: [min(p[name] for p in low), max(p[name] for p in low)] for name in proxy_names},
              'high_error_ranges': {name: [min(p[name] for p in high), max(p[name] for p in high)] for name in proxy_names},
              'cells': rows}
    target = STUDY / 'executed/saved_evidence_verification.json'
    if target.exists():
        assert json.loads(target.read_text()) == output, 'Saved verification differs; inspect without overwriting'
    else:
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'cells'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
