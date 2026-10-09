import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    assert sha(Path(__file__)) == prereg['verification_source_sha256']
    cells = {}
    max_energy_error = 0.0
    for path in sorted((STUDY / 'results').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        request = metadata['request']
        source = PROGRAM / request['source_arrays_path']
        assert sha(source) == request['source_arrays_sha256']
        assert sha(path.parent / 'endpoints.json') == metadata['endpoints_sha256']
        with np.load(source, allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            loss = arrays['loss'] / arrays['loss'][0]
            stored = arrays['normalized_loss']
            assert np.max(np.abs(loss - stored)) <= 1e-10
            max_energy_error = max(max_energy_error, float(np.max(np.abs(arrays['normalized_modal_energies'].sum(axis=1) - stored))))
            under = [i for i, value in enumerate(stored) if value <= prereg['threshold']]
            last_above = next((i for i in range(len(stored)-1, -1, -1) if stored[i] > prereg['threshold']), -1)
            expected = {'first': under[0] if under else None,
                        'sustained_to_T': last_above + 1 if last_above + 1 < len(stored) else None}
            output = json.loads((path.parent / 'endpoints.json').read_text())
            assert output['endpoints'] == expected
            recomputed = [i for i, value in enumerate(loss) if value > prereg['threshold']]
            assert expected['sustained_to_T'] == (recomputed[-1] + 1 if recomputed else 0)
        cells[(request['alpha'], request['seed'], request['momentum'])] = expected
    assert len(cells) == 36 and max_energy_error <= 1e-10
    for alpha in prereg['alphas']:
        for mu in prereg['momenta']:
            assert all(cells[(alpha, seed, mu)] == cells[(alpha, prereg['seeds'][0], mu)] for seed in prereg['seeds'])
    for pair in summary['paired_endpoints']:
        t0 = cells[(pair['alpha'], pair['seed'], 0.0)][pair['endpoint']]
        tm = cells[(pair['alpha'], pair['seed'], 0.9)][pair['endpoint']]
        steady = (t0 + 9) // 10
        accumulated, velocity, startup = 0.0, 0.0, 0
        while accumulated < t0:
            velocity = .9*velocity + 1.0
            accumulated += velocity
            startup += 1
        assert t0 == pair['t_plain'] and tm == pair['t_momentum']
        assert steady == pair['nominal_prediction'] and startup == pair['startup_prediction']
        for key, predicted in [('nominal', steady), ('startup', startup)]:
            assert abs(abs(predicted - tm)/tm - pair[f'{key}_relative_abs_error']) <= 1e-15
    primary = [p for p in summary['paired_endpoints'] if p['endpoint'] == 'sustained_to_T']
    p1 = all(max(p['nominal_relative_abs_error'], p['startup_relative_abs_error']) <= .05 for p in primary if p['alpha'] > .001)
    p2 = all(p['nominal_relative_abs_error'] <= .05 and .05 < p['startup_relative_abs_error'] <= .25 for p in primary if p['alpha'] == .001)
    assert summary['predictions']['P1_alpha_above_q']['status'] == ('supported' if p1 else 'refuted')
    assert summary['predictions']['P2_alpha_equals_q']['status'] == ('supported' if p2 else 'refuted')
    for row in prereg['historical_manifest'] + summary['result_manifest']:
        path = PROGRAM / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_mtime_ns == row['mtime_ns']
    assert subprocess.check_output(['git', 'show', f"{summary['preregistration_commit']}:{Path(__file__).relative_to(PROGRAM)}"], cwd=PROGRAM) == Path(__file__).read_bytes()
    output = {'status': 'passed', 'saved_endpoint_cells': len(cells), 'new_training_cells': 0,
              'independent_endpoints_and_conversion_match': True, 'coordinate_endpoints_identical': True,
              'max_modal_energy_sum_error': max_energy_error,
              'historical_files_hash_mtime_unchanged': len(prereg['historical_manifest']),
              'new_result_files_hash_mtime_unchanged': len(summary['result_manifest']),
              'summary_sha256': sha(STUDY / 'summary.json')}
    target = STUDY / 'executed/saved_evidence_verification.json'
    if target.exists():
        assert json.loads(target.read_text()) == output
    else:
        with target.open('x') as handle:
            handle.write(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
