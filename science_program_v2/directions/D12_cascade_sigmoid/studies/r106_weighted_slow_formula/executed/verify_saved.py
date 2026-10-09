from datetime import datetime
import json
import math
import subprocess

import numpy as np

from run import ROOT, STUDY, check_frozen, scan_results, sha, write_json


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit, = summary['preregistration_commits']
    prereg = check_frozen(commit)
    pins = scan_results(prereg, commit)
    assert len(pins) == len(summary['cells']) == 3
    committed_time = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip())
    controls = {r['cell_id']: r for r in json.loads((ROOT / prereg['previous_summary']).read_text())['cells']}
    rows = {r['cell_id']: r for r in summary['cells']}
    differences = []
    for request in prereg['cells']:
        result_dir = STUDY / 'results' / request['cell_id']
        result = json.loads((result_dir / 'evaluation.json').read_text())
        assert committed_time < datetime.fromisoformat(result['evaluation_started_at']) < datetime.fromisoformat(result['completed_at'])
        with np.load(ROOT / request['arrays_path'], allow_pickle=False) as original:
            values = original['normalized_loss'][prereg['evaluation']['grid']]
            weights = original['mode_weights'].copy()
            initial_mode_loss = .5 * original['eigenvalues'] * original['mode_delta'][0] ** 2
            assert np.max(np.abs(initial_mode_loss / initial_mode_loss.sum() - weights)) <= 1e-14
        with np.load(result_dir / 'arrays.npz', allow_pickle=False) as saved:
            assert np.array_equal(saved['grid'], prereg['evaluation']['grid'])
            assert np.array_equal(saved['observed'], values)
            assert np.array_equal(saved['weights'], weights)
            assert np.array_equal(saved['eigenvalues'], [.01, 1.])
            errors, curve_diffs, exact_diffs, component_diffs = [], [], [], []
            for index, step in enumerate(prereg['evaluation']['grid']):
                components, exact = [], []
                for mode, eigenvalue in enumerate([.01, 1.]):
                    rate = -2 * math.log1p(-.1 * eigenvalue)
                    half_time = math.log(2) / rate
                    logistic = 1. if step == 0 else 1 / (1 + math.exp(
                        (2 * math.log(2)) * (math.log(step) - math.log(half_time))))
                    assert abs(saved['rates'][mode] - rate) <= 1e-14
                    assert abs(saved['half_times'][mode] - half_time) <= 1e-12
                    assert abs(saved['slopes'][mode] - 2 * math.log(2)) <= 1e-14
                    component_diffs.append(abs(logistic - saved['components'][index, mode]))
                    components.append(float(weights[mode]) * logistic)
                    exact.append(float(weights[mode]) * (1 - .1 * eigenvalue) ** (2 * step))
                prediction = math.fsum(components)
                curve_diffs.append(abs(prediction - saved['prediction'][index]))
                exact_diffs.append(abs(math.fsum(exact) - float(values[index])))
                errors.append(prediction - float(values[index]))
            rmse = math.sqrt(math.fsum(error ** 2 for error in errors) / len(errors))
            maxabs = max(map(abs, errors))
            metric_difference = max(abs(rmse - rows[request['cell_id']]['rmse']),
                                    abs(maxabs - rows[request['cell_id']]['maxabs']))
            assert max(curve_diffs + component_diffs) <= 1e-14 and metric_difference <= 1e-14
            assert max(exact_diffs) <= 1e-10
            assert rows[request['cell_id']]['adequate'] == (rmse <= .03 and maxabs <= .05)
            assert result['fitted_control'] == controls[request['cell_id']]['best_errors']
            for metric, value in [('rmse', rmse), ('maxabs', maxabs)]:
                assert abs(result['paired_formula_minus_fitted'][metric] - (value - result['fitted_control'][metric])) <= 1e-14
            differences.append({'cell_id': request['cell_id'], 'curve_maxabs': max(curve_diffs),
                                'component_maxabs': max(component_diffs), 'metric_maxabs': metric_difference,
                                'exact_recurrence_maxabs': max(exact_diffs),
                                'commit_lead_seconds': (datetime.fromisoformat(result['evaluation_started_at']) - committed_time).total_seconds()})
    assert summary['predictions']['P1']['status'] == ('supported' if all(r['rmse'] <= .03 and r['maxabs'] > .05 for r in summary['cells']) else 'refuted')
    intervals = prereg['predictions'][1]['intervals']
    assert summary['predictions']['P2']['status'] == ('supported' if all(
        intervals[m][0] <= r[m] <= intervals[m][1] for r in summary['cells'] for m in intervals) else 'refuted')
    audit = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    verification = {'status': 'passed', 'preregistration_commit': commit,
                    'commit_before_every_evaluation': True, 'cells_independently_recomputed': 3,
                    'previous_study_hash_and_mtime_unchanged': len(audit['previous_study_file_manifest']),
                    'initial_saved_loss_weights_verified': True, 'differences': differences}
    write_json(STUDY / 'executed/independent_verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
