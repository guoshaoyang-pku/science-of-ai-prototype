from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit, = summary['preregistration_commits']
    committed_time = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip())
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for path, expected in {str(prereg_path.relative_to(ROOT)): sha(prereg_path),
                           **prereg['source_hashes']}.items():
        frozen = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest() == expected == sha(ROOT / path), path
    for pin in prereg['input_manifest']:
        assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
    audit = json.loads((STUDY / 'executed/r028_closeout_audit.json').read_text())
    for pin in audit['previous_study_file_manifest']:
        path = ROOT / pin['path']
        assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns'], pin['path']
    rows = {r['cell_id']: r for r in summary['cells']}
    differences = []
    for request in prereg['cells']:
        cell_id = request['cell_id']
        result_dir = STUDY / 'results' / cell_id
        result = json.loads((result_dir / 'evaluation.json').read_text())
        assert committed_time < datetime.fromisoformat(result['completed_at'])
        assert result['contract']['preregistration_sha256'] == sha(prereg_path)
        assert result['contract']['preregistration_commit'] == commit
        assert result['contract']['source_hashes'] == prereg['source_hashes']
        assert result['contract']['request'] == request
        assert sha(result_dir / 'arrays.npz') == result['arrays_sha256']
        with np.load(ROOT / request['arrays_path'], allow_pickle=False) as original:
            values = original['normalized_loss'][prereg['evaluation']['grid']]
        with np.load(result_dir / 'arrays.npz', allow_pickle=False) as saved:
            assert np.array_equal(saved['grid'], prereg['evaluation']['grid'])
            assert np.array_equal(saved['observed'], values)
            errors = []
            maximum_curve_difference = 0.0
            maximum_exact_difference = 0.0
            for index, step in enumerate(prereg['evaluation']['grid']):
                components = []
                exact = []
                for eigenvalue in [.01, .01 * request['ratio']]:
                    rate = -2 * math.log1p(-.1 * eigenvalue)
                    half_time = math.log(2) / rate
                    logistic = 1.0 if step == 0 else 1 / (1 + math.exp(
                        (2 * math.log(2)) * (math.log(step) - math.log(half_time))))
                    components.append(logistic)
                    exact.append((1 - .1 * eigenvalue) ** (2 * step))
                prediction = math.fsum(components) / 2
                maximum_curve_difference = max(maximum_curve_difference,
                                                abs(prediction - saved['prediction'][index]))
                maximum_exact_difference = max(maximum_exact_difference,
                                                abs(math.fsum(exact) / 2 - values[index]))
                errors.append(prediction - float(values[index]))
            rmse = math.sqrt(math.fsum(error ** 2 for error in errors) / len(errors))
            maxabs = max(map(abs, errors))
            metric_difference = max(abs(rmse - rows[cell_id]['rmse']),
                                    abs(maxabs - rows[cell_id]['maxabs']))
            assert maximum_curve_difference <= 1e-14 and metric_difference <= 1e-14
            assert maximum_exact_difference <= 1e-10
            assert rows[cell_id]['adequate'] == (rmse <= .03 and maxabs <= .05)
            control = rows[cell_id]['fitted_control']
            assert abs(rows[cell_id]['paired_formula_minus_fitted']['rmse']
                       - (rmse - control['rmse'])) <= 1e-14
            differences.append({'cell_id': cell_id, 'curve_maxabs': maximum_curve_difference,
                                'metric_maxabs': metric_difference,
                                'exact_recurrence_maxabs': maximum_exact_difference})
    verification = {'status': 'passed', 'preregistration_commit': commit,
                    'commit_before_every_evaluation': True,
                    'previous_study_hash_and_mtime_unchanged': len(audit['previous_study_file_manifest']),
                    'cells_independently_recomputed': len(differences), 'differences': differences}
    destination = STUDY / 'executed/independent_verification.json'
    destination.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
