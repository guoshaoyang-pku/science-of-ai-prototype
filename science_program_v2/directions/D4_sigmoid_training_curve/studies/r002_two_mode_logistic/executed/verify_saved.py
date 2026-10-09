import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
FREEZE = '91fc940b310612a5b0a157366ffb97168160db8e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = STUDY / 'executed' / args.output
    assert output.parent == STUDY / 'executed'
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    source_paths = [prereg_path, STUDY / 'executed/run.py', STUDY / 'analysis.py']
    assert sha(source_paths[1]) == prereg['execution_source_sha256']
    assert sha(source_paths[2]) == prereg['analysis_source_sha256']
    for path in source_paths:
        original = subprocess.check_output(['git', 'show', f'{FREEZE}:{path.relative_to(REPO)}'], cwd=REPO)
        assert original == path.read_bytes()
    freeze_time = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', FREEZE], cwd=REPO, text=True).strip())
    audit_time = datetime.fromisoformat(json.loads(
        (STUDY / 'executed/round010_audit.json').read_text())['checked_at'])
    receipts = [json.loads(p.read_text()) for p in sorted(
        (STUDY / 'executed').glob('run_receipt_*.json'))]
    pins = {}
    for receipt in receipts:
        for cell in receipt['new_successes'] + receipt['resumed_successes']:
            if cell['cell_id'] in pins:
                assert pins[cell['cell_id']] == cell
            pins[cell['cell_id']] = cell
    expected = {(r, s) for r in prereg['fast_to_slow_eigenvalue_ratios'] for s in prereg['seeds']}
    seen, rows = set(), []
    for directory in sorted((STUDY / 'results').iterdir()):
        metadata_path, arrays_path = directory / 'metadata.json', directory / 'arrays.npz'
        metadata = json.loads(metadata_path.read_text())
        cell_id, request = metadata['cell_id'], metadata['request']
        pair = request['ratio'], request['seed']
        assert pair in expected and pair not in seen
        seen.add(pair)
        assert cell_id == directory.name and metadata['status'] == 'success'
        assert sha(metadata_path) == pins[cell_id]['metadata_sha256']
        assert sha(arrays_path) == pins[cell_id]['arrays_sha256'] == metadata['arrays_sha256']
        assert request['preregistration_sha256'] == sha(prereg_path)
        assert request['source_sha256'] == prereg['execution_source_sha256']
        assert request['steps'] == prereg['steps']
        assert request['learning_rate'] == prereg['learning_rate']
        assert request['slow_eigenvalue'] == prereg['slow_eigenvalue']
        assert metadata['request_sha256'] == hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()).hexdigest()
        commit = metadata['preregistration_commit']
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', FREEZE, commit],
                              cwd=REPO, check=False).returncode == 0
        for path in source_paths:
            assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(REPO)}'],
                                           cwd=REPO) == path.read_bytes()
        assert freeze_time < audit_time < datetime.fromisoformat(metadata['completed_at'])
        with np.load(arrays_path) as a:
            n = prereg['data_contract']['sample_count']
            eta = prereg['learning_rate']
            eigenvalues = prereg['slow_eigenvalue'] * np.array([1.0, request['ratio']])
            assert np.array_equal(a['step'], np.arange(prereg['steps'] + 1))
            assert np.array_equal(a['eigenvalues'], eigenvalues)
            assert a['loss'].dtype == np.float64 and a['mode_delta'].dtype == np.float64
            rotation = a['rotation']
            assert np.max(np.abs(rotation.T @ rotation - np.eye(2))) < 1e-12
            assert np.max(np.abs(a['features'] - np.diag(np.sqrt(n * eigenvalues)) @ rotation.T)) < 1e-12
            assert np.max(np.abs(a['target_parameter'] - rotation @ np.sqrt(1 / eigenvalues))) < 1e-12
            assert np.max(np.abs(a['target'] - a['features'] @ a['target_parameter'])) < 1e-12
            assert np.max(np.abs(a['initial_delta'] + a['target_parameter'])) < 1e-12
            modes = a['mode_delta']
            recurrence_error = float(np.max(np.abs(modes[1:] - modes[:-1] * (1 - eta * eigenvalues))))
            loss_recomputed = .5 * np.sum(eigenvalues * modes ** 2, axis=1)
            loss_error = float(np.max(np.abs(loss_recomputed - a['loss'])))
            exact = .5 * np.sum((1 - eta * eigenvalues) ** (2 * a['step'][:, None]), axis=1)
            exact_error = float(np.max(np.abs(a['normalized_loss'] - exact)))
            assert recurrence_error < 1e-12 and loss_error < 1e-12 and exact_error < 1e-10
            assert abs(float(a['loss'][0]) - 1) < 1e-12
            assert np.array_equal(a['normalized_loss'], a['loss'] / a['loss'][0])
        rows.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path),
                     'metadata_sha256': sha(metadata_path), 'mode_recurrence_maxabs': recurrence_error,
                     'loss_recomputed_maxabs': loss_error, 'exact_loss_maxabs': exact_error})
    assert rows
    result = {'round': 10, 'study': prereg['study'], 'status': 'verified',
              'checked_at': datetime.now().astimezone().isoformat(),
              'original_preregistration_commit': FREEZE, 'freeze_time': freeze_time.isoformat(),
              'pretraining_audit_time': audit_time.isoformat(), 'verified_cells': len(rows),
              'planned_cells': prereg['cell_count'], 'all_cells_saved': seen == expected,
              'successful_call_seconds': sum(r['elapsed_seconds'] for r in receipts), 'cells': rows}
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write(chr(10))
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
