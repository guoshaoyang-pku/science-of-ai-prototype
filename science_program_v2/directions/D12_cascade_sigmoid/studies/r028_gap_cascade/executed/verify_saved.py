import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['first', 'all'], required=True)
    args = parser.parse_args()
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    receipt_pins = {}
    for path in sorted((STUDY / 'executed').glob('run_receipt_*.json')):
        receipt = json.loads(path.read_text())
        for pin in receipt['new_successes'] + receipt['resumed_successes']:
            if pin['cell_id'] in receipt_pins and receipt_pins[pin['cell_id']] != pin:
                raise RuntimeError('Receipt pin conflict')
            receipt_pins[pin['cell_id']] = pin
    rows = []
    for path in sorted((STUDY / 'results').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        cell_id = metadata['cell_id']
        arrays_path = path.parent / 'arrays.npz'
        pin = receipt_pins[cell_id]
        assert sha(path) == pin['metadata_sha256']
        assert sha(arrays_path) == pin['arrays_sha256'] == metadata['arrays_sha256']
        commit = metadata['preregistration_commit']
        for local in [prereg_path, STUDY / 'executed/run.py', STUDY / 'analysis.py']:
            recorded = subprocess.check_output(
                ['git', 'show', f'{commit}:{local.relative_to(REPO)}'], cwd=REPO)
            assert hashlib.sha256(recorded).hexdigest() == sha(local)
        commit_time = int(subprocess.check_output(
            ['git', 'show', '-s', '--format=%ct', commit], cwd=REPO))
        assert commit_time < datetime.fromisoformat(metadata['completed_at']).timestamp()
        with np.load(arrays_path, allow_pickle=False) as saved:
            assert all(np.isfinite(saved[name]).all() for name in saved.files)
            step, loss = saved['step'], saved['loss']
            x, rotation = saved['features'], saved['rotation']
            theta, lam = saved['target_parameter'], saved['eigenvalues']
            request = metadata['request']
            assert np.array_equal(step, np.arange(prereg['steps'] + 1))
            assert np.array_equal(lam, .01 * np.array([1., request['ratio']]))
            assert request['preregistration_sha256'] == sha(prereg_path)
            assert request['source_sha256'] == sha(STUDY / 'executed/run.py')
            assert metadata['request_sha256'] == hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()).hexdigest()
            gram = np.einsum('ij,ik->jk', x, x) / 2
            spectrum_error = float(np.max(np.abs(np.linalg.eigvalsh(gram) - lam)))
            delta = -theta.copy()
            independent_loss = np.empty(len(step))
            independent_modes = np.empty((len(step), 2))
            for t in step:
                residual = np.einsum('ij,j->i', x, delta)
                independent_loss[t] = np.einsum('i,i->', residual, residual) / 4
                independent_modes[t] = np.einsum('ji,j->i', rotation, delta)
                delta -= .1 * np.einsum('ij,i->j', x, residual) / 2
            recurrence_error = float(np.max(np.abs(independent_loss - loss)))
            mode_error = float(np.max(np.abs(independent_modes - saved['mode_delta'])))
            exact = .5 * (1 - .1 * lam[0]) ** (2 * step) + .5 * (1 - .1 * lam[1]) ** (2 * step)
            exact_error = float(np.max(np.abs(exact - saved['normalized_loss'])))
            assert np.max(np.abs(rotation.T @ rotation - np.eye(2))) < 1e-12
            assert np.max(np.abs(x @ theta - saved['target'])) < 1e-12
            assert spectrum_error < 1e-12 and recurrence_error < 1e-12
            assert mode_error < 1e-10 and exact_error < 1e-10
        row = {'cell_id': cell_id, 'spectrum_maxabs': spectrum_error,
               'independent_loss_maxabs': recurrence_error,
               'independent_mode_maxabs': mode_error, 'exact_curve_maxabs': exact_error}
        rows.append(row)
    assert len(rows) == (1 if args.stage == 'first' else 12)
    report = {'status': 'passed', 'cells': rows,
              'count': len(rows), 'preregistration_before_training': True}
    output = STUDY / 'executed' / f'{args.stage}_saved_verification.json'
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
