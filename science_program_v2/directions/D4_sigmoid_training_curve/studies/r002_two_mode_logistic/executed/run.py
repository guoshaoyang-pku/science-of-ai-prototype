import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def committed_contract(prereg):
    pinned = {STUDY / 'preregistration.json': sha(STUDY / 'preregistration.json'),
              Path(__file__).resolve(): prereg['execution_source_sha256'],
              STUDY / 'analysis.py': prereg['analysis_source_sha256']}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    for path, expected in pinned.items():
        if sha(path) != expected:
            raise RuntimeError(f'Local source/contract hash mismatch: {path}')
        result = subprocess.run(['git', 'show', f'{head}:{path.relative_to(REPO)}'],
                                cwd=REPO, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise RuntimeError(f'Not committed with the registered content: {path}')
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int, default=15)
    args = parser.parse_args()
    if args.max_new_cells < 0:
        raise ValueError('max-new-cells must be nonnegative')
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    head = committed_contract(prereg)
    import numpy as np

    started = time.monotonic()
    receipts = sorted((STUDY / 'executed').glob('run_receipt_*.json'))
    metadata_pins = {}
    previous_run_seconds = 0.0
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        previous_run_seconds += receipt['elapsed_seconds']
        for cell in receipt['new_successes'] + receipt['resumed_successes']:
            key = cell['cell_id']
            if key in metadata_pins and metadata_pins[key] != cell['metadata_sha256']:
                raise RuntimeError('Conflicting metadata receipt hashes')
            metadata_pins[key] = cell['metadata_sha256']
    results = STUDY / 'results'
    results.mkdir(exist_ok=True)
    new_successes, resumed = [], []
    for ratio in prereg['fast_to_slow_eigenvalue_ratios']:
        for seed in prereg['seeds']:
            cell_id = f'ratio_{ratio:03d}_seed_{seed}'
            cell_dir = results / cell_id
            request = {'ratio': ratio, 'seed': seed, 'steps': prereg['steps'],
                       'learning_rate': prereg['learning_rate'],
                       'slow_eigenvalue': prereg['slow_eigenvalue'],
                       'source_sha256': sha(Path(__file__)),
                       'preregistration_sha256': sha(STUDY / 'preregistration.json')}
            request_sha = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            if cell_dir.exists():
                metadata_path, arrays_path = cell_dir / 'metadata.json', cell_dir / 'arrays.npz'
                if not metadata_path.exists() or not arrays_path.exists():
                    raise RuntimeError(f'Incomplete cell; do not overwrite: {cell_id}')
                metadata = json.loads(metadata_path.read_text())
                if (metadata['status'] != 'success' or metadata['request'] != request
                        or metadata['request_sha256'] != request_sha
                        or metadata['arrays_sha256'] != sha(arrays_path)
                        or metadata_pins.get(cell_id) != sha(metadata_path)):
                    raise RuntimeError(f'Resume hash mismatch: {cell_id}')
                resumed.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path),
                                'metadata_sha256': sha(metadata_path)})
                continue
            if len(new_successes) >= args.max_new_cells:
                continue
            if previous_run_seconds + time.monotonic() - started >= prereg['compute_budget']['maximum_seconds']:
                raise RuntimeError('Registered compute budget exhausted; keep saved cells')
            cell_dir.mkdir()
            cell_started = time.monotonic()
            eigenvalues = prereg['slow_eigenvalue'] * np.array([1.0, ratio])
            rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(2, 2)))
            features = np.diag(np.sqrt(2 * eigenvalues)) @ rotation.T
            target_parameter = rotation @ np.sqrt(1 / eigenvalues)
            target = features @ target_parameter
            initial_delta = -target_parameter
            delta = initial_delta.copy()
            loss = np.empty(prereg['steps'] + 1)
            mode_delta = np.empty((prereg['steps'] + 1, 2))
            for step in range(prereg['steps'] + 1):
                prediction_error = features @ delta
                loss[step] = prediction_error @ prediction_error / 4
                mode_delta[step] = rotation.T @ delta
                if step < prereg['steps']:
                    delta -= prereg['learning_rate'] * features.T @ prediction_error / 2
            arrays_path = cell_dir / 'arrays.npz'
            np.savez_compressed(arrays_path, step=np.arange(prereg['steps'] + 1),
                                loss=loss, normalized_loss=loss / loss[0],
                                mode_delta=mode_delta, eigenvalues=eigenvalues,
                                features=features, target=target, rotation=rotation,
                                target_parameter=target_parameter, initial_delta=initial_delta)
            metadata = {'status': 'success', 'cell_id': cell_id, 'request': request,
                        'request_sha256': request_sha, 'arrays_sha256': sha(arrays_path),
                        'preregistration_commit': head,
                        'initial_loss': float(loss[0]),
                        'elapsed_seconds': time.monotonic() - cell_started,
                        'completed_at': datetime.now().astimezone().isoformat()}
            metadata_path = cell_dir / 'metadata.json'
            write_new_json(metadata_path, metadata)
            new_successes.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path),
                                  'metadata_sha256': sha(metadata_path)})
            print(json.dumps({'cell': cell_id, 'status': 'saved'}, ensure_ascii=False), flush=True)
    receipt = {'preregistration_commit': head, 'argv': sys.argv, 'python': sys.version,
               'numpy': np.__version__, 'new_successes': new_successes,
               'resumed_successes': resumed, 'elapsed_seconds': time.monotonic() - started,
               'completed_at': datetime.now().astimezone().isoformat()}
    write_new_json(STUDY / 'executed' / f'run_receipt_{len(receipts) + 1:03d}.json', receipt)
    print(json.dumps({'new': len(new_successes), 'resumed': len(resumed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
