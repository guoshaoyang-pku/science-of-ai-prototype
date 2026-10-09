import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(PROGRAM), *args]).strip()


def execute(request):
    eigenvalues = np.array(request['spectrum'], dtype=np.float64)
    dimension = len(eigenvalues)
    rng = np.random.default_rng(request['seed'])
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    features = np.sqrt(dimension * eigenvalues)[:, None] * rotation.T
    index = request['target_index']
    curvature = eigenvalues[index]
    target_parameter = rotation[:, index] * request['initial_gradient_norm'] / curvature
    targets = features @ target_parameter
    residual_parameter = -target_parameter.copy()
    velocity = np.zeros(dimension, dtype=np.float64)
    target_norm = np.linalg.norm(targets)
    arrays = {key: np.empty(request['steps'] + 1) for key in
              ['normalized_loss', 'target_mode_residual', 'gradient_norm', 'normalized_function_distance']}
    for step in range(request['steps'] + 1):
        function_residual = features @ residual_parameter
        gradient = features.T @ function_residual / dimension
        arrays['normalized_loss'][step] = (np.linalg.norm(function_residual) / target_norm) ** 2
        arrays['target_mode_residual'][step] = (rotation[:, index] @ residual_parameter) * curvature / request['initial_gradient_norm']
        arrays['gradient_norm'][step] = np.linalg.norm(gradient)
        arrays['normalized_function_distance'][step] = np.linalg.norm(function_residual + targets) / target_norm
        if step < request['steps']:
            velocity = request['momentum'] * velocity + gradient
            residual_parameter -= request['learning_rate'] * velocity
    arrays.update(features=features, targets=targets, target_parameter=target_parameter,
                  rotation=rotation, eigenvalues=eigenvalues)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Nonfinite result; do not mark success')
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    source_sha = sha(Path(__file__))
    if source_sha != prereg['execution_source_sha256']:
        raise RuntimeError('Execution source differs from preregistration')
    relative_prereg = str(prereg_path.relative_to(PROGRAM))
    prereg_commit = git('log', '-1', '--format=%H', '--', relative_prereg).decode()
    committed_prereg = subprocess.check_output(['git', '-C', str(PROGRAM), 'show', f'{prereg_commit}:{relative_prereg}'])
    if hashlib.sha256(committed_prereg).hexdigest() != sha(prereg_path):
        raise RuntimeError('Preregistration is not committed unchanged')
    started = time.time()
    results = STUDY / 'results'
    results.mkdir(exist_ok=True)
    resumed, new_cells, cells = [], [], []
    for condition in prereg['conditions']:
        for seed in prereg['seeds']:
            for momentum in prereg['momenta']:
                request = {**condition, 'seed': seed, 'momentum': momentum,
                           'spectrum': prereg['spectrum'], 'steps': prereg['steps'],
                           'learning_rate': prereg['learning_rate'], 'domain': prereg['domain'],
                           'source_sha256': source_sha, 'preregistration_sha256': sha(prereg_path),
                           'preregistration_commit': prereg_commit}
                canonical = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
                cell_id = hashlib.sha256(canonical).hexdigest()[:16]
                cell = results / cell_id
                metadata_path, arrays_path = cell / 'metadata.json', cell / 'arrays.npz'
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata['status'] != 'success' or metadata['request'] != request or metadata['arrays_sha256'] != sha(arrays_path):
                        raise RuntimeError(f'Saved cell hash/contract mismatch: {cell_id}')
                    resumed.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path), 'metadata_sha256': sha(metadata_path)})
                else:
                    if cell.exists():
                        raise RuntimeError(f'Incomplete cell requires inspection: {cell_id}')
                    cell.mkdir()
                    cell_started = time.time()
                    write_json(metadata_path, {'status': 'running', 'request': request, 'started_at_epoch': cell_started})
                    try:
                        arrays = execute(request)
                        np.savez_compressed(arrays_path, **arrays)
                    except Exception as error:
                        write_json(metadata_path, {'status': 'failed', 'request': request, 'error': repr(error)})
                        raise
                    metadata = {'cell_id': cell_id, 'status': 'success', 'request': request,
                                'started_at_epoch': cell_started, 'finished_at_epoch': time.time(),
                                'elapsed_seconds': time.time() - cell_started, 'arrays_sha256': sha(arrays_path),
                                'initial_half_mse': float(arrays['targets'] @ arrays['targets'] / (2 * len(prereg['spectrum']))),
                                'measured_initial_gradient_norm': float(arrays['gradient_norm'][0])}
                    write_json(metadata_path, metadata)
                    new_cells.append(cell_id)
                    print(json.dumps({'cell': cell_id, 'condition': condition['id'], 'seed': seed,
                                      'momentum': momentum, 'seconds': metadata['elapsed_seconds']}), flush=True)
                cells.append(cell_id)
                if args.max_new_cells and len(new_cells) >= args.max_new_cells:
                    break
            else:
                continue
            break
        else:
            continue
        break
    receipt_index = len(list((STUDY / 'executed').glob('run_receipt_*.json'))) + 1
    write_json(STUDY / 'executed' / f'run_receipt_{receipt_index:03d}.json', {
        'status': 'complete' if len(cells) == prereg['cell_count'] else 'checkpointed',
        'finished_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'started_at_epoch': started, 'elapsed_seconds': time.time() - started,
        'preregistration_commit': prereg_commit, 'preregistration_sha256': sha(prereg_path),
        'execution_source_sha256': source_sha, 'python': platform.python_version(), 'numpy': np.__version__,
        'thread_environment': {key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'resumed_successes': resumed, 'new_cells': new_cells, 'verified_or_new_cells': len(cells)})


if __name__ == '__main__':
    main()
