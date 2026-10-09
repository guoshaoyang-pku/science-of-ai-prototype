import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def execute(request):
    eigenvalues = np.array(request['spectrum'], dtype=np.float64)
    n = len(eigenvalues)
    rotation, _ = np.linalg.qr(np.random.default_rng(request['seed']).normal(size=(n, n)))
    features = np.sqrt(n * eigenvalues)[:, None] * rotation.T
    weights = np.array([request['slow_energy_fraction'], 1 - request['slow_energy_fraction']])
    target_modes = np.sqrt(2 * request['initial_loss'] * weights / eigenvalues)
    target_parameter = np.einsum('ij,j->i', rotation, target_modes)
    targets = np.einsum('ij,j->i', features, target_parameter)
    delta, velocity = -target_parameter.copy(), np.zeros(n, dtype=np.float64)
    length = request['steps'] + 1
    loss = np.empty(length)
    modes = np.empty((length, n))
    gradient_norm = np.empty(length)
    for step in range(length):
        residual = np.einsum('ij,j->i', features, delta)
        gradient = np.einsum('ij,i->j', features, residual) / n
        loss[step] = np.sum(residual ** 2) / (2 * n)
        modes[step] = np.einsum('ij,i->j', rotation, delta)
        gradient_norm[step] = np.sqrt(np.sum(gradient ** 2))
        if step < request['steps']:
            velocity = request['momentum'] * velocity + gradient
            delta = delta - request['learning_rate'] * velocity
    arrays = {'loss': loss, 'normalized_loss': loss / loss[0], 'modal_residuals': modes,
              'normalized_modal_energies': modes ** 2 * eigenvalues / (2 * loss[0]),
              'gradient_norm': gradient_norm, 'features': features, 'rotation': rotation,
              'eigenvalues': eigenvalues, 'targets': targets, 'target_parameter': target_parameter,
              'target_modes': target_modes, 'energy_weights': weights}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Non-finite saved arrays')
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--max-new-cells', type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    prereg_path = STUDY / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', args.preregistration_commit], cwd=PROGRAM).decode().strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=PROGRAM, check=True)
    for name in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / name
        saved = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(PROGRAM)}'], cwd=PROGRAM)
        if saved != path.read_bytes():
            raise RuntimeError(f'Committed bytes differ: {name}')
    if sha(Path(__file__)) != prereg['execution_source_sha256'] or sha(STUDY / 'analysis.py') != prereg['analysis_source_sha256']:
        raise RuntimeError('Pinned source hash mismatch')
    if int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=PROGRAM)) >= started:
        raise RuntimeError('Preregistration must precede execution')
    new_cells, resumed, visited = [], [], 0
    for condition in prereg['conditions']:
        for seed in prereg['seeds']:
            for momentum in prereg['momenta']:
                request = {**condition, 'seed': seed, 'momentum': momentum,
                           **{key: prereg[key] for key in ['spectrum', 'steps', 'learning_rate', 'initial_loss', 'domain']},
                           'preregistration_commit': commit, 'preregistration_sha256': sha(prereg_path),
                           'source_sha256': sha(Path(__file__))}
                canonical = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
                cell_id = hashlib.sha256(canonical).hexdigest()[:16]
                cell = STUDY / 'results' / cell_id
                metadata_path, arrays_path = cell / 'metadata.json', cell / 'arrays.npz'
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata['status'] != 'success' or metadata['request'] != request or metadata['arrays_sha256'] != sha(arrays_path):
                        raise RuntimeError(f'Saved cell mismatch: {cell_id}')
                    resumed.append({'cell_id': cell_id, 'arrays_sha256': sha(arrays_path), 'metadata_sha256': sha(metadata_path)})
                else:
                    cell.mkdir()
                    cell_started = time.time()
                    write_json(metadata_path, {'status': 'running', 'request': request, 'started_at_epoch': cell_started})
                    try:
                        arrays = execute(request)
                        np.savez_compressed(arrays_path, **arrays)
                    except Exception as error:
                        write_json(metadata_path, {'status': 'failed', 'request': request, 'error': repr(error)})
                        raise
                    finished = time.time()
                    metadata = {'cell_id': cell_id, 'status': 'success', 'request': request,
                                'started_at_epoch': cell_started, 'finished_at_epoch': finished,
                                'elapsed_seconds': finished - cell_started, 'arrays_sha256': sha(arrays_path),
                                'initial_loss': float(arrays['loss'][0]),
                                'initial_gradient_norm': float(arrays['gradient_norm'][0])}
                    write_json(metadata_path, metadata)
                    new_cells.append(cell_id)
                    print(json.dumps({'cell': cell_id, 'condition': condition['id'], 'seed': seed,
                                      'momentum': momentum, 'seconds': metadata['elapsed_seconds']}), flush=True)
                visited += 1
                if args.max_new_cells and len(new_cells) >= args.max_new_cells:
                    break
            else:
                continue
            break
        else:
            continue
        break
    index = len(list((STUDY / 'executed').glob('run_receipt_*.json'))) + 1
    write_json(STUDY / 'executed' / f'run_receipt_{index:03d}.json', {
        'status': 'complete' if visited == prereg['cell_count'] else 'checkpointed',
        'preregistration_commit': commit, 'preregistration_sha256': sha(prereg_path),
        'execution_source_sha256': sha(Path(__file__)), 'started_at_epoch': started,
        'finished_at_epoch': time.time(), 'elapsed_seconds': time.time() - started,
        'python': platform.python_version(), 'numpy': np.__version__,
        'thread_environment': {key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'new_cells': new_cells, 'resumed_successes': resumed, 'verified_or_new_cells': visited})


if __name__ == '__main__':
    main()
