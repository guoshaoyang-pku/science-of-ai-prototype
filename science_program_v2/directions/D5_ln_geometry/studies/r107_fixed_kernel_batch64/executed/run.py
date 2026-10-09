from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
FUNCTIONS = ('trigonometric8', 'quadratic8', 'interaction12', 'radial12')
LABELS = ('LN010_w64', 'noLN_w64')
SEEDS = (100, 101, 102, 103, 104)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def save(path, value):
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def gate(commit):
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    for rel in ('preregistration.json', 'executed/input_manifest.json', *cfg['source_sha256']):
        path = STUDY / rel
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes(), rel
        if rel in cfg['source_sha256']:
            assert sha(path) == cfg['source_sha256'][rel], rel
    assert sha(STUDY / 'executed/input_manifest.json') == cfg['manifest_sha256']
    manifest = json.loads((STUDY / 'executed/input_manifest.json').read_text())
    for rel, item in manifest['files'].items():
        path = ROOT / rel
        assert sha(path) == item['sha256'] and path.stat().st_mtime_ns == item['mtime_ns'], rel
    fixed = {'preregistration_commit': commit, 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
             'manifest_sha256': cfg['manifest_sha256'], 'source_sha256': cfg['source_sha256']}
    receipt_path = STUDY / 'executed/receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert all(receipt[k] == v for k, v in fixed.items())
    else:
        receipt = {**fixed, 'commit_epoch': int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT, text=True)),
                   'gate_epoch': time.time(), 'python': os.sys.executable, 'numpy': np.__version__, 'device': 'cpu', 'threads': 1}
        save(receipt_path, receipt)
    return cfg, manifest, receipt


def inputs(function, label, seed, manifest, receipt):
    spec = manifest['cells'][f'{function}_{label}_{seed}']
    kernel_path = ROOT / spec['kernel_result_json']
    kernel_row = json.loads(kernel_path.read_text())
    assert kernel_row['status'] == 'completed' and sha(kernel_path.with_suffix('.npz')) == kernel_row['arrays_sha256']
    with np.load(kernel_path.with_suffix('.npz'), allow_pickle=False) as data:
        kernel = data[spec['kernel_array']].copy()
        full = data[spec['full_vector_array']].copy()
        arrays = {key: data[key].copy() for key in data.files if key.startswith('train_')}
    assert pin(kernel) == spec['kernel_pin']
    batch_path = ROOT / spec['batch_result_json']
    batch_row = json.loads(batch_path.read_text())
    assert batch_row['status'] == 'completed' and sha(batch_path.with_suffix('.npz')) == batch_row['arrays_sha256']
    with np.load(batch_path.with_suffix('.npz'), allow_pickle=False) as data:
        batches = data['batch_indices'].copy()
        assert np.array_equal(data['train_x'], arrays['train_x'])
    assert pin(batches) == spec['batch_pin']
    assert batches.shape == (256, 64) and batches.dtype == np.int64
    assert np.all((batches >= 0) & (batches < 256))
    arrays.update(kernel=kernel, batch_indices=batches, full_propagated_ones_256=full)
    request = {'function': function, 'label': label, 'seed': seed, 'recipe': kernel_row['contract']['recipe'],
               'source': spec, **{key: receipt[key] for key in ('preregistration_commit', 'preregistration_sha256', 'manifest_sha256', 'source_sha256')}}
    return request, arrays


def evaluate(path, request, arrays, receipt):
    assert not path.exists() and not path.with_suffix('.npz').exists()
    started, timer = time.time(), time.monotonic()
    kernel = arrays['kernel']
    trajectory = np.empty((257, 256), dtype=np.float64)
    trajectory[0] = 1.0
    for step, indices in enumerate(arrays['batch_indices']):
        previous = trajectory[step]
        change = np.einsum('ij,j->i', kernel[:, indices], previous[indices], optimize=False)
        trajectory[step + 1] = previous - (2 * .001 * 256 / 64) * change
    centered = trajectory[-1] - trajectory[-1].mean()
    arrays.update(propagated_ones_trajectory=trajectory, centered_propagated_ones_256=centered)
    assert all(np.isfinite(value).all() for value in arrays.values())
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path.with_suffix('.npz'))
    row = {'status': 'completed', 'contract': request, 'arrays_sha256': sha(path.with_suffix('.npz')),
           'batch64_train_chord_256': float(9 * np.mean(centered ** 2)), 'new_training_cells': 0,
           'new_kernel_cells': 0, 'started_epoch': started, 'saved_epoch': time.time(), 'seconds': time.monotonic() - timer}
    assert started >= receipt['gate_epoch'] >= receipt['commit_epoch']
    save(path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--max-new-cells', type=int)
    args = parser.parse_args()
    _, manifest, receipt = gate(args.commit)
    cells = []
    for function in FUNCTIONS:
        for label in LABELS:
            for seed in SEEDS:
                request, arrays = inputs(function, label, seed, manifest, receipt)
                path = STUDY / 'results' / f'{function}_{label}_{seed}.json'
                cells.append((path, request, arrays))
    expected = {p for path, _, _ in cells for p in (path, path.with_suffix('.npz'))}
    assert set((STUDY / 'results').iterdir()) <= expected, 'unexpected or temporary result file'
    done = {}
    for path, request, _ in cells:
        if path.exists():
            row = json.loads(path.read_text())
            assert row['status'] == 'completed' and row['contract'] == request
            assert sha(path.with_suffix('.npz')) == row['arrays_sha256']
            assert row['started_epoch'] >= receipt['gate_epoch']
            done[path] = row
        else:
            assert not path.with_suffix('.npz').exists(), 'orphan NPZ'
    new, reused = 0, len(done)
    used = sum(row['seconds'] for row in done.values())
    for path, request, arrays in cells:
        if path in done:
            continue
        if args.max_new_cells is not None and new >= args.max_new_cells:
            break
        assert used < 1200
        row = evaluate(path, request, arrays, receipt)
        done[path] = row
        used += row['seconds']
        new += 1
        print(f'{len(done)}/40 {path.stem} {row["seconds"]:.4f}s', flush=True)
    state = {'status': 'complete' if len(done) == 40 else 'partial', 'saved_evaluation_cells': len(done),
             'new_evaluation_cells': new, 'reused_evaluation_cells': reused,
             'new_training_cells': 0, 'new_kernel_cells': 0, 'evaluation_seconds': used}
    save(STUDY / 'current.json', state)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
