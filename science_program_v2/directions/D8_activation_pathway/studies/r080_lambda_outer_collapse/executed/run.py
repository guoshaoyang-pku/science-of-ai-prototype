import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def cells(config):
    for seed in config['seeds']:
        for index, lam in enumerate(config['lambdas']):
            for scale in config['scales']:
                yield {'seed': seed, 'lambda_index': index, 'lambda': lam, 'scale': scale,
                       'delta': lam * scale**2}


def label(cell):
    return f"s{cell['seed']}_l{cell['lambda_index']}_a{cell['scale']:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((STUDY / 'preregistration.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    for name in ['preregistration.json', *config['source_sha256']]:
        path = STUDY / name
        if name in config['source_sha256'] and sha(path) != config['source_sha256'][name]:
            raise RuntimeError(f'Pinned source mismatch: {name}')
        blob = subprocess.run(['git', 'show', f'{commit}:{path.relative_to(REPO)}'], cwd=REPO, capture_output=True, check=True).stdout
        if blob != path.read_bytes():
            raise RuntimeError('Matching preregistration, pinned source and input must be committed before measurement')
    audit = json.loads((STUDY / 'executed/condition_audit.json').read_text())
    assert audit['planned_cells'] == config['planned_cells'] == 36
    assert audit['overlap_cells'] == 0 and audit['new_measurement_cells'] == 36
    actual = []
    for cell in cells(config):
        b = config['bias_root'] + cell['delta']
        assert not any(r['seed']==cell['seed'] and r['scale']==cell['scale'] and abs(r['bias']-b)<1e-14 for r in audit['historical_requests']), 'Historical condition: activation forbidden'
        actual.append({**cell, 'bias': b, 'cell_id': label(cell), 'historical_overlap': [], 'mode': 'new_measurement'})
    assert actual == audit['new_cells'], 'Cartesian cells differ from precommitted audit'
    baseline = json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())['files']
    for name, expected in baseline.items():
        p = REPO / name
        if sha(p) != expected['sha256'] or p.stat().st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError(f'Old evidence changed: {name}')
    forecasts = json.loads((STUDY / 'executed/analytic_forecasts.json').read_text())
    with np.load(STUDY / 'executed/input_snapshot.npz', allow_pickle=False) as z:
        inputs = {name: z[name].copy() for name in z.files}
    if set(inputs) != set(forecasts['input_arrays']):
        raise RuntimeError('Input names differ')
    for name, a in inputs.items():
        expected = forecasts['input_arrays'][name]
        if hashlib.sha256(a.tobytes()).hexdigest() != expected['sha256'] or list(a.shape) != expected['shape'] or str(a.dtype) != expected['dtype']:
            raise RuntimeError('Pinned input array differs')
    for seed in config['seeds']:
        rebuilt = np.einsum('nd,jd->nj', inputs['train_x'], inputs[f'weight_{seed}'])
        if np.max(np.abs(rebuilt-inputs[f'u_{seed}'])) > 2e-14:
            raise RuntimeError('Saved projection reconstruction failed')
    manifest = {'preregistration_commit': commit, 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
                'source_sha256': config['source_sha256'], 'input_arrays': forecasts['input_arrays'],
                'torch_version': torch.__version__, 'numpy_version': np.__version__,
                'device': 'cpu', 'threads': 1, 'training_steps': 0}
    manifest_path = STUDY / 'executed/source_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ('preregistration_sha256', 'source_sha256', 'input_arrays'):
            if old[key] != manifest[key]:
                raise RuntimeError('Existing manifest differs')
        manifest = old
    else:
        save(manifest_path, manifest)
    result_dir = STUDY / 'results'
    result_dir.mkdir(exist_ok=True)
    expected_stems = {label(c) for c in cells(config)}
    if any(p.stem not in expected_stems or p.suffix not in ('.json', '.npz') for p in result_dir.iterdir()):
        raise RuntimeError('Unexpected result file')
    new, reused = 0, 0
    started = time.monotonic()
    for cell in cells(config):
        path = result_dir / (label(cell) + '.json')
        request = {'cell': cell, 'bias': config['bias_root'] + cell['delta'], 'manifest_sha256': sha(manifest_path)}
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if path.exists():
            row = json.loads(path.read_text())
            if row['status'] != 'success' or row['cell_id'] != label(cell) or row['request'] != request or row['contract_sha256'] != contract or row['arrays_sha256'] != sha(path.with_suffix('.npz')):
                raise RuntimeError('Saved result contract/hash mismatch; refusing overwrite')
            reused += 1
        else:
            if path.with_suffix('.npz').exists():
                raise RuntimeError('Orphan NPZ requires audit; refusing overwrite')
            cell_started = time.monotonic()
            u = torch.from_numpy(inputs[f"u_{cell['seed']}"])
            raw = torch.nn.functional.silu(request['bias'] + cell['scale']*u)
            reflected = torch.nn.functional.silu(request['bias'] - cell['scale']*u)
            arrays = {'odd_raw': ((raw-reflected)/2).numpy(),
                      'even_centered_raw': ((raw+reflected)/2-raw.mean(0)).numpy(),
                      'feature_center': raw.mean(0).numpy()}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError('Nonfinite measurement')
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            save(path, {'status': 'success', 'cell_id': label(cell), 'request': request,
                        'contract_sha256': contract, 'arrays_sha256': sha(path.with_suffix('.npz')),
                        'seconds': time.monotonic()-cell_started, 'training_steps': 0})
            new += 1
        state = {'completed': new+reused, 'new': new, 'reused': reused, 'planned_cells': config['planned_cells'],
                 'last_cell': label(cell), 'seconds': time.monotonic()-started, 'training_steps': 0}
        save(STUDY / 'current.json', state)
        if args.limit is not None and new >= args.limit:
            break
    if time.monotonic()-started > config['execution']['maximum_seconds']:
        raise RuntimeError('Measurement time budget exceeded')
    print(json.dumps(state))


if __name__ == '__main__':
    main()
