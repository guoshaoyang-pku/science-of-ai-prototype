from __future__ import annotations

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
        for index, delta in enumerate(config['bias_offsets']):
            for scale in config['scales']:
                yield {'seed': seed, 'offset_index': index, 'delta': delta, 'scale': scale}


def label(cell):
    return f"s{cell['seed']}_d{cell['offset_index']:02d}_a{cell['scale']:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((STUDY / 'preregistration.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                            check=True, text=True).stdout.strip()
    for name in ['preregistration.json', *config['source_sha256']]:
        path = STUDY / name
        if name in config['source_sha256'] and sha(path) != config['source_sha256'][name]:
            raise RuntimeError(f'Pinned source mismatch: {name}')
        saved = subprocess.run(['git', 'show', f'{commit}:{path.relative_to(REPO)}'],
                               cwd=REPO, capture_output=True, check=True).stdout
        if saved != path.read_bytes():
            raise RuntimeError('Matching preregistration and pinned files must be committed before measurement')
    baseline = json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())
    for name, expected in baseline['files'].items():
        if sha(REPO / name) != expected['sha256'] or (REPO / name).stat().st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError(f'Old evidence changed: {name}')
    old = REPO / config['reference_study']
    with np.load(old / 'executed/data.npz', allow_pickle=False) as z:
        inputs = {'train_x': z['train_x'].copy()}
    for seed in config['seeds']:
        with np.load(old / f'results/s{seed}_inflection_a0.025.npz', allow_pickle=False) as z:
            inputs[f'weight_{seed}'] = z['hidden_weight'].copy()
            inputs[f'u_{seed}'] = z['train_u'].copy()
        reproduced = np.einsum('nd,jd->nj', inputs['train_x'], inputs[f'weight_{seed}'])
        if np.max(np.abs(reproduced - inputs[f'u_{seed}'])) > 2e-14:
            raise RuntimeError('Old projection reconstruction failed')
    input_path = STUDY / 'executed/input_snapshot.npz'
    if input_path.exists():
        with np.load(input_path, allow_pickle=False) as z:
            if set(z.files) != set(inputs) or any(not np.array_equal(z[name], a) for name, a in inputs.items()):
                raise RuntimeError('Saved input snapshot differs')
    else:
        np.savez_compressed(input_path, **inputs)
    manifest = {'preregistration_commit': commit, 'preregistration_sha256': sha(STUDY / 'preregistration.json'),
                'source_sha256': config['source_sha256'], 'input_sha256': sha(input_path),
                'input_arrays': {name: {'sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                                       'shape': list(a.shape), 'dtype': str(a.dtype)} for name, a in inputs.items()},
                'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': 'cpu',
                'threads': 1, 'training_steps': 0}
    manifest_path = STUDY / 'executed/source_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for name in ('preregistration_sha256', 'source_sha256', 'input_sha256', 'input_arrays'):
            if previous[name] != manifest[name]:
                raise RuntimeError('Manifest differs')
        manifest = previous
    else:
        save(manifest_path, manifest)
    new, reused = 0, 0
    started = time.monotonic()
    for cell in cells(config):
        path = STUDY / 'results' / (label(cell) + '.json')
        request = {'cell': cell, 'bias': config['bias_root'] + cell['delta'],
                   'manifest_sha256': sha(manifest_path)}
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if path.exists():
            row = json.loads(path.read_text())
            if row['status'] != 'success' or row['request'] != request or row['contract_sha256'] != contract:
                raise RuntimeError('Saved cell contract differs')
            if row['arrays_sha256'] != sha(path.with_suffix('.npz')):
                raise RuntimeError('Saved cell array hash differs')
            reused += 1
        else:
            if path.with_suffix('.npz').exists():
                raise RuntimeError('Orphan arrays require inspection; refusing overwrite')
            cell_started = time.monotonic()
            u = torch.from_numpy(inputs[f"u_{cell['seed']}"])
            raw = torch.nn.functional.silu(request['bias'] + cell['scale'] * u)
            reflected = torch.nn.functional.silu(request['bias'] - cell['scale'] * u)
            odd = (raw - reflected) / 2
            even = (raw + reflected) / 2 - raw.mean(0)
            arrays = {'odd_raw': odd.numpy(), 'even_centered_raw': even.numpy(),
                      'feature_center': raw.mean(0).numpy()}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError('Nonfinite measurement; not saved as successful')
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            save(path, {'status': 'success', 'cell_id': label(cell), 'request': request,
                        'contract_sha256': contract, 'arrays_sha256': sha(path.with_suffix('.npz')),
                        'seconds': time.monotonic() - cell_started, 'training_steps': 0})
            new += 1
        state = {'completed': new + reused, 'new': new, 'reused': reused,
                 'planned_cells': config['planned_cells'], 'seconds': time.monotonic() - started,
                 'last_cell': label(cell), 'training_steps': 0}
        save(STUDY / 'current.json', state)
        if args.limit is not None and new >= args.limit:
            print(json.dumps(state), flush=True)
            break
    else:
        print(json.dumps(state), flush=True)
    if time.monotonic() - started > config['execution']['maximum_seconds']:
        raise RuntimeError('Measurement time budget exceeded')


if __name__ == '__main__':
    main()
