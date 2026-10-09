from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r077_fixed_kernel_train_chord'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}
def save(path, value):
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

class Block(nn.Module):
    def __init__(self, width, use_ln):
        super().__init__(); self.norm = nn.LayerNorm(width) if use_ln else nn.Identity(); self.linear = nn.Linear(width, width); self.act = nn.GELU()
    def forward(self, x): return self.act(self.linear(self.norm(x)))
class Model(nn.Module):
    def __init__(self, dimension, recipe):
        super().__init__(); width = recipe['width']
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.GELU(), *[Block(width, flag) for flag in recipe['layer_norm']], nn.Linear(width, 1))
    def forward(self, x): return self.net(x)

def gate(commit):
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    assert cfg['inputs']['manifest_sha256'] == sha(STUDY / 'executed/input_manifest.json')
    for rel in ('preregistration.json', 'executed/input_manifest.json', 'executed/run.py', 'analysis.py', 'executed/verify.py'):
        path = STUDY / rel
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes(), rel
        if rel in cfg['inputs']['source_sha256']: assert sha(path) == cfg['inputs']['source_sha256'][rel], rel
    manifest = json.loads((STUDY / 'executed/input_manifest.json').read_text())
    assert manifest['snapshot_commit'] == cfg['inputs']['snapshot_commit']
    for rel, item in manifest['files'].items():
        path = ROOT / rel
        assert sha(path) == item['sha256'] and path.stat().st_mtime_ns == item['mtime_ns'], rel
        assert subprocess.check_output(['git', 'show', f"{manifest['snapshot_commit']}:{rel}"], cwd=ROOT) == path.read_bytes(), rel
    receipt_path = STUDY / 'executed/receipt.json'
    fixed = {'preregistration_commit': commit, 'preregistration_sha256': sha(STUDY / 'preregistration.json'), 'source_sha256': cfg['inputs']['source_sha256'], 'manifest_sha256': sha(STUDY / 'executed/input_manifest.json')}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text()); assert all(receipt[k] == v for k, v in fixed.items())
    else:
        receipt = {**fixed, 'commit_epoch': int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT, text=True)), 'gate_epoch': time.time(), 'python': os.sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'threads': 1}
        save(receipt_path, receipt)
    return receipt

def inputs(function, seed, receipt):
    label = 'LN010_w64'; path = OLD / 'results' / f'{function}_{label}_{seed}.json'
    row = json.loads(path.read_text()); npz = path.with_suffix('.npz')
    assert row['status'] == 'completed' and sha(npz) == row['arrays_sha256']
    with np.load(npz) as d: arrays = {name: d[name].copy() for name in d.files}
    assert pin(arrays['kernel']) == row['kernel_pin']
    req = {'function': function, 'seed': seed, 'label': label, 'recipe': row['contract']['recipe'], 'initial_linear': row['contract']['initial_linear'], 'linear_kernel_pin': row['kernel_pin'], 'old_result_json_sha256': sha(path), 'old_result_npz_sha256': sha(npz), 'preregistration_commit': receipt['preregistration_commit'], 'preregistration_sha256': receipt['preregistration_sha256'], 'source_sha256': receipt['source_sha256'], 'manifest_sha256': receipt['manifest_sha256']}
    return req, arrays

def measure(path, req, arrays, receipt):
    assert not path.exists() and not path.with_suffix('.npz').exists()
    started, timer = time.time(), time.monotonic(); torch.manual_seed(req['seed'])
    model = Model(arrays['train_x'].shape[1], req['recipe']); named = dict(model.named_parameters())
    names = ('net.3.norm.weight', 'net.3.norm.bias'); assert tuple(n for n in named if '.norm.' in n) == names
    params = [named[n] for n in names]
    for name, p in named.items(): p.requires_grad_(name in names)
    pred = model(torch.from_numpy(arrays['train_x'])).ravel()
    assert np.array_equal(pred.detach().numpy(), arrays['train_predictions_0'].ravel())
    jac32 = np.stack([np.concatenate([g.detach().numpy().ravel() for g in torch.autograd.grad(pred[i], params, retain_graph=True)]) for i in range(pred.numel())])
    jac = jac32.astype(np.float64); n = pred.numel()
    ka = np.einsum('ip,jp->ij', jac, jac, optimize=False) / n
    kt = arrays['kernel'].astype(np.float64) + ka
    eig = np.linalg.eigvalsh(kt); assert eig[0] >= -1e-10 and .001 * eig[-1] < 1
    vecs = {}; v = np.ones(n, dtype=np.float64)
    for step in range(1, 257):
        v = v - .002 * np.einsum('ij,j->i', kt, v, optimize=False)
        if step in (1, 256): vecs[step] = v.copy()
    chords = {}
    for step, v in vecs.items():
        centered = v - v.mean(); arrays[f'affine_total_propagated_ones_{step}'] = v
        arrays[f'affine_total_centered_propagated_ones_{step}'] = centered
        chords[str(step)] = float(9 * np.mean(centered ** 2))
    arrays.update({'ln_affine_jacobian': jac, 'ln_affine_kernel': ka, 'total_kernel': kt, 'total_kernel_eigenvalues': eig, 'prediction_initial_reconstructed': pred.detach().numpy().copy()})
    assert all(np.isfinite(a).all() for a in arrays.values())
    tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp, **arrays); os.replace(tmp, path.with_suffix('.npz'))
    row = {'status': 'completed', 'contract': req, 'arrays_sha256': sha(path.with_suffix('.npz')), 'parameter_names': list(names), 'parameter_count': sum(p.numel() for p in params), 'affine_jacobian_shape': list(jac.shape), 'affine_kernel_pin': pin(ka), 'total_kernel_pin': pin(kt), 'kernel_eigenvalue_min': float(eig[0]), 'kernel_eigenvalue_max': float(eig[-1]), 'fixed_kernel_train_chord': chords, 'started_epoch': started, 'saved_epoch': time.time(), 'seconds': time.monotonic() - timer}
    save(path, row); return row

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--commit', required=True); parser.add_argument('--max-new-cells', type=int); args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); receipt = gate(args.commit)
    cells = []
    for function in ('trigonometric8', 'quadratic8', 'interaction12', 'radial12'):
        for seed in (100, 101, 102, 103, 104):
            req, arr = inputs(function, seed, receipt); path = STUDY / 'results' / f'{function}_LN010_w64_{seed}.json'; cells.append((path, req, arr))
    done = {}
    for path, req, _ in cells:
        if path.exists():
            row = json.loads(path.read_text()); assert row['status'] == 'completed' and row['contract'] == req and sha(path.with_suffix('.npz')) == row['arrays_sha256']; done[path] = row
        else: assert not path.with_suffix('.npz').exists(), 'orphan NPZ'
    assert set((STUDY / 'results').glob('*.json')) <= {p for p, _, _ in cells}
    used = sum(row['seconds'] for row in done.values()); new = 0
    for path, req, arr in cells:
        if path in done: continue
        if args.max_new_cells is not None and new >= args.max_new_cells: break
        assert used < 1200; row = measure(path, req, arr, receipt)
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        done[path] = row; used += row['seconds']; new += 1; print(f'{len(done)}/20 {path.stem} {row["seconds"]:.4f}s', flush=True)
    state = {'status': 'complete' if len(done) == 20 else 'partial', 'saved_measurement_cells': len(done), 'new_measurement_cells': new, 'new_training_cells': 0, 'reused_training_cells': 120, 'measurement_seconds': used}
    save(STUDY / 'current.json', state); print(json.dumps(state))

if __name__ == '__main__': main()
