import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def commit_witness():
    files = [STUDY / 'preregistration.json', STUDY / 'executed/manifest.json']
    manifest = json.loads(files[1].read_text())
    for name, digest in manifest['sources'].items():
        file = STUDY / name
        if sha(file) != digest:
            raise ValueError(f'Source hash mismatch: {name}')
        files.append(file)
    relative = [str(p.relative_to(ROOT)) for p in files]
    commits = subprocess.check_output(['git', 'log', '--format=%H', '--', relative[0]], cwd=ROOT, text=True).splitlines()
    for commit in commits:
        matched = True
        for file, rel in zip(files, relative):
            result = subprocess.run(['git', 'show', f'{commit}:{rel}'], cwd=ROOT, capture_output=True)
            if result.returncode or result.stdout != file.read_bytes():
                matched = False
                break
        if matched:
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head], cwd=ROOT, check=True)
            return commit, head
    raise ValueError('No ancestor commit matches preregistration, manifest and sources; no experiment allowed')


def weights(depth, seed, recipe):
    width = recipe['width']
    matrices = []
    for layer in range(depth):
        fan_in = recipe['input_dim'] if layer == 0 else width
        gen = torch.Generator(device='cpu').manual_seed(seed + 10000 * layer)
        value = torch.randn(width, fan_in, generator=gen) * np.sqrt(2 / fan_in)
        matrices.append(value.requires_grad_())
    gen = torch.Generator(device='cpu').manual_seed(seed + 900000)
    matrices.append((torch.randn(1, width, generator=gen) / np.sqrt(width)).requires_grad_())
    return matrices


def forward(x, matrices):
    hidden_rms = []
    hidden = x
    for matrix in matrices[:-1]:
        hidden = torch.nn.functional.silu(hidden @ matrix.T)
        hidden_rms.append(float(hidden.detach().square().mean().sqrt()))
    return (hidden @ matrices[-1].T).flatten(), hidden_rms


def measure(x, y, matrices, step, arrays):
    prediction, rms = forward(x, matrices)
    jacobian = [[] for _ in matrices]
    for i in range(len(x)):
        grads = torch.autograd.grad(prediction[i], matrices, retain_graph=i < len(x) - 1)
        for block, grad in zip(jacobian, grads):
            block.append(grad.detach().flatten().numpy())
    residual = prediction.detach().numpy() - y.numpy()
    arrays[f'prediction_{step}'] = prediction.detach().numpy()
    arrays[f'residual_{step}'] = residual
    arrays[f'hidden_rms_{step}'] = np.array(rms)
    kernels = []
    for layer, block in enumerate(jacobian):
        jac = np.stack(block)
        kernel = np.einsum('ip,jp->ij', jac, jac) / len(x)
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        arrays[f'J_{step}_{layer}'] = jac
        arrays[f'K_{step}_{layer}'] = kernel
        arrays[f'eigenvalues_{step}_{layer}'] = eigenvalues
        arrays[f'eigenvectors_{step}_{layer}'] = eigenvectors
        arrays[f'residual_energy_{step}_{layer}'] = np.einsum('ij,i->j', eigenvectors, residual) ** 2
        kernels.append(kernel)
    arrays[f'K_sum_{step}'] = np.sum(kernels, axis=0)


def main():
    commit, head = commit_witness()
    preregistration = json.loads((STUDY / 'preregistration.json').read_text())
    recipe = preregistration['recipe']
    generator = np.random.default_rng(recipe['data_seed'])
    x_np = generator.normal(size=(recipe['n'], recipe['input_dim']))
    raw = x_np[:, 0] + .5 * x_np[:, 0] * x_np[:, 1]
    mean, rms = raw.mean(), np.sqrt(np.mean((raw - raw.mean()) ** 2))
    y_np = (raw - mean) / rms
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
    contract = {'preregistration_sha256': sha(STUDY / 'preregistration.json'),
                'manifest_sha256': sha(STUDY / 'executed/manifest.json'),
                'preregistration_commit': commit,
                'x_sha256': hashlib.sha256(x_np.tobytes()).hexdigest(),
                'y_sha256': hashlib.sha256(y_np.tobytes()).hexdigest()}
    total_start = time.monotonic()
    for depth in recipe['hidden_depths']:
        for seed in recipe['init_seeds']:
            cell = f'depth{depth}_seed{seed}'
            path = STUDY / 'results' / f'{cell}.json'
            npz = path.with_suffix('.npz')
            cell_contract = dict(contract, depth=depth, seed=seed)
            if path.exists():
                row = json.loads(path.read_text())
                if row['contract'] != cell_contract or sha(npz) != row['arrays_sha256']:
                    raise ValueError(f'Existing cell hash/contract mismatch: {cell}')
                print(f'{cell}: verified, skipped', flush=True)
                continue
            if npz.exists():
                raise ValueError(f'Orphan arrays need audit: {cell}')
            start = time.monotonic()
            matrices = weights(depth, seed, recipe)
            arrays = {'x': x_np, 'y': y_np, 'target_mean_rms': np.array([mean, rms]),
                      'parameter_shapes': np.array([[m.shape[0], m.shape[1]] for m in matrices]),
                      'parameters_0': torch.cat([m.detach().flatten() for m in matrices]).numpy().copy()}
            losses = []
            divergence = None
            for step in range(recipe['steps'] + 1):
                prediction, _ = forward(x, matrices)
                loss = .5 * (prediction - y).square().mean()
                value = float(loss.detach())
                if not np.isfinite(value) or value > 1e12:
                    divergence = {'step': step, 'reason': 'nonfinite or loss > 1e12'}
                    break
                losses.append(value)
                if step in recipe['checkpoints']:
                    measure(x, y, matrices, step, arrays)
                if step < recipe['steps']:
                    grads = torch.autograd.grad(loss, matrices)
                    with torch.no_grad():
                        for matrix, grad in zip(matrices, grads):
                            matrix.sub_(recipe['eta'] * grad)
            arrays['loss'] = np.array(losses)
            arrays['parameters_final'] = torch.cat([m.detach().flatten() for m in matrices]).numpy()
            temp = npz.with_suffix('.tmp.npz')
            np.savez_compressed(temp, **arrays)
            os.replace(temp, npz)
            row = {'id': cell, 'contract': cell_contract, 'execution_head': head,
                   'arrays_sha256': sha(npz), 'status': 'complete' if divergence is None else 'diverged',
                   'divergence': divergence, 'parameter_count': sum(m.numel() for m in matrices),
                   'seconds': time.monotonic() - start, 'saved_at_unix': time.time(),
                   'torch_version': torch.__version__, 'numpy_version': np.__version__}
            save_json(path, row)
            print(f'{cell}: {row["status"]}, {row["seconds"]:.3f}s', flush=True)
    print(f'Total elapsed {time.monotonic() - total_start:.3f}s', flush=True)


if __name__ == '__main__':
    main()
