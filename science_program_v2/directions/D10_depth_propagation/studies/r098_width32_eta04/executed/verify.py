import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def unpack(flat, shapes):
    matrices, start = [], 0
    for rows, columns in shapes:
        size = int(rows * columns)
        matrices.append(flat[start:start + size].reshape(int(rows), int(columns)))
        start += size
    assert start == len(flat)
    return matrices


def predict_and_jacobian(x, matrices):
    hidden = [x]
    derivatives = []
    for matrix in matrices[:-1]:
        z = np.einsum('nj,ij->ni', hidden[-1], matrix)
        sigmoid = 1 / (1 + np.exp(-z))
        hidden.append(z * sigmoid)
        derivatives.append(sigmoid + z * sigmoid * (1 - sigmoid))
    prediction = np.einsum('nj,ij->ni', hidden[-1], matrices[-1]).flatten()
    jacobians = [None] * len(matrices)
    jacobians[-1] = hidden[-1].copy()
    delta = np.broadcast_to(matrices[-1], hidden[-1].shape).copy()
    for layer in range(len(matrices) - 2, -1, -1):
        dz = delta * derivatives[layer]
        jacobians[layer] = np.einsum('ni,nj->nij', dz, hidden[layer]).reshape(len(x), -1)
        delta = np.einsum('ni,ij->nj', dz, matrices[layer])
    return prediction, jacobians


def independent_time(kernel, residual, eta):
    update = np.eye(len(residual)) - eta * kernel
    norm = np.dot(residual, residual)

    def fraction(step):
        power = np.eye(len(residual))
        base = update.copy()
        remaining = step
        while remaining:
            if remaining % 2:
                power = np.einsum('ij,jk->ik', power, base)
            base = np.einsum('ij,jk->ik', base, base)
            remaining //= 2
        assert np.isfinite(power).all()
        r = np.einsum('ij,j->i', power, residual)
        return float(np.dot(r, r) / norm)

    high = 1
    while fraction(high) > .5:
        high *= 2
        if high > 2 ** 40:
            raise ValueError('independent search exceeded audit limit')
    low = 0
    while high - low > 1:
        middle = (high + low) // 2
        if fraction(middle) <= .5:
            high = middle
        else:
            low = middle
    return high, fraction(high), fraction(high - 1)


def main():
    preregistration = json.loads((STUDY / 'preregistration.json').read_text())
    recipe = preregistration['recipe']
    summary = json.loads((STUDY / 'summary.json').read_text())
    errors = {name: 0.0 for name in ['prediction', 'jacobian', 'first_step_prediction', 'final_prediction',
              'loss', 'kernel', 'kernel_sum', 'eigen_reconstruction', 'residual_energy', 'frozen_fraction']}
    maximum_stability = 0.0
    comparisons = []
    not_verified = []
    checkpoint_stats = []
    prefix_checks = {}
    generator = np.random.default_rng(recipe['data_seed'])
    expected_x = generator.normal(size=(recipe['n'], recipe['input_dim']))
    raw = expected_x[:, 0] + .5 * expected_x[:, 0] * expected_x[:, 1]
    expected_y = (raw - raw.mean()) / np.sqrt(np.mean((raw - raw.mean()) ** 2))
    for cell in summary['cells']:
        name = cell['id']
        json_path = STUDY / 'results' / (name + '.json')
        row = json.loads(json_path.read_text())
        npz_path = json_path.with_suffix('.npz')
        assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == row['arrays_sha256']
        if row['status'] != 'complete':
            not_verified.append({'id': name, 'reason': row['status'], 'divergence': row['divergence']})
            continue
        contract = row['contract']
        assert hashlib.sha256((STUDY / 'preregistration.json').read_bytes()).hexdigest() == contract['preregistration_sha256']
        commit = contract['preregistration_commit']
        for relative in ['preregistration.json', 'executed/manifest.json', 'executed/run.py', 'executed/analysis.py']:
            saved = subprocess.check_output(['git', 'show', f'{commit}:{(STUDY / relative).relative_to(ROOT)}'], cwd=ROOT)
            assert saved == (STUDY / relative).read_bytes()
        with np.load(npz_path) as arrays:
            baseline_path = STUDY.parent / preregistration['baseline']['study'] / 'results' / (name + '.npz')
            with np.load(baseline_path) as baseline:
                for key in ['x','y','target_mean_rms','parameter_shapes','parameters_0','prediction_0','residual_0','hidden_rms_0','K_sum_0'] + [k for k in arrays.files if k.startswith(('J_0_', 'K_0_', 'eigenvalues_0_', 'eigenvectors_0_', 'residual_energy_0_'))]:
                    assert np.array_equal(arrays[key], baseline[key]), (name, key)
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            assert np.array_equal(arrays['x'], expected_x) and np.array_equal(arrays['y'], expected_y)
            initial = unpack(arrays['parameters_0'], arrays['parameter_shapes'])
            for layer, matrix in enumerate(initial):
                is_head = layer == cell['depth']
                seed = cell['seed'] + (900000 if is_head else 10000 * layer)
                gen = torch.Generator().manual_seed(seed)
                expected = torch.randn(*matrix.shape, generator=gen)
                expected = expected / np.sqrt(recipe['width']) if is_head else expected * np.sqrt(2 / matrix.shape[1])
                assert np.array_equal(matrix, expected.numpy())
                prefix_key = (cell['seed'], 'head' if is_head else layer)
                digest = hashlib.sha256(matrix.tobytes()).hexdigest()
                assert prefix_key not in prefix_checks or prefix_checks[prefix_key] == digest
                prefix_checks[prefix_key] = digest
            prediction, jacobians = predict_and_jacobian(arrays['x'], initial)
            errors['prediction'] = max(errors['prediction'], float(np.max(np.abs(prediction - arrays['prediction_0']))))
            residual = prediction - arrays['y']
            updated = []
            for layer, (matrix, jacobian) in enumerate(zip(initial, jacobians)):
                errors['jacobian'] = max(errors['jacobian'], float(np.max(np.abs(jacobian - arrays[f'J_0_{layer}']))))
                gradient = np.einsum('np,n->p', jacobian, residual) / recipe['n']
                updated.append(matrix - recipe['eta'] * gradient.reshape(matrix.shape))
            one_step, _ = predict_and_jacobian(arrays['x'], updated)
            errors['first_step_prediction'] = max(errors['first_step_prediction'], float(np.max(np.abs(one_step - arrays['prediction_1']))))
            final = unpack(arrays['parameters_final'], arrays['parameter_shapes'])
            final_prediction, _ = predict_and_jacobian(arrays['x'], final)
            errors['final_prediction'] = max(errors['final_prediction'], float(np.max(np.abs(final_prediction - arrays['prediction_4096']))))
            hits = np.flatnonzero(arrays['loss'] <= arrays['loss'][0] / 2)
            actual = int(hits[0]) if len(hits) else None
            assert actual == cell['actual_t50']
            layer_times = []
            for step in recipe['checkpoints']:
                r = arrays[f'residual_{step}']
                errors['loss'] = max(errors['loss'], abs(float(.5 * np.mean(r ** 2)) - arrays['loss'][step]))
                assert np.allclose(r, arrays[f'prediction_{step}'] - arrays['y'], atol=1e-14, rtol=0)
                kernels = []
                qs = []
                for layer in range(cell['depth'] + 1):
                    j = arrays[f'J_{step}_{layer}']
                    k = np.einsum('ip,jp->ij', j, j) / recipe['n']
                    errors['kernel'] = max(errors['kernel'], float(np.max(np.abs(k - arrays[f'K_{step}_{layer}']))))
                    vals = arrays[f'eigenvalues_{step}_{layer}']
                    vecs = arrays[f'eigenvectors_{step}_{layer}']
                    reconstruction = np.einsum('ij,j,kj->ik', vecs, vals, vecs)
                    errors['eigen_reconstruction'] = max(errors['eigen_reconstruction'], float(np.max(np.abs(reconstruction - k))))
                    energy = np.einsum('ij,i->j', vecs, r) ** 2
                    errors['residual_energy'] = max(errors['residual_energy'], float(np.max(np.abs(energy - arrays[f'residual_energy_{step}_{layer}']))))
                    kernels.append(k)
                    qs.append(float(np.einsum('i,ij,j->', r, k, r) / np.dot(r, r)))
                    if step == 0:
                        maximum_stability = max(maximum_stability, recipe['eta'] * float(vals[-1]))
                        reported = cell['layer_times'][layer]
                        if reported['status'] != 'finite':
                            not_verified.append({'id': name, 'group': layer, 'reason': reported['status']})
                            layer_times.append(None)
                            continue
                        independent, at, before = independent_time(k, r, recipe['eta'])
                        assert independent == reported['t50'] and at <= .5 < before
                        errors['frozen_fraction'] = max(errors['frozen_fraction'], abs(at - reported['loss_fraction_at']), abs(before - reported['loss_fraction_before']))
                        layer_times.append(independent)
                summed = np.sum(kernels, axis=0)
                errors['kernel_sum'] = max(errors['kernel_sum'], float(np.max(np.abs(summed - arrays[f'K_sum_{step}']))))
                checkpoint_stats.append({'id': name, 'step': step, 'layer_q': qs,
                   'relative_kernel_drift': float(np.linalg.norm(summed - arrays['K_sum_0']) / np.linalg.norm(arrays['K_sum_0']))})
            reported = cell['sum_kernel']
            if reported['status'] != 'finite':
                not_verified.append({'id': name, 'group': 'sum', 'reason': reported['status']})
                continue
            independent, at, before = independent_time(arrays['K_sum_0'], arrays['residual_0'], recipe['eta'])
            assert independent == reported['t50'] and at <= .5 < before
            maximum_stability = max(maximum_stability, recipe['eta'] * float(np.linalg.eigvalsh(arrays['K_sum_0'])[-1]))
            comparisons.append({'id': name, 'actual_t50': actual, 'layer_t50': layer_times,
                  'sum_t50': independent, 'slowest_ratio': max(layer_times) / actual if actual is not None and all(t is not None for t in layer_times) else None, 'sum_ratio': independent / actual if actual is not None else None})
    assert all(value < 1e-8 for value in errors.values()), errors
    paired = []
    for seed in recipe['init_seeds']:
        times = {c['depth']: c['actual_t50'] for c in summary['cells'] if c['seed'] == seed}
        for shallow, deep in [(1, 2), (2, 4), (4, 6)]:
            if times[deep] is None or times[shallow] is None:
                continue
            paired.append({'seed': seed, 'depths': [shallow, deep], 'actual_time_difference': times[deep] - times[shallow], 'actual_time_ratio': times[deep] / times[shallow]})
    result = {'status': 'passed' if not not_verified else 'partial', 'not_verified': not_verified, 'cells_verified': len(comparisons), 'checkpoints_verified': len(checkpoint_stats),
              'new_training_cells': 0, 'eta_baseline_bitwise_initial_pairing': len(comparisons), 'one_step_numpy_reconstructions': len(comparisons), 'maximum_absolute_errors': errors,
              'maximum_initial_eta_lambda': maximum_stability, 'comparisons': comparisons,
              'paired_depth_changes': paired, 'checkpoint_diagnostics': checkpoint_stats,
              'scope': 'matrix-power half-time audit independent of pinned eigenspectrum search; no extra training'}
    (STUDY / 'executed/verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    print(json.dumps({'status': result['status'], 'errors': errors, 'maximum_initial_eta_lambda': maximum_stability}, indent=2))


if __name__ == '__main__':
    main()
