from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f'Evidence recomputation differs: {label}')


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'seed_95pct_t_interval': [mean - radius, mean + radius]}


def quadratic(v, k):
    return float(np.einsum('i,ij,j->', v, k, v, optimize=False))


def design(rows, name):
    if name == 'E':
        return np.array([[1., r['E']] for r in rows])
    if name == 'W':
        return np.array([[1., r['E'], float(r['width'] == 8)] for r in rows])
    return np.array([[1., r['E'], r['Q']] for r in rows])


def calibration_audit(config):
    calibration = json.loads((STUDY / 'executed/calibration.json').read_text())
    old_rows = []
    old = REPO / config['calibration_study']
    for path in sorted((old / 'results').glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != meta['arrays_sha256']:
            raise RuntimeError('Old cell array hash differs')
        with np.load(path, allow_pickle=False) as z:
            k = np.einsum('tip,tjp->tij', z['jacobians'], z['jacobians'], optimize=False) / len(z['train_y'])
            close(k, z['kernels'], 'old K')
            scale = np.trace(k, axis1=1, axis2=2) / np.trace(k[0])
            r0, y = z['residuals'][0], z['train_y']
            early, late = z['checkpoints'].tolist().index(32), z['checkpoints'].tolist().index(256)
            fixed = [quadratic(r0, candidate) / (a * quadratic(r0, k[0]))
                     for candidate, a in zip(k, scale)]
            target = [quadratic(y, candidate) / (a * quadratic(y, k[0]))
                      for candidate, a in zip(k, scale)]
            close(fixed, z['fixed_r0_ratios'], 'old fixed r0')
            close(target, z['fixed_target_ratios'], 'old fixed y')
            request = meta['request']
            old_rows.append({'cell_id': path.stem, 'function': request['function'],
                             'width': request['width'], 'seed': request['seed'],
                             'E': float(np.log(fixed[early])), 'Q': float(np.log(target[early])),
                             'L': float(np.log(fixed[late]))})
    for actual, expected in zip(old_rows, calibration['rows']):
        if actual['cell_id'] != expected['cell_id']:
            raise RuntimeError('Old calibration row order differs')
        close([actual[k] for k in ['E', 'Q', 'L']],
              [expected[k] for k in ['E', 'Q', 'L']], 'calibration row')
    for name in config['frozen_coefficients']:
        x = design(old_rows, name)
        late = np.array([r['L'] for r in old_rows])
        coefficients = np.linalg.lstsq(x, late, rcond=None)[0]
        close(coefficients, config['frozen_coefficients'][name], 'frozen coefficients')
        errors = []
        for seed in config['calibration_seeds']:
            held = np.array([r['seed'] == seed for r in old_rows])
            beta = np.linalg.lstsq(x[~held], late[~held], rcond=None)[0]
            errors.extend((x[held] @ beta - late[held]).tolist())
        close(np.mean(np.abs(errors)), calibration['models'][name]['leave_seed_group_out_mae'],
              'seed-group-out MAE')
    return calibration['models']


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    for name, expected in config['source_sha256'].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f'Pinned source differs: {name}')
    for name, expected in config['input_sha256'].items():
        if sha(REPO / name) != expected:
            raise RuntimeError(f'Old input hash differs: {name}')
    calibration = calibration_audit(config)
    expected = {f'{fn}_w{w}_s{s}': (fn, w, s) for fn in config['functions']
                for w in config['widths'] for s in config['seeds']}
    rows, commits = [], set()
    for cell, (function, width, seed) in expected.items():
        path = STUDY / 'results' / f'{cell}.json'
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        early_path = path.with_name(f'{cell}.early.json')
        early_record = json.loads(early_path.read_text())
        request = meta['request']
        contract = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if (meta['status'] != 'success' or meta['cell_id'] != cell
                or request['function'] != function or request['width'] != width or request['seed'] != seed
                or request['preregistration_sha256'] != sha(STUDY / 'preregistration.json')
                or request['source_sha256'] != config['source_sha256']
                or meta['contract_sha256'] != contract or meta['arrays_sha256'] != sha(path.with_suffix('.npz'))
                or meta['early_sha256'] != sha(early_path) or early_record['request'] != request
                or early_record['preregistration_commit'] != meta['preregistration_commit']
                or not meta['start_time_ns'] < early_record['time_ns'] < meta['finish_time_ns']
                or early_record['late_step_executed'] is not False):
            raise RuntimeError('Saved contract, early record or hashes differ')
        commit = meta['preregistration_commit']
        commits.add(commit)
        for name in ['preregistration.json', *config['source_sha256']]:
            relative = str((STUDY / name).relative_to(REPO))
            blob = subprocess.run(['git', 'show', f'{commit}:{relative}'],
                                  cwd=REPO, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(STUDY / name):
                raise RuntimeError('Saved commit differs from contract')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            if not all(np.isfinite(z[k]).all() for k in z.files):
                raise RuntimeError('Nonfinite arrays')
            if z['checkpoints'].tolist() != config['checkpoints']:
                raise RuntimeError('Checkpoint contract differs')
            d = config['data_contract']
            x = np.random.default_rng(d['data_seed']).normal(size=(d['train_n'], d['input_dim']))
            raw = (np.tanh(x[:, 0] * x[:, 1]) if function == 'product' else
                   np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2]))
            y = (raw - raw.mean()) / raw.std(ddof=0)
            close(z['train_x'], x, 'train x')
            close(z['train_y'], y, 'train y')
            close(z['target_normalization'], [raw.mean(), raw.std(ddof=0)], 'target scaling')
            for name in ['train_x', 'train_y']:
                if hashlib.sha256(z[name].tobytes()).hexdigest() != request['inputs_sha256'][name]:
                    raise RuntimeError('Input hash differs')
            kernels = np.einsum('tip,tjp->tij', z['jacobians'], z['jacobians'], optimize=False) / len(x)
            close(z['kernels'], kernels, 'K=JJ^T/n')
            scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
            close(z['trace_scales'], scales, 'trace scales')
            close(z['nonlinear_loss'][z['checkpoints']], np.mean(z['residuals'] ** 2, axis=1) / 2, 'loss')
            r0 = z['residuals'][0]
            fixed = [quadratic(r0, k) / (a * quadratic(r0, kernels[0])) for k, a in zip(kernels, scales)]
            target = [quadratic(y, k) / (a * quadratic(y, kernels[0])) for k, a in zip(kernels, scales)]
            close(fixed, z['fixed_r0_ratios'], 'fixed r0 ratios')
            close(target, z['fixed_target_ratios'], 'target ratios')
            early, late = config['checkpoints'].index(32), config['checkpoints'].index(256)
            row = {'cell_id': cell, 'function': function, 'width': width, 'seed': seed,
                   'E': float(np.log(fixed[early])), 'Q': float(np.log(target[early])),
                   'L': float(np.log(fixed[late])), 'seconds': meta['seconds']}
            close([row['E'], row['Q']], [early_record['E'], early_record['Q']], 'early features')
            predictions, errors = {}, {}
            for name, coefficients in config['frozen_coefficients'].items():
                value = float((design([row], name) @ np.asarray(coefficients))[0])
                close(value, early_record['predictions'][name], 'timestamped frozen forecast')
                predictions[name], errors[name] = value, value - row['L']
            row['predictions'], row['signed_errors'] = predictions, errors
            rows.append(row)
    units = []
    for function in config['functions']:
        for width in config['widths']:
            paired = [r for r in rows if r['function'] == function and r['width'] == width]
            unit = {'function': function, 'width': width, 'saved_seeds': len(paired)}
            if paired:
                unit['mae'] = {name: float(np.mean([abs(r['signed_errors'][name]) for r in paired]))
                               for name in config['frozen_coefficients']}
            if len(paired) == len(config['seeds']):
                unit['paired_error_reduction'] = {name: interval([abs(r['signed_errors']['E']) -
                                                       abs(r['signed_errors'][name]) for r in paired])
                                                   for name in ['W', 'T']}
            units.append(unit)
    complete = len(rows) == config['planned_cells']
    mae = {name: float(np.mean([abs(r['signed_errors'][name]) for r in rows]))
           for name in config['frozen_coefficients']} if rows else {}
    reduction = {name: mae['E'] - mae[name] for name in ['W', 'T']} if rows else {}
    p1 = bool(mae['T'] <= .70 and reduction['T'] >= .08) if complete else None
    p2 = bool(reduction['W'] < .08) if complete else None
    result = {'study': config['study'], 'round': 22, 'direction_round': 3, 'domain': 'development',
              'status': 'measurements_complete' if complete else 'partial_measurements',
              'saved_cells': len(rows), 'planned_cells': config['planned_cells'],
              'planned_recipe_units': 4, 'mae': mae, 'paired_mae_reduction': reduction,
              'predictions': {'P1_target_amplitude': 'supported' if p1 else 'refuted' if complete else 'not_evaluated',
                              'P2_width_insufficient': 'supported' if p2 else 'refuted' if complete else 'not_evaluated'},
              'frozen_coefficients': config['frozen_coefficients'], 'calibration': calibration,
              'units': units, 'rows': rows, 'training_seconds': sum(r['seconds'] for r in rows),
              'preregistration_commits': sorted(commits),
              'boundary': '4个旧recipe/新初始化development验证；无新函数或data seed，不作OOD或因果归因。'}
    (STUDY / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'saved_cells', 'mae', 'paired_mae_reduction', 'predictions']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
