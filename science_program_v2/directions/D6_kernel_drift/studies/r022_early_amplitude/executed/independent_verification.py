from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
CONFIG = json.loads((STUDY / 'preregistration.json').read_text())
SUMMARY = json.loads((STUDY / 'summary.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def features(row, name):
    return {'E': [1., row['E']], 'W': [1., row['E'], float(row['width'] == 8)],
            'T': [1., row['E'], row['Q']]}[name]


def read_arrays(path, width):
    with np.load(path, allow_pickle=False) as source:
        z = {name: source[name] for name in source.files}
    assert all(np.isfinite(value).all() for value in z.values())
    kernels = np.einsum('tip,tjp->tij', z['jacobians'], z['jacobians'], optimize=False) / len(z['train_y'])
    check(kernels, z['kernels'])
    scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
    check(scales, z['trace_scales'])
    ratios = []
    for vector, saved in [(z['residuals'][0], 'fixed_r0_ratios'), (z['train_y'], 'fixed_target_ratios')]:
        quadratic = np.einsum('i,tij,j->t', vector, kernels, vector, optimize=False)
        value = quadratic / (scales * quadratic[0])
        check(value, z[saved])
        ratios.append(value)
    early, late = z['checkpoints'].tolist().index(32), z['checkpoints'].tolist().index(256)
    row = {'width': width, 'E': float(np.log(ratios[0][early])),
           'Q': float(np.log(ratios[1][early])), 'L': float(np.log(ratios[0][late]))}
    return z, row


def forward(flat, x, width):
    sizes = [width * 3, width, width * width, width, width, 1]
    w1, b1, w2, b2, w3, b3 = flat.split(sizes)
    h1 = torch.nn.functional.silu(x @ w1.reshape(width, 3).T + b1)
    h2 = torch.nn.functional.silu(h1 @ w2.reshape(width, width).T + b2)
    return (h2 @ w3.reshape(1, width).T + b3).reshape(-1)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for relative, expected in CONFIG['source_sha256'].items():
        assert sha(STUDY / relative) == expected
    for relative, expected in CONFIG['input_sha256'].items():
        assert sha(REPO / relative) == expected
    old_rows = []
    for function in CONFIG['functions']:
        for width in CONFIG['widths']:
            for seed in CONFIG['calibration_seeds']:
                path = REPO / CONFIG['calibration_study'] / 'results' / f'{function}_w{width}_s{seed}.npz'
                _, row = read_arrays(path, width)
                row['seed'] = seed
                old_rows.append(row)
    calibration_cv = {}
    for name, coefficients in CONFIG['frozen_coefficients'].items():
        x = np.array([features(row, name) for row in old_rows])
        late = np.array([row['L'] for row in old_rows])
        check(np.linalg.lstsq(x, late, rcond=None)[0], coefficients)
        errors = []
        for seed in CONFIG['calibration_seeds']:
            held = np.array([row['seed'] == seed for row in old_rows])
            beta = np.linalg.lstsq(x[~held], late[~held], rcond=None)[0]
            errors.extend((x[held] @ beta - late[held]).tolist())
        calibration_cv[name] = float(np.mean(np.abs(errors)))
        check(calibration_cv[name], CONFIG['feasibility']['leave_seed_group_out_mae'][name])
    rows, commits = [], set()
    output_max_error = jacobian_max_error = init_max_error = 0.
    min_commit_lead_seconds = float('inf')
    for function in CONFIG['functions']:
        for width in CONFIG['widths']:
            for seed in CONFIG['seeds']:
                cell = f'{function}_w{width}_s{seed}'
                path = STUDY / 'results' / f'{cell}.json'
                meta = json.loads(path.read_text())
                early_path = path.with_name(f'{cell}.early.json')
                early = json.loads(early_path.read_text())
                assert meta['status'] == 'success' and meta['cell_id'] == cell
                assert meta['arrays_sha256'] == sha(path.with_suffix('.npz'))
                assert meta['early_sha256'] == sha(early_path)
                request = meta['request']
                assert request['function'] == function and request['width'] == width and request['seed'] == seed
                assert request['preregistration_sha256'] == sha(STUDY / 'preregistration.json')
                assert request['source_sha256'] == CONFIG['source_sha256']
                assert request == early['request']
                digest = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
                assert digest == meta['contract_sha256']
                commit = meta['preregistration_commit']
                commits.add(commit)
                assert commit == early['preregistration_commit']
                for relative in ['preregistration.json', *CONFIG['source_sha256']]:
                    archived = subprocess.run(['git', 'show', f'{commit}:{(STUDY / relative).relative_to(REPO)}'],
                                              cwd=REPO, check=True, capture_output=True).stdout
                    assert archived == (STUDY / relative).read_bytes()
                commit_stamp = subprocess.run(['git', 'show', '-s', '--format=%cI', commit], cwd=REPO,
                                             check=True, capture_output=True, text=True).stdout.strip()
                commit_ns = int(datetime.fromisoformat(commit_stamp).timestamp() * 1e9)
                assert commit_ns < meta['start_time_ns'] < early['time_ns'] < meta['finish_time_ns']
                assert early['step'] == 32 and early['late_step_executed'] is False
                min_commit_lead_seconds = min(min_commit_lead_seconds, (meta['start_time_ns'] - commit_ns) / 1e9)
                z, row = read_arrays(path.with_suffix('.npz'), width)
                assert z['checkpoints'].tolist() == CONFIG['checkpoints']
                d = CONFIG['data_contract']
                x = np.random.default_rng(d['data_seed']).standard_normal((d['train_n'], d['input_dim']))
                raw = (np.tanh(x[:, 0] * x[:, 1]) if function == 'product' else
                       np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2]))
                y = (raw - raw.mean()) / raw.std(ddof=0)
                check(x, z['train_x'])
                check(y, z['train_y'])
                check([raw.mean(), raw.std(ddof=0)], z['target_normalization'])
                for name in ['train_x', 'train_y']:
                    assert hashlib.sha256(z[name].tobytes()).hexdigest() == request['inputs_sha256'][name]
                torch.manual_seed(seed)
                initial = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width),
                                        nn.SiLU(), nn.Linear(width, 1)).double()
                expected_init = torch.cat([p.detach().reshape(-1) for p in initial.parameters()]).numpy()
                init_max_error = max(init_max_error, check(expected_init, z['parameters'][0]))
                inputs = torch.tensor(x, dtype=torch.float64)
                for position, checkpoint in enumerate(z['checkpoints']):
                    flat = torch.tensor(z['parameters'][position], dtype=torch.float64, requires_grad=True)
                    prediction = forward(flat, inputs, width).detach().numpy()
                    output_max_error = max(output_max_error, check(prediction - y, z['residuals'][position]))
                    jacobian = torch.autograd.functional.jacobian(lambda p: forward(p, inputs, width),
                                                                 flat, vectorize=True).detach().numpy()
                    jacobian_max_error = max(jacobian_max_error, check(jacobian, z['jacobians'][position]))
                    check(np.mean((prediction - y) ** 2) / 2, z['nonlinear_loss'][checkpoint])
                check([row['E'], row['Q']], [early['E'], early['Q']])
                row.update(cell_id=cell, function=function, seed=seed)
                row['errors'] = {}
                for name, coefficients in CONFIG['frozen_coefficients'].items():
                    prediction = float(np.dot(features(row, name), coefficients))
                    check(prediction, early['predictions'][name])
                    row['errors'][name] = abs(prediction - row['L'])
                saved_row = next(saved for saved in SUMMARY['rows'] if saved['cell_id'] == cell)
                check([row['E'], row['Q'], row['L']], [saved_row[name] for name in ['E', 'Q', 'L']])
                rows.append(row)
    mae = {name: float(np.mean([row['errors'][name] for row in rows])) for name in CONFIG['frozen_coefficients']}
    reduction = {name: mae['E'] - mae[name] for name in ['W', 'T']}
    for name in mae:
        check(mae[name], SUMMARY['mae'][name])
    for name in reduction:
        check(reduction[name], SUMMARY['paired_mae_reduction'][name])
    for unit in SUMMARY['units']:
        matched = [row for row in rows if row['function'] == unit['function'] and row['width'] == unit['width']]
        assert len(matched) == unit['saved_seeds'] == 3
        for name in mae:
            check(np.mean([row['errors'][name] for row in matched]), unit['mae'][name])
        for name in reduction:
            differences = np.array([row['errors']['E'] - row['errors'][name] for row in matched])
            mean = differences.mean()
            radius = t.ppf(.975, 2) * differences.std(ddof=1) / np.sqrt(3)
            check(mean, unit['paired_error_reduction'][name]['mean'])
            check([mean - radius, mean + radius], unit['paired_error_reduction'][name]['seed_95pct_t_interval'])
    predictions = {'P1_target_amplitude': 'supported' if mae['T'] <= .70 and reduction['T'] >= .08 else 'refuted',
                   'P2_width_insufficient': 'supported' if reduction['W'] < .08 else 'refuted'}
    assert predictions == SUMMARY['predictions']
    assert len(rows) == SUMMARY['saved_cells'] == CONFIG['planned_cells'] == 12
    assert len(list((STUDY / 'results').glob('*.npz'))) == 12
    result = {'status': 'passed', 'cells_checked': len(rows), 'checkpoints_rebuilt': 3 * len(rows),
              'new_training_cells': 0, 'preregistration_commits': sorted(commits),
              'min_preregistration_commit_lead_seconds': min_commit_lead_seconds,
              'calibration_group_cv_mae': calibration_cv, 'mae': mae, 'paired_mae_reduction': reduction,
              'predictions': predictions, 'max_initial_parameter_error': init_max_error,
              'max_output_error': output_max_error, 'max_jacobian_error': jacobian_max_error,
              'verified': ['pinned source/input/arrays/early hashes', 'commit before cell start and early forecast before finish',
                           'all saved outputs/J rebuilt from parameters', 'data/labels/normalization/seed initialization',
                           'kernel/trace/fixed residual and target ratios', 'calibration coefficients/group CV',
                           'frozen forecasts/MAE/recipe seed t intervals/P1/P2'],
              'boundary': '仅核验保存证据；无新训练，无独立数据或OOD。',
              'verification_source_sha256': sha(Path(__file__)),
              'summary_sha256': sha(STUDY / 'summary.json')}
    (STUDY / 'executed/independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
