import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
contract = json.loads((ROOT / 'preregistration.json').read_text())
summary = json.loads((ROOT / 'summary.json').read_text())
freeze = summary['freeze_commit']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mm(a, b):
    return np.einsum('ij,jk->ik', a, b, optimize=False)


def independent_loss_grad(x, y, weights):
    acts = [x]
    for w in weights:
        acts.append(mm(acts[-1], w))
    residual = acts[-1] - y
    loss = math.fsum(float(v * v) for v in residual.ravel()) / residual.size
    backward = 2 * residual / len(x)
    gradients = [None] * 4
    for j in range(3, -1, -1):
        gradients[j] = mm(acts[j].T, backward)
        backward = mm(backward, weights[j].T)
    return loss, gradients


def peak(loss, clock):
    smooth = np.array([math.fsum(loss[max(0, i - 5):min(len(loss), i + 6)]) /
                       (min(len(loss), i + 6) - max(0, i - 5)) for i in range(len(loss))])
    x = np.log1p(clock)
    d = np.empty(len(x))
    left, right = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    d[1:-1] = (-right / (left * (left + right))) * smooth[:-2]
    d[1:-1] += ((right - left) / (left * right)) * smooth[1:-1]
    d[1:-1] += (left / (right * (left + right))) * smooth[2:]
    d[0] = (smooth[1] - smooth[0]) / (x[1] - x[0])
    d[-1] = (smooth[-1] - smooth[-2]) / (x[-1] - x[-2])
    return int(np.argmin(d))


max_loss_error = 0.
max_update_error = 0.
max_r2_error = 0.
max_rmse_error = 0.
checkpoint_count = 0
cell_records = []
for cell in contract['cells']:
    path = ROOT / 'results' / (cell['id'] + '.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert receipt['npz_sha256'] == digest(path)
    assert receipt['contract_sha256'] == digest(ROOT / 'preregistration.json')
    assert receipt['freeze_commit'] == freeze
    archive = np.load(path)
    assert np.isfinite(archive['loss']).all()
    rng = np.random.default_rng(cell['seed'])
    weights = [rng.normal(0, .8 / np.sqrt(a), size=(a, b)) for a, b in [(4, 32), (32, 32), (32, 32), (32, 4)]]
    for j, w in enumerate(weights):
        assert np.array_equal(w, archive[f'initial_{j}'])
    steps = archive['checkpoint_steps'].tolist()
    for k, step in enumerate(steps):
        snapshot = [archive[f'weights_{j}'][k] for j in range(4)]
        assert all(np.isfinite(w).all() for w in snapshot)
        loss, gradients = independent_loss_grad(archive['x'], archive['y'], snapshot)
        max_loss_error = max(max_loss_error, abs(loss - archive['loss'][step]))
        checkpoint_count += 1
        if step in [0, 1]:
            following = [w - cell['eta'] * g for w, g in zip(snapshot, gradients)]
            for j, w in enumerate(following):
                max_update_error = max(max_update_error, float(np.max(np.abs(w - archive[f'weights_{j}'][k + 1]))))
    cell_records.append({'id': cell['id'], 'npz_sha256': digest(path), 'checkpoint_count': len(steps),
                         'minimum_finite_loss': float(archive['loss'].min()), 'maximum_finite_loss': float(archive['loss'].max())})

for row in summary['rows']:
    cell = row['cell']
    source = ROOT.parent / 'r001_chain_tc/results' / f'L4_s{cell["seed"]}.npz' if cell['old'] else ROOT / 'results' / (cell['id'] + '.npz')
    loss = np.load(source)['loss']
    clock = np.arange(len(loss), dtype=float)
    assert peak(loss, clock) == int(row['B']['tc'])
    grid = np.arange(3001, dtype=float)
    grid_loss = np.interp(.02 * grid, cell['eta'] * clock, loss)
    assert peak(grid_loss, grid) == int(row['reference']['B']['tc'])
    for fitted, fit_clock in [(row['A'], clock), (row['reference']['A'], cell['eta'] * clock / .02)]:
        pred = fitted['c'] + (fitted['a'] - fitted['c']) * expit(-fitted['k'] * (np.log1p(fit_clock) - fitted['logtc']))
        squared = (pred - loss) ** 2
        rmse = math.sqrt(math.fsum(squared) / len(loss))
        mean = math.fsum(loss) / len(loss)
        r2 = 1 - math.fsum(squared) / math.fsum((loss - mean) ** 2)
        max_rmse_error = max(max_rmse_error, abs(rmse - fitted['rmse']))
        max_r2_error = max(max_r2_error, abs(r2 - fitted['r2']))
assert max_loss_error <= 1e-12
assert max_update_error <= 1e-12
assert max_r2_error <= 1e-12 and max_rmse_error <= 1e-12
for rel, pin in contract['old_input_pins'].items():
    p = REPO / rel
    assert digest(p) == pin['sha256'] and p.stat().st_mtime_ns == pin['mtime_ns']
current = json.loads((REPO / 'central/state.json').read_text())
assert current['round'] == contract['state_guard']['round']
assert {k: v['rounds_done'] for k, v in current['directions'].items()} == contract['state_guard']['directions_rounds_done']
frozen_sec = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', freeze], cwd=REPO))
receipts = [json.loads((ROOT / 'results' / (c['id'] + '.json')).read_text()) for c in contract['cells']]
checks = {'status': 'passed', 'freeze_commit': freeze, 'preregistration_sha256': digest(ROOT / 'preregistration.json'),
          'old_files_unchanged': len(contract['old_input_pins']), 'new_cells': len(cell_records), 'checkpoint_count': checkpoint_count,
          'initial_adjacent_updates_checked': 12, 'max_independent_loss_error': max_loss_error,
          'max_einsum_update_error': max_update_error, 'max_rmse_error': max_rmse_error, 'max_r2_error': max_r2_error,
          'all_B_argmin_indices_match': True, 'sum_training_seconds': sum(r['elapsed_sec'] for r in receipts),
          'minimum_freeze_to_training_seconds': min(r['started_unix'] - frozen_sec for r in receipts),
          'central_round_and_rounds_done_unchanged': True, 'cells': cell_records,
          'warning': '首次训练发出matmul除零/overflow/invalid RuntimeWarning；所有保存loss及checkpoint权重finite，einsum重构及相邻更新通过；警告根因未定，不宣称消除。',
          'verification_scope': '未重训；只重构保存checkpoint、核验最初两次局部更新、重算已有拟合误差与B峰值；不证明local拟合全局最优。'}
(ROOT / 'executed/independent_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + chr(10))
print(json.dumps({k: v for k, v in checks.items() if k != 'cells'}, ensure_ascii=False, indent=2))
