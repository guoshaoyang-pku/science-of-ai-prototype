import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
contract = json.loads((ROOT / 'preregistration.json').read_text())
summary = json.loads((ROOT / 'summary.json').read_text())
freeze = summary['freeze_commit']
from run_experiment import frozen_contract
frozen_contract(freeze)


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


cell = contract['cells'][0]
path = ROOT / 'results' / (cell['id'] + '.npz')
receipt = json.loads(path.with_suffix('.json').read_text())
assert receipt['npz_sha256'] == digest(path)
assert receipt['freeze_commit'] == freeze
assert receipt['contract_sha256'] == digest(ROOT / 'preregistration.json')
archive = np.load(path)
assert np.isfinite(archive['loss']).all()
baseline = np.load(REPO / contract['baseline_npz'])
for key, expected in contract['array_sha256'].items():
    assert np.array_equal(archive[key], baseline[key])
    assert hashlib.sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest() == expected
steps = archive['checkpoint_steps'].tolist()
max_loss_error = max_update_error = 0.
for k, step in enumerate(steps):
    weights = [archive[f'weights_{j}'][k] for j in range(4)]
    assert all(np.isfinite(w).all() for w in weights)
    loss, gradients = independent_loss_grad(archive['x'], archive['y'], weights)
    max_loss_error = max(max_loss_error, abs(loss - archive['loss'][step]))
    if step in [0, 1]:
        for j, (w, g) in enumerate(zip(weights, gradients)):
            error = np.max(np.abs(w - cell['eta'] * g - archive[f'weights_{j}'][k + 1]))
            max_update_error = max(max_update_error, float(error))
row = summary['row']
fitted = row['reference']['A']
loss = archive['loss']
tau = cell['eta'] * np.arange(len(loss), dtype=float)
pred = fitted['c'] + (fitted['a'] - fitted['c']) * expit(-fitted['k'] * (np.log1p(tau/.02) - fitted['logtc']))
squared = (pred - loss) ** 2
rmse = math.sqrt(math.fsum(squared) / len(loss))
mean = math.fsum(loss) / len(loss)
r2 = 1 - math.fsum(squared) / math.fsum((loss - mean) ** 2)
grid = np.arange(3001, dtype=float)
assert peak(np.interp(.02 * grid, tau, loss), grid) == row['reference']['B']['index']
base = json.loads((REPO / contract['baseline_fit']).read_text())['reference']['tau_A']
new_tau = .02 * math.expm1(fitted['logtc'])
relative = abs(new_tau - base) / abs(base)
assert abs(new_tau - summary['comparison']['new_tau_A']) <= 1e-14
assert abs(relative - summary['comparison']['relative_difference']) <= 1e-12
assert summary['predictions']['P1_reference_A_within_5pct'] == ('supported' if fitted['success'] and relative <= .05 else 'refuted')
assert max_loss_error <= 1e-12 and max_update_error <= 1e-12
assert abs(rmse - fitted['rmse']) <= 1e-12 and abs(r2 - fitted['r2']) <= 1e-12
for rel, pin in contract['old_input_pins'].items():
    p = REPO / rel
    assert digest(p) == pin['sha256'] and p.stat().st_mtime_ns == pin['mtime_ns']
current = json.loads((REPO / 'central/state.json').read_text())
assert current['round'] == contract['state_guard']['round']
assert {k:v['rounds_done'] for k,v in current['directions'].items()} == contract['state_guard']['directions_rounds_done']
frozen_sec = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', freeze], cwd=REPO))
assert receipt['started_unix'] > frozen_sec and row['started_unix'] > receipt['finished_unix']
checks = {'status': 'passed', 'freeze_commit': freeze,
          'preregistration_sha256': digest(ROOT / 'preregistration.json'),
          'old_files_hash_mtime_unchanged': len(contract['old_input_pins']),
          'new_cells': 1, 'checkpoint_count': len(steps), 'initial_adjacent_updates_checked': 2,
          'initial_data_and_layers_equal_to_baseline': True,
          'max_independent_loss_error': max_loss_error, 'max_einsum_update_error': max_update_error,
          'max_rmse_error': abs(rmse-fitted['rmse']), 'max_r2_error': abs(r2-fitted['r2']),
          'reference_B_argmin_matches': True, 'prediction_recomputed': True,
          'freeze_to_training_seconds': receipt['started_unix'] - frozen_sec,
          'training_seconds': receipt['elapsed_sec'], 'central_round_and_rounds_done_unchanged': True,
          'scope': '保存checkpoint及2次局部更新重构；已存拟合误差重算；0训练、0拟合。不证明local拟合全局最优或B真峰收敛。'}
out = ROOT / 'executed/independent_verification.json'
if out.exists():
    assert json.loads(out.read_text()) == checks
else:
    with out.open('x') as stream:
        json.dump(checks, stream, ensure_ascii=False, indent=2)
        stream.write(chr(10))
print(json.dumps(checks,ensure_ascii=False,indent=2))
