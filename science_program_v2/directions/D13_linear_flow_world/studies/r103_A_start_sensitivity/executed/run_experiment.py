import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_contract(freeze):
    path = ROOT / 'preregistration.json'
    contract = json.loads(path.read_text())
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', freeze, 'HEAD'], cwd=REPO)
    rel = str(path.relative_to(REPO))
    additions = subprocess.check_output(
        ['git', 'log', '--diff-filter=A', '--format=%H', '--', rel], cwd=REPO).decode().splitlines()
    assert additions == [freeze], 'freeze must uniquely introduce the contract'
    assert subprocess.check_output(['git', 'show', f'{freeze}:{rel}'], cwd=REPO) == path.read_bytes()
    for rel, expected in contract['source_sha256'].items():
        source = ROOT / rel
        assert sha(source) == expected, f'source hash mismatch: {rel}'
        assert subprocess.check_output(
            ['git', 'show', f'{freeze}:{source.relative_to(REPO)}'], cwd=REPO) == source.read_bytes()
    for rel, pin in contract['old_input_pins'].items():
        old = REPO / rel
        assert sha(old) == pin['sha256'] and old.stat().st_mtime_ns == pin['mtime_ns'], rel
    state = json.loads((REPO / 'central/state.json').read_text())
    assert state['round'] == contract['state_guard']['round']
    assert {k: v['rounds_done'] for k, v in state['directions'].items()} == contract['state_guard']['rounds_done']
    return contract, sha(path)


def fit_a(loss, clock, k0):
    ymax = float(np.max(loss))
    p0 = np.array([loss[0], loss[-1], k0, np.log(11.)])
    lb = np.array([0., 0., 1e-5, 0.])
    ub = np.array([max(1., ymax * 3), max(1., ymax * 3), 30., np.log1p(1e7)])
    logtime = np.log1p(clock)

    def curve(p):
        a, c, k, logtc = p
        return c + (a - c) / (1 + np.exp(np.clip(k * (logtime - logtc), -700, 700)))

    out = least_squares(lambda p: curve(p) - loss, np.clip(p0, lb + 1e-9, ub - 1e-9),
                        bounds=(lb, ub), max_nfev=4000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    pred = curve(out.x)
    rmse = float(np.sqrt(np.mean((pred - loss) ** 2)))
    r2 = 1 - float(np.sum((pred - loss) ** 2)) / float(np.sum((loss - loss.mean()) ** 2))
    return {'a': float(out.x[0]), 'c': float(out.x[1]), 'k': float(out.x[2]),
            'logtc': float(out.x[3]), 'tc': float(np.expm1(out.x[3])), 'rmse': rmse,
            'normalized_rmse': rmse / max(float(loss[0] - loss[-1]), 1e-15),
            'r2': r2, 'success': bool(out.success), 'nfev': out.nfev,
            'pass': bool(out.success and rmse / max(float(loss[0] - loss[-1]), 1e-15) <= .05 and r2 >= .99)}


def saved_rows(contract, contract_sha, freeze):
    expected = {cell['id'] + '.json': cell for cell in contract['cells']}
    paths = list((ROOT / 'results').iterdir())
    assert {p.name for p in paths} <= set(expected), 'unknown/orphan result file'
    rows = {}
    for path in paths:
        assert path.is_file()
        row = json.loads(path.read_text())
        assert row['cell'] == expected[path.name]
        assert row['freeze_commit'] == freeze and row['contract_sha256'] == contract_sha
        assert row['npz_sha256'] == contract['old_input_pins'][contract['source_npz']]['sha256']
        assert row['source_sha256'] == contract['source_sha256']
        rows[row['cell']['id']] = row
    summary_path = ROOT / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        assert len(rows) == len(expected), 'summary with missing results'
        assert summary['freeze_commit'] == freeze and summary['preregistration_sha256'] == contract_sha
        for rel, expected_sha in summary['result_sha256'].items():
            assert sha(ROOT / rel) == expected_sha, f'saved result changed: {rel}'
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    rows = saved_rows(contract, contract_sha, args.freeze)
    freeze_time = int(subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', args.freeze], cwd=REPO))
    counts = {'new_training_cells': 0, 'new_fits': 0, 'reused_fits': 0}
    loss = None
    for cell in contract['cells']:
        if cell['id'] in rows:
            counts['reused_fits'] += 1
            continue
        if loss is None:
            with np.load(REPO / contract['source_npz']) as archive:
                loss = archive['loss'].copy()
            assert loss.dtype == np.float64 and len(loss) == 192001 and np.isfinite(loss).all()
            clock = .0003125 * np.arange(len(loss), dtype=float) / .02
        started = time.time()
        assert started > freeze_time
        fitted = fit_a(loss, clock, cell['k0'])
        row = {'cell': cell, 'freeze_commit': args.freeze, 'contract_sha256': contract_sha,
               'source_sha256': contract['source_sha256'],
               'npz_sha256': sha(REPO / contract['source_npz']),
               'started_unix': started, 'finished_unix': time.time(), 'A': fitted,
               'tau_A': .02 * fitted['tc'], 'loss0': float(loss[0]), 'loss_final': float(loss[-1])}
        path = ROOT / 'results' / (cell['id'] + '.json')
        with path.open('x') as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
            stream.write(chr(10))
        counts['new_fits'] += 1
        print(json.dumps({'saved': cell['id'], 'success': fitted['success'], 'tau_A': row['tau_A']}), flush=True)
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
