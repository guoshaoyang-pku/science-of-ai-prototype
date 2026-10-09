import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import least_squares
from executed.run_experiment import frozen_contract, sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def fit_a(loss, clock):
    ymax = float(np.max(loss))
    half = max(10, clock[int(np.argmax(loss < ((loss[0] + loss[-1]) / 2)))])
    p0 = np.array([loss[0], loss[-1], 1., np.log1p(half)])
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


def fit_b(loss, clock):
    sm = np.array([loss[max(0, i - 5):min(len(loss), i + 6)].mean() for i in range(len(loss))])
    slope = np.gradient(sm, np.log1p(clock))
    idx = int(np.argmin(slope))
    return {'tc': float(clock[idx]), 'index': idx, 'final_loss': float(loss[-1]), 'slope': float(slope[idx])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    cell = contract['cells'][0]
    path = ROOT / 'results' / (cell['id'] + '.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert receipt['npz_sha256'] == sha(path)
    assert receipt['freeze_commit'] == args.freeze and receipt['contract_sha256'] == contract_sha
    fit_path = ROOT / 'fits' / (cell['id'] + '.json')
    assert {p.name for p in (ROOT / 'fits').iterdir()} <= {fit_path.name}
    assert {p.name for p in (ROOT / 'results').iterdir()} == {path.name, path.with_suffix('.json').name}
    if fit_path.exists():
        row = json.loads(fit_path.read_text())
        assert row['npz_sha256'] == sha(path) and row['contract_sha256'] == contract_sha
        assert row['freeze_commit'] == args.freeze and row['cell'] == cell
        new_fits = 0
    else:
        assert not (ROOT / 'summary.json').exists(), 'summary with missing fit'
        loss = np.load(path)['loss']
        assert len(loss) == cell['steps'] + 1 and loss.dtype == np.float64 and np.isfinite(loss).all()
        tau = cell['eta'] * np.arange(len(loss), dtype=float)
        started = time.time()
        ref_a = fit_a(loss, tau / .02)
        grid = np.arange(3001, dtype=float)
        ref_b = fit_b(np.interp(.02 * grid, tau, loss), grid)
        row = {'cell': cell, 'npz_sha256': sha(path), 'contract_sha256': contract_sha,
               'freeze_commit': args.freeze, 'source_sha256': contract['source_sha256'], 'started_unix': started, 'finished_unix': time.time(),
               'reference': {'A': ref_a, 'B': ref_b, 'tau_A': .02 * ref_a['tc'],
                             'tau_B': .02 * ref_b['tc'], 'AB_ratio': ref_a['tc'] / ref_b['tc']},
               'loss0': float(loss[0]), 'loss_final': float(loss[-1])}
        with fit_path.open('x') as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
            stream.write(chr(10))
        new_fits = 1
    baseline_path = REPO / contract['baseline_fit']
    base = json.loads(baseline_path.read_text())
    old_tau, new_tau = base['reference']['tau_A'], row['reference']['tau_A']
    relative = abs(new_tau - old_tau) / abs(old_tau)
    summary = {'study_id': contract['study_id'], 'freeze_commit': args.freeze,
               'preregistration_sha256': contract_sha, 'training_cells': 1, 'new_A_fits': 1,
               'baseline_fit_sha256': sha(baseline_path), 'fit_sha256': sha(fit_path), 'row': row,
               'comparison': {'baseline_eta': .00125, 'new_eta': cell['eta'],
                              'baseline_tau_A': old_tau, 'new_tau_A': new_tau,
                              'signed_difference': new_tau - old_tau, 'relative_difference': relative,
                              'relative_denominator': 'baseline_tau_A',
                              'prediction_point': contract['predictions']['P1']['point_tau_A'],
                              'point_error': new_tau - contract['predictions']['P1']['point_tau_A']},
               'predictions': {'P1_reference_A_within_5pct': 'supported' if row['reference']['A']['success'] and relative <= .05 else 'refuted'},
               'boundary': contract['boundary']}
    summary_path = ROOT / 'summary.json'
    if summary_path.exists():
        assert json.loads(summary_path.read_text()) == summary, 'saved summary differs'
    else:
        with summary_path.open('x') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write(chr(10))
    print(json.dumps({'new_fits': new_fits, 'reused_fits': 1-new_fits,
                      'comparison': summary['comparison'], 'predictions': summary['predictions']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
