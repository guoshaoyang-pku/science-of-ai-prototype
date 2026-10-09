import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t
from executed.run_experiment import frozen_contract, sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'r001_chain_tc'


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


def interval(values):
    a = np.array(values)
    margin = float(student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max()),
            'paired_95pct_t_interval': [float(a.mean() - margin), float(a.mean() + margin)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    old_summary = json.loads((OLD / 'analysis/summary.json').read_text())
    rows = []
    fits_dir = ROOT / 'fits'
    fits_dir.mkdir(exist_ok=True)
    cases = [{'id': f'L4_s{s}_eta020', 'seed': s, 'eta': .02, 'steps': 3000, 'old': True} for s in [11, 22, 33]]
    cases += [dict(c, old=False) for c in contract['cells']]
    for cell in cases:
        eta = cell['eta']
        path = OLD / 'results' / f'L4_s{cell["seed"]}.npz' if cell['old'] else ROOT / 'results' / (cell['id'] + '.npz')
        if not cell['old']:
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['npz_sha256'] == sha(path)
            assert receipt['freeze_commit'] == args.freeze and receipt['contract_sha256'] == contract_sha
        fit_path = fits_dir / (cell['id'] + '.json')
        if fit_path.exists():
            row = json.loads(fit_path.read_text())
            assert row['npz_sha256'] == sha(path) and row['contract_sha256'] == contract_sha
            assert row['freeze_commit'] == args.freeze
        else:
            loss = np.load(path)['loss']
            step_clock = np.arange(len(loss), dtype=float)
            tau = eta * step_clock
            if cell['old']:
                previous = next(r for r in old_summary['rows'] if r['depth'] == 4 and r['seed'] == cell['seed'])
                original_a, original_b = previous['A'].copy(), previous['B'].copy()
                original_a['pass'] = previous['A_pass']
            else:
                original_a, original_b = fit_a(loss, step_clock), fit_b(loss, step_clock)
            ref_a = fit_a(loss, tau / .02) if not cell['old'] else original_a.copy()
            grid_tau = np.arange(3001) * .02
            grid_loss = np.interp(grid_tau, tau, loss)
            ref_b = fit_b(grid_loss, np.arange(3001, dtype=float)) if not cell['old'] else original_b.copy()
            ratio = original_a['tc'] / original_b['tc']
            row = {'cell': cell, 'npz_sha256': sha(path), 'contract_sha256': contract_sha, 'freeze_commit': args.freeze,
                   'A': original_a, 'B': original_b, 'AB_ratio': ratio, 'AB_agree': bool(.5 <= ratio <= 2),
                   'tau_A': eta * original_a['tc'], 'tau_B': eta * original_b['tc'],
                   'reference': {'A': ref_a, 'B': ref_b, 'tau_A': .02 * ref_a['tc'], 'tau_B': .02 * ref_b['tc'],
                                 'AB_ratio': ref_a['tc'] / ref_b['tc']},
                   'loss0': float(loss[0]), 'loss_final': float(loss[-1])}
            with fit_path.open('x') as stream:
                json.dump(row, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        rows.append(row)
    by_eta = {}
    for eta in [.02, .005, .00125]:
        z = [r for r in rows if r['cell']['eta'] == eta]
        times = np.array([r['A']['tc'] for r in z])
        by_eta[str(eta)] = {'A_pass_count': sum(r['A']['pass'] for r in z), 'AB_agree_count': sum(r['AB_agree'] for r in z),
                            'AB_ratio_range': [min(r['AB_ratio'] for r in z), max(r['AB_ratio'] for r in z)],
                            'tc_A_mean': float(times.mean()), 'tc_A_cv': float(times.std(ddof=1) / times.mean()),
                            'reference_AB_ratio_range': [min(r['reference']['AB_ratio'] for r in z), max(r['reference']['AB_ratio'] for r in z)]}
    pairs = []
    for seed in [11, 22, 33]:
        selected = {r['cell']['eta']: r for r in rows if r['cell']['seed'] == seed}
        base, mid, fine = [selected[e] for e in [.02, .005, .00125]]
        reduction = 1 - abs(np.log(fine['AB_ratio'])) / abs(np.log(base['AB_ratio']))
        conv = {p: abs(fine['reference']['tau_' + p] - mid['reference']['tau_' + p]) / max(abs(fine['reference']['tau_' + p]), 1e-12) for p in ['A', 'B']}
        pairs.append({'seed': seed, 'fine_minus_base_ratio': fine['AB_ratio'] - base['AB_ratio'],
                      'log_offset_reduction': float(reduction), 'reference_convergence_relative': conv,
                      'P1_pass': bool(fine['AB_agree'] and reduction >= .25), 'P2_pass': bool(max(conv.values()) <= .10)})
    summary = {'study_id': contract['study_id'], 'freeze_commit': args.freeze, 'contract_sha256': contract_sha,
               'training_cells': 6, 'old_training_cells_reused': 3, 'rows': rows, 'by_eta': by_eta, 'pairs': pairs,
               'paired_ratio_change': interval([p['fine_minus_base_ratio'] for p in pairs]),
               'paired_log_offset_reduction': interval([p['log_offset_reduction'] for p in pairs]),
               'predictions': {'P1_learning_rate_resolves_original_offset': 'supported' if all(p['P1_pass'] for p in pairs) else 'refuted',
                               'P2_reference_protocol_convergence': 'supported' if all(p['P2_pass'] for p in pairs) else 'refuted'},
               'fit_sha256': {p.name: sha(p) for p in sorted(fits_dir.glob('*.json'))}}
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'fit_sha256')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
