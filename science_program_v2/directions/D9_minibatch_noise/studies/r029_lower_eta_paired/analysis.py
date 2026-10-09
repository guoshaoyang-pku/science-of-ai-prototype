import hashlib
import json
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]
OLD = ROOT / 'directions/D9_minibatch_noise/studies/r025_batch_risk_shift_corrected'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    rec = json.loads((STUDY / 'results/receipt.json').read_text())
    old_summary = json.loads((OLD / 'summary.json').read_text())
    old_rows = {(r['seed'], r['batch'], r['variance']): r for r in old_summary['cells']}
    saved, controls = {}, []
    for label, digest in rec['cells'].items():
        path = STUDY / 'results' / (label + '.json')
        row = json.loads(path.read_text())
        if sha(path) != digest or sha(path.with_suffix('.npz')) != row['arrays_sha256']:
            raise RuntimeError('Hash mismatch: ' + label)
        with np.load(path.with_suffix('.npz')) as z:
            arrays = {k: z[k] for k in z.files}
        c = row['cell']
        saved[(c['seed'], c['lr'], c['batch'], c['noise_variance'])] = arrays
        if c['batch'] == 64:
            controls.append({**c, 'realized_t_star': int(np.argmin(arrays['mean_risk'])),
                             'expected_t_star': int(np.argmin(arrays['expected_risk']))})
    if len(saved) != cfg['counts']['planned_cells']:
        raise RuntimeError('Incomplete cells')
    rows = []
    for seed in cfg['seeds']:
        for batch in cfg['minibatches']:
            for variance in cfg['variances']:
                current = saved[(seed, .1, batch, variance)]
                with np.load(OLD / 'results' / f'n64_seed{seed}_B{batch}_var{variance:g}.npz') as z:
                    old_risk = z['replicate_risk'].copy()
                full = saved[(seed, .1, 64, variance)]
                old_full = saved[(seed, .3, 64, variance)]
                ts = int(np.argmin(current['mean_risk']))
                full_real = int(np.argmin(full['mean_risk']))
                full_expected = int(np.argmin(full['expected_risk']))
                old_ts = int(np.argmin(old_risk.mean(0)))
                old_full_real = int(np.argmin(old_full['mean_risk']))
                old = old_rows[(seed, batch, variance)]
                if old_ts != old['t_star'] or int(np.argmin(old_full['expected_risk'])) != old['full_t_star']:
                    raise RuntimeError('Historical t_star not reproduced')
                ratio, legacy = ts / full_real, ts / full_expected
                old_ratio = old_ts / old_full_real
                rng = np.random.default_rng(cfg['bootstrap_seed_base'] + 100000 * seed + 1000 * batch + int(variance * 100))
                indices = rng.integers(0, cfg['replicates'], size=(cfg['bootstrap_replicates'], cfg['replicates']))
                new_boot = np.argmin(current['replicate_risk'][indices].mean(1), axis=1) / full_real
                old_boot = np.argmin(old_risk[indices].mean(1), axis=1) / old_full_real
                change = np.abs(new_boot - 1) - np.abs(old_boot - 1)
                rows.append({'seed': seed, 'batch': batch, 'variance': variance, 't_star': ts,
                             'full_realized_t_star': full_real, 'full_expected_t_star': full_expected,
                             'aligned_ratio': ratio, 'legacy_ratio': legacy,
                             'old_t_star': old_ts, 'old_full_realized_t_star': old_full_real,
                             'old_aligned_ratio': old_ratio, 'old_legacy_ratio': old['ratio'],
                             'absolute_deviation_change': abs(ratio - 1) - abs(old_ratio - 1),
                             'aligned_ratio_bootstrap_95pct': np.quantile(new_boot, [.025, .975]).tolist(),
                             'paired_deviation_change_bootstrap_95pct': np.quantile(change, [.025, .975]).tolist(),
                             'replicate_t_stars': np.argmin(current['replicate_risk'], axis=1).tolist(),
                             'se_at_min': float(current['standard_error'][ts]),
                             'min_risk': float(current['mean_risk'][ts]),
                             'interior': bool(0 < ts < cfg['conditions'][0]['steps'])})
    groups = []
    for batch in cfg['minibatches']:
        for variance in cfg['variances']:
            selected = [r for r in rows if r['batch'] == batch and r['variance'] == variance]
            groups.append({'batch': batch, 'variance': variance, 'seed_count': len(selected),
                           'aligned_range': [min(r['aligned_ratio'] for r in selected), max(r['aligned_ratio'] for r in selected)],
                           'legacy_range': [min(r['legacy_ratio'] for r in selected), max(r['legacy_ratio'] for r in selected)],
                           'old_aligned_range': [min(r['old_aligned_ratio'] for r in selected), max(r['old_aligned_ratio'] for r in selected)],
                           'aligned_passed': sum(.5 <= r['aligned_ratio'] <= 2 for r in selected),
                           'legacy_passed': sum(.5 <= r['legacy_ratio'] <= 2 for r in selected),
                           'deviation_decreased': sum(r['absolute_deviation_change'] < 0 for r in selected)})
    legacy_pass = sum(.5 <= r['legacy_ratio'] <= 2 for r in rows)
    aligned_pass = sum(.5 <= r['aligned_ratio'] <= 2 for r in rows)
    finite = all(np.isfinite(a).all() for z in saved.values() for a in z.values())
    interior = all(r['interior'] for r in rows) and all(0 < r['realized_t_star'] < (2304 if r['lr'] == .1 else 768)
                                                     and 0 < r['expected_t_star'] < (2304 if r['lr'] == .1 else 768) for r in controls)
    summary = {'study': cfg['study'], 'round': 29, 'direction_round': 2, 'direction': 'D9_minibatch_noise',
               'domain': 'development', 'status': 'complete', 'round_result': 'ok',
               'counts': {'planned_cells': 48, 'saved_cells': len(saved), 'new_minibatch_trajectories': 192,
                          'new_full_batch_trajectories': 24, 'reused_minibatch_cells': 24, 'recipe_units': 6},
               'compute_seconds': rec['compute_seconds'],
               'predictions': {'P1_legacy_ratio_half_to_two': {'status': 'supported' if legacy_pass == 24 else 'refuted', 'passed': legacy_pass, 'total': 24},
                               'P2_aligned_ratio_half_to_two': {'status': 'supported' if aligned_pass == 24 else 'refuted', 'passed': aligned_pass, 'total': 24},
                               'P3_finite_interior': {'status': 'supported' if finite and interior else 'refuted', 'finite': bool(finite), 'interior': bool(interior)}},
               'cells': rows, 'full_batch_controls': controls, 'groups': groups,
               'bootstrap': {'replicates': cfg['bootstrap_replicates'], 'scope': 'conditional on fixed data/epsilon; paired batch trajectory index resampling; descriptive percentile interval, not a confidence interval across recipes'},
               'sources': cfg['input_sha256'], 'method': 'first argmin of batch-trajectory mean audit MSE; legacy divides by label-noise expected full risk, aligned divides by same fixed-label full risk; fixed eta*T=230.4'}
    (STUDY / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['predictions'], ensure_ascii=False, indent=2))
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
