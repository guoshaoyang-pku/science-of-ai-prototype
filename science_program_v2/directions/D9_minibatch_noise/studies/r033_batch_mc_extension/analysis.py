import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from executed.run import ROOT, STUDY, contract, save, sha


def bootstrap(risk, full_t, seed, count):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(risk), size=(count, len(risk)))
    t_stars = np.empty(count, dtype=np.int64)
    for start in range(0, count, 50):
        chosen = indices[start:start + 50]
        t_stars[start:start + len(chosen)] = np.argmin(risk[chosen].mean(1), axis=1)
    return t_stars, np.quantile(t_stars / full_t, [.025, .975])


def main():
    cfg, pins = contract()
    rec = json.loads((STUDY / 'results/receipt.json').read_text())
    if len(rec['cells']) != 2:
        raise RuntimeError('Incomplete cells')
    rows, bootstrap_arrays = [], {}
    for c in cfg['cells']:
        seed = c['seed']
        label = f'n64_seed{seed}_eta0.1_B8_var1_M64'
        path = STUDY / 'results' / (label + '.json')
        meta = json.loads(path.read_text())
        if sha(path) != rec['cells'][label] or sha(path.with_suffix('.npz')) != meta['arrays_sha256']:
            raise RuntimeError('Hash mismatch')
        with np.load(path.with_suffix('.npz')) as z:
            current = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c['old_mini']) as z:
            old = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c['full']) as z:
            full_t = int(np.argmin(z['mean_risk']))
            expected_t = int(np.argmin(z['expected_risk']))
        rng_seed = cfg['bootstrap_seed_base'] + 100000 * seed + 8000 + 100
        old_boot, old_ci = bootstrap(old['replicate_risk'], full_t, rng_seed, cfg['bootstrap_replicates'])
        new_boot, ci = bootstrap(current['replicate_risk'], full_t, rng_seed, cfg['bootstrap_replicates'])
        ts, old_ts = int(np.argmin(current['mean_risk'])), int(np.argmin(old['mean_risk']))
        prefix = all(np.array_equal(current[k][:8], old[k]) for k in ['replicate_risk', 'final_heads', 'batch_seeds'])
        width_ratio = float((ci[1] - ci[0]) / (old_ci[1] - old_ci[0]))
        envelope = c['predicted_bootstrap_envelope']
        finite = all(np.isfinite(a).all() for a in current.values())
        rows.append({'seed': seed, 'batch': 8, 'variance': 1, 'eta': .1, 'replicates': 64,
                     't_star': ts, 'full_realized_t_star': full_t, 'full_expected_t_star': expected_t,
                     'aligned_ratio': ts / full_t, 'legacy_ratio': ts / expected_t,
                     'old_t_star': old_ts, 'old_aligned_ratio': old_ts / full_t,
                     'aligned_ratio_bootstrap_95pct': ci.tolist(), 'old_bootstrap_95pct': old_ci.tolist(),
                     'bootstrap_width_ratio': width_ratio, 'bootstrap_crosses_two': bool(ci[0] <= 2 <= ci[1]),
                     'bootstrap_fraction_over_two': float(np.mean(new_boot / full_t > 2)),
                     'P1_pass': bool(.5 <= ts / full_t <= 2),
                     'P2_pass': bool(width_ratio <= .5 and ci[0] >= envelope[0] and ci[1] <= envelope[1]),
                     'prefix_unchanged': prefix, 'finite': bool(finite), 'interior': bool(0 < ts < cfg['steps']),
                     'min_risk': float(current['mean_risk'][ts]),
                     'se_at_min': float(current['standard_error'][ts]),
                     'replicate_t_stars': np.argmin(current['replicate_risk'], axis=1).tolist()})
        bootstrap_arrays[f'seed{seed}_M8_t_star'] = old_boot
        bootstrap_arrays[f'seed{seed}_M64_t_star'] = new_boot
    bootstrap_path = STUDY / 'results/bootstrap.npz'
    if bootstrap_path.exists():
        with np.load(bootstrap_path) as z:
            if set(z.files) != set(bootstrap_arrays) or not all(np.array_equal(z[k], a) for k, a in bootstrap_arrays.items()):
                raise RuntimeError('Saved bootstrap mismatch; refusing overwrite')
    else:
        np.savez_compressed(bootstrap_path, **bootstrap_arrays)
    predictions = {}
    for name in ['P1', 'P2']:
        passed = sum(r[name + '_pass'] for r in rows)
        predictions[name] = {'status': 'supported' if passed == 2 else 'refuted', 'passed': passed, 'total': 2}
    integrity = all(r['prefix_unchanged'] and r['finite'] and r['interior'] for r in rows)
    predictions['P3'] = {'status': 'supported' if integrity else 'refuted'}
    summary = {'study': cfg['study'], 'round': 33, 'direction_round': 3, 'direction': 'D9_minibatch_noise',
               'domain': 'development', 'status': 'complete', 'round_result': 'ok', 'predictions': predictions,
               'counts': {'recipe_units': 1, 'paired_data_seeds': 2, 'saved_cells': 2,
                          'new_batch_trajectories': 112, 'retained_batch_trajectories': 16, 'total_batch_trajectories': 128,
                          'reused_full_controls': 2, 'new_full_trajectories': 0},
               'compute_seconds': rec['compute_seconds'], 'cells': rows, 'pins': pins,
               'bootstrap': {'replicates': cfg['bootstrap_replicates'], 'arrays_sha256': sha(bootstrap_path),
                             'scope': 'fixed data/epsilon conditional trajectory bootstrap, percentile descriptive interval; nested M8/M64 samples are not independent'},
               'method': 'first argmin of the mean audit risk; M64 retains M8 prefix; aligned denominator uses the same epsilon; no exact batch expectation claim'}
    save(STUDY / 'summary.json', summary)
    print(json.dumps({'predictions': predictions, 'cells': rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
