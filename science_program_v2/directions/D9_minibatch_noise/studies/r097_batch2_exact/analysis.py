import json
from executed.run import ROOT, STUDY, cell_label, contract, save, scan_results, sha
import numpy as np


def main():
    cfg, pins = contract()
    receipt = scan_results(cfg, pins)
    if len(receipt['cells']) != len(cfg['cells']):
        raise RuntimeError('Incomplete study')
    rows = []
    for condition in cfg['cells']:
        path = STUDY / 'results' / (cell_label(condition['seed']) + '.npz')
        with np.load(path) as z, np.load(ROOT / condition['baseline']) as baseline, np.load(ROOT / condition['full']) as full:
            risk = z['expected_risk']
            ts = int(np.argmin(risk))
            old_ts = int(np.argmin(baseline['expected_risk']))
            full_ts = int(np.argmin(full['mean_risk']))
            if full_ts != condition['known_full_t_star'] or old_ts != condition['known_B4_t_star']:
                raise RuntimeError('Saved reference endpoint changed')
            lo, hi = condition['predicted_integer_t_star_range']
            row = {'seed': condition['seed'], 'rank': int(z['basis'].shape[1]), 'batch': cfg['batch'],
                   't_star': ts, 'B4_t_star': old_ts, 'full_t_star': full_ts,
                   'aligned_ratio': ts / full_ts, 'B2_minus_B4_steps': ts-old_ts, 'B2_over_B4_time': ts/old_ts,
                   'predicted_integer_t_star_range': [lo, hi], 'P1_pass': bool(lo <= ts <= hi),
                   'twofold_descriptive_pass': bool(.5 <= ts/full_ts <= 2),
                   'min_expected_risk': float(risk[ts]), 'mean_risk_at_min': float(z['mean_risk'][ts]),
                   'covariance_risk_at_min': float(z['covariance_risk'][ts]),
                   'B4_min_expected_risk': float(baseline['expected_risk'][old_ts]),
                   'B4_covariance_risk_at_min': float(baseline['covariance_risk'][old_ts]),
                   'min_risk_difference_B2_minus_B4': float(risk[ts]-baseline['expected_risk'][old_ts]),
                   'minimum_neighbor_margin': float(min(risk[ts-1], risk[ts+1])-risk[ts]) if 0 < ts < cfg['steps'] else None,
                   'mean_coordinates_B4_maxabs': float(np.max(np.abs(z['mean_coordinates']-baseline['mean_coordinates']))),
                   'mean_risk_B4_maxabs': float(np.max(np.abs(z['mean_risk']-baseline['mean_risk']))),
                   'B2_minus_B4_covariance_risk_min': float(np.min(z['covariance_risk']-baseline['covariance_risk'])),
                   'minimum_covariance_eigenvalue': float(z['minimum_covariance_eigenvalue'].min()),
                   'full_risk_saved_control_maxabs': float(np.max(np.abs(z['full_risk']-full['mean_risk']))),
                   'registered_C_basis': bool(z['basis'].flags.c_contiguous),
                   'full_arrays_equal_baseline': bool(np.array_equal(z['full_risk'],baseline['full_risk'])
                       and np.array_equal(z['full_mean_coordinates'],baseline['full_mean_coordinates'])
                       and np.array_equal(z['basis'],baseline['basis'])
                       and np.array_equal(z['full_covariance_maxabs'],baseline['full_covariance_maxabs'])),
                   'full_covariance_maxabs_inherited': float(z['full_covariance_maxabs']),
                   'finite': bool(all(np.isfinite(z[k]).all() for k in z.files)), 'interior': bool(0 < ts < cfg['steps'])}
            rows.append(row)
    audit = json.loads((STUDY / 'executed/input_audit.json').read_text())
    valid = all(row['finite'] and row['interior'] and row['minimum_covariance_eigenvalue'] >= -1e-10
                and row['registered_C_basis'] and row['full_arrays_equal_baseline'] and row['full_risk_saved_control_maxabs'] <= 2e-8
                and row['mean_coordinates_B4_maxabs'] <= cfg['validation']['mean_coordinates_atol']
                and row['mean_risk_B4_maxabs'] <= cfg['validation']['mean_risk_atol'] for row in rows)
    passed = sum(row['P1_pass'] for row in rows)
    result = {'study': cfg['study'], 'round': cfg['round'], 'direction_round': cfg['direction_round'],
              'direction': cfg['direction'], 'domain': cfg['domain'], 'status': 'complete',
              'round_result': 'ok' if valid else 'partial',
              'predictions': {'P1': {'status': 'supported' if passed == len(rows) else 'refuted', 'passed': passed, 'total': len(rows)}},
              'validation': {'passed': bool(valid), 'old_files_unchanged': True, 'old_file_count': len(audit['old_files'])},
              'counts': {'recipe_units': 1, 'paired_data_seeds': 2, 'saved_exact_cells': 2, 'new_training': 0,
                         'new_batch_trajectories': 0, 'reused_full_controls': 2, 'reused_B4_exact_cells': 2,
                         'old_B4_B8_full_recomputed': 0, 'new_MC_or_bootstrap': 0},
              'gamma_B2': 62/126, 'gamma_B4': 60/252, 'gamma_ratio': 31/15,
              'compute_seconds': receipt['compute_seconds'], 'cells': rows, 'pins': pins,
              'scope': cfg['boundaries']}
    save(STUDY / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not valid:
        raise RuntimeError('Analysis validation failed')


if __name__ == '__main__':
    main()
