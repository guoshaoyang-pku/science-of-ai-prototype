import json
import numpy as np
from executed.run import ROOT, STUDY, contract, save, sha


def main():
    cfg, pins = contract()
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    if len(receipt['cells']) != 2:
        raise RuntimeError('Incomplete study')
    rows = []
    for condition in cfg['cells']:
        seed = condition['seed']
        label = f'n64_seed{seed}_eta0.1_B8_var1_exact'
        path = STUDY / 'results' / (label + '.json')
        metadata = json.loads(path.read_text())
        if sha(path) != receipt['cells'][label] or sha(path.with_suffix('.npz')) != metadata['arrays_sha256']:
            raise RuntimeError('Cell hash mismatch')
        with np.load(path.with_suffix('.npz')) as z:
            risk = z['expected_risk']
            mean_risk = z['mean_risk']
            covariance_risk = z['covariance_risk']
            ts = int(np.argmin(risk))
            full_curve_error = None
            with np.load(ROOT / condition['full']) as full:
                full_t = int(np.argmin(full['mean_risk']))
                full_curve_error = float(np.max(np.abs(z['full_risk'] - full['mean_risk'])))
            if full_t != condition['known_full_t_star']:
                raise RuntimeError('Fixed denominator changed')
            row = {'seed': seed, 'rank': metadata['rank'], 't_star': ts, 'full_t_star': full_t,
                   'aligned_ratio': ts / full_t, 'P1_pass': bool(.5 <= ts / full_t <= 2),
                   'min_expected_risk': float(risk[ts]), 'mean_risk_at_min': float(mean_risk[ts]),
                   'covariance_risk_at_min': float(covariance_risk[ts]),
                   'full_curve_maxabs_error': full_curve_error,
                   'mean_equals_full_maxabs_error': float(np.max(np.abs(mean_risk - z['full_risk']))),
                   'full_covariance_maxabs': float(z['full_covariance_maxabs']),
                   'minimum_covariance_eigenvalue': float(z['minimum_covariance_eigenvalue'].min()),
                   'finite': bool(all(np.isfinite(z[k]).all() for k in z.files)),
                   'interior': bool(0 < ts < cfg['steps']),
                   'minimum_neighbor_margin': float(min(risk[ts-1], risk[ts+1])-risk[ts]) if 0 < ts < cfg['steps'] else None}
            blocks = []
            for name, range_ in [('old_block', [0, 63]), ('new_block', [64, 127])]:
                with np.load(ROOT / condition[name]) as block:
                    curve = block['mean_risk']
                    bt = int(np.argmin(curve))
                    blocks.append({'replicate_range': range_, 'M': 64, 't_star': bt, 'aligned_ratio': bt / full_t,
                                   'mc_minus_exact_t_star': bt-ts, 'mc_minus_exact_ratio': (bt-ts)/full_t,
                                   'curve_difference_maxabs': float(np.max(np.abs(curve-risk))),
                                   'curve_difference_rmse': float(np.sqrt(np.mean((curve-risk)**2)))})
            row['independent_blocks_descriptive_only'] = blocks
            rows.append(row)
    audit = json.loads((STUDY / 'executed/input_audit.json').read_text())
    unchanged = all(sha(ROOT / p) == value['sha256'] and (ROOT / p).stat().st_mtime_ns == value['mtime_ns']
                    for p, value in audit['old_files'].items())
    passed = sum(row['P1_pass'] for row in rows)
    valid = unchanged and all(row['finite'] and row['interior'] and row['full_curve_maxabs_error'] <= 2e-8
                              and row['full_covariance_maxabs'] == 0 and row['minimum_covariance_eigenvalue'] >= -1e-10 for row in rows)
    summary = {'study': cfg['study'], 'round': 74, 'direction_round': 5, 'direction': 'D9_minibatch_noise',
               'domain': 'development', 'status': 'complete', 'round_result': 'ok' if valid else 'partial',
               'predictions': {'P1': {'status': 'supported' if passed == 2 else 'refuted', 'passed': passed, 'total': 2}},
               'validation': {'passed': valid, 'old_files_unchanged': unchanged, 'old_file_count': len(audit['old_files'])},
               'counts': {'recipe_units': 1, 'paired_data_seeds': 2, 'saved_exact_cells': 2, 'new_training': 0,
                          'new_batch_trajectories': 0, 'new_full_trajectories': 0, 'reused_full_controls': 2,
                          'reused_mc_blocks': 4, 'reused_mc_trajectories': 256},
               'compute_seconds': receipt['compute_seconds'], 'cells': rows, 'pins': pins,
               'scope': '固定data/epsilon的有限窗条件batch期望；非epsilon期望、非跨数据统计保证。两个MC块分别描述，不合并，不追溯改判旧M8/M64。已知矩递推不登记新发现。'}
    save(STUDY / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
