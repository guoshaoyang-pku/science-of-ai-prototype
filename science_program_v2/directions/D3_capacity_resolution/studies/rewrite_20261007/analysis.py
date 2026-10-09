from pathlib import Path
import hashlib
import json

import numpy as np


OUTPUT = Path(__file__).resolve().parent
STUDIES = OUTPUT.parent
SOURCE_STUDIES = (
    'r004_matched_rayleigh',
    'r020_matched_two_moments',
    'r040_matched_norm',
    'r052_matched_three_moments_norm',
)
rows = []
for study in SOURCE_STUDIES:
    for path in sorted((STUDIES / study / 'results').glob('*.npz')):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            eigenvalues = saved['eigenvalues']
            weights = saved['modal_weights']
            observed = (saved['normalized_loss'] if 'normalized_loss' in saved
                        else saved['loss'] / saved['loss'][0])
            assert observed.shape == (1001,)
            assert np.all(eigenvalues > 0)
            assert np.all(weights >= 0)
            steps = np.arange(len(observed))
            terms = weights[:, None] * (1 - 0.5 * eigenvalues[:, None]) ** (2 * steps)
            predicted = np.sum(terms, axis=0)
            hit = int(np.flatnonzero(observed <= 0.01)[0])
            tail_errors = np.abs(observed[hit:] - predicted[hit:])
            active = np.flatnonzero(weights > 0)
            slowest = active[np.argmin(eigenvalues[active])]
            final_theta = (saved['final_theta'] if 'final_theta' in saved
                           else saved['theta_history'][-1])
            final_residual = saved['features'] @ final_theta - saved['target']
            final_ratio_recomputed = float(final_residual @ final_residual
                                           / (saved['target'] @ saved['target']))
            rows.append({
                'source_npz': str(path.relative_to(STUDIES.parent)),
                'source_sha256': before,
                'first_1percent_step': hit,
                'post_threshold_points': len(tail_errors),
                'post_threshold_max_absolute_error': float(np.max(tail_errors)),
                'full_curve_max_absolute_error': float(np.max(np.abs(observed - predicted))),
                'final_observed_ratio': float(observed[-1]),
                'final_exact_series_ratio': float(predicted[-1]),
                'final_ratio_recomputed': final_ratio_recomputed,
                'slowest_active_eigenvalue': float(eigenvalues[slowest]),
                'slowest_active_weight': float(weights[slowest]),
                'slowest_final_fraction_of_exact_series': float(terms[slowest, -1] / predicted[-1])
                    if predicted[-1] > 0 else None,
                'null_target_mass': 0.0,
                'numerical_plateau_measured': False,
            })
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before

summary = {
    'label': '保存数据事后拟合（development）',
    'analysis_kind': '已知精确谱递推的事后核验；未拟合参数',
    'data_contract': {
        'inputs': list(SOURCE_STUDIES),
        'window_steps': [0, 1000],
        'post_threshold_window': '首次达到1%阈值至1000步，包含端点',
        'eta': 0.5,
        'normalization': '每条保存曲线除以其初始损失',
        'new_training_cells': 0,
        'new_seeds': 0,
        'input_files_unchanged': True,
        'no_extrapolated_training_points': True,
    },
    'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'saved_training_curves': len(rows),
    'post_threshold_points': sum(r['post_threshold_points'] for r in rows),
    'curves_with_post_threshold_error_at_most_1e_minus10': sum(
        r['post_threshold_max_absolute_error'] <= 1e-10 for r in rows),
    'maximum_post_threshold_absolute_error': max(
        r['post_threshold_max_absolute_error'] for r in rows),
    'maximum_full_curve_absolute_error': max(r['full_curve_max_absolute_error'] for r in rows),
    'rows': rows,
    'boundaries': [
        '全部保存谱满秩；零特征值残余公式是解析结果，未在奇异谱上测量。',
        '1000步终点不是无限训练；未量得数值plateau或通用数值底。',
        '精确级数不是新发现；验证不属于封存OOD或盲预测。',
    ],
}
(OUTPUT / 'summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))
for row in rows:
    if 'r052' in row['source_npz']:
        print(json.dumps(row, ensure_ascii=False))
