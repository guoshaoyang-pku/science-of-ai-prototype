import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
STUDIES = OUT.parent
ETA = 0.3
NAMES = [
    'r003_noise_sample_scaling',
    'r019_population_label_normalization',
    'r051_noise_halving',
    'r083_low_noise_u_threshold',
    'r095_low_noise_seed413',
]
sources = {}
cells = []
for name in NAMES:
    summary_path = STUDIES / name / 'summary.json'
    sources[str(summary_path.relative_to(STUDIES))] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    old = json.loads(summary_path.read_text())
    metadata = {c['label']: c for c in old.get('cells', [])}
    for path in sorted((STUDIES / name / 'results').glob('*.npz')):
        sources[str(path.relative_to(STUDIES))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            risk = data['expected_risk'].copy()
        assert len(risk) == 16385 and np.all(np.isfinite(risk))
        ts = int(np.argmin(risk))
        meta = metadata.get(path.stem, old.get('cell'))
        assert ts == (meta['t_star'] if 't_star' in meta else old['new']['t_star'])
        offsets = np.unique(np.rint(np.geomspace(max(1, 0.1 * ts), ts, 40)).astype(int))
        x = offsets / ts
        coefficient = float(risk[2 * ts] - risk[ts])
        y = (risk[ts + offsets] - risk[ts]) / coefficient
        assert np.all(y > 0) and coefficient > 0
        logx, logy = np.log(x), np.log(y)
        slope = float(logx @ logy / (logx @ logx))
        positive = np.flatnonzero(np.diff(risk) > 0)
        cell = {
            'study': name, 'label': path.stem,
            'n': meta['n'], 'seed': meta['seed'], 'noise_variance': meta['noise_variance'],
            't_star': ts, 'u_rise': ETA * ts,
            'first_rising_interval_left_step': int(positive[0]),
            'decreasing_intervals_after_minimum': int(np.sum(np.diff(risk[ts:]) < 0)),
            'minimum_risk': float(risk[ts]), 'endpoint_excess': float(risk[-1] - risk[ts]),
            'C_at_twice_t_star': coefficient, 'individual_exponent': slope,
            'offset_steps': offsets.tolist(), 'normalized_excess': y.tolist(),
        }
        cells.append(cell)

base = [c for c in cells if c['study'] == NAMES[0]]
assert len(base) == 60 and len(cells) == 78
denominator = sum(float(np.log(np.array(c['offset_steps']) / c['t_star']) @
                        np.log(np.array(c['offset_steps']) / c['t_star'])) for c in base)
numerator = sum(float(np.log(np.array(c['offset_steps']) / c['t_star']) @
                      np.log(c['normalized_excess'])) for c in base)
power = numerator / denominator
for cell in cells:
    x = np.array(cell['offset_steps']) / cell['t_star']
    y = np.array(cell['normalized_excess'])
    error = float(np.max(np.exp(np.abs(power * np.log(x) - np.log(y)))))
    tail_ratio = cell['C_at_twice_t_star'] * ((16384 - cell['t_star']) / cell['t_star']) ** power / cell['endpoint_excess']
    cell.update(max_factor_error=error, all_sampled_points_within_factor2=error <= 2,
                endpoint_prediction_over_observation=float(tail_ratio))

old_base = json.loads((STUDIES / NAMES[0] / 'summary.json').read_text())
onset_fits = [dict(f, u_coefficient=ETA * float(np.exp(f['a'])))
              for f in old_base['leave_one_n_out_fits']]
groups = []
for name in NAMES:
    group = [c for c in cells if c['study'] == name]
    groups.append({
        'study': name, 'curves': len(group),
        'passed': sum(c['all_sampled_points_within_factor2'] for c in group),
        'individual_exponent_range': [min(c['individual_exponent'] for c in group), max(c['individual_exponent'] for c in group)],
        'coefficient_range': [min(c['C_at_twice_t_star'] for c in group), max(c['C_at_twice_t_star'] for c in group)],
        'maximum_factor_error': max(c['max_factor_error'] for c in group),
        'tail_ratio_range': [min(c['endpoint_prediction_over_observation'] for c in group), max(c['endpoint_prediction_over_observation'] for c in group)],
    })
result = {
    'label': '保存数据事后拟合（development）',
    'new_training_cells': 0, 'new_evaluation_cells': 0, 'new_seeds': 0,
    'data_contract': '仅读取五个已完成 study 的全部78条 expected_risk；不组合新噪声条件，不调用训练或旧执行器。',
    'selection_history': '首次探索检视使用同一锚定幂律、40个几何间隔候选点、相对偏移0.1到1；本脚本保留该模型及尾部失败，未搜索窗口或模型。',
    'formula': 'L_test(U)-L_min = C*((U-U_rise)/U_rise)**p',
    'coefficient_definition': 'C=L_test(2*U_rise)-L_min，使用每条曲线自身标签；不是跨条件无标签系数预测。',
    'fit_contract': '原60条曲线，合并所有去重后的整数取样点，锚定x=1/y=1，log误差平方和最小；样本点取round(geomspace(max(1,0.1*t_star),t_star,40))，网格边界误差≤0.5步。',
    'criterion': '一条曲线全部取样点的预测/实测均在[0.5,2]才通过；事后描述判据，非预注册检验。',
    'tail_check': '同一公式直接评价已保存第16384步；不重新拟合。',
    'eta': ETA, 'common_exponent': power, 'onset_fits': onset_fits,
    'onset_factor2_passed': old_base['predictions']['P3'],
    'first_rise_matches_minimum': sum(c['first_rising_interval_left_step'] == c['t_star'] for c in cells),
    'monotone_after_minimum': sum(c['decreasing_intervals_after_minimum'] == 0 for c in cells),
    'groups': groups, 'cells': cells, 'source_sha256': sources,
    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
for relative, digest in sources.items():
    assert hashlib.sha256((STUDIES / relative).read_bytes()).hexdigest() == digest
(OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['common_exponent', 'onset_fits', 'first_rise_matches_minimum', 'monotone_after_minimum', 'groups']}, ensure_ascii=False, indent=2))
