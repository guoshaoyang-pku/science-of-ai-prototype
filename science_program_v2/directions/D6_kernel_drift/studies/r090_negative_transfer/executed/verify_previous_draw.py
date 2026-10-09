import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def errors(path, coefficients, width):
    with np.load(path, allow_pickle=False) as z:
        kernels = np.einsum('tip,tjp->tij', z['jacobians'], z['jacobians'], optimize=False) / len(z['train_y'])
        scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
        logs = []
        for vector in [z['residuals'][0], z['train_y']]:
            quadratic = np.einsum('i,tij,j->t', vector, kernels, vector, optimize=False)
            logs.append(np.log(quadratic / (scales * quadratic[0])))
        e, q, late = logs[0][1], logs[1][1], logs[0][2]
        features = {'E': [1., e], 'W': [1., e, float(width == 8)], 'T': [1., e, q]}
        return {name: abs(np.dot(features[name], beta) - late) for name, beta in coefficients.items()}, z['parameters'][0].copy()


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    verified = []
    for unit in summary['paired_previous_draw_error_change']:
        function, width = unit['function'], unit['width']
        changes = {name: [] for name in config['frozen_coefficients']}
        for seed in config['seeds']:
            cell = f'{function}_w{width}_s{seed}.npz'
            old, old_initial = errors(REPO / config['previous_draw_study'] / 'results' / cell, config['frozen_coefficients'], width)
            new, new_initial = errors(STUDY / 'results' / cell, config['frozen_coefficients'], width)
            assert np.array_equal(new_initial, old_initial)
            for name in changes:
                changes[name].append(new[name] - old[name])
        intervals = {}
        for name, values in changes.items():
            values = np.asarray(values)
            mean = float(values.mean())
            radius = float(t.isf(.025, 2) * values.std(ddof=1) / np.sqrt(3))
            expected = unit['new_minus_old_absolute_error'][name]
            np.testing.assert_allclose(mean, expected['mean'], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose([mean - radius, mean + radius], expected['seed_95pct_t_interval'], rtol=1e-10, atol=1e-12)
            intervals[name] = {'mean': mean, 'seed_95pct_t_interval': [mean - radius, mean + radius]}
        verified.append({'function': function, 'width': width, 'intervals': intervals})
    components = {'T_reduction_lt_0': bool(summary['paired_mae_reduction']['T'] < 0.)}
    assert components == summary['registered_components']
    assert summary['predictions']['P1_negative_reduction'] == ('supported' if all(components.values()) else 'refuted')
    for name, expected in config['input_sha256'].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected
    result = {'status': 'passed', 'cells_paired': 12, 'new_training_cells': 0,
              'previous_data_seed': 260608, 'current_data_seed': 260609,
              'all_initial_parameters_equal': True, 'registered_components': components,
              'paired_intervals': verified, 'historical_input_hashes_unchanged': len(config['input_sha256']),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': '事后核验保存J与参数；0训练/0拟合；不改变注册公式、判据或任一旧预测判定。'}
    (STUDY / 'executed/previous_draw_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'paired_intervals'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
