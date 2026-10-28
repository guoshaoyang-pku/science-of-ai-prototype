"""Exact finite CD examples. No training, network access or generated files."""
from fractions import Fraction as F
import json


def main():
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    data = [F(2, 5), F(1, 20), F(1, 20), F(1, 2)]
    model = [F(1, 6), F(1, 6), F(1, 6), F(1, 2)]
    # w = log(3): each active-neighbor conditional is 3/4.
    active = F(3, 4)
    kernel = []
    for _, old_y in states:
        row = []
        for new_x, new_y in states:
            px = active if old_y else F(1, 2)
            py = active if new_x else F(1, 2)
            row.append((px if new_x else 1 - px) * (py if new_y else 1 - py))
        assert sum(row) == 1 and all(x > 0 for x in row)
        kernel.append(row)

    def advance(distribution):
        return [sum(distribution[i] * kernel[i][j] for i in range(4)) for j in range(4)]

    assert advance(model) == model
    negative = advance(data)
    assert sum(data) == sum(negative) == 1
    ml_update = data[3] - model[3]
    cd_update = data[3] - negative[3]
    assert ml_update == 0
    assert negative[3] == F(153, 320)
    assert cd_update == F(7, 320)

    # Separate one-visible/one-hidden RBM, hidden bias fixed to 0, data visible=0.
    s, q, r = F(3, 4), F(1, 2), F(3, 4)
    partial_w_delta_b = -r * (1 - r) / 2
    partial_b_delta_w = -s * (q * (1 - q) + r * (1 - r)) / 2
    curl = partial_w_delta_b - partial_b_delta_w
    assert partial_w_delta_b == F(-3, 32)
    assert partial_b_delta_w == F(-21, 128)
    assert curl == F(9, 128)
    print(json.dumps({
        'visible_bm_at_log3': {
            'maximum_likelihood_update': str(ml_update),
            'expected_cd1_update': str(cd_update),
            'one_sweep_probability_11': str(negative[3]),
        },
        'binary_rbm_at_log3_and_zero_bias': {
            'partial_w_delta_b': str(partial_w_delta_b),
            'partial_b_delta_w': str(partial_b_delta_w),
            'mixed_partial_difference': str(curl),
        },
        'scope': 'Exact expectations for two specified finite models; no training convergence claim.'
    }, indent=2))


if __name__ == '__main__':
    main()
