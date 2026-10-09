#!/usr/bin/env python3
"""Exact moment feasibility and analytic predictions; no training or data."""
from fractions import Fraction as F
from functools import reduce
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
modes = [1, 2, 3, 9]
eigenvalues = [F(1, i * i) for i in modes]
null_vector = [
    1 / reduce(lambda a, b: a * b, (x - y for j, y in enumerate(eigenvalues) if j != i), F(1))
    for i, x in enumerate(eigenvalues)
]
assert all(sum(v * x ** k for v, x in zip(null_vector, eigenvalues)) == 0 for k in range(3))
mass = sum(v for v in null_vector if v > 0)
targets = {}
for name, sign in [("fast_pair", 1), ("slow_pair", -1)]:
    weights = [max(sign * v, F(0)) / mass for v in null_vector]
    moments = [sum(w * x ** k for w, x in zip(weights, eigenvalues)) for k in range(4)]
    assert all(w >= 0 for w in weights) and moments[0] == 1
    ratio = lambda t: sum(float(w) * (1 - .5 * float(x)) ** (2 * t) for w, x in zip(weights, eigenvalues))
    hit = next(t for t in range(1001) if ratio(t) <= .01)
    targets[name] = {
        "weights_fraction_on_modes": [str(w) for w in weights],
        "weights_float_on_modes": [float(w) for w in weights],
        "moments_fraction_k0_to_3": [str(m) for m in moments],
        "moments_float_k0_to_3": [float(m) for m in moments],
        "predicted_threshold_step": hit,
        "predicted_loss_ratio_before_threshold": ratio(hit - 1),
        "predicted_loss_ratio_at_threshold": ratio(hit),
        "predicted_loss1": .5 * ratio(1),
        "predicted_loss_drop1": .5 * (1 - ratio(1)),
        "theta_star_norm_squared_fraction": str(sum(w / x for w, x in zip(weights, eigenvalues))),
    }
assert targets["fast_pair"]["moments_fraction_k0_to_3"][:3] == targets["slow_pair"]["moments_fraction_k0_to_3"][:3]
output = {
    "kind": "pretraining_analytic_feasibility_only",
    "training_ran": False,
    "selection_rule": "固定四模态1/2/3/9，取三矩Vandermonde零空间向量正负部分并分别归一化；未扫描训练结果。",
    "modes_one_indexed": modes,
    "eigenvalues_fraction": [str(x) for x in eigenvalues],
    "null_vector_fraction": [str(v) for v in null_vector],
    "null_moments_exactly_zero_k0_to_2": True,
    "targets": targets,
    "predicted_step_difference": targets["slow_pair"]["predicted_threshold_step"] - targets["fast_pair"]["predicted_threshold_step"],
    "predicted_step_ratio": targets["slow_pair"]["predicted_threshold_step"] / targets["fast_pair"]["predicted_threshold_step"],
}
(STUDY / "executed/feasibility.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(output, ensure_ascii=False))
