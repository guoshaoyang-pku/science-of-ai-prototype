#!/usr/bin/env python3
"""Exact constraint review and known spectral predictions; no training."""
from fractions import Fraction
import json
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
MODES = [1, 2, 3, 4, 5, 9]
lam = [Fraction(1, i * i) for i in MODES]
v = [x / math.prod(x - z for j, z in enumerate(lam) if j != k)
     for k, x in enumerate(lam)]
assert all(sum(a * x ** k for a, x in zip(v, lam)) == 0 for k in [-1, 0, 1, 2, 3])
positive = sum(a for a in v if a > 0)
negative = -sum(a for a in v if a < 0)
assert positive == negative
weights = {
    "slow_pair": [max(-a, 0) / negative for a in v],
    "fast_pair": [max(a, 0) / positive for a in v],
}
moments = {}
for name, w in weights.items():
    assert all(a >= 0 for a in w) and sum(w) == 1
    moments[name] = {str(k): sum(a * x ** k for a, x in zip(w, lam)) for k in [-1, 0, 1, 2, 3, 4]}
assert all(moments["slow_pair"][str(k)] == moments["fast_pair"][str(k)] for k in [-1, 0, 1, 2, 3])

eta = Fraction(1, 2)
threshold = Fraction(1, 100)
endpoints = {}
for name, w in weights.items():
    predicted = next(t for t in range(1001)
                     if math.fsum(float(a) * float(1 - eta * x) ** (2 * t)
                                  for a, x in zip(w, lam)) <= float(threshold))
    adjacent = {}
    for t in [predicted - 1, predicted]:
        value = sum(a * (1 - eta * x) ** (2 * t) for a, x in zip(w, lam))
        adjacent[str(t)] = {"normalized_loss": float(value), "exact_threshold_comparison": value <= threshold}
    assert not adjacent[str(predicted - 1)]["exact_threshold_comparison"]
    assert adjacent[str(predicted)]["exact_threshold_comparison"]
    endpoints[name] = {"step": predicted, "exact_adjacent_check": adjacent}

targets = {}
for name, w in weights.items():
    full = [Fraction(0)] * 9
    for mode, value in zip(MODES, w):
        full[mode - 1] = value
    targets[name] = [str(a) for a in full]
common = moments["slow_pair"]
value = {
    "available_modes": MODES, "targets": targets,
    "construction": "v_j=lambda_j/prod_{k!=j}(lambda_j-lambda_k), positive/negative parts normalized",
    "constraint_identity": "sum(v_j*lambda_j^k)=0 for k=-1,0,1,2,3; degree<=4 barycentric identity on six distinct nodes",
    "minimal_support_union_proof": "For <=5 distinct positive nodes, multiply each column of [1/lambda,1,lambda,lambda^2,lambda^3] by lambda. The resulting powers0..4 have full column rank by the Vandermonde determinant. Thus no nonzero weight difference exists on <=5 nodes; six is the minimum support union, not the support size of each target.",
    "moments_exact": {name: {k: str(a) for k, a in m.items()} for name, m in moments.items()},
    "common_theta_star_norm_squared": str(common["-1"]),
    "common_first_step_loss_drop": str(Fraction(1, 2) * (2 * eta * common["1"] - eta ** 2 * common["2"])),
    "endpoints": endpoints,
    "predicted_ratio": endpoints["slow_pair"]["step"] / endpoints["fast_pair"]["step"],
    "prediction_provenance": "Registered before training; endpoints calculated from known fixed-feature scalar spectral recurrence, not a blind discovery. Fixed modes 1/2/3/4/5/9; no training-results search.",
    "new_training_cells": 0, "domain": "development",
}
(STUDY / "executed/feasibility.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"targets": targets, "endpoints": endpoints, "moments": value["moments_exact"]}))
