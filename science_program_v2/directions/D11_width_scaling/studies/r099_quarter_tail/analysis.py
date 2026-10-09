#!/usr/bin/env python3
"""只读保存的实际参数训练结果，检验同 Rayleigh 目标配对。"""
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent / "executed"))
from run import STUDY, ROOT, contract, label, sha, save
import numpy as np


def spectral_hit(lam, weights, eta, threshold, limit):
    occupied = weights > 0
    rates = 2 * np.log1p(-eta * lam[occupied])
    energy = weights[occupied]
    def relative_loss(t):
        return float(np.sum(energy * np.exp(t * rates)))
    if relative_loss(limit) > threshold:
        return None
    lo, hi = 0, limit
    while lo < hi:
        mid = (lo + hi) // 2
        if relative_loss(mid) <= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    config, pins = contract()
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    assert receipt["pins"] == pins and len(receipt["cells"]) == 33
    rows = []
    eta, epsilon, n = config["optimizer"]["eta"], config["threshold"], config["sample_count"]
    for d in config["dimensions"]:
        for seed in config["coordinate_seeds"]:
            cell = {"dimension": d, "seed": seed, "target": "quarter_tail"}
            name = label(cell)
            meta_path = STUDY / f"results/{name}.json"
            meta = json.loads(meta_path.read_text())
            npz_path = STUDY / f"results/{name}.npz"
            assert receipt["cells"][name] == sha(meta_path)
            assert meta["pins"] == pins and meta["arrays_sha256"] == sha(npz_path)
            old_path = ROOT / config["baseline_study"] / f"results/d{d}_seed{seed}_middle.npz"
            old_meta = json.loads(old_path.with_suffix(".json").read_text())
            with np.load(old_path) as old:
                old_arrays = {key: old[key] for key in old.files}
            with np.load(npz_path) as result:
                a = {key: result[key] for key in result.files}
            assert all(np.isfinite(value).all() for value in a.values())
            for key in ("eigenvalues", "permutation", "signs", "feature_scale", "parameter_scale"):
                assert np.array_equal(a[key], old_arrays[key]), key
            weights = np.zeros(d)
            weights[d // 4 - 1], weights[d - 1] = 1 / 3, 2 / 3
            assert np.array_equal(a["modal_weights"], weights)
            assert np.array_equal(a["target"], np.sqrt(n * weights))
            lam, loss = a["eigenvalues"], a["loss"]
            occupied = weights > 0
            steps = np.arange(len(loss))
            theory = np.sum(weights[occupied, None] * np.exp(
                2 * steps[None, :] * np.log1p(-eta * lam[occupied, None])), axis=0)
            curve_error = float(np.max(np.abs(loss / loss[0] - theory)))
            hits = np.flatnonzero(loss / loss[0] <= epsilon)
            hit = int(hits[0]) if len(hits) else None
            full_hit = spectral_hit(lam, weights, eta, epsilon, int(np.ceil(5 * d * meta["values"]["harmonic_number"])))
            assert hit == meta["values"]["measured_step"] == full_hit and hit is not None
            assert len(loss) - 1 == hit + 1
            old_hit = old_meta["values"]["measured_step"]
            m1 = float(np.sum(weights * lam))
            old_m1 = float(np.sum(old_arrays["modal_weights"] * lam))
            rayleigh_hit = math.ceil(math.log(epsilon) / (2 * math.log1p(-eta * m1)))
            h = meta["values"]["harmonic_number"]
            rows.append({"dimension": d, "seed": seed, "measured_step": hit,
                         "middle_step": old_hit, "ratio_to_middle": hit / old_hit,
                         "rayleigh_step": rayleigh_hit, "rayleigh_underestimate": 1 - rayleigh_hit / hit,
                         "rayleigh_pair_difference": m1 - old_m1,
                         "initial_loss_pair_difference": float(loss[0] - old_arrays["loss"][0]),
                         "curve_maxabs": curve_error, "scaled_budget": hit / (d * h),
                         "harmonic_number": h, "m1": m1,
                         "m2": float(np.sum(weights * lam**2)),
                         "parameter_target_norm_squared": float(np.sum(weights / lam)),
                         "old_parameter_target_norm_squared": float(np.sum(old_arrays["modal_weights"] / lam))})
    assert all(len({r["measured_step"] for r in rows if r["dimension"] == d}) == 1 for d in config["dimensions"])
    verification = json.loads((STUDY / "executed/independent_verification.json").read_text())
    assert verification["passed"] and verification["cells"] == 33
    p1 = all(r["curve_maxabs"] <= 1e-10 for r in rows)
    p2 = all(1.75 <= r["ratio_to_middle"] <= 1.90 and r["rayleigh_underestimate"] >= .40
             and abs(r["rayleigh_pair_difference"]) <= 1e-14
             and abs(r["initial_loss_pair_difference"]) <= 1e-14 for r in rows)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.exp(-4 * mid) / 3 + 2 * math.exp(-mid) / 3 <= epsilon:
            hi = mid
        else:
            lo = mid
    coefficient = (lo + hi) / 2
    summary = {"round": 99, "direction": "D11", "domain": "development", "new_training_cells": 33,
               "conditions": 11, "coordinate_seeds": 3, "old_middle_pairs": 33,
               "pins": pins, "predictions": {"P1": "supported" if p1 else "refuted", "P2": "supported" if p2 else "refuted"},
               "ratio_to_middle_range": [min(r["ratio_to_middle"] for r in rows), max(r["ratio_to_middle"] for r in rows)],
               "rayleigh_underestimate_range": [min(r["rayleigh_underestimate"] for r in rows), max(r["rayleigh_underestimate"] for r in rows)],
               "curve_maxabs": max(r["curve_maxabs"] for r in rows),
               "rayleigh_pair_maxabs": max(abs(r["rayleigh_pair_difference"]) for r in rows),
               "initial_loss_pair_maxabs": max(abs(r["initial_loss_pair_difference"]) for r in rows),
               "continuous_coefficient_descriptive": coefficient,
               "continuous_coefficient_max_relative_error": max(abs(r["scaled_budget"] / coefficient - 1) for r in rows),
               "d_ge_128_coefficient_max_relative_error": max(abs(r["scaled_budget"] / coefficient - 1) for r in rows if r["dimension"] >= 128),
               "compute_seconds": receipt["compute_seconds"], "rows": rows,
               "boundaries": config["boundaries"]}
    save(STUDY / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("rows", "pins", "boundaries")}))


if __name__ == "__main__":
    main()
