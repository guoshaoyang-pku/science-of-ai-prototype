#!/usr/bin/env python3
"""从保存数组复算阈值、配对和注册的 d h_d 标度判据。"""
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    rows = []
    for name, digest in sorted(receipt["cells"].items()):
        meta_path, npz_path = STUDY / "results" / f"{name}.json", STUDY / "results" / f"{name}.npz"
        meta = json.loads(meta_path.read_text())
        request = hashlib.sha256(json.dumps({"cell": meta["cell"], "pins": receipt["pins"]["sha256"]}, sort_keys=True).encode()).hexdigest()
        assert sha(meta_path) == digest and sha(npz_path) == meta["arrays_sha256"]
        assert meta["pins"] == receipt["pins"] and request == meta["request_sha256"]
        with np.load(npz_path) as saved:
            loss = saved["loss"]
            hits = np.flatnonzero(loss / loss[0] <= config["threshold"])
            hit = int(hits[0]) if len(hits) else None
            assert hit == meta["values"]["measured_step"]
            d, target = meta["cell"]["dimension"], meta["cell"]["target"]
            harmonic = float(np.sum(np.arange(1, d + 1, dtype=float) ** -1))
            coefficient = np.log(100) / (2 if target == "middle" else 1)
            coefficient_error = None if hit is None else hit / (d * harmonic * coefficient) - 1
            old_meta = json.loads((ROOT / config["baseline_study"] / "results" / f"{name}.json").read_text())
            old_hit = old_meta["values"]["measured_step"]
            lam, w = saved["eigenvalues"], saved["modal_weights"]
            rayleigh = float(np.sum(lam * w))
            ray_step = int(np.ceil(np.log(config["threshold"]) / (2 * np.log1p(-config["optimizer"]["eta"] * rayleigh))))
            old_fit = config["frozen_old_d_fits"][target]
            rows.append(meta["cell"] | {"name": name, "measured_step": hit, "harmonic_number": harmonic,
                "T_over_dh": None if hit is None else hit / (d * harmonic),
                "coefficient": float(coefficient), "coefficient_relative_error": coefficient_error,
                "initial_loss": float(loss[0]), "trace": float(np.sum(lam)), "rayleigh": rayleigh,
                "rayleigh_step": ray_step, "rayleigh_relative_error": None if hit is None else ray_step / hit - 1,
                "old_step": old_hit, "new_over_old_budget": None if hit is None else hit / old_hit,
                "old_frozen_fit_relative_error": None if hit is None else old_fit["coefficient"] * d**old_fit["exponent"] / hit - 1})
    expected = len(config["dimensions"]) * len(config["coordinate_seeds"]) * len(config["targets"])
    complete = len(rows) == expected and all(row["measured_step"] is not None for row in rows)
    aggregate = {}
    fits = {}
    for target in config["targets"]:
        subset = [row for row in rows if row["target"] == target and row["measured_step"] is not None]
        aggregate[target] = {"max_abs_coefficient_relative_error": max(abs(row["coefficient_relative_error"]) for row in subset),
                            "d_ge_128_max_abs_coefficient_relative_error": max([abs(row["coefficient_relative_error"]) for row in subset if row["dimension"] >= 128], default=None),
                            "old_frozen_fit_relative_error_range": [min(row["old_frozen_fit_relative_error"] for row in subset), max(row["old_frozen_fit_relative_error"] for row in subset)],
                            "new_over_old_budget_range": [min(row["new_over_old_budget"] for row in subset), max(row["new_over_old_budget"] for row in subset)]}
        fitted = []
        for seed in config["coordinate_seeds"]:
            selected = sorted([row for row in subset if row["seed"] == seed], key=lambda row: row["dimension"])
            if len(selected) != len(config["dimensions"]):
                continue
            d = np.array([row["dimension"] for row in selected])
            t = np.array([row["measured_step"] for row in selected])
            exponent, intercept = np.polyfit(np.log(d), np.log(t), 1)
            fitted.append({"seed": seed, "coefficient": float(np.exp(intercept)), "exponent": float(exponent),
                           "max_relative_error": float(np.max(np.abs(np.exp(intercept) * d**exponent / t - 1)))})
        fits[target] = fitted
    index = {(row["dimension"], row["seed"], row["target"]): row for row in rows}
    pairs = []
    for d in config["dimensions"]:
        for seed in config["coordinate_seeds"]:
            a, b = index.get((d, seed, "middle")), index.get((d, seed, "endpoints"))
            if a and b and a["measured_step"] is not None and b["measured_step"] is not None:
                pairs.append({"dimension": d, "seed": seed, "budget_ratio": b["measured_step"] / a["measured_step"],
                              "rayleigh_difference": b["rayleigh"] - a["rayleigh"], "initial_loss_difference": b["initial_loss"] - a["initial_loss"]})
    verify_path = STUDY / "executed/independent_verification.json"
    verification = json.loads(verify_path.read_text()) if verify_path.exists() else None
    p2 = complete and all(aggregate[target]["max_abs_coefficient_relative_error"] <= config["criteria"][target] and
                         aggregate[target]["d_ge_128_max_abs_coefficient_relative_error"] <= config["criteria"]["d_ge_128"] for target in config["targets"])
    summary = {"study": config["study"], "round": 73, "direction": "D11_width_scaling", "domain": "development",
               "complete": complete, "counts": {"conditions": 22, "coordinate_seeds": 3, "saved_cells": len(rows), "target_pairs": len(pairs)},
               "predictions": {"P1": "supported" if complete and verification and verification["all_passed"] else "not_evaluated",
                               "P2": "supported" if p2 else ("refuted" if complete else "not_evaluated")},
               "preregistration_commit": receipt["pins"]["git_commit"], "pins": receipt["pins"], "compute_seconds": receipt["compute_seconds"],
               "rows": rows, "target_pairs": pairs, "aggregate": aggregate, "d_only_fits_descriptive": fits,
               "known_before_registration": config["known_before_registration"], "boundaries": config["boundaries"]}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    print(json.dumps({"saved_cells": len(rows), "predictions": summary["predictions"], "aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
