#!/usr/bin/env python3
"""Recompute registered endpoints and audit saved parameter measurements."""
import hashlib
import json
import os
from pathlib import Path
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results"
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    for rel, digest in config["baseline_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"baseline hash mismatch: {rel}")
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    rows, checks, baseline_pairs = [], [], []
    for name, digest in sorted(receipt["cells"].items()):
        meta_path, npz_path = RESULTS / f"{name}.json", RESULTS / f"{name}.npz"
        meta = json.loads(meta_path.read_text())
        if digest != sha(meta_path) or meta["arrays_sha256"] != sha(npz_path):
            raise RuntimeError(f"saved hash mismatch: {name}")
        if meta["pins"] != receipt["pins"]:
            raise RuntimeError(f"contract mismatch: {name}")
        with np.load(npz_path) as a:
            d, eta = meta["cell"]["dimension"], config["optimizer"]["eta"]
            n = config["sample_count"]
            lam, w, loss = a["eigenvalues"], a["modal_weights"], a["loss"]
            expected_lam = np.arange(1, d + 1, dtype=float) ** -1
            expected_w = np.zeros(d)
            if meta["cell"]["target"] == "middle":
                expected_w[d // 2 - 1] = 1
            else:
                expected_w[0], expected_w[-1] = 1 / (d - 1), (d - 2) / (d - 1)
            target = a["target"]
            perm, signs = a["permutation"], a["signs"]
            scale = a["parameter_scale"]
            parameter_target = a["parameter_target"]
            rng = np.random.default_rng(meta["cell"]["seed"])
            expected_perm = rng.permutation(d)
            expected_signs = rng.choice(np.array([-1.0, 1.0]), d)
            if not np.array_equal(expected_perm, perm) or not np.array_equal(expected_signs, signs):
                raise RuntimeError(f"coordinate seed mismatch: {name}")
            active = w > 0
            log_decay = 2 * np.log1p(-eta * lam[active])
            steps = np.arange(len(loss))
            curve = np.sum(w[active, None] * np.exp(log_decay[:, None] * steps), axis=0)
            hits = np.flatnonzero(loss / loss[0] <= config["threshold"])
            measured = int(hits[0]) if len(hits) else None
            max_steps = config["optimizer"]["max_steps_factor"] * d
            def spectrum_loss(t):
                return float(np.sum(w[active] * np.exp(t * log_decay)))
            full = None
            if spectrum_loss(max_steps) <= config["threshold"]:
                lo, hi = 0, max_steps
                while hi - lo > 1:
                    midpoint = (lo + hi) // 2
                    if spectrum_loss(midpoint) <= config["threshold"]:
                        hi = midpoint
                    else:
                        lo = midpoint
                full = hi
            rayleigh = float(np.sum(w * lam))
            ray_step = int(np.ceil(np.log(config["threshold"]) / (2 * np.log1p(-eta * rayleigh))))
            checkpoint_error, step_error, theta_error = 0.0, 0.0, 0.0
            checkpoint_steps = a["checkpoint_steps"]
            theta_checkpoints = a["theta_checkpoints"]
            for t, theta in zip(checkpoint_steps, theta_checkpoints):
                residual = scale * theta - parameter_target
                checkpoint_error = max(checkpoint_error, abs(float(np.sum(residual ** 2) / (2 * n)) - loss[t]))
                reference = parameter_target / scale * (-np.expm1(t * np.log1p(-eta * scale ** 2 / n)))
                theta_error = max(theta_error, float(np.max(np.abs(theta - reference))))
            for k in range(len(checkpoint_steps) - 1):
                if checkpoint_steps[k + 1] == checkpoint_steps[k] + 1:
                    theta = theta_checkpoints[k]
                    update = theta - eta * scale * (scale * theta - parameter_target) / n
                    step_error = max(step_error, float(np.max(np.abs(update - theta_checkpoints[k + 1]))))
            check = {"name": name, "spectrum_error": float(np.max(np.abs(lam - expected_lam))),
                     "weights_error": float(np.max(np.abs(w - expected_w))),
                     "target_error": float(np.max(np.abs(target - np.sqrt(n * w)))),
                     "feature_scale_error": float(np.max(np.abs(a["feature_scale"] ** 2 / n - lam))),
                     "parameter_mapping_error": float(np.max(np.abs(scale[perm] - signs * a["feature_scale"]))),
                     "target_mapping_error": float(np.max(np.abs(parameter_target[perm] - target))),
                     "curve_error": float(np.max(np.abs(loss / loss[0] - curve))),
                     "checkpoint_loss_error": checkpoint_error, "one_step_update_error": step_error,
                     "closed_form_parameter_error": theta_error,
                     "threshold_matches_full_spectrum": measured == full,
                     "metadata_matches": measured == meta["values"]["measured_step"] and full == meta["values"]["full_spectrum_step"] and ray_step == meta["values"]["rayleigh_step"],
                     "initial_loss_error": abs(float(loss[0]) - .5)}
            check["passed"] = bool(measured is not None and all(check[k] <= 1e-10 for k in [
                "spectrum_error", "weights_error", "target_error", "feature_scale_error",
                "parameter_mapping_error", "target_mapping_error", "curve_error",
                "checkpoint_loss_error", "one_step_update_error", "initial_loss_error"]) and
                theta_error <= 1e-7 and check["threshold_matches_full_spectrum"] and check["metadata_matches"])
            old_results = ROOT / config["baseline_study"] / "results"
            old_meta = json.loads((old_results / f"{name}.json").read_text())
            with np.load(old_results / f"{name}.npz") as old:
                old_loss = old["loss"]
                old_hits = np.flatnonzero(old_loss / old_loss[0] <= config["threshold"])
                old_hit = int(old_hits[0]) if len(old_hits) else None
                common_length = min(len(loss), len(old_loss))
                baseline_pairs.append({"name": name, "dimension": d, "seed": meta["cell"]["seed"],
                    "target": meta["cell"]["target"], "old_sample_count": d, "new_sample_count": n,
                    "old_step": old_hit, "new_step": measured,
                    "step_difference": None if measured is None or old_hit is None else measured - old_hit,
                    "old_metadata_matches": old_hit == old_meta["values"]["measured_step"],
                    "coordinate_contract_equal": bool(np.array_equal(perm, old["permutation"]) and np.array_equal(signs, old["signs"]) and np.array_equal(w, old["modal_weights"]) and np.array_equal(lam, old["eigenvalues"])),
                    "common_curve_length": common_length,
                    "normalized_curve_maxabs": float(np.max(np.abs(loss[:common_length] / loss[0] - old_loss[:common_length] / old_loss[0]))),
                    "normalized_hessian_maxabs": float(np.max(np.abs(scale ** 2 / n - old["parameter_scale"] ** 2 / d))),
                    "initial_gradient_maxabs": float(np.max(np.abs(scale * parameter_target / n - old["parameter_scale"] * old["parameter_target"] / d)))})
            checks.append(check)
            rows.append(meta["cell"] | {"name": name, "measured_step": measured, "full_spectrum_step": full,
                "rayleigh_step": ray_step, "rayleigh": rayleigh, "initial_loss": float(loss[0]),
                "rayleigh_relative_error": None if measured is None else (ray_step - measured) / measured,
                "target_parameter_norm_squared": float(np.sum(w / lam)),
                "executed_updates": len(loss) - 1})
    row_index = {(r["dimension"], r["seed"], r["target"]): r for r in rows}
    pairs = []
    for d in config["dimensions"]:
        for seed in config["coordinate_seeds"]:
            a = row_index.get((d, seed, "middle"))
            b = row_index.get((d, seed, "endpoints"))
            if a is not None and b is not None:
                pairs.append({"dimension": d, "seed": seed,
                    "rayleigh_difference": b["rayleigh"] - a["rayleigh"],
                    "initial_loss_difference": b["initial_loss"] - a["initial_loss"],
                    "step_difference": None if a["measured_step"] is None or b["measured_step"] is None else b["measured_step"] - a["measured_step"],
                    "step_ratio": None if a["measured_step"] is None or b["measured_step"] is None else b["measured_step"] / a["measured_step"]})
    fits = {}
    for target in config["targets"]:
        seed_fits = []
        for seed in config["coordinate_seeds"]:
            subset = sorted((r for r in rows if r["target"] == target and r["seed"] == seed), key=lambda r: r["dimension"])
            if len(subset) != len(config["dimensions"]) or any(r["measured_step"] is None for r in subset):
                continue
            dimensions = np.array([r["dimension"] for r in subset])
            endpoints = np.array([r["measured_step"] for r in subset])
            slope, intercept = np.polyfit(np.log(dimensions), np.log(endpoints), 1)
            fitted = np.exp(intercept) * dimensions ** slope
            seed_fits.append({"seed": seed, "exponent": float(slope), "coefficient": float(np.exp(intercept)),
                "max_relative_fit_error": float(np.max(np.abs(fitted / endpoints - 1))),
                "log_rmse": float(np.sqrt(np.mean((np.log(fitted) - np.log(endpoints)) ** 2))),
                "adjacent_exponents": (np.diff(np.log(endpoints)) / np.diff(np.log(dimensions))).tolist()})
        fits[target] = seed_fits
    completed = len(rows) == config["counts"]["planned_cells"]
    p1 = all(c["passed"] for c in checks) and all(abs(p["rayleigh_difference"]) <= 1e-12 and abs(p["initial_loss_difference"]) <= 1e-12 for p in pairs)
    p2 = all(p["step_difference"] is not None and abs(p["step_difference"]) <= 1 and p["old_metadata_matches"] and p["coordinate_contract_equal"] for p in baseline_pairs)
    differences = [p["step_difference"] for p in baseline_pairs if p["step_difference"] is not None]
    predictions = {k: ("supported" if value else "refuted") if completed else "not_evaluated" for k, value in [("P1", p1), ("P2", p2)]}
    summary = {"study": config["study"], "round": 31, "domain": "development",
        "counts": {"completed_cells": len(rows), "planned_cells": config["counts"]["planned_cells"], "conditions": 22, "coordinate_seeds": 3, "completed_pairs": len(pairs)},
        "rows": rows, "pairs": pairs, "power_fits": fits, "predictions": predictions,
        "baseline_pairs": baseline_pairs,
        "exact_budget_matches": sum(p["step_difference"] == 0 for p in baseline_pairs),
        "step_difference_range": [min(differences), max(differences)] if differences else None,
        "max_paired_curve_error": max(p["normalized_curve_maxabs"] for p in baseline_pairs),
        "checks": checks, "all_saved_checks_passed": p1, "compute_seconds": receipt["compute_seconds"],
        "max_curve_error": max(c["curve_error"] for c in checks),
        "max_closed_form_parameter_error": max(c["closed_form_parameter_error"] for c in checks),
        "pins": receipt["pins"],
        "boundary": "n=8192 with d nonzero rows scaled by sqrt(n); structured fixed features; coordinate seeds are not independent datasets; paired sample-accounting control, no independent data increase or learned width/lazy boundary/OOD"}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": len(rows), "predictions": predictions, "power_fits": fits,
        "all_saved_checks_passed": p1, "compute_seconds": receipt["compute_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
