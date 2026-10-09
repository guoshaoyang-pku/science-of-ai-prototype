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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    rows, checks = [], []
    for name, digest in sorted(receipt["cells"].items()):
        meta_path, npz_path = RESULTS / f"{name}.json", RESULTS / f"{name}.npz"
        meta = json.loads(meta_path.read_text())
        if digest != sha(meta_path) or meta["arrays_sha256"] != sha(npz_path):
            raise RuntimeError(f"saved hash mismatch: {name}")
        if meta["pins"] != receipt["pins"]:
            raise RuntimeError(f"contract mismatch: {name}")
        with np.load(npz_path) as a:
            d, eta = meta["cell"]["dimension"], config["optimizer"]["eta"]
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
                checkpoint_error = max(checkpoint_error, abs(float(np.sum(residual ** 2) / (2 * d)) - loss[t]))
                reference = parameter_target / scale * (-np.expm1(t * np.log1p(-eta * scale ** 2 / d)))
                theta_error = max(theta_error, float(np.max(np.abs(theta - reference))))
            for k in range(len(checkpoint_steps) - 1):
                if checkpoint_steps[k + 1] == checkpoint_steps[k] + 1:
                    theta = theta_checkpoints[k]
                    update = theta - eta * scale * (scale * theta - parameter_target) / d
                    step_error = max(step_error, float(np.max(np.abs(update - theta_checkpoints[k + 1]))))
            check = {"name": name, "spectrum_error": float(np.max(np.abs(lam - expected_lam))),
                     "weights_error": float(np.max(np.abs(w - expected_w))),
                     "target_error": float(np.max(np.abs(target - np.sqrt(d * w)))),
                     "feature_scale_error": float(np.max(np.abs(a["feature_scale"] ** 2 / d - lam))),
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
    p2 = all(len(fs) == len(config["coordinate_seeds"]) and all(.98 <= f["exponent"] <= 1.04 and f["max_relative_fit_error"] <= .06 for f in fs) for fs in fits.values())
    p3 = all(r["measured_step"] is not None and (abs(r["rayleigh_relative_error"]) >= .35 if r["target"] == "endpoints" else r["rayleigh_relative_error"] == 0) for r in rows)
    p3 = p3 and all(p["step_ratio"] is not None and 1.95 <= p["step_ratio"] <= 2.05 for p in pairs if p["dimension"] >= 256)
    predictions = {k: ("supported" if value else "refuted") if completed else "not_evaluated" for k, value in [("P1", p1), ("P2", p2), ("P3", p3)]}
    summary = {"study": config["study"], "round": 27, "domain": "development",
        "counts": {"completed_cells": len(rows), "planned_cells": config["counts"]["planned_cells"], "conditions": 22, "coordinate_seeds": 3, "completed_pairs": len(pairs)},
        "rows": rows, "pairs": pairs, "power_fits": fits, "predictions": predictions,
        "checks": checks, "all_saved_checks_passed": p1, "compute_seconds": receipt["compute_seconds"],
        "max_curve_error": max(c["curve_error"] for c in checks),
        "max_closed_form_parameter_error": max(c["closed_form_parameter_error"] for c in checks),
        "pins": receipt["pins"],
        "boundary": "joint n=d axis; structured fixed full-rank features; coordinate seeds are not independent datasets; no learned width/lazy boundary/OOD"}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": len(rows), "predictions": predictions, "power_fits": fits,
        "all_saved_checks_passed": p1, "compute_seconds": receipt["compute_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
