#!/usr/bin/env python3
"""只读保存样本空间数组，独立核验谱、整数阈值、参数及一步 SGD。"""
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    checks = []
    checkpoint_count, adjacent_count = 0, 0
    for name, digest in receipt["cells"].items():
        meta_path = STUDY / "results" / f"{name}.json"
        npz_path = STUDY / "results" / f"{name}.npz"
        meta = json.loads(meta_path.read_text())
        assert hashlib.sha256(meta_path.read_bytes()).hexdigest() == digest
        assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == meta["arrays_sha256"]
        with np.load(npz_path) as archive:
            saved = {key: archive[key] for key in archive.files}
        with np.load(ROOT / config["baseline_study"] / "results" / f"{name}.npz") as archive:
            old = {key: archive[key] for key in ("eigenvalues", "modal_weights", "permutation", "signs", "target", "feature_scale")}
        d, n, eta = meta["cell"]["dimension"], config["sample_count"], config["optimizer"]["eta"]
        harmonic = float(np.sum(np.arange(1, d + 1, dtype=float) ** -1))
        lam, weights = saved["eigenvalues"], saved["modal_weights"]
        losses, steps = saved["loss"], saved["checkpoint_steps"]
        rng = np.random.default_rng(meta["cell"]["seed"])
        permutation = rng.permutation(d)
        signs = rng.choice(np.array([-1.0, 1.0]), d)
        expected_w = np.zeros(d)
        if meta["cell"]["target"] == "middle":
            expected_w[d // 2 - 1] = 1
        else:
            expected_w[0], expected_w[-1] = 1 / (d - 1), (d - 2) / (d - 1)
        active = weights > 0
        curve = np.sum(weights[active, None] * (1 - eta * lam[active, None])**(2 * np.arange(len(losses))), axis=0)
        hit_indices = np.flatnonzero(losses / losses[0] <= config["threshold"])
        hit = int(hit_indices[0]) if len(hit_indices) else None
        references = {row["dimension"]: row for row in config["known_before_registration"]["integer_reference_table"]}
        expected_hit = references[d][meta["cell"]["target"]]
        checkpoint_loss_error, theta_error, update_error = 0.0, 0.0, 0.0
        for t, theta in zip(steps, saved["theta_checkpoints"]):
            residual = saved["feature_scale"] * signs * theta[permutation] - saved["target"]
            checkpoint_loss_error = max(checkpoint_loss_error, abs(float(np.sum(residual**2) / (2*n)) - losses[t]))
            expected_theta_modal = saved["target"] / (saved["feature_scale"] * signs) * (1 - (1 - eta*lam)**t)
            theta_error = max(theta_error, float(np.max(np.abs(theta[permutation] - expected_theta_modal))))
            checkpoint_count += 1
        for k in range(len(steps) - 1):
            if steps[k + 1] == steps[k] + 1:
                theta = saved["theta_checkpoints"][k]
                residual = saved["feature_scale"] * signs * theta[permutation] - saved["target"]
                expected = theta.copy()
                expected[permutation] -= eta * saved["feature_scale"] * signs * residual / n
                update_error = max(update_error, float(np.max(np.abs(expected - saved["theta_checkpoints"][k + 1]))))
                adjacent_count += 1
        errors = {"spectrum": float(np.max(np.abs(lam - np.arange(1,d+1,dtype=float)**-1/harmonic))),
                  "weights": float(np.max(np.abs(weights-expected_w))), "trace": abs(float(np.sum(lam))-1),
                  "initial_loss": abs(float(losses[0])-.5), "curve": float(np.max(np.abs(losses/losses[0]-curve))),
                  "checkpoint_loss": checkpoint_loss_error, "closed_form_theta": theta_error, "one_step_update": update_error,
                  "scaled_features": float(np.max(np.abs(saved["feature_scale"] - old["feature_scale"]/np.sqrt(harmonic))))}
        unchanged = all(np.array_equal(saved[key], old[key]) for key in ("modal_weights", "permutation", "signs", "target"))
        passed = (unchanged and np.array_equal(permutation,saved["permutation"]) and np.array_equal(signs,saved["signs"]) and
                  hit == expected_hit == meta["values"]["measured_step"] and len(losses) == hit+2 and
                  all(value <= (1e-7 if key == "closed_form_theta" else 1e-10) for key,value in errors.items()))
        checks.append({"name": name, "passed": bool(passed), "unchanged_old_arrays": unchanged, "threshold": hit, "errors": errors})
    verification = {"all_passed": bool(checks and all(check["passed"] for check in checks)), "saved_cells": len(checks),
                    "checkpoint_count": checkpoint_count, "adjacent_update_count": adjacent_count, "checks": checks,
                    "max_errors": {key: max(check["errors"][key] for check in checks) for key in checks[0]["errors"]}}
    path = STUDY / "executed" / ("first_cell_verification.json" if len(checks) == 1 else "independent_verification.json")
    path.write_text(json.dumps(verification,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
    print(json.dumps({key:value for key,value in verification.items() if key != "checks"}))
    if not verification["all_passed"]:
        raise RuntimeError("saved evidence verification failed")


if __name__ == "__main__":
    main()
