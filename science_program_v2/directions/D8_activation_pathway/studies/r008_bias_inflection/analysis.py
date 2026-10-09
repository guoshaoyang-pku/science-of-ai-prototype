from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-11, rtol=1e-8):
        raise RuntimeError(f"Evidence recomputation differs: {label}")


def interval(values):
    a = np.array(values, dtype=float)
    radius = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / len(a) ** .5) if len(a) > 1 else None
    mean = float(a.mean())
    return {"mean": mean, "seed_95pct_t_interval": [mean - radius, mean + radius] if radius is not None else None,
            "min": float(a.min()), "max": float(a.max())}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    for name, expected in config["source_sha256"].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f"Pinned source differs: {name}")
    paths = sorted((STUDY / "results").glob("*.json"))
    rows, weights = [], {}
    if paths:
        manifest_path = STUDY / "executed/source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["preregistration_sha256"] != sha(STUDY / "preregistration.json") or manifest["source_sha256"] != config["source_sha256"]:
            raise RuntimeError("Manifest source mismatch")
        for name in ["preregistration.json", *config["source_sha256"]]:
            result = subprocess.run(["git", "show", f"{manifest['preregistration_commit']}:{(STUDY / name).relative_to(REPO)}"], cwd=REPO, capture_output=True, check=True)
            if result.stdout != (STUDY / name).read_bytes():
                raise RuntimeError("Execution lacks matching preregistration commit")
        if sha(STUDY / "executed/data.npz") != manifest["data_file_sha256"]:
            raise RuntimeError("Data file hash mismatch")
        with np.load(STUDY / "executed/data.npz", allow_pickle=False) as z:
            data = {n: z[n].copy() for n in z.files}
        for name, a in data.items():
            if hashlib.sha256(a.tobytes()).hexdigest() != manifest["data"][name]["sha256"]:
                raise RuntimeError("Data array hash mismatch")
        expected_cells = {(s, b, a) for s in config["seeds"] for b in config["bias_labels"] for a in config["scales"]}
    for path in paths:
        row = json.loads(path.read_text())
        request = row["request"]
        cell = request["cell"]
        key = (cell["seed"], cell["bias_label"], cell["scale"])
        if key not in expected_cells:
            raise RuntimeError("Unexpected or duplicate cell")
        expected_cells.remove(key)
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if (row["status"] != "success" or row["cell_id"] != path.stem or request["manifest_sha256"] != sha(manifest_path)
                or row["contract_sha256"] != contract or sha(path.with_suffix(".npz")) != row["arrays_sha256"]):
            raise RuntimeError("Cell contract/arrays hash mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            if not all(np.isfinite(z[n]).all() for n in z.files):
                raise RuntimeError("Nonfinite evidence")
            b, a, u = request["bias"], cell["scale"], z["train_u"]
            if ((cell["bias_label"] == "zero" and b != 0) or (cell["bias_label"] == "one" and b != 1)
                    or (cell["bias_label"] == "inflection" and abs(b * np.tanh(b / 2) - 2) > 1e-12)):
                raise RuntimeError("Bias condition differs")
            close(u, data["train_x"] @ z["hidden_weight"].T, "preactivation")
            positive = (b + a * u) / (1 + np.exp(-(b + a * u)))
            negative = (b - a * u) / (1 + np.exp(-(b - a * u)))
            center = positive.mean(axis=0)
            norm = np.sqrt(np.mean((positive - center) ** 2))
            close(z["odd_raw"], (positive - negative) / 2, "odd features")
            close(z["even_centered_raw"], (positive + negative) / 2 - center, "even features")
            close(z["train_features"], (positive - center) / norm / config["architecture"]["width"] ** .5, "normalized features")
            close(z["train_y"], data["train_y"], "target")
            ratio = float(np.mean(z["even_centered_raw"] ** 2) / np.mean(z["odd_raw"] ** 2))
            sig = 1 / (1 + np.exp(-b))
            q, h = sig * (1 - sig), 1 - 2 * sig
            first, second = sig + b * q, q * (2 + b * h)
            fourth = q * (4 * (h * h - 2 * q) + b * (h ** 3 - 8 * q * h))
            order = 4 if cell["bias_label"] == "inflection" else 2
            leading_even = (fourth if order == 4 else second) * a ** order * (u ** order - np.mean(u ** order, axis=0)) / (24 if order == 4 else 2)
            leading_ratio = float(np.mean(leading_even ** 2) / np.mean((first * a * u) ** 2))
            phi, y = z["train_features"], z["train_y"]
            kernel = phi @ phi.T / len(phi)
            energy = np.sum(y * (kernel @ y), axis=0) / np.sum(y * y, axis=0)
            residual, curve = -y.copy(), []
            for step in range(config["optimizer"]["steps"] + 1):
                curve.append(np.mean(residual ** 2, axis=0))
                residual -= 2 * config["optimizer"]["lr"] * kernel @ residual
            close(z["train_curve"], curve, "independent kernel recurrence")
            train_mse = np.mean((z["checkpoint_outputs"] - y) ** 2, axis=1)
            test_mse = np.mean((z["checkpoint_test_outputs"] - z["test_y"]) ** 2, axis=1)
            close(train_mse, z["train_curve"][config["checkpoints"]], "checkpoint MSE")
            close(phi @ z["final_head"], z["checkpoint_outputs"][-1], "final head")
            if cell["seed"] in weights:
                close(z["hidden_weight"], weights[cell["seed"]], "same-seed bias/scale pairing")
            weights[cell["seed"]] = z["hidden_weight"].copy()
            rows.append({**cell, "bias": b, "ratio": ratio, "leading_ratio": leading_ratio,
                         "leading_relative_error": abs(ratio / leading_ratio - 1), "target_kernel_energy": energy.tolist(),
                         "initial_progress": (z["train_curve"][0] - z["train_curve"][1]).tolist(),
                         "final_train_mse": train_mse[-1].tolist(), "final_test_mse": test_mse[-1].tolist(),
                         "spectral_max_error": float(np.max(np.abs(z["train_curve"] - z["spectral_curve"]))), "seconds": row["seconds"]})
    pairs = []
    for bias in config["bias_labels"]:
        for seed in config["seeds"]:
            small = sorted([r for r in rows if r["bias_label"] == bias and r["seed"] == seed and r["scale"] in config["primary_scales"]], key=lambda r: r["scale"])
            if len(small) == 2:
                growth = small[1]["ratio"] / small[0]["ratio"]
                pairs.append({"bias_label": bias, "seed": seed, "growth": growth, "exponent": float(np.log(growth) / np.log(small[1]["scale"] / small[0]["scale"])),
                              "small_scale_leading_relative_error": small[0]["leading_relative_error"]})
    complete = len(rows) == config["planned_cells"]
    ordinary = [p for p in pairs if p["bias_label"] != "inflection"]
    special = [p for p in pairs if p["bias_label"] == "inflection"]
    criteria = config["criteria"]
    passed = {"P1": all(criteria["ordinary_growth"][0] <= p["growth"] <= criteria["ordinary_growth"][1] for p in ordinary),
              "P2": all(criteria["inflection_growth"][0] <= p["growth"] <= criteria["inflection_growth"][1] for p in special),
              "P3": all(p["small_scale_leading_relative_error"] <= criteria["leading_relative_error"] for p in pairs)}
    predictions = [{"id": p["id"], "status": ("pass" if passed[p["id"]] else "refuted") if complete else "not_evaluated"} for p in config["predictions"]]
    units = []
    for bias in config["bias_labels"]:
        matched = [p for p in pairs if p["bias_label"] == bias]
        units.append({"bias_label": bias, "paired_seeds": len(matched), "growth": interval([p["growth"] for p in matched]) if matched else None,
                      "exponent": interval([p["exponent"] for p in matched]) if matched else None})
    attempts = {}
    for phase in ("preregistration", "final"):
        path = STUDY / "executed" / f"{phase}_commit_attempt.json"
        if path.exists():
            attempts[phase] = json.loads(path.read_text())
    blocked = attempts.get("preregistration", {}).get("returncode", 0) != 0
    summary = {"study": config["study"], "round": 8, "direction_round": 1, "domain": "development", "question": config["question"],
               "status": "measurements_complete" if complete else "partial_measurements" if rows else "blocked_before_training" if blocked else "planned_not_run",
               "saved_cells": len(rows), "planned_cells": config["planned_cells"], "trained_trajectories": 2 * len(rows), "recipe_scale_target_units": 24,
               "bias_recipes": 3, "predictions": predictions, "units": units, "paired_seed_results": pairs, "rows": rows,
               "execution_check": {"spectral_error_max": max((r["spectral_max_error"] for r in rows), default=None), "tolerance": criteria["spectral_tolerance"]},
               "training_seconds": sum(r["seconds"] for r in rows), "commit_attempts": attempts,
               "source_sha256": {"preregistration.json": sha(STUDY / "preregistration.json"), **config["source_sha256"]}, "boundary": config["boundary"]}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "saved_cells": len(rows), "predictions": predictions}))


if __name__ == "__main__":
    main()
