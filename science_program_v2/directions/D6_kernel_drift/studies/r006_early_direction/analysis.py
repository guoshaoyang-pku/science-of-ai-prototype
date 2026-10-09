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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"Evidence recomputation differs: {label}")


def mean_interval(values):
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "seed_95pct_t_interval": None}
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "seed_95pct_t_interval": [mean - radius, mean + radius]}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    for name, expected in config["source_sha256"].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f"Pinned source differs: {name}")
    expected_cells = {f"{fn}_w{w}_s{s}": (fn, w, s) for fn in config["functions"]
                      for w in config["widths"] for s in config["seeds"]}
    rows = []
    early = config["checkpoints"].index(config["primary"]["early_step"])
    late = config["checkpoints"].index(config["primary"]["late_step"])
    threshold = np.log(config["primary"]["deadband_ratio"])
    for path in sorted((STUDY / "results").glob("*.json")):
        if path.name.endswith(".failure.json"):
            continue
        meta = json.loads(path.read_text())
        cell = path.stem
        if cell not in expected_cells or meta["status"] != "success":
            raise RuntimeError(f"Unexpected result: {cell}")
        fn, width, seed = expected_cells[cell]
        request = meta["request"]
        if (request["function"], request["width"], request["seed"]) != (fn, width, seed):
            raise RuntimeError("Cell identity differs")
        if (request["preregistration_sha256"] != sha(STUDY / "preregistration.json")
                or request["source_sha256"] != config["source_sha256"]
                or meta["contract_sha256"] != digest(request)
                or meta["arrays_sha256"] != sha(path.with_suffix(".npz"))):
            raise RuntimeError("Saved contract/array hash differs")
        for name in ["preregistration.json", *config["source_sha256"]]:
            relative = str((STUDY / name).relative_to(REPO))
            blob = subprocess.run(["git", "show", f"{meta['preregistration_commit']}:{relative}"],
                                  cwd=REPO, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(STUDY / name):
                raise RuntimeError("Saved commit does not match contract")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            if not all(np.isfinite(z[name]).all() for name in z.files):
                raise RuntimeError("Nonfinite evidence")
            if z["checkpoints"].tolist() != config["checkpoints"]:
                raise RuntimeError("Checkpoint contract differs")
            for name in ["train_x", "train_y"]:
                if hashlib.sha256(z[name].tobytes()).hexdigest() != request["inputs_sha256"][name]:
                    raise RuntimeError("Saved data hash differs")
            data = config["data_contract"]
            x = np.random.default_rng(data["data_seed"]).normal(size=(data["train_n"], data["input_dim"]))
            raw = (np.tanh(x[:, 0] * x[:, 1]) if fn == "product" else
                   np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2]))
            y = (raw - raw.mean()) / raw.std(ddof=0)
            close(z["train_x"], x, "data seed")
            close(z["train_y"], y, "targets")
            close(z["target_normalization"], [raw.mean(), raw.std(ddof=0)], "normalization")
            js = z["jacobians"]
            kernels = js @ js.transpose(0, 2, 1) / len(x)
            close(z["kernels"], kernels, "K=JJ^T/n")
            eigenvalues = np.linalg.eigvalsh(kernels)
            close(z["eigenvalues"], eigenvalues, "spectrum")
            scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
            close(z["trace_scales"], scales, "trace scaling")
            matched = scales[:, None, None] * kernels[0]
            r0 = z["residuals"][0]
            close(z["nonlinear_loss"][z["checkpoints"]],
                  np.mean(z["residuals"] ** 2, axis=1) / 2, "checkpoint loss")
            fixed, moving, target, gains = [], [], [], []
            eta = config["probe"]["stability_fraction"] / max(
                eigenvalues[:, -1].max(), (scales * eigenvalues[0, -1]).max())
            close(z["probe_lr"], eta, "shared probe eta")
            for i, (k, a0, r) in enumerate(zip(kernels, matched, z["residuals"])):
                fixed.append(float((r0 @ k @ r0) / (r0 @ a0 @ r0)))
                moving.append(float((r @ k @ r) / (r @ a0 @ r)))
                target.append(float((y @ k @ y) / (y @ a0 @ y)))
                for p, pseed in enumerate(config["probe"]["permutation_seeds"]):
                    perm = np.random.default_rng(pseed).permutation(r)
                    close(z["permutation_ratios"][i, p],
                          (perm @ k @ perm) / (perm @ a0 @ perm), "permutation ratio")
                for candidate_index, candidate in enumerate([kernels[0], a0, k]):
                    current = r.copy()
                    curve = [np.mean(current ** 2) / 2]
                    for _ in range(config["probe"]["steps"]):
                        current = current - eta * candidate @ current
                        curve.append(np.mean(current ** 2) / 2)
                    close(z["probe_loss"][i, candidate_index], curve, "probe recurrence")
                gains.append(float((z["probe_loss"][i, 1, -1] - z["probe_loss"][i, 2, -1])
                                   / z["probe_loss"][i, 0, 0]))
            close(z["fixed_r0_ratios"], fixed, "fixed residual ratio")
            close(z["moving_rt_ratios"], moving, "moving residual ratio")
            close(z["fixed_target_ratios"], target, "target ratio")
            log_early, log_late = float(np.log(fixed[early])), float(np.log(fixed[late]))
            direction = lambda value: int(np.sign(value)) if abs(value) > threshold else 0
            signs = [direction(log_early), direction(log_late)]
            rows.append({"cell_id": cell, "function": fn, "width": width, "seed": seed,
                         "early_log_ratio": log_early, "late_log_ratio": log_late,
                         "directions": signs, "same_nonzero_direction": signs[0] != 0 and signs[0] == signs[1],
                         "fixed_r0_ratios": fixed, "moving_rt_ratios": moving,
                         "fixed_target_ratios": target, "trace_scales": scales.tolist(),
                         "late_minus_early_log_ratio": log_late - log_early,
                         "probe_normalized_gains": gains, "seconds": meta["seconds"]})
    units = []
    for fn in config["functions"]:
        for width in config["widths"]:
            paired = [r for r in rows if r["function"] == fn and r["width"] == width]
            matches = sum(r["same_nonzero_direction"] for r in paired)
            complete = len(paired) == len(config["seeds"])
            units.append({"function": fn, "width": width, "saved_seeds": len(paired),
                          "matching_seeds": matches,
                          "pass": matches >= config["primary"]["unit_min_matching_seeds"] if complete else None,
                          "paired_late_minus_early_log_ratio": mean_interval(
                              [r["late_minus_early_log_ratio"] for r in paired]) if paired else None})
    attempts = {}
    for phase in ["preregistration", "final"]:
        path = STUDY / "executed" / f"{phase}_commit_attempt.json"
        if path.exists():
            attempts[phase] = json.loads(path.read_text())
    complete = len(rows) == config["planned_cells"]
    failed_commit = attempts.get("preregistration", {}).get("returncode", 0) != 0
    status = ("measurements_complete" if complete else "partial_measurements" if rows else
              "blocked_before_training" if failed_commit else "planned_not_run")
    result = {"study": config["study"], "round": 6, "direction_round": 1,
              "domain": "development", "status": status, "question": config["question"],
              "saved_cells": len(rows), "planned_cells": config["planned_cells"],
              "saved_recipe_units": sum(u["saved_seeds"] == len(config["seeds"]) for u in units),
              "planned_recipe_units": 4, "checkpoints": config["checkpoints"],
              "primary_prediction": {"id": "P1_direction_persistence",
                                     "status": ("pass" if sum(u["pass"] is True for u in units) >=
                                                config["primary"]["min_passing_units"] else "refuted")
                                     if complete else "not_evaluated",
                                     "passing_units": sum(u["pass"] is True for u in units) if complete else None},
              "units": units, "rows": rows,
              "training_seconds": sum(r["seconds"] for r in rows),
              "commit_attempts": attempts,
              "science_claims_added": 0 if not complete else None,
              "source_sha256": {"preregistration.json": sha(STUDY / "preregistration.json"),
                                **{name: sha(STUDY / name) for name in config["source_sha256"]}},
              "boundary": "固定r0、trace匹配、4个开发function×width单位；不预测泛化，不声称封存OOD验证。"}
    (STUDY / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "saved_cells": len(rows),
                      "primary_prediction": result["primary_prediction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
