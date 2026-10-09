from __future__ import annotations

import hashlib
import importlib.util
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
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11):
        raise RuntimeError(f"Evidence recomputation differs: {label}")


def interval(values):
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {"mean": mean, "seed_95pct_t_interval": [mean - radius, mean + radius] if radius is not None else None}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    for name, expected in config["source_sha256"].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f"Pinned source differs: {name}")
    expected_cells = {f"{fn}_w{w}_s{s}_sigma{sigma:g}": (fn, w, s, sigma)
                      for fn in config["functions"] for w in config["widths"]
                      for s in config["seeds"] for sigma in config["noise_sigmas"]}
    paths = sorted((STUDY / "results").glob("*.json"))
    paths = [p for p in paths if not p.name.endswith(".failure.json")]
    runner = None
    if paths:
        spec = importlib.util.spec_from_file_location("r007_runner", STUDY / "executed/run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    rows, pairing_arrays = [], {}
    for path in paths:
        if path.stem not in expected_cells:
            raise RuntimeError(f"Unexpected cell: {path.stem}")
        fn, width, seed, sigma = expected_cells[path.stem]
        meta = json.loads(path.read_text())
        data = runner.make_data(config, fn, sigma)
        expected_request = {"function": fn, "width": width, "seed": seed, "sigma": sigma,
                            "preregistration_sha256": sha(STUDY / "preregistration.json"),
                            "source_sha256": config["source_sha256"],
                            "inputs_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in data.items()}}
        if (meta["cell_id"] != path.stem or meta["status"] != "success"
                or meta["request"] != expected_request
                or meta["contract_sha256"] != runner.digest(expected_request)
                or meta["arrays_sha256"] != sha(path.with_suffix(".npz"))):
            raise RuntimeError("Saved contract/array hash differs")
        for name in ["preregistration.json", *config["source_sha256"]]:
            relative = str((STUDY / name).relative_to(REPO))
            blob = subprocess.run(["git", "show", f"{meta['preregistration_commit']}:{relative}"],
                                  cwd=REPO, capture_output=True, check=True).stdout
            if blob != (STUDY / name).read_bytes():
                raise RuntimeError("Saved commit differs from executable contract")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            if not all(np.isfinite(z[n]).all() for n in z.files):
                raise RuntimeError("Nonfinite evidence")
            for name, expected in data.items():
                close(z[name], expected, f"data contract: {name}")
            if z["checkpoints"].tolist() != config["checkpoints"]:
                raise RuntimeError("Checkpoint contract differs")
            train_loss = np.mean((z["train_outputs"] - z["train_y"]) ** 2, axis=2) / 2
            clean_train_loss = np.mean((z["train_outputs"] - z["clean_train_y"]) ** 2, axis=2) / 2
            test_loss = np.mean((z["test_outputs"] - z["test_y"]) ** 2, axis=2) / 2
            close(train_loss, z["train_loss"][z["checkpoints"]], "checkpoint loss")
            close(z["train_outputs"][0, 0], z["train_outputs"][0, 1], "matched initial output")
            j, jt = z["initial_jacobian_train"], z["initial_jacobian_test"]
            close(z["train_outputs"][-1, 1], z["train_outputs"][0, 1] + j @ z["final_tangent_delta"], "tangent train endpoint")
            close(z["test_outputs"][-1, 1], z["test_outputs"][0, 1] + jt @ z["final_tangent_delta"], "tangent test endpoint")
            r = z["train_outputs"][0, 1] - z["train_y"]
            kernel = j @ j.T / len(r)
            recurrence = []
            for step in range(config["recipe"]["steps"] + 1):
                recurrence.append(np.mean(r ** 2) / 2)
                r = r - config["recipe"]["lr"] * kernel @ r
            close(z["train_loss"][:, 1], recurrence, "exact tangent recurrence")
            pairing_arrays[(fn, width, seed, sigma)] = {n: z[n].copy() for n in
                ["initial_parameters", "initial_jacobian_train", "initial_jacobian_test", "train_x", "test_x", "test_y", "clean_train_y"]}
            late = config["checkpoints"].index(config["secondary"]["late_start_step"])
            rows.append({"cell_id": path.stem, "function": fn, "width": width, "seed": seed, "sigma": sigma,
                         "train_loss": train_loss.tolist(), "clean_train_loss": clean_train_loss.tolist(),
                         "test_loss": test_loss.tolist(),
                         "train_gain": float(train_loss[-1, 1] - train_loss[-1, 0]),
                         "test_gap": float(test_loss[-1, 0] - test_loss[-1, 1]),
                         "late_test_change_real": float(test_loss[-1, 0] - test_loss[late, 0]),
                         "late_test_change_tangent": float(test_loss[-1, 1] - test_loss[late, 1]),
                         "seconds": meta["seconds"]})
    pairs = []
    for fn in config["functions"]:
        for width in config["widths"]:
            for seed in config["seeds"]:
                matched = [r for r in rows if (r["function"], r["width"], r["seed"]) == (fn, width, seed)]
                if len(matched) != 2:
                    continue
                clean = next(r for r in matched if r["sigma"] == 0)
                noisy = next(r for r in matched if r["sigma"] == 1)
                for name in pairing_arrays[(fn, width, seed, 0)]:
                    close(pairing_arrays[(fn, width, seed, 0)][name],
                          pairing_arrays[(fn, width, seed, 1)][name], f"noise pairing: {name}")
                pairs.append({"function": fn, "width": width, "seed": seed,
                              "noise_amplification": noisy["test_gap"] - clean["test_gap"],
                              "clean_train_gain": clean["train_gain"], "noisy_train_gain": noisy["train_gain"],
                              "clean_test_gap": clean["test_gap"], "noisy_test_gap": noisy["test_gap"],
                              "noise_risk_change_real": noisy["test_loss"][-1][0] - clean["test_loss"][-1][0],
                              "noise_risk_change_tangent": noisy["test_loss"][-1][1] - clean["test_loss"][-1][1]})
    units = []
    metrics = ["noise_amplification", "clean_train_gain", "noisy_train_gain", "clean_test_gap",
               "noisy_test_gap", "noise_risk_change_real", "noise_risk_change_tangent"]
    for fn in config["functions"]:
        for width in config["widths"]:
            paired = [p for p in pairs if (p["function"], p["width"]) == (fn, width)]
            measured = {m: interval([p[m] for p in paired]) for m in metrics} if paired else {}
            hits = sum(p["noise_amplification"] > 0 for p in paired)
            complete = len(paired) == len(config["seeds"])
            passed = (measured["noise_amplification"]["mean"] >= config["primary"]["minimum_mean_amplification"]
                      and hits >= config["primary"]["minimum_positive_seeds"]) if complete else None
            units.append({"function": fn, "width": width, "paired_seeds": len(paired),
                          "positive_amplification_seeds": hits, "pass": passed, "metrics": measured})
    attempts = {}
    for phase in ["preregistration", "final"]:
        path = STUDY / "executed" / f"{phase}_commit_attempt.json"
        if path.exists():
            attempts[phase] = json.loads(path.read_text())
    complete = len(rows) == config["planned_cells"]
    status = ("measurements_complete" if complete else "partial_measurements" if rows else
              "blocked_before_training" if attempts.get("preregistration", {}).get("returncode", 0) != 0 else "planned_not_run")
    passing = sum(u["pass"] is True for u in units)
    summary = {"study": config["study"], "round": 7, "direction_round": 1, "domain": "development",
               "status": status, "question": config["question"], "saved_cells": len(rows),
               "planned_cells": config["planned_cells"], "trained_trajectories": 2 * len(rows),
               "planned_recipe_units": 4, "complete_noise_pairs": len(pairs),
               "prediction": {"id": "P1_noise_amplification",
                              "status": ("pass" if passing >= config["primary"]["minimum_passing_units"] else "refuted") if complete else "not_evaluated",
                              "passing_units": passing if complete else None},
               "units": units, "paired_seed_results": pairs, "rows": rows,
               "training_seconds": sum(r["seconds"] for r in rows), "commit_attempts": attempts,
               "source_sha256": {"preregistration.json": sha(STUDY / "preregistration.json"), **config["source_sha256"]},
               "reference_audit": "executed/b02_audit.json",
               "boundary": "仅开发条件、固定data/noise seed；test gap是配对终点差，晚期test增量另报；不区分核方向与隐式偏置，不视为封存OOD。"}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "saved_cells": len(rows), "prediction": summary["prediction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
