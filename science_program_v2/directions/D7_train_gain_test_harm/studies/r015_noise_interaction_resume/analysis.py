from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def interval(values):
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {"mean": mean, "seed_95pct_t_interval": [mean-radius, mean+radius] if radius is not None else None}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    original = REPO / config["resume"]["original_study_path"]
    for relative, expected in config["source_sha256"].items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f"Pinned source differs: {relative}")
    receipt = json.loads((STUDY / "executed/pre_execution_audit.json").read_text())
    if receipt["preregistration_sha256"] != sha(STUDY / "preregistration.json"):
        raise RuntimeError("Resume receipt contract differs")
    for relative in [str((STUDY / "preregistration.json").relative_to(REPO)), *config["source_sha256"]]:
        blob = subprocess.check_output(["git", "show", f"{receipt['resume_commit']}:{relative}"], cwd=REPO)
        if blob != (REPO / relative).read_bytes():
            raise RuntimeError(f"Resume commit differs: {relative}")
    process = subprocess.run([sys.executable, "-B", str(original / "analysis.py")],
                             cwd=REPO, capture_output=True, text=True)
    save(STUDY / "executed/original_analysis_log.json", {
        "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
    if process.returncode:
        raise RuntimeError("Original analysis failed; see original_analysis_log.json")
    summary = json.loads((original / "summary.json").read_text())
    plan = config["original_contract"]
    torch.set_num_threads(1)
    maxima = {"initial_parameters": 0.0, "initial_train_outputs": 0.0,
              "initial_test_outputs": 0.0, "initial_test_pairing": 0.0, "real_train_endpoint": 0.0,
              "real_test_endpoint": 0.0, "tangent_train_checkpoints": 0.0,
              "tangent_test_checkpoints": 0.0}
    checked = []
    for row in summary["rows"]:
        meta_path = original / "results" / (row["cell_id"] + ".json")
        metadata = json.loads(meta_path.read_text())
        with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as z:
            torch.manual_seed(row["seed"])
            width = row["width"]
            model = nn.Sequential(nn.Linear(plan["data_contract"]["input_dim"], width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)).double()
            tx, vx = torch.from_numpy(z["train_x"]), torch.from_numpy(z["test_x"])
            with torch.no_grad():
                initial = torch.cat([p.reshape(-1) for p in model.parameters()]).numpy()
                comparisons = {
                    "initial_parameters": (initial, z["initial_parameters"]),
                    "initial_train_outputs": (model(tx).reshape(-1).numpy(), z["train_outputs"][0, 0]),
                    "initial_test_outputs": (model(vx).reshape(-1).numpy(), z["test_outputs"][0, 0]),
                    "initial_test_pairing": (z["test_outputs"][0, 0], z["test_outputs"][0, 1])}
                offset = 0
                for parameter in model.parameters():
                    count = parameter.numel()
                    parameter.copy_(torch.from_numpy(z["final_parameters"][offset:offset+count]).reshape_as(parameter))
                    offset += count
                if offset != len(z["final_parameters"]):
                    raise RuntimeError("Final parameter shape differs")
                comparisons["real_train_endpoint"] = (model(tx).reshape(-1).numpy(), z["train_outputs"][-1, 0])
                comparisons["real_test_endpoint"] = (model(vx).reshape(-1).numpy(), z["test_outputs"][-1, 0])
            j, jt = z["initial_jacobian_train"], z["initial_jacobian_test"]
            kernel = np.einsum("ip,jp->ij", j, j, optimize=False) / len(tx)
            cross_kernel = np.einsum("ip,jp->ij", jt, j, optimize=False) / len(tx)
            residual = z["train_outputs"][0, 1] - z["train_y"]
            test_prediction = z["test_outputs"][0, 1].copy()
            train_predictions, test_predictions = [], []
            for step in range(plan["recipe"]["steps"] + 1):
                if step in plan["checkpoints"]:
                    train_predictions.append(residual + z["train_y"])
                    test_predictions.append(test_prediction.copy())
                if step == plan["recipe"]["steps"]:
                    break
                test_prediction -= plan["recipe"]["lr"] * np.einsum("ij,j->i", cross_kernel, residual, optimize=False)
                residual -= plan["recipe"]["lr"] * np.einsum("ij,j->i", kernel, residual, optimize=False)
            comparisons["tangent_train_checkpoints"] = (np.array(train_predictions), z["train_outputs"][:, 1])
            comparisons["tangent_test_checkpoints"] = (np.array(test_predictions), z["test_outputs"][:, 1])
            for name, (actual, expected) in comparisons.items():
                maxima[name] = max(maxima[name], float(np.max(np.abs(actual - expected))))
                if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11):
                    raise RuntimeError(f"Independent evidence check failed: {row['cell_id']} {name}")
        checked.append({"cell_id": row["cell_id"], "metadata_sha256": sha(meta_path),
                        "arrays_sha256": sha(meta_path.with_suffix(".npz")),
                        "preregistration_commit": metadata["preregistration_commit"]})
    late = plan["checkpoints"].index(plan["secondary"]["late_start_step"])
    late_units = []
    for function in plan["functions"]:
        for width in plan["widths"]:
            for sigma in plan["noise_sigmas"]:
                rows = [r for r in summary["rows"] if (r["function"], r["width"], r["sigma"]) == (function, width, sigma)]
                if not rows:
                    continue
                metrics = {}
                for model_index, label in enumerate(["real", "tangent"]):
                    values = [r["test_loss"][-1][model_index] - r["test_loss"][late][model_index] for r in rows]
                    train_values = [r["train_loss"][-1][model_index] - r["train_loss"][late][model_index] for r in rows]
                    clean_values = [r["clean_train_loss"][-1][model_index] for r in rows]
                    metrics[label] = {"late_test_change": interval(values), "late_train_change": interval(train_values),
                                      "endpoint_clean_train_loss": interval(clean_values),
                                      "positive_late_test_seeds": sum(v > 0 for v in values)}
                late_units.append({"function": function, "width": width, "sigma": sigma,
                                   "seeds": len(rows), "metrics": metrics,
                                   "positive_endpoint_gap_seeds": sum(r["test_gap"] > 0 for r in rows),
                                   "positive_gap_but_real_test_decreased_seeds": sum(
                                       r["test_gap"] > 0 and r["late_test_change_real"] < 0 for r in rows)})
    verification = {"checked_cells": len(checked), "maximum_absolute_errors": maxima,
                    "cells": checked, "status": "passed", "method": "原analysis核验hash/配对/半MSE/切线train递推；独立重建初始与终点真实模型；einsum复算切线全部train/test checkpoint"}
    save(STUDY / "executed/saved_evidence_verification.json", verification)
    summary.update({"study": config["study"], "round": config["round"], "direction_round": config["direction_round"],
                    "original_study": plan["study"], "original_design_round": plan["round"],
                    "resume_commit": receipt["resume_commit"], "execution_head": receipt["resume_commit"],
                    "original_preregistration_commit": receipt["original_commit"],
                    "historical_commit_attempts": summary.pop("commit_attempts"),
                    "late_units": late_units, "independent_verification": {
                        "status": "passed", "checked_cells": len(checked), "maximum_absolute_errors": maxima},
                    "resume_source_sha256": config["source_sha256"],
                    "original_summary_sha256": sha(original / "summary.json"),
                    "original_analysis_warnings": process.stderr,
                    "reference_audit": str((original / "executed/b02_audit.json").relative_to(REPO))})
    save(STUDY / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "saved_cells": summary["saved_cells"],
                      "prediction": summary["prediction"], "independent_verification": summary["independent_verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
