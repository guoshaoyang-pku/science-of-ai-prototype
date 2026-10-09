#!/usr/bin/env python3
"""Recompute registered comparisons from saved cells, never train."""
import importlib.util
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("execution", STUDY / "executed" / "run.py")
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)
np = execution.np


def first_hit(curve, threshold):
    hits = np.flatnonzero(curve <= threshold)
    return int(hits[0]) if len(hits) else None


def analyze():
    config = json.loads((STUDY / "preregistration.json").read_text())
    pins = {name: execution.sha(STUDY / name) for name in ("preregistration.json", "executed/run.py", "analysis.py")}
    if config["execution_source_sha256"] != pins["executed/run.py"] or config["analysis_source_sha256"] != pins["analysis.py"]:
        raise RuntimeError("Source pin mismatch")
    receipt_path = STUDY / "results" / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0}
    requested = list(execution.cells(config))
    if set(receipt["cells"]) - {execution.label(c) for c in requested}:
        raise RuntimeError("Unregistered receipt cell")
    entries = []
    for cell in requested:
        name = execution.label(cell)
        path = STUDY / "results" / f"{name}.json"
        exists = [path.exists(), path.with_suffix(".npz").exists(), name in receipt["cells"]]
        if not any(exists):
            continue
        if not all(exists):
            raise RuntimeError(f"Partial cell: {name}")
        row = execution.verify_saved(path, receipt, cell, pins)
        with np.load(path.with_suffix(".npz")) as arrays:
            x, y = arrays["features"], arrays["target"]
            kernel = x @ x.T / len(y)
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
            weights = (eigenvectors.T @ y) ** 2 / (y @ y)
            times = np.arange(config["optimizer"]["steps"] + 1)
            reference = (weights[:, None] * (1 - config["optimizer"]["eta"] * eigenvalues[:, None]) ** (2 * times)).sum(axis=0)
            loss = arrays["normalized_loss"]
            actual = first_hit(loss, config["threshold"])
            prediction = first_hit(reference, config["threshold"])
            if loss.shape != times.shape or not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                raise RuntimeError("Invalid loss array")
            curve_error = float(np.max(np.abs(loss - reference)))
            saved_prediction_error = float(np.max(np.abs(reference - arrays["predicted_curve"])))
            probability = eigenvalues / eigenvalues.sum()
            descriptors = {"parameter_count": x.shape[1], "trace": float(eigenvalues.sum()),
                           "effective_rank": float(np.exp(-np.sum(probability * np.log(probability)))),
                           "rayleigh": float(y @ kernel @ y / (y @ y)), "initial_loss": float(y @ y / (2 * len(y)))}
            final_residual = x @ arrays["final_theta"] - y
            final_error = abs(float(final_residual @ final_residual / (y @ y)) - float(loss[-1]))
            audit_pass = curve_error <= 1e-10 and saved_prediction_error <= 1e-10 and final_error <= 1e-12 and actual is not None and actual == prediction == row["predicted_threshold_step"]
            entries.append({**cell, "label": name, "threshold_step": actual, "predicted_threshold_step": prediction,
                            "censored": actual is None, "descriptors": descriptors,
                            "max_curve_error": curve_error, "saved_prediction_error": saved_prediction_error,
                            "final_residual_error": final_error, "execution_audit_pass": bool(audit_pass)})
    pairs = []
    for exponent in config["spectral_exponents"]:
        for seed in config["seeds"]:
            matched = {e["target"]: e for e in entries if e["exponent"] == exponent and e["seed"] == seed}
            if len(matched) != 2:
                continue
            left, right = matched["middle"], matched["endpoints"]
            a, b = left["threshold_step"], right["threshold_step"]
            gaps = {k: right["descriptors"][k] - left["descriptors"][k] for k in left["descriptors"]}
            pairs.append({"exponent": exponent, "seed": seed, "middle_step": a, "endpoints_step": b,
                          "step_difference": b - a if a is not None and b is not None else None,
                          "step_ratio": b / a if a is not None and b is not None and a > 0 else None,
                          "descriptor_differences": gaps, "matched": all(abs(v) <= 1e-12 for v in gaps.values())})
    complete = len(entries) == config["counts"]["planned_cells"]
    predictions = {p["id"]: {"status": "not_evaluated", "reason": "No complete committed experiment"} for p in config["predictions"]}
    if complete:
        flat = [p for p in pairs if p["exponent"] == 0]
        decaying = [p for p in pairs if p["exponent"] > 0]
        predictions = {"P1": {"status": "supported" if all(e["execution_audit_pass"] for e in entries) and all(p["matched"] for p in pairs) else "refuted"},
                       "P2": {"status": "supported" if all(p["middle_step"] == p["endpoints_step"] == 4 for p in flat) else "refuted", "pairs": len(flat)},
                       "P3": {"status": "supported" if all(p["step_ratio"] is not None and p["step_ratio"] >= config["minimum_step_ratios"][str(p["exponent"])] for p in decaying) else "refuted", "pairs": len(decaying)}}
    attempts = [json.loads(p.read_text()) for p in sorted((STUDY / "executed").glob("*commit_attempt*.json"))]
    summary = {"study": config["study"], "round": 4, "direction_round": 1, "direction": config["direction"],
               "domain": "development", "status": "complete" if complete else ("partial" if entries else "blocked_before_training"),
               "round_result": "ok" if complete else ("partial" if entries else "failed"), "training_started": bool(entries),
               "hashes": pins, "counts": {**config["counts"], "saved_cells": len(entries)},
               "compute_seconds": receipt["compute_seconds"], "predictions": predictions, "cells": entries,
               "paired_comparisons": pairs, "commit_attempts": attempts, "boundaries": config["boundaries"],
               "known_theory_is_new_claim": False,
               "next_action": "先成功commit合同与pinned源码，再运行executed/run.py --max-new-cells 1；恢复核验hash后续跑，最后运行analysis.py。" if not complete else "从配对结果划定标量描述量边界；下轮另选一个小问题并预注册。"}
    execution.save(STUDY / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "saved_cells": len(entries), "predictions": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
