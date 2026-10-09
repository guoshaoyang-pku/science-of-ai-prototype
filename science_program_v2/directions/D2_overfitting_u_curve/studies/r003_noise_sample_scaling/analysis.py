#!/usr/bin/env python3
"""Recompute oracle minima and registered predictions from saved cells."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("execution", STUDY / "executed" / "run.py")
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def analyze():
    config = json.loads((STUDY / "preregistration.json").read_text())
    results = STUDY / "results"
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0}
    entries = []
    for n, seed, variance in itertools.product(config["data"]["train_rows"], config["seeds"], config["data"]["noise_variance"]):
        label = f"n{n}_seed{seed}_var{variance:g}"
        path = results / f"{label}.json"
        if not path.exists():
            if label in receipt["cells"] or path.with_suffix(".npz").exists():
                raise RuntimeError(f"Partial saved cell: {label}")
            continue
        row = json.loads(path.read_text())
        if receipt["cells"].get(label) != execution.sha(path) or row["arrays_sha256"] != execution.sha(path.with_suffix(".npz")):
            raise RuntimeError(f"Result hash mismatch: {label}")
        current_pins = {k: execution.sha(STUDY / k) for k in row["pins"]["sha256"]}
        if current_pins != row["pins"]["sha256"]:
            raise RuntimeError("Sources changed after execution")
        if row["data_sha256"] != execution.sha(STUDY / row["data_file"]):
            raise RuntimeError("Saved data hash mismatch")
        contract = hashlib.sha256(json.dumps({"cell": row["cell"], "pins": current_pins,
                                             "data_sha256": row["data_sha256"]}, sort_keys=True).encode()).hexdigest()
        if contract != row["contract_sha256"] or row["cell"] != {"n": n, "seed": seed, "noise_variance": variance}:
            raise RuntimeError("Cell contract mismatch")
        with np.load(path.with_suffix(".npz")) as arrays:
            bias = arrays["signal_bias"]
            variance_unit = arrays["variance_unit"]
            risk = bias + variance * variance_unit
            if len(risk) != config["optimizer"]["steps"] + 1 or not np.allclose(risk, arrays["expected_risk"], rtol=0, atol=1e-12):
                raise RuntimeError("Saved risk contract mismatch")
            q = 1 - 2 * config["optimizer"]["lr"] * arrays["eigenvalues"]
            factors = np.zeros(len(q))
            max_signal_error, max_variance_error = 0., 0.
            for step in range(len(risk)):
                coefficients = factors * arrays["signal_projection"]
                spectral_bias = coefficients @ arrays["basis_gram"] @ coefficients - 2 * arrays["basis_target"] @ coefficients + float(arrays["audit_target_square"])
                spectral_variance = np.diag(arrays["basis_gram"]) @ (factors ** 2)
                max_signal_error = max(max_signal_error, abs(spectral_bias - bias[step]))
                max_variance_error = max(max_variance_error, abs(spectral_variance - variance_unit[step]))
                factors = q * factors + 2 * config["optimizer"]["lr"]
            with np.load(STUDY / row["data_file"]) as data:
                direct_final = float(np.mean((data["audit_features"] @ arrays["final_heads"][:, 1] - data["audit_y"]) ** 2))
            t = int(np.argmin(risk))
            audit = {"max_train_increase": float(np.diff(arrays["train_mse"]).max()),
                     "max_spectral_signal_error": max_signal_error,
                     "max_spectral_variance_error": max_variance_error,
                     "final_direct_bias_error": abs(direct_final - bias[-1])}
            p1 = all(np.isfinite(arrays[k]).all() for k in arrays.files) and audit["max_train_increase"] <= 1e-10 and audit["max_spectral_signal_error"] <= 1e-9 and audit["final_direct_bias_error"] <= 1e-9 and audit["max_spectral_variance_error"] <= 1e-9
            entries.append({**row["cell"], "label": label, "t_star": t, "n_over_variance": n / variance,
                            "interior": 0 < t < config["optimizer"]["steps"],
                            "minimum_expected_risk": float(risk[t]), "final_expected_risk": float(risk[-1]),
                            "u_shape": 0 < t < len(risk) - 1 and float(risk[-1] - risk[t]) >= .05,
                            "audit": audit, "P1_pass": bool(p1)})
    complete = len(entries) == config["counts"]["planned_cells"]
    pairs = []
    for left, right in itertools.combinations(entries, 2):
        if left["seed"] != right["seed"] or left["n"] == right["n"] or left["n_over_variance"] != right["n_over_variance"]:
            continue
        ratio = right["t_star"] / left["t_star"] if left["t_star"] > 0 else None
        passed = left["interior"] and right["interior"] and ratio is not None and .5 <= ratio <= 2
        pairs.append({"left": left["label"], "right": right["label"], "t_star_ratio": ratio,
                      "log2_ratio": float(np.log2(ratio)) if ratio is not None and ratio > 0 else None, "pass": bool(passed)})
    forecasts, fits = [], []
    for held_n in config["data"]["train_rows"]:
        train = [e for e in entries if e["n"] != held_n and e["interior"]]
        valid_fit = len(train) >= 2 and len({e["n_over_variance"] for e in train}) >= 2
        beta = None
        if valid_fit:
            design = np.column_stack([np.ones(len(train)), np.log([e["n_over_variance"] for e in train])])
            beta = np.linalg.lstsq(design, np.log([e["t_star"] for e in train]), rcond=None)[0]
        fits.append({"held_n": held_n, "training_cells": len(train), "a": float(beta[0]) if beta is not None else None,
                     "b": float(beta[1]) if beta is not None else None})
        for entry in (e for e in entries if e["n"] == held_n):
            prediction = float(np.exp(np.clip(beta[0] + beta[1] * np.log(entry["n_over_variance"]), 0, np.log(config["optimizer"]["steps"])))) if beta is not None else None
            factor = max(prediction / entry["t_star"], entry["t_star"] / prediction) if prediction is not None and entry["t_star"] > 0 else None
            forecasts.append({"label": entry["label"], "prediction": prediction, "factor_error": factor,
                              "pass": bool(entry["interior"] and factor is not None and factor <= 2)})
    predictions = {p["id"]: {"status": "not_evaluated", "reason": "No complete committed experiment"} for p in config["predictions"]}
    if complete:
        predictions = {"P1": {"status": "supported" if all(e["P1_pass"] for e in entries) else "refuted"},
                       "P2": {"status": "supported" if all(p["pass"] for p in pairs) else "refuted", "passed": sum(p["pass"] for p in pairs), "total": len(pairs)},
                       "P3": {"status": "supported" if sum(f["pass"] for f in forecasts) >= 48 else "refuted", "passed": sum(f["pass"] for f in forecasts), "total": 60}}
    elif any(not p["pass"] for p in pairs):
        predictions["P2"] = {"status": "refuted", "reason": "Saved partial experiment contains a counterexample"}
    attempts = [json.loads(p.read_text()) for p in sorted((STUDY / "executed").glob("*commit_attempt*.json"))]
    summary = {"study": config["study"], "round": 3, "direction": config["direction"], "domain": "development",
               "status": "complete" if complete else ("partial" if entries else "blocked_before_training"),
               "round_result": "ok" if complete else ("partial" if entries else "failed"),
               "training_started": bool(entries), "fit_started": bool(entries),
               "preregistration_sha256": execution.sha(STUDY / "preregistration.json"),
               "execution_source_sha256": execution.sha(STUDY / "executed" / "run.py"),
               "analysis_source_sha256": execution.sha(__file__),
               "counts": {**config["counts"], "saved_cells": len(entries), "boundary_minima": sum(not e["interior"] for e in entries)},
               "compute_seconds": receipt["compute_seconds"], "predictions": predictions,
               "cells": entries, "equal_ratio_pairs": pairs, "leave_one_n_out_fits": fits,
               "forecasts": forecasts, "boundaries": config["boundaries"],
               "commit_attempts": attempts, "measured_claims": [],
               "next_action": "先在本repo成功commit预注册与pinned源码，运行executed/run.py --max-new-cells 1，核验恢复后续跑，再运行analysis.py。"}
    execution.save(STUDY / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "saved_cells": len(entries), "predictions": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    analyze()
