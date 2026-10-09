#!/usr/bin/env python3
"""Summarize immutable saved threshold evaluations."""
import argparse
import importlib.util
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("saved_endpoint", STUDY / "executed/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def analyze(commit):
    config, pins, names = run.verify(commit)
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    assert receipt["pins"] == pins and set(receipt["cells"]) == names
    rows = []
    for name in sorted(names):
        path = STUDY / f"results/{name}.json"
        assert run.sha(path) == receipt["cells"][name]
        row = json.loads(path.read_text())
        assert row["pins"] == pins
        rows.append(row)
    pairs = []
    for seed in config["seeds"]:
        fast = next(r for r in rows if r["seed"] == seed and r["target"] == "fast_pair")
        slow = next(r for r in rows if r["seed"] == seed and r["target"] == "slow_pair")
        f, s = fast["threshold_step"], slow["threshold_step"]
        valid = f is not None and s is not None
        pairs.append({"seed": seed, "fast_step": f, "slow_step": s,
                      "step_difference": s - f if valid else None, "step_ratio": s / f if valid and f else None,
                      "same_features_bytes": fast["features_bytes_sha256"] == slow["features_bytes_sha256"]})
    complete = all(not r["censored"] for r in rows)
    p1 = complete and all(r["audit_passed"] for r in rows) and all(p["same_features_bytes"] for p in pairs)
    p2 = complete and all(p["step_ratio"] >= config["minimum_step_ratio"] for p in pairs)
    intervals = config["numeric_prediction"]
    p3 = complete and all(intervals["fast_steps"][0] <= p["fast_step"] <= intervals["fast_steps"][1]
                          and intervals["slow_steps"][0] <= p["slow_step"] <= intervals["slow_steps"][1]
                          and intervals["step_ratio"][0] <= p["step_ratio"] <= intervals["step_ratio"][1]
                          for p in pairs)
    summary = {"study": config["study"], "round": 96, "direction_round": 7, "direction": "D3_capacity_resolution",
               "domain": "development", "preregistration_commit": commit,
               "preregistration_sha256": run.sha(STUDY / "preregistration.json"),
               "source_preregistration_sha256": rows[0]["source_preregistration_sha256"],
               "source_scientific_commit": config["original_scientific_commit"],
               "threshold": config["threshold"], "saved_evaluation_cells": len(rows), "saved_source_training_cells": 6,
               "source_training_history": {"r052_old_cells": 1, "r084_new_cells": 5}, "new_training_cells": 0,
               "rows": rows, "pairs": pairs, "predictions": {k: {"status": "supported" if ok else "refuted"}
                   for k, ok in [("P1", p1), ("P2", p2), ("P3", p3)]},
               "ranges": {key: [min(p[key] for p in pairs), max(p[key] for p in pairs)] if complete else None
                          for key in ("fast_step", "slow_step", "step_difference", "step_ratio")},
               "maximum_spectral_curve_error": max(r["spectral_maxabs"] for r in rows),
               "compute_seconds": sum(r["seconds"] for r in rows), "boundaries": config["boundaries"]}
    run.dump(STUDY / "summary.json", summary)
    print(json.dumps({"predictions": summary["predictions"], "ranges": summary["ranges"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-commit", required=True)
    analyze(parser.parse_args().preregistration_commit)
