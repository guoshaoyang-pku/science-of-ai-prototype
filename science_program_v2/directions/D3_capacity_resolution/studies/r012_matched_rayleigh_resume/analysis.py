#!/usr/bin/env python3
"""Recompute the frozen r004 experiment and label its r012 execution."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-cell", action="store_true")
    args = parser.parse_args()
    config = json.loads((STUDY / "preregistration.json").read_text())
    resume = config["resume_execution"]
    original = ROOT / resume["original_study_path"]
    start = json.loads((STUDY / "executed" / "experiment_start.json").read_text())
    commit = start["preregistration_commit"]
    frozen = {
        original / "preregistration.json": resume["original_preregistration_sha256"],
        original / "executed" / "run.py": config["execution_source_sha256"],
        original / "analysis.py": config["analysis_source_sha256"],
        Path(__file__): config["resume_analysis_source_sha256"],
        STUDY / "preregistration.json": start["resume_preregistration_sha256"],
    }
    for path, expected in frozen.items():
        stored = subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        if sha(path) != expected or stored != path.read_bytes():
            raise RuntimeError(f"Frozen file mismatch: {path}")
    subprocess.run([sys.executable, "-B", str(original / "analysis.py")], cwd=ROOT, check=True)
    result = json.loads((original / "summary.json").read_text())
    source_hashes = result["hashes"]
    if source_hashes["preregistration.json"] != resume["original_preregistration_sha256"]:
        raise RuntimeError("Original analysis used a different contract")
    for metadata in (original / "results").glob("p*_seed*_*.json"):
        if json.loads(metadata.read_text())["pins"]["git_commit"] != commit:
            raise RuntimeError("Cell execution commit differs from the pre-training commit")
    if args.first_cell and result["counts"]["saved_cells"] != 1:
        raise RuntimeError("First-cell checkpoint requires exactly one saved cell")
    if args.first_cell and not result["cells"][0]["execution_audit_pass"]:
        raise RuntimeError("First-cell spectral execution audit failed")
    ranges = []
    for exponent in config["spectral_exponents"]:
        pairs = [pair for pair in result["paired_comparisons"] if pair["exponent"] == exponent]
        if not pairs:
            continue
        row = {"exponent": exponent, "paired_coordinate_seeds": len(pairs)}
        for field in ("middle_step", "endpoints_step", "step_difference", "step_ratio"):
            values = [pair[field] for pair in pairs if pair[field] is not None]
            row[field + "_range"] = [min(values), max(values)] if values else None
        row["max_absolute_descriptor_difference"] = max(
            abs(value) for pair in pairs for value in pair["descriptor_differences"].values()
        )
        ranges.append(row)
    output = {key: value for key, value in result.items() if key not in ("commit_attempts", "next_action")}
    output.update({
        "study": config["study"], "round": config["round"], "direction_round": config["direction_round"],
        "original_design_study": resume["original_study_path"],
        "original_summary_path": (original / "summary.json").relative_to(ROOT).as_posix(),
        "preregistration_commit": commit,
        "resume_preregistration_sha256": start["resume_preregistration_sha256"],
        "resume_analysis_source_sha256": sha(Path(__file__)),
        "checked_at": datetime.now().astimezone().isoformat(),
        "hash_audit": "passed: committed source/contract, request, metadata, NPZ, receipt, and cell execution commit",
        "paired_seed_ranges": ranges,
        "maximum_curve_error": max((cell["max_curve_error"] for cell in result["cells"]), default=None),
        "maximum_saved_prediction_error": max((cell["saved_prediction_error"] for cell in result["cells"]), default=None),
        "counting": config["counting"],
    })
    destination = STUDY / "executed" / "first_cell_audit.json" if args.first_cell else STUDY / "summary.json"
    if args.first_cell and destination.exists():
        raise RuntimeError("First-cell checkpoint already exists; refusing overwrite")
    save(destination, output)
    print(json.dumps({"checkpoint": destination.name, "hash_audit": "passed",
                      "saved_cells": result["counts"]["saved_cells"], "paired_seed_ranges": ranges}, ensure_ascii=False))


if __name__ == "__main__":
    main()
