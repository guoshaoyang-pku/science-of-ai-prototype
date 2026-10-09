#!/usr/bin/env python3
"""Resume the unchanged r052 experiment after verifying both commits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
ORIGINAL = STUDY.parent / "r052_matched_three_moments_norm"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_initial(audit):
    mutable = set(audit["mutable_during_resume"])
    for relative, record in audit["immutable_artifacts"].items():
        if relative in mutable:
            continue
        path = ROOT / relative
        if sha(path) != record["sha256"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"Previous evidence changed: {relative}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-commit", required=True)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    config = json.loads((STUDY / "preregistration.json").read_text())
    original_commit = config["original_scientific_commit"]
    for commit in [original_commit, args.resume_commit]:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    own_pins = {}
    for relative in ["preregistration.json", *config["source_sha256"], *config["audit_sha256"]]:
        path = STUDY / relative
        expected = subprocess.check_output(["git", "show", f"{args.resume_commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        if expected != path.read_bytes():
            raise RuntimeError(f"Resume contract not committed: {relative}")
        own_pins[relative] = sha(path)
    for relative, digest in {**config["source_sha256"], **config["audit_sha256"]}.items():
        if own_pins[relative] != digest:
            raise RuntimeError(f"Resume pin mismatch: {relative}")
    for relative, digest in config["original_pins"]["sha256"].items():
        path = ORIGINAL / relative
        expected = subprocess.check_output(["git", "show", f"{original_commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        if expected != path.read_bytes() or sha(path) != digest:
            raise RuntimeError(f"Original scientific pin mismatch: {relative}")
    audit = json.loads((STUDY / "executed/pre_execution_audit.json").read_text())
    verify_initial(audit)
    receipt_path = ORIGINAL / "results/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["pins"] != config["original_pins"]:
        raise RuntimeError("Mixed original scientific contracts")
    before = {p.name: {"sha256": sha(p), "mtime_ns": p.stat().st_mtime_ns}
              for p in (ORIGINAL / "results").iterdir() if p.name != "receipt.json"}
    start_path = STUDY / "executed/experiment_start.json"
    if start_path.exists():
        if json.loads(start_path.read_text())["resume_commit"] != args.resume_commit:
            raise RuntimeError("Resume commit changed")
    else:
        save(start_path, {"resume_commit": args.resume_commit, "original_scientific_commit": original_commit,
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          "source_and_contract_exact_bytes": True, "resume_pins": own_pins,
                          "saved_cells_before": sorted(receipt["cells"]), "new_training_cells_before": 0,
                          "python": sys.executable, "immutable_results_before": before})
    command = [sys.executable, "-B", str(ORIGINAL / "executed/run.py"),
               "--preregistration-commit", original_commit]
    if args.max_new_cells is not None:
        command += ["--max-new-cells", str(args.max_new_cells)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path = STUDY / "executed/execution_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"commands": []}
    log["commands"].append({"command": command, "returncode": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
    save(log_path, log)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError("Original execution failed; see saved log")
    verify_initial(audit)
    for name, record in before.items():
        path = ORIGINAL / "results" / name
        if sha(path) != record["sha256"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"Successful cell overwritten: {name}")
    after = json.loads(receipt_path.read_text())
    log["commands"][-1].update({"saved_before": len(receipt["cells"]), "saved_after": len(after["cells"]),
                                  "new_cells": len(after["cells"]) - len(receipt["cells"]),
                                  "all_prior_successful_cell_hash_mtime_unchanged": True})
    save(log_path, log)


if __name__ == "__main__":
    main()
