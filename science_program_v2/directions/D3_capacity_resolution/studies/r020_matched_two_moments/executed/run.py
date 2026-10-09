#!/usr/bin/env python3
"""Committed paired fixed-feature SGD; successful cells are immutable."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def contract(commit):
    subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT)
    config = json.loads((STUDY / "preregistration.json").read_text())
    paths = ["preregistration.json", *config["source_sha256"], "executed/feasibility.json", "executed/previous_evidence_audit.json"]
    pins = {}
    for relative in paths:
        path = STUDY / relative
        stored = subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        if stored != path.read_bytes():
            raise RuntimeError(f"Uncommitted contract: {relative}")
        pins[relative] = sha(path)
    for relative, expected in config["source_sha256"].items():
        if pins[relative] != expected:
            raise RuntimeError(f"Source pin mismatch: {relative}")
    return config, {"preregistration_commit": commit, "sha256": pins}


def requested_cells(config):
    return [{"seed": seed, "target": target} for seed in config["seeds"] for target in config["targets"]]


def label(cell):
    return f"seed{cell['seed']}_{cell['target']}"


def request_hash(cell, pins):
    return hashlib.sha256(json.dumps({"cell": cell, "pins": pins}, sort_keys=True).encode()).hexdigest()


def verify_saved(path, receipt, cell, pins):
    row = json.loads(path.read_text())
    if row["cell"] != cell or row["status"] != "completed" or row["pins"] != pins:
        raise RuntimeError(f"Invalid saved cell: {path.name}")
    if receipt["cells"].get(label(cell)) != sha(path) or row["arrays_sha256"] != sha(path.with_suffix(".npz")):
        raise RuntimeError(f"Saved hash mismatch: {path.name}")
    if row["request_sha256"] != request_hash(cell, pins):
        raise RuntimeError(f"Saved request mismatch: {path.name}")


def measure(config, cell):
    n = config["data_contract"]["sample_count"]
    eigenvalues = np.arange(1, n + 1, dtype=float) ** -2
    rotation, _ = np.linalg.qr(np.random.default_rng(cell["seed"]).normal(size=(n, n)))
    features = np.sqrt(n * eigenvalues)[:, None] * rotation.T
    weights = np.array([float(Fraction(x)) for x in config["targets"][cell["target"]]])
    target = np.sqrt(n * weights)
    steps, eta = config["optimizer"]["steps"], config["optimizer"]["eta"]
    theta = np.zeros(n)
    theta_history = np.empty((steps + 1, n))
    losses = np.empty(steps + 1)
    for step in range(steps + 1):
        theta_history[step] = theta
        residual = np.einsum("ij,j->i", features, theta) - target
        losses[step] = np.sum(residual ** 2) / (2 * n)
        if step < steps:
            gradient = np.einsum("ij,i->j", features, residual) / n
            theta -= eta * gradient
    arrays = {"features": features, "rotation": rotation, "eigenvalues": eigenvalues,
              "target": target, "modal_weights": weights, "theta_history": theta_history, "loss": losses}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Nonfinite result; refusing completed cell")
    hits = np.flatnonzero(losses / losses[0] <= config["threshold"])
    return arrays, int(hits[0]) if len(hits) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    config, pins = contract(args.preregistration_commit)
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0, "pins": pins}
    requested = requested_cells(config)
    if receipt["pins"] != pins or set(receipt["cells"]) - {label(c) for c in requested}:
        raise RuntimeError("Unregistered receipt")
    for cell in requested:
        name = label(cell)
        path = results / f"{name}.json"
        exists = [path.exists(), path.with_suffix(".npz").exists(), name in receipt["cells"]]
        if any(exists):
            if not all(exists):
                raise RuntimeError(f"Partial cell; refusing overwrite: {name}")
            verify_saved(path, receipt, cell, pins)
    start, previous_seconds, new = time.monotonic(), receipt["compute_seconds"], 0
    for cell in requested:
        name = label(cell)
        if name in receipt["cells"]:
            continue
        if previous_seconds + time.monotonic() - start >= config["compute_budget"]["maximum_seconds"]:
            raise RuntimeError("Compute budget reached")
        cell_start = time.monotonic()
        arrays, hit = measure(config, cell)
        path = results / f"{name}.json"
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        save(path, {"cell": cell, "status": "completed", "pins": pins,
                    "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "request_sha256": request_hash(cell, pins), "arrays_sha256": sha(path.with_suffix(".npz")),
                    "measured_threshold_step": hit, "numpy": np.__version__, "seconds": time.monotonic() - cell_start})
        receipt["cells"][name] = sha(path)
        receipt["compute_seconds"] = previous_seconds + time.monotonic() - start
        save(receipt_path, receipt)
        new += 1
        print(json.dumps({"saved": name, "completed": len(receipt["cells"])}), flush=True)
        if args.max_new_cells is not None and new >= args.max_new_cells:
            return


if __name__ == "__main__":
    main()
