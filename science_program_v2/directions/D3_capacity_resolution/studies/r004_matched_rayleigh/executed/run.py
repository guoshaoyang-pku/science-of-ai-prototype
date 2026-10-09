#!/usr/bin/env python3
"""Run committed, paired fixed-feature SGD cells on CPU."""
import argparse
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


def committed_contract():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = [STUDY / "preregistration.json", Path(__file__), STUDY / "analysis.py"]
    pins = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        stored = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if stored != path.read_bytes():
            raise RuntimeError(f"Uncommitted contract: {relative}")
        pins[path.relative_to(STUDY).as_posix()] = sha(path)
    config = json.loads(paths[0].read_text())
    if config["execution_source_sha256"] != sha(__file__) or config["analysis_source_sha256"] != sha(paths[2]):
        raise RuntimeError("Source differs from preregistration pin")
    return config, {"git_commit": commit, "sha256": pins}


def cells(config):
    for exponent in config["spectral_exponents"]:
        for seed in config["seeds"]:
            for target in config["targets"]:
                yield {"exponent": exponent, "seed": seed, "target": target}


def label(cell):
    return f"p{cell['exponent']}_seed{cell['seed']}_{cell['target']}"


def request_hash(cell, pins):
    return hashlib.sha256(json.dumps({"cell": cell, "pins": pins}, sort_keys=True).encode()).hexdigest()


def verify_saved(path, receipt, cell, pins):
    name = label(cell)
    row = json.loads(path.read_text())
    if row["cell"] != cell or row["status"] != "completed":
        raise RuntimeError(f"Invalid saved cell: {name}")
    if receipt["cells"].get(name) != sha(path) or row["arrays_sha256"] != sha(path.with_suffix(".npz")):
        raise RuntimeError(f"Saved result hash mismatch: {name}")
    if row["pins"]["sha256"] != pins or row["request_sha256"] != request_hash(cell, pins):
        raise RuntimeError(f"Saved contract hash mismatch: {name}")
    return row


def measure(config, cell):
    n = config["data_contract"]["dimension"]
    eigenvalues = np.arange(1, n + 1, dtype=float) ** (-cell["exponent"])
    rotation, _ = np.linalg.qr(np.random.default_rng(cell["seed"]).normal(size=(n, n)))
    features = np.sqrt(n) * np.diag(np.sqrt(eigenvalues)) @ rotation.T
    weights = np.zeros(n)
    if cell["target"] == "middle":
        weights[2] = 1
    else:
        weights[0] = .5 if cell["exponent"] == 0 else (eigenvalues[2] - eigenvalues[-1]) / (eigenvalues[0] - eigenvalues[-1])
        weights[-1] = 1 - weights[0]
    target = np.sqrt(n * weights)
    steps, eta = config["optimizer"]["steps"], config["optimizer"]["eta"]
    predicted_curve = (weights[:, None] * (1 - eta * eigenvalues[:, None]) ** (2 * np.arange(steps + 1))).sum(axis=0)
    theta = np.zeros(n)
    normalized_loss = np.empty(steps + 1)
    for step in range(steps + 1):
        residual = features @ theta - target
        normalized_loss[step] = residual @ residual / (target @ target)
        if step < steps:
            theta -= eta * features.T @ residual / n
    arrays = {"features": features, "rotation": rotation, "eigenvalues": eigenvalues,
              "target": target, "modal_weights": weights, "initial_theta": np.zeros(n),
              "final_theta": theta, "normalized_loss": normalized_loss, "predicted_curve": predicted_curve}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Nonfinite result; no completed cell saved")
    hits = np.flatnonzero(predicted_curve <= config["threshold"])
    return arrays, int(hits[0]) if len(hits) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    config, pins = committed_contract()
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0}
    requested = list(cells(config))
    expected_names = {label(cell) for cell in requested}
    if set(receipt["cells"]) - expected_names:
        raise RuntimeError("Receipt contains an unregistered cell")
    for cell in requested:
        name = label(cell)
        path = results / f"{name}.json"
        exists = [path.exists(), path.with_suffix(".npz").exists(), name in receipt["cells"]]
        if any(exists):
            if not all(exists):
                raise RuntimeError(f"Partial saved cell; refusing overwrite: {name}")
            verify_saved(path, receipt, cell, pins["sha256"])
    started, previous_seconds, new = time.monotonic(), receipt["compute_seconds"], 0
    for cell in requested:
        name = label(cell)
        if name in receipt["cells"]:
            continue
        if previous_seconds + time.monotonic() - started >= config["compute_budget"]["maximum_seconds"]:
            raise RuntimeError("Cumulative compute budget reached")
        cell_started = time.monotonic()
        arrays, prediction = measure(config, cell)
        path = results / f"{name}.json"
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        save(path, {"cell": cell, "status": "completed", "pins": pins,
                    "request_sha256": request_hash(cell, pins["sha256"]),
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                    "predicted_threshold_step": prediction, "numpy": np.__version__,
                    "seconds": time.monotonic() - cell_started})
        receipt["cells"][name] = sha(path)
        receipt["compute_seconds"] = previous_seconds + time.monotonic() - started
        save(receipt_path, receipt)
        new += 1
        print(json.dumps({"saved": name, "completed": len(receipt["cells"])}), flush=True)
        if args.max_new_cells is not None and new >= args.max_new_cells:
            return


if __name__ == "__main__":
    main()
