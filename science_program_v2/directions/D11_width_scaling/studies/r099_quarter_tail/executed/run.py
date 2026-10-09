#!/usr/bin/env python3
"""实际参数 SGD；沿用 trace=1 特征，仅改为固定权重的 quarter_tail 目标。"""
import argparse
from datetime import datetime
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    temporary.replace(path)


def contract():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    start_path = STUDY / "executed/experiment_start.json"
    start = json.loads(start_path.read_text()) if start_path.exists() else None
    commits = subprocess.check_output(["git", "log", "--format=%H", "--", str((STUDY / "preregistration.json").relative_to(ROOT))], cwd=ROOT, text=True).splitlines()
    if len(commits) != 1:
        raise RuntimeError("preregistration must have one unique freeze commit")
    commit = start["git_commit"] if start else commits[0]
    if commits != [commit]:
        raise RuntimeError("execution must bind the unique freeze commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT, check=True)
    paths = ("preregistration.json", "executed/run.py", "analysis.py", "executed/verify.py", "executed/baseline_manifest.json")
    for rel in paths:
        path = STUDY / rel
        if subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT) != path.read_bytes():
            raise RuntimeError(f"uncommitted contract: {rel}")
    config = json.loads((STUDY / "preregistration.json").read_text())
    manifest_path = STUDY / "executed/baseline_manifest.json"
    if sha(manifest_path) != config["baseline_manifest_sha256"]:
        raise RuntimeError("baseline manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    for rel, info in manifest.items():
        path = ROOT / rel
        if sha(path) != info["sha256"] or path.stat().st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"baseline hash/mtime mismatch: {rel}")
    pins = {rel: sha(STUDY / rel) for rel in paths[1:4]}
    if any(pins[rel] != digest for rel, digest in config["source_sha256"].items()):
        raise RuntimeError("source hash mismatch")
    if start:
        if start["sha256"] != pins:
            raise RuntimeError("execution start source mismatch")
    else:
        if any((STUDY / "results").iterdir()):
            raise RuntimeError("first execution requires empty results")
        save(start_path, {"git_commit": commit, "sha256": pins, "started_at": datetime.now().astimezone().isoformat(),
                          "empty_results": True, "baseline_hash_mtime_verified": len(manifest),
                          "all_contract_bytes_match_commit": True})
    return config, {"git_commit": commit, "sha256": pins, "preregistration_sha256": sha(STUDY / "preregistration.json")}


def label(cell):
    return f"d{cell['dimension']}_seed{cell['seed']}_{cell['target']}"


def request_hash(cell, pins):
    return hashlib.sha256(json.dumps({"cell": cell, "pins": pins}, sort_keys=True).encode()).hexdigest()


def measure(config, cell):
    d, n = cell["dimension"], config["sample_count"]
    baseline = ROOT / config["baseline_study"] / "results" / f"d{d}_seed{cell['seed']}_middle.npz"
    with np.load(baseline) as old:
        lam = old["eigenvalues"]
        permutation, signs = old["permutation"], old["signs"]
    harmonic = float(np.sum(1.0 / np.arange(1, d + 1)))
    weights = np.zeros(d, dtype=np.float64)
    weights[d // 4 - 1], weights[d - 1] = 1.0 / 3.0, 2.0 / 3.0
    target = np.sqrt(n * weights)
    inverse = np.argsort(permutation)
    feature_scale = np.sqrt(n * lam)
    parameter_scale = (feature_scale * signs)[inverse]
    parameter_target = target[inverse]
    theta = np.zeros(d, dtype=np.float64)
    eta, threshold = config["optimizer"]["eta"], config["threshold"]
    max_steps = int(np.ceil(config["optimizer"]["max_steps_factor"] * d * harmonic))
    checkpoint_set = {0, 1, 2, *[2**k for k in range(max_steps.bit_length())]}
    checkpoint_steps, checkpoints, losses = [], [], []
    hit = None
    started = time.monotonic()
    for step in range(max_steps + 1):
        residual = parameter_scale * theta - parameter_target
        loss = float(np.sum(residual * residual) / (2 * n))
        losses.append(loss)
        if step in checkpoint_set:
            checkpoint_steps.append(step)
            checkpoints.append(theta.copy())
        if hit is None and loss / losses[0] <= threshold:
            hit = step
            if step - 1 not in checkpoint_steps:
                checkpoint_steps.append(step - 1)
                checkpoints.append(previous_theta.copy())
            if step not in checkpoint_steps:
                checkpoint_steps.append(step)
                checkpoints.append(theta.copy())
        if hit is not None and step == hit + 1:
            if step not in checkpoint_steps:
                checkpoint_steps.append(step)
                checkpoints.append(theta.copy())
            break
        if step < max_steps:
            previous_theta = theta.copy()
            theta -= eta * parameter_scale * residual / n
        if step % 1024 == 0 and time.monotonic() - started > config["cell_timeout_seconds"]:
            raise RuntimeError("cell time limit reached")
    order = np.argsort(checkpoint_steps)
    arrays = {"eigenvalues": lam, "modal_weights": weights, "loss": np.array(losses),
              "permutation": permutation, "signs": signs, "target": target,
              "feature_scale": feature_scale, "parameter_scale": parameter_scale,
              "parameter_target": parameter_target, "checkpoint_steps": np.array(checkpoint_steps)[order],
              "theta_checkpoints": np.array(checkpoints)[order]}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("nonfinite result")
    rayleigh = float(np.sum(weights * lam))
    return arrays, {"measured_step": hit, "censored": hit is None, "dimension": d, "sample_count": n,
                    "nonzero_sample_count": d, "harmonic_number": harmonic, "trace": float(np.sum(lam)),
                    "rayleigh": rayleigh, "executed_updates": len(losses) - 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    config, pins = contract()
    results = STUDY / "results"
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0.0, "pins": pins}
    if receipt["pins"] != pins:
        raise RuntimeError("receipt contract mismatch")
    requested = [{"dimension": d, "seed": seed, "target": target} for d in config["dimensions"]
                 for seed in config["coordinate_seeds"] for target in config["targets"]]
    expected = {label(cell) for cell in requested}
    if set(receipt["cells"]) - expected:
        raise RuntimeError("unregistered saved cell")
    allowed = {"receipt.json"} | {f"{name}.{suffix}" for name in expected for suffix in ("json", "npz")}
    if {path.name for path in results.iterdir()} - allowed:
        raise RuntimeError("unexpected result files")
    for cell in requested:
        name = label(cell)
        meta_path, npz_path = results / f"{name}.json", results / f"{name}.npz"
        exists = [meta_path.exists(), npz_path.exists(), name in receipt["cells"]]
        if any(exists):
            if not all(exists):
                raise RuntimeError(f"partial cell; refusing overwrite: {name}")
            meta = json.loads(meta_path.read_text())
            if (meta["cell"] != cell or meta["status"] != "completed" or meta["pins"] != pins or
                    meta["arrays_sha256"] != sha(npz_path) or receipt["cells"][name] != sha(meta_path) or
                    meta["request_sha256"] != request_hash(cell, pins["sha256"])):
                raise RuntimeError(f"saved cell contract/hash mismatch: {name}")
    started, previous, new = time.monotonic(), receipt["compute_seconds"], 0
    for cell in requested:
        name = label(cell)
        if name in receipt["cells"]:
            continue
        if previous + time.monotonic() - started >= config["compute_budget_seconds"]:
            raise RuntimeError("compute budget reached")
        cell_start = time.monotonic()
        started_at = datetime.now().astimezone().isoformat()
        arrays, values = measure(config, cell)
        npz_path, meta_path = results / f"{name}.npz", results / f"{name}.json"
        np.savez_compressed(npz_path, **arrays)
        save(meta_path, {"cell": cell, "status": "completed", "pins": pins,
                         "started_at": started_at,
                         "request_sha256": request_hash(cell, pins["sha256"]), "arrays_sha256": sha(npz_path),
                         "values": values, "numpy": np.__version__, "seconds": time.monotonic() - cell_start})
        receipt["cells"][name] = sha(meta_path)
        receipt["compute_seconds"] = previous + time.monotonic() - started
        save(receipt_path, receipt)
        new += 1
        print(json.dumps({"saved": name, "completed": len(receipt["cells"])}), flush=True)
        if args.max_new_cells is not None and new >= args.max_new_cells:
            return
    print(json.dumps({"new_cells": new, "saved_cells": len(receipt["cells"])}))


if __name__ == "__main__":
    main()
