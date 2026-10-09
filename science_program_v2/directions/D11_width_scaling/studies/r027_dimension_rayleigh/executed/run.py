#!/usr/bin/env python3
"""Actual parameter SGD with a signed-permutation fixed feature matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def contract():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_path = STUDY / "preregistration.json"
    analysis_path = STUDY / "analysis.py"
    for path in (config_path, Path(__file__), analysis_path):
        rel = path.relative_to(ROOT).as_posix()
        if subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT) != path.read_bytes():
            raise RuntimeError(f"uncommitted contract: {rel}")
    config = json.loads(config_path.read_text())
    pins = {"preregistration.json": sha(config_path), "executed/run.py": sha(Path(__file__)), "analysis.py": sha(analysis_path)}
    for rel, digest in config["source_sha256"].items():
        if pins[rel] != digest:
            raise RuntimeError("source hash differs from preregistration")
    return config, {"git_commit": commit, "sha256": pins}


def cells(config):
    for dimension in config["dimensions"]:
        for seed in config["coordinate_seeds"]:
            for target in config["targets"]:
                yield {"dimension": dimension, "seed": seed, "target": target}


def label(cell):
    return f"d{cell['dimension']}_seed{cell['seed']}_{cell['target']}"


def request_hash(cell, pins):
    return hashlib.sha256(json.dumps({"cell": cell, "pins": pins}, sort_keys=True).encode()).hexdigest()


def modal_contract(config, dimension, target):
    d = dimension
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-config["spectral_exponent"])
    middle = d // 2 - 1
    weights = np.zeros(d, dtype=np.float64)
    if target == "middle":
        weights[middle] = 1.0
    else:
        weights[0] = (lam[middle] - lam[-1]) / (lam[0] - lam[-1])
        weights[-1] = 1.0 - weights[0]
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("invalid target weights")
    return lam, weights, middle


def measure(config, cell):
    d, target_name = cell["dimension"], cell["target"]
    lam, weights, middle = modal_contract(config, d, target_name)
    eta, threshold = config["optimizer"]["eta"], config["threshold"]
    max_steps = int(config["optimizer"]["max_steps_factor"] * d)
    rng = np.random.default_rng(cell["seed"])
    permutation = rng.permutation(d)
    signs = rng.choice(np.array([-1.0, 1.0]), d)
    inverse = np.argsort(permutation)
    feature_scale = np.sqrt(d * lam)
    target = np.sqrt(d * weights)
    parameter_scale = (feature_scale * signs)[inverse]
    parameter_target = target[inverse]
    theta = np.zeros(d, dtype=np.float64)
    checkpoint_steps, checkpoints, losses = [], [], []
    hit = None
    checkpoint_set = {0, 1, 2, *[2**k for k in range(max_steps.bit_length())]}
    started = time.monotonic()
    for step in range(max_steps + 1):
        residual = parameter_scale * theta - parameter_target
        loss = float(np.sum(residual * residual) / (2 * d))
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
            theta -= eta * parameter_scale * residual / d
        if step % 1024 == 0 and time.monotonic() - started > config["cell_timeout_seconds"]:
            raise RuntimeError("cell time limit reached")
    losses = np.array(losses)
    rayleigh = float(np.dot(weights, lam))
    rayleigh_steps = int(np.ceil(np.log(threshold) / (2.0 * np.log1p(-eta * rayleigh))))
    active = weights > 0
    log_decay = 2.0 * np.log1p(-eta * lam[active])
    def predicted_loss(step):
        return float(np.sum(weights[active] * np.exp(step * log_decay)))
    full_steps = None
    if predicted_loss(max_steps) <= threshold:
        lo, hi = 0, max_steps
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if predicted_loss(mid) <= threshold:
                hi = mid
            else:
                lo = mid
        full_steps = hi
    order = np.argsort(checkpoint_steps)
    arrays = {"eigenvalues": lam, "modal_weights": weights, "loss": losses,
              "permutation": permutation, "signs": signs, "target": target,
              "feature_scale": feature_scale, "parameter_scale": parameter_scale,
              "parameter_target": parameter_target,
              "checkpoint_steps": np.array(checkpoint_steps)[order],
              "theta_checkpoints": np.array(checkpoints)[order]}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("nonfinite result")
    return arrays, {"dimension": d, "seed": cell["seed"], "target": target_name, "middle_index": middle,
                    "rayleigh": rayleigh, "measured_full_step": full_steps,
                    "measured_step": hit, "full_spectrum_step": full_steps, "rayleigh_step": rayleigh_steps,
                    "rayleigh_relative_error": None if hit is None else (rayleigh_steps - hit) / hit,
                    "censored": hit is None, "executed_updates": len(losses) - 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    config, pins = contract()
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0.0, "pins": pins}
    if receipt.get("pins") != pins:
        raise RuntimeError("receipt contract mismatch")
    requested = list(cells(config))
    expected = {label(c) for c in requested}
    if set(receipt["cells"]) - expected:
        raise RuntimeError("receipt contains unregistered cell")
    for cell in requested:
        name = label(cell)
        meta_path, npz_path = results / f"{name}.json", results / f"{name}.npz"
        exists = [meta_path.exists(), npz_path.exists(), name in receipt["cells"]]
        if any(exists):
            if not all(exists):
                raise RuntimeError(f"partial cell; refusing overwrite: {name}")
            meta = json.loads(meta_path.read_text())
            if meta["cell"] != cell or meta["status"] != "completed" or meta["arrays_sha256"] != sha(npz_path) or receipt["cells"][name] != sha(meta_path):
                raise RuntimeError(f"saved cell hash mismatch: {name}")
            if meta["request_sha256"] != request_hash(cell, pins["sha256"]):
                raise RuntimeError(f"saved request mismatch: {name}")
    started, previous, new = time.monotonic(), receipt["compute_seconds"], 0
    for cell in requested:
        name = label(cell)
        if name in receipt["cells"]:
            continue
        if previous + time.monotonic() - started >= config["compute_budget_seconds"]:
            raise RuntimeError("compute budget reached")
        cell_start = time.monotonic()
        arrays, values = measure(config, cell)
        npz_path = results / f"{name}.npz"
        meta_path = results / f"{name}.json"
        np.savez_compressed(npz_path, **arrays)
        meta = {"cell": cell, "status": "completed", "pins": pins,
                "request_sha256": request_hash(cell, pins["sha256"]), "arrays_sha256": sha(npz_path),
                "values": values, "numpy": np.__version__, "seconds": time.monotonic() - cell_start}
        save(meta_path, meta)
        receipt["cells"][name] = sha(meta_path)
        receipt["compute_seconds"] = previous + time.monotonic() - started
        save(receipt_path, receipt)
        new += 1
        print(json.dumps({"saved": name, "completed": len(receipt["cells"])}, ensure_ascii=False), flush=True)
        if args.max_new_cells is not None and new >= args.max_new_cells:
            return


if __name__ == "__main__":
    main()
