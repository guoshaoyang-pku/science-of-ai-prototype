#!/usr/bin/env python3
"""Execute committed C04-style fixed-feature cells on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def committed_contract():
    paths = [STUDY / "preregistration.json", Path(__file__), STUDY / "analysis.py"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pins = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        stored = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if stored != path.read_bytes():
            raise RuntimeError(f"Uncommitted execution contract: {relative}")
        pins[path.relative_to(STUDY).as_posix()] = sha(path)
    config = json.loads(paths[0].read_text())
    if config["execution_source_sha256"] != sha(__file__):
        raise RuntimeError("Execution source differs from preregistration pin")
    if config["analysis_source_sha256"] != sha(STUDY / "analysis.py"):
        raise RuntimeError("Analysis source differs from preregistration pin")
    return config, {"git_commit": commit, "sha256": pins}


def make_data(config, n, seed):
    rng = torch.Generator().manual_seed(34000 + seed + 17 * n)
    xs = []
    for count in (n, config["data"]["risk_audit_rows"]):
        half = torch.randn(count // 2, 4, generator=rng, dtype=torch.float64)
        xs.append(torch.cat([half, -half]))
    ys = [x[:, 0] + .5 * x[:, 0] * x[:, 1] for x in xs]
    mean = ys[0].mean()
    rms = (ys[0] - mean).square().mean().sqrt()
    ys = [(y - mean) / rms for y in ys]
    epsilon = torch.randn(n, generator=rng, dtype=torch.float64)
    weight = torch.randn(128, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) / 2
    raw = [torch.relu(.03 * x @ weight.T) for x in xs]
    center = raw[0].mean(0)
    norm = (raw[0] - center).square().mean().sqrt()
    features = [(a - center) / norm / 128 ** .5 for a in raw]
    features = [torch.cat([a, torch.ones(len(a), 1, dtype=torch.float64)], 1) for a in features]
    return {"train_x": xs[0].numpy(), "audit_x": xs[1].numpy(),
            "train_y": ys[0].numpy(), "audit_y": ys[1].numpy(),
            "epsilon": epsilon.numpy(), "hidden_weight": weight.numpy(),
            "train_features": features[0].numpy(), "audit_features": features[1].numpy(),
            "target_mean_rms": np.array([float(mean), float(rms)])}


def saved_data(config, n, seed):
    arrays = make_data(config, n, seed)
    path = STUDY / "executed" / f"data_n{n}_seed{seed}.npz"
    manifest = path.with_suffix(".json")
    array_pins = {k: {"sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                      "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
    if path.exists() or manifest.exists():
        if not path.exists() or not manifest.exists():
            raise RuntimeError("Partial saved data; refusing replacement")
        old = json.loads(manifest.read_text())
        if old["file_sha256"] != sha(path) or old["arrays"] != array_pins:
            raise RuntimeError("Saved data contract mismatch")
        with np.load(path) as saved:
            if any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise RuntimeError("Saved data arrays differ")
    else:
        np.savez_compressed(path, **arrays)
        save(manifest, {"n": n, "seed": seed, "arrays": array_pins, "file_sha256": sha(path)})
    return arrays, path, sha(path)


def measure(config, data, variance):
    x, tx = data["train_features"], data["audit_features"]
    y, ty, eps = data["train_y"], data["audit_y"], data["epsilon"]
    n, lr, steps = len(x), config["optimizer"]["lr"], config["optimizer"]["steps"]
    eigenvalues, eigenvectors = np.linalg.eigh(x @ x.T / n)
    q = 1 - 2 * lr * eigenvalues
    if q.min() < -1e-12 or q.max() > 1 + 1e-12:
        raise RuntimeError("Registered positive stable GD factors violated")
    audit_gram = tx.T @ tx / len(tx)
    audit_target_projection = tx.T @ ty / len(tx)
    audit_target_square = float(ty @ ty / len(tx))
    spectral_map = x.T @ eigenvectors / n
    basis_gram = spectral_map.T @ audit_gram @ spectral_map
    basis_target = audit_target_projection @ spectral_map
    signal_projection = eigenvectors.T @ y
    targets = np.column_stack([y + variance ** .5 * eps, y, variance ** .5 * eps])
    weights = np.zeros((x.shape[1], 3))
    train_mse = np.empty(steps + 1)
    signal_bias = np.empty(steps + 1)
    variance_unit = np.empty(steps + 1)
    spectral_bias = np.empty(steps + 1)
    factors = np.zeros(n)
    for step in range(steps + 1):
        residual = x @ weights - targets
        train_mse[step] = np.mean(residual[:, 0] ** 2)
        w = weights[:, 1]
        signal_bias[step] = w @ audit_gram @ w - 2 * audit_target_projection @ w + audit_target_square
        coefficients = factors * signal_projection
        spectral_bias[step] = coefficients @ basis_gram @ coefficients - 2 * basis_target @ coefficients + audit_target_square
        variance_unit[step] = np.diag(basis_gram) @ (factors ** 2)
        if step < steps:
            weights -= 2 * lr * x.T @ residual / n
            factors = q * factors + 2 * lr
    direct_final_bias = float(np.mean((tx @ weights[:, 1] - ty) ** 2))
    risk = signal_bias + variance * variance_unit
    arrays = {"train_mse": train_mse, "signal_bias": signal_bias,
              "variance_unit": variance_unit, "expected_risk": risk,
              "eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
              "basis_gram": basis_gram, "basis_target": basis_target,
              "signal_projection": signal_projection, "audit_gram": audit_gram,
              "audit_target_projection": audit_target_projection,
              "audit_target_square": np.array(audit_target_square), "final_heads": weights}
    audit = {"max_train_increase": float(np.diff(train_mse).max()),
             "max_spectral_signal_error": float(np.max(np.abs(signal_bias - spectral_bias))),
             "final_direct_bias_error": abs(direct_final_bias - signal_bias[-1]),
             "direct_final_bias": direct_final_bias, "q_min": float(q.min()), "q_max": float(q.max())}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Nonfinite cell; no completed result saved")
    return arrays, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells < 1:
        parser.error("--max-new-cells must be positive")
    config, pins = committed_contract()
    torch.set_num_threads(1)
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    receipt_path = results / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0}
    started, previous_seconds, new = time.monotonic(), receipt["compute_seconds"], 0
    save(STUDY / "executed" / "source_manifest.json", {**pins, "numpy": np.__version__, "torch": torch.__version__})
    for n in config["data"]["train_rows"]:
        for seed in config["seeds"]:
            data, data_path, data_sha = saved_data(config, n, seed)
            for variance in config["data"]["noise_variance"]:
                cell = {"n": n, "seed": seed, "noise_variance": variance}
                label = f"n{n}_seed{seed}_var{variance:g}"
                path = results / f"{label}.json"
                contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins["sha256"],
                                                     "data_sha256": data_sha}, sort_keys=True).encode()).hexdigest()
                if path.exists() or path.with_suffix(".npz").exists() or label in receipt["cells"]:
                    if not path.exists() or not path.with_suffix(".npz").exists() or label not in receipt["cells"]:
                        raise RuntimeError(f"Partial cell {label}; refusing overwrite")
                    row = json.loads(path.read_text())
                    if sha(path) != receipt["cells"][label] or row["contract_sha256"] != contract or row["arrays_sha256"] != sha(path.with_suffix(".npz")):
                        raise RuntimeError(f"Saved cell hash mismatch: {label}")
                    continue
                if previous_seconds + time.monotonic() - started >= config["compute_budget"]["maximum_seconds"]:
                    raise RuntimeError("Cumulative compute budget reached")
                cell_started = time.monotonic()
                arrays, audit = measure(config, data, variance)
                np.savez_compressed(path.with_suffix(".npz"), **arrays)
                save(path, {"cell": cell, "status": "completed", "finite": True, "audit": audit,
                            "contract_sha256": contract, "pins": pins,
                            "data_file": data_path.relative_to(STUDY).as_posix(), "data_sha256": data_sha,
                            "arrays_sha256": sha(path.with_suffix(".npz")),
                            "seconds": time.monotonic() - cell_started})
                receipt["cells"][label] = sha(path)
                receipt["compute_seconds"] = previous_seconds + time.monotonic() - started
                save(receipt_path, receipt)
                new += 1
                print(json.dumps({"saved": label, "completed": len(receipt["cells"]),
                                  "seconds": receipt["compute_seconds"]}), flush=True)
                if args.max_new_cells is not None and new >= args.max_new_cells:
                    return


if __name__ == "__main__":
    main()
