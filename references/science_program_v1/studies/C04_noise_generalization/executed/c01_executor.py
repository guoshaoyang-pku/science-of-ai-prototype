#!/usr/bin/env python3
"""Fixed-feature, target-specific activation study with resumable paired cells."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write(chr(10))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def dataset(spec):
    rng = torch.Generator().manual_seed(spec["data"]["dataset_seed"])
    dimension = spec["architecture"]["input_dim"]
    xs = []
    for count in (spec["data"]["train_rows"], spec["data"]["test_rows"]):
        half = torch.randn(count // 2, dimension, generator=rng, dtype=torch.float64)
        xs.append(torch.cat([half, -half]))
    targets = {}
    for name in spec["data"]["targets"]:
        values = [x[:, 0] if name == "linear" else x[:, 0].square() for x in xs]
        mean, rms = values[0].mean(), (values[0] - values[0].mean()).square().mean().sqrt()
        targets[name] = [(y - mean) / rms for y in values]
    return xs, targets


def features(xs, seed, scale, activation, control, width):
    rng = torch.Generator().manual_seed(seed)
    weight = torch.randn(width, xs[0].shape[1], generator=rng, dtype=torch.float64)
    weight /= xs[0].shape[1] ** .5
    zs = [scale * x @ weight.T for x in xs]
    if control == "preactivation_ln":
        zs = [(z - z.mean(1, keepdim=True)) / (z.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt() for z in zs]
    act = torch.relu if activation == "relu" else torch.nn.functional.silu
    raw = [act(z) for z in zs]
    opposite = [act(-z) for z in zs]
    if control == "linear_skip":
        raw = [torch.cat([a, z / 2], 1) for a, z in zip(raw, zs)]
        opposite = [torch.cat([a, -z / 2], 1) for a, z in zip(opposite, zs)]
    center = raw[0].mean(0)
    norm = (raw[0] - center).square().mean().sqrt()
    matrix = [(a - center) / norm / a.shape[1] ** .5 for a in raw]
    odd = (raw[0] - opposite[0]) / 2
    even = (raw[0] + opposite[0]) / 2 - center
    if activation == "relu":
        derivative = (zs[0] > 0).double()
    else:
        sigmoid = torch.sigmoid(zs[0])
        derivative = sigmoid * (1 + zs[0] * (1 - sigmoid))
    info = {"preactivation_rms": float(zs[0].square().mean().sqrt()),
            "raw_feature_rms": float(norm), "normalized_row_energy": float(matrix[0].square().sum(1).mean()),
            "derivative_square_mean": float(derivative.square().mean()),
            "even_energy": float(even.square().mean()), "odd_energy": float(odd.square().mean()),
            "even_odd_energy_ratio": float(even.square().mean() / odd.square().mean())}
    return matrix, info


def run_cell(spec, cell, matrices, ys, info):
    x, tx = matrices
    y, ty = ys
    kernel = x @ x.T / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    coefficients = eigenvectors.T @ y
    lr = spec["optimizer"]["lr"]
    steps = spec["optimizer"]["steps"]
    weights = torch.zeros(x.shape[1], dtype=torch.float64)
    checkpoints = set(spec["checkpoints"])
    curve, records = [], []
    predicted = []
    for step in range(steps + 1):
        residual = x @ weights - y
        loss = float(residual.square().mean())
        curve.append(loss)
        spectral_loss = float((coefficients.square() * (1 - 2 * lr * eigenvalues).pow(2 * step)).sum() / len(x))
        predicted.append(spectral_loss)
        if step in checkpoints:
            update = -2 * lr * kernel @ residual
            test_loss = float((tx @ weights - ty).square().mean())
            records.append({"step": step, "train_mse": loss, "test_mse": test_loss,
                            "function_update_rms": float(update.square().mean().sqrt()),
                            "head_norm": float(weights.norm())})
        if step < steps:
            weights -= 2 * lr * (x.T @ residual) / len(x)
    row = {"cell": cell, "features": info, "records": records,
           "initial_target_kernel_energy": float(y @ kernel @ y / (y @ y)),
           "initial_progress": curve[0] - curve[1],
           "kernel_max_eigenvalue": float(eigenvalues[-1]),
           "train_curve": curve, "spectral_train_curve": predicted,
           "spectral_max_error": float(np.max(np.abs(np.array(curve) - predicted))),
           "status": "completed", "finite": bool(np.isfinite(curve).all())}
    if not row["finite"]:
        raise RuntimeError("Nonfinite fixed-feature run")
    return row, {"train_features": x.numpy(), "test_features": tx.numpy(),
                 "train_target": y.numpy(), "test_target": ty.numpy(),
                 "final_head": weights.numpy(), "eigenvalues": eigenvalues.numpy(),
                 "target_spectral_coefficients": coefficients.numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many newly saved cells for recovery audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not json.loads((ROOT / "state/current.json").read_text()).get("first_chain_complete"):
        raise RuntimeError("Coordinator has not saved first_chain_complete")
    with (HERE / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pins = {"run_study.py": sha(__file__), "preregistration.json": sha(HERE / "preregistration.json")}
        directory = HERE / "executed"
        directory.mkdir(exist_ok=True)
        for name in pins:
            frozen = directory / name
            if frozen.exists() and sha(frozen) != pins[name]:
                raise ValueError("Execution changed after source was frozen")
            if not frozen.exists():
                shutil.copy2(HERE / name, frozen)
        spec = json.loads((HERE / "preregistration.json").read_text())
        xs, targets = dataset(spec)
        data_arrays = {"train_x": xs[0].numpy(), "test_x": xs[1].numpy()}
        for name, ys in targets.items():
            data_arrays.update({f"train_y_{name}": ys[0].numpy(), f"test_y_{name}": ys[1].numpy()})
        datapins = {name: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)} for name, a in data_arrays.items()}
        path = HERE / "data.npz"
        if not path.exists():
            np.savez_compressed(path, **data_arrays)
        else:
            with np.load(path) as saved:
                if any(not np.array_equal(saved[name], a) for name, a in data_arrays.items()):
                    raise ValueError("Saved synthetic data changed")
        save(HERE / "source_manifest.json", {"executable": pins, "data": datapins, "data_file_sha256": sha(path), "preregistered": True})
        count = len(spec["seeds"]) * len(spec["scales"]) * len(spec["activations"]) * len(spec["controls"]) * len(spec["data"]["targets"])
        started = time.monotonic()
        complete, reused, new = 0, 0, 0
        for seed in spec["seeds"]:
            for scale in spec["scales"]:
                for activation in spec["activations"]:
                    for control in spec["controls"]:
                        matrices, info = features(xs, seed, scale, activation, control, spec["architecture"]["width"])
                        for target, ys in targets.items():
                            cell = {"seed": seed, "scale": scale, "activation": activation, "control": control, "target": target}
                            label = f"{seed}_{scale}_{activation}_{control}_{target}"
                            path = HERE / "results" / f"{label}.json"
                            contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
                            if path.exists():
                                row = json.loads(path.read_text())
                                if row["contract_sha256"] != contract or sha(path.with_suffix(".npz")) != row["arrays_sha256"]:
                                    raise ValueError("Saved success contract or arrays mismatch")
                                reused += 1
                            else:
                                cell_started = time.monotonic()
                                row, arrays = run_cell(spec, cell, matrices, ys, info)
                                path.parent.mkdir(exist_ok=True)
                                np.savez_compressed(path.with_suffix(".npz"), **arrays)
                                row.update(contract_sha256=contract, arrays_sha256=sha(path.with_suffix(".npz")),
                                           seconds=time.monotonic() - cell_started)
                                save(path, row)
                                new += 1
                            complete += 1
                            state = {"status": "measurements_complete" if complete == count else "running", "pid": os.getpid(),
                                     "completed": complete, "total": count, "reused": reused, "new": new, "last_cell": cell,
                                     "seconds": time.monotonic() - started, "updated_at": time.time(), "solver_evaluation": "not_run"}
                            save(HERE / "current.json", state)
                            print(json.dumps(state), flush=True)
                            if args.limit is not None and new >= args.limit:
                                state["status"] = "checkpoint_stop"
                                save(HERE / "current.json", state)
                                return


if __name__ == "__main__":
    main()
