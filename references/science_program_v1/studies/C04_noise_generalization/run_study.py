#!/usr/bin/env python3
"""Stable fixed-feature GD and explicit bias/noise/test-risk recorder."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c01_executor", HERE.parent / "C01_activation_scale/executed/run_study.py")
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)


def make_data(config, n, seed):
    rng = torch.Generator().manual_seed(34000 + seed + 17 * n)
    xs = []
    for count in [n, config["data"]["test_rows"]]:
        half = torch.randn(count // 2, config["architecture"]["input_dim"], generator=rng, dtype=torch.float64)
        xs.append(torch.cat([half, -half]))
    ys = [x[:, 0] + .5 * x[:, 0] * x[:, 1] for x in xs]
    mean, rms = ys[0].mean(), (ys[0] - ys[0].mean()).square().mean().sqrt()
    ys = [(y - mean) / rms for y in ys]
    epsilon = torch.randn(n, generator=rng, dtype=torch.float64)
    return xs, ys, epsilon


def run(config, cell, xs, ys, eps):
    matrices, info = c01.features(xs, cell["seed"], cell["scale"], cell["activation"], "none", config["architecture"]["width"])
    matrices = [torch.cat([a, torch.ones(len(a), 1, dtype=torch.float64)], 1) for a in matrices]
    x, tx = matrices
    y, ty = ys
    sigma = cell["noise_sd"]
    noise = sigma * eps
    target = y + noise
    kernel = x @ x.T / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    q = 1 - 2 * config["optimizer"]["lr"] * eigenvalues
    if q.abs().max() > 1 + 1e-12:
        raise RuntimeError("Fixed GD is not spectrally stable")
    basis_test = tx @ x.T / len(x) @ eigenvectors
    basis_square = basis_test.square().mean(0)
    weights = torch.zeros(x.shape[1], 3, dtype=torch.float64)
    targets = torch.stack([target, y, noise], 1)
    curves, records = [], []
    checkpoints = set(config["checkpoints"])
    for step in range(config["optimizer"]["steps"] + 1):
        prediction = x @ weights
        residuals = prediction - targets
        loss = float(residuals[:, 0].square().mean())
        curves.append(loss)
        if step in checkpoints:
            test_prediction = tx @ weights
            signal_residual = test_prediction[:, 1] - ty
            noise_prediction = test_prediction[:, 2]
            factors = torch.where(eigenvalues > 1e-12, (1 - q.pow(step)) / eigenvalues, torch.full_like(eigenvalues, 2 * config["optimizer"]["lr"] * step))
            expected_noise = float(sigma ** 2 * (basis_square * factors.square()).sum())
            bias = float(signal_residual.square().mean())
            noise_energy = float(noise_prediction.square().mean())
            cross = float(2 * (signal_residual * noise_prediction).mean())
            clean_risk = float((test_prediction[:, 0] - ty).square().mean())
            record = {"step": step, "train_mse": loss, "train_clean_mse": float((prediction[:, 0] - y).square().mean()),
                      "test_clean_mse": clean_risk, "signal_bias_mse": bias, "realized_noise_energy": noise_energy, "cross_term": cross,
                      "expected_noise_energy": expected_noise, "expected_test_clean_mse": bias + expected_noise,
                      "decomposition_error": abs(clean_risk - (bias + noise_energy + cross)),
                      "head_additivity_error": float((weights[:, 0] - weights[:, 1] - weights[:, 2]).abs().max()),
                      "initial_nominal_U": config["optimizer"]["lr"] * step}
            records.append(record)
        if step < config["optimizer"]["steps"]:
            weights -= 2 * config["optimizer"]["lr"] * x.T @ residuals / len(x)
    arrays = {"eigenvalues": eigenvalues.numpy(), "target_spectral_projection": (eigenvectors.T @ y).numpy(),
              "test_basis_mean_square": basis_square.numpy(), "final_heads": weights.numpy(), "train_clean": y.numpy(),
              "train_noisy": target.numpy(), "test_clean": ty.numpy(), "final_test_predictions": (tx @ weights).numpy()}
    row = {"cell": cell, "status": "completed", "finite": bool(np.isfinite(curves).all()), "features": info, "records": records,
           "train_curve": curves, "max_train_increase": float(np.diff(curves).max()), "stable_multiplier_abs_max": float(q.abs().max())}
    return row, arrays


def main():
    torch.set_num_threads(1)
    with (HERE / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((HERE / "preregistration.json").read_text())
        pins = {"run_study.py": c01.sha(__file__), "preregistration.json": c01.sha(HERE / "preregistration.json"),
                "c01_executor.py": c01.sha(HERE.parent / "C01_activation_scale/executed/run_study.py")}
        (HERE / "executed").mkdir(exist_ok=True)
        for name, source in [("run_study.py", Path(__file__)), ("preregistration.json", HERE / "preregistration.json"),
                             ("c01_executor.py", HERE.parent / "C01_activation_scale/executed/run_study.py")]:
            target = HERE / "executed" / name
            if target.exists() and c01.sha(target) != pins[name]:
                raise ValueError("C04 source changed after measurement")
            if not target.exists():
                shutil.copy2(source, target)
        count, reused, started = 0, 0, time.monotonic()
        total = len(config["data"]["train_rows"]) * len(config["data"]["noise_sd"]) * len(config["seeds"]) * len(config["scales"]) * len(config["activations"])
        manifests = {}
        for n in config["data"]["train_rows"]:
            for seed in config["seeds"]:
                xs, ys, epsilon = make_data(config, n, seed)
                arrays = {"train_x": xs[0].numpy(), "test_x": xs[1].numpy(), "train_y_clean": ys[0].numpy(), "test_y_clean": ys[1].numpy(), "epsilon": epsilon.numpy()}
                datapins = {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)} for k, a in arrays.items()}
                datapath = HERE / f"data_n{n}_seed{seed}.npz"
                if not datapath.exists():
                    np.savez_compressed(datapath, **arrays)
                else:
                    with np.load(datapath) as saved:
                        if any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
                            raise ValueError("C04 saved data changed")
                manifests[f"n{n}_seed{seed}"] = {"arrays": datapins, "file_sha256": c01.sha(datapath)}
                for activation in config["activations"]:
                    for scale in config["scales"]:
                        for sigma in config["data"]["noise_sd"]:
                            cell = {"n": n, "seed": seed, "scale": scale, "activation": activation, "noise_sd": sigma}
                            path = HERE / "results" / f"n{n}_{seed}_{activation}_{scale}_noise{sigma}.json"
                            contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
                            if path.exists():
                                row = json.loads(path.read_text())
                                if row["contract_sha256"] != contract or row["arrays_sha256"] != c01.sha(path.with_suffix(".npz")):
                                    raise ValueError("C04 saved success mismatch")
                                reused += 1
                            else:
                                t0 = time.monotonic()
                                row, output = run(config, cell, xs, ys, epsilon)
                                path.parent.mkdir(exist_ok=True)
                                np.savez_compressed(path.with_suffix(".npz"), **output)
                                row.update(contract_sha256=contract, arrays_sha256=c01.sha(path.with_suffix(".npz")), seconds=time.monotonic() - t0)
                                c01.save(path, row)
                            count += 1
                            current = {"status": "measurements_complete" if count == total else "running", "pid": os.getpid(), "completed": count, "total": total,
                                       "reused": reused, "seconds": time.monotonic()-started, "updated_at": time.time(), "last_cell": cell, "solver_evaluation": "not_run"}
                            c01.save(HERE / "current.json", current)
                            print(json.dumps(current), flush=True)
        c01.save(HERE / "source_manifest.json", {"executable": pins, "data": manifests, "preregistered": True})


if __name__ == "__main__":
    main()
