#!/usr/bin/env python3
"""Freeze prospective full risk forecasts, then measure separate head fits."""
import argparse
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
C01 = HERE.parent / "C01_activation_scale/executed/run_study.py"
spec = importlib.util.spec_from_file_location("c01_executor", C01)
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)


def make_data(config, seed):
    rng = torch.Generator().manual_seed(41000 + seed)
    xs = []
    for n in [config["data"]["train_rows"], config["data"]["test_rows"]]:
        half = (2 * torch.rand(n // 2, config["architecture"]["input_dim"], generator=rng, dtype=torch.float64) - 1) * 3 ** .5
        xs.append(torch.cat([half, -half]))
    ys = [torch.sin(x[:, 0]) + .7 * x[:, 0] * x[:, 1] + .2 * x[:, 2].square() for x in xs]
    mean, rms = ys[0].mean(), (ys[0] - ys[0].mean()).square().mean().sqrt()
    ys = [(y - mean) / rms for y in ys]
    noise = []
    for i in range(config["data"]["noise_repeats"]):
        noise.append(torch.randn(len(xs[0]), generator=torch.Generator().manual_seed(42000 + seed + i), dtype=torch.float64))
    return xs, ys, torch.stack(noise, 1)


def features(config, cell, xs):
    matrices, info = c01.features(xs, cell["seed"], cell["scale"], cell["activation"], "none", config["architecture"]["width"])
    return [torch.cat([a, torch.ones(len(a), 1, dtype=torch.float64)], 1) for a in matrices], info


def predict(config, cell, matrices, ys):
    x, tx = matrices
    y, ty = ys
    kernel = x @ x.T / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    q = 1 - 2 * config["optimizer"]["lr"] * eigenvalues
    if q.min() < -1e-12 or q.max() > 1 + 1e-12:
        raise ValueError("Forecast requires stable positive spectral factors")
    target = eigenvectors.T @ y
    basis = tx @ x.T / len(x) @ eigenvectors
    records = []
    for step in config["checkpoints"]:
        factors = torch.where(eigenvalues > 1e-12, (1 - q.pow(step)) / eigenvalues, torch.full_like(eigenvalues, 2 * config["optimizer"]["lr"] * step))
        signal_prediction = basis @ (factors * target)
        signal_test = float((signal_prediction - ty).square().mean())
        variance = float(cell["noise_sd"] ** 2 * (basis.square().mean(0) * factors.square()).sum())
        train_signal = float((target.square() * q.pow(2 * step)).sum() / len(x))
        records.append({"step": step, "forecast_train_signal_mse": train_signal, "forecast_test_signal_mse": signal_test,
                        "forecast_noise_variance": variance, "forecast_expected_test_risk": signal_test + variance})
    best = min(records, key=lambda r:r["forecast_expected_test_risk"])
    return {"cell": cell, "records": records, "forecast_best_step": best["step"], "initial_kernel_max_eigenvalue": float(eigenvalues[-1]),
            "saved_before_training": True, "saved_at": time.time()}


def measure(config, cell, matrices, ys, noise):
    x, tx = matrices
    y, ty = ys
    total_targets = y[:, None] + cell["noise_sd"] * noise
    targets = torch.cat([y[:, None], total_targets], 1)
    weights = torch.zeros(x.shape[1], targets.shape[1], dtype=torch.float64)
    curves, records = [], []
    for step in range(config["optimizer"]["steps"] + 1):
        residuals = x @ weights - targets
        noisy_losses = residuals[:, 1:].square().mean(0)
        curves.append(noisy_losses.numpy().tolist())
        if step in config["checkpoints"]:
            test_prediction = tx @ weights
            risk = (test_prediction[:, 1:] - ty[:, None]).square().mean(0)
            records.append({"step": step, "train_signal_mse": float(residuals[:, 0].square().mean()),
                            "test_signal_mse": float((test_prediction[:, 0] - ty).square().mean()),
                            "train_noisy_mse_draws": noisy_losses.tolist(), "test_risk_draws": risk.tolist(),
                            "mean_test_risk": float(risk.mean()), "se_mean_test_risk": float(risk.std(unbiased=True) / risk.numel() ** .5)})
        if step < config["optimizer"]["steps"]:
            weights -= 2 * config["optimizer"]["lr"] * x.T @ residuals / len(x)
    return {"cell": cell, "status": "completed", "finite": bool(np.isfinite(curves).all()), "records": records,
            "max_noisy_train_increase": float(np.diff(curves, axis=0).max())}, {"final_heads": weights.numpy(), "final_test_predictions": (tx @ weights).numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["forecast", "train"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (HERE / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((HERE / "preregistration.json").read_text())
        previous = HERE.parent / "C04_noise_generalization/report.md"
        if c01.sha(previous) != config["development_report_sha256"]:
            raise ValueError("Previous report changed after prediction sealing")
        pins = {"run_study.py": c01.sha(__file__), "preregistration.json": c01.sha(HERE / "preregistration.json"),
                "c01_executor.py": c01.sha(C01), "development_report.md": c01.sha(previous)}
        (HERE / "executed").mkdir(exist_ok=True)
        for name, source in [("run_study.py", Path(__file__)), ("preregistration.json", HERE / "preregistration.json"),
                             ("c01_executor.py", C01), ("development_report.md", previous)]:
            target = HERE / "executed" / name
            if target.exists() and c01.sha(target) != pins[name]:
                raise ValueError("C05 execution changed")
            if not target.exists():
                shutil.copy2(source, target)
        sealpath = HERE / "forecast_seal.json"
        seal = json.loads(sealpath.read_text()) if sealpath.exists() else None
        if args.phase == "train" and seal is None:
            raise ValueError("All predictions must be sealed before first training")
        if args.phase == "forecast" and seal is not None:
            print(json.dumps({"status": "forecasts_already_sealed", "sha256": c01.sha(sealpath)}))
            return
        forecasts, manifests = {}, {}
        count, reused, started = 0, 0, time.monotonic()
        total = len(config["seeds"]) * len(config["scales"]) * len(config["activations"]) * len(config["data"]["noise_sd"])
        for seed in config["seeds"]:
            xs, ys, noise = make_data(config, seed)
            arrays = {"train_x": xs[0].numpy(), "test_x": xs[1].numpy(), "train_y": ys[0].numpy(), "test_y": ys[1].numpy(), "noise_samples": noise.numpy()}
            datapins = {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)} for k, a in arrays.items()}
            datapath = HERE / f"data_seed{seed}.npz"
            if not datapath.exists():
                np.savez_compressed(datapath, **arrays)
            else:
                with np.load(datapath) as saved:
                    if any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
                        raise ValueError("C05 OOD data changed")
            manifests[str(seed)] = {"arrays": datapins, "file_sha256": c01.sha(datapath)}
            for scale in config["scales"]:
                for activation in config["activations"]:
                    for sigma in config["data"]["noise_sd"]:
                        cell = {"seed": seed, "scale": scale, "activation": activation, "noise_sd": sigma}
                        label = f"{seed}_{scale}_{activation}_noise{sigma}"
                        forecastpath = HERE / "forecasts" / f"{label}.json"
                        contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
                        matrices, info = features(config, cell, xs)
                        if args.phase == "forecast":
                            if forecastpath.exists():
                                forecast = json.loads(forecastpath.read_text())
                                if forecast["contract_sha256"] != contract:
                                    raise ValueError("Previously saved forecast contract mismatch")
                                reused += 1
                            else:
                                forecast = predict(config, cell, matrices, ys)
                                forecast.update(contract_sha256=contract, initial_features=info)
                                c01.save(forecastpath, forecast)
                            forecasts[str(forecastpath.relative_to(HERE))] = c01.sha(forecastpath)
                        else:
                            if c01.sha(forecastpath) != seal["forecasts"][str(forecastpath.relative_to(HERE))]:
                                raise ValueError("Sealed forecast changed before training")
                            forecast = json.loads(forecastpath.read_text())
                            if forecast["contract_sha256"] != contract:
                                raise ValueError("Prediction/train contract mismatch")
                            path = HERE / "results" / f"{label}.json"
                            if path.exists():
                                row = json.loads(path.read_text())
                                if row["contract_sha256"] != contract or row["arrays_sha256"] != c01.sha(path.with_suffix(".npz")):
                                    raise ValueError("Saved OOD training mismatch")
                                reused += 1
                            else:
                                t0 = time.monotonic()
                                row, output = measure(config, cell, matrices, ys, noise)
                                path.parent.mkdir(exist_ok=True)
                                np.savez_compressed(path.with_suffix(".npz"), **output)
                                row.update(contract_sha256=contract, arrays_sha256=c01.sha(path.with_suffix(".npz")), forecast_sha256=c01.sha(forecastpath),
                                           forecast_seal_sha256=c01.sha(sealpath), seconds=time.monotonic()-t0)
                                c01.save(path, row)
                        count += 1
                        current = {"status": args.phase + "_complete" if count == total else "running", "phase": args.phase, "pid": os.getpid(), "completed": count, "total": total,
                                   "reused": reused, "seconds": time.monotonic()-started, "updated_at": time.time(), "last_cell": cell, "solver_evaluation": "not_run"}
                        c01.save(HERE / "current.json", current)
                        print(json.dumps(current), flush=True)
        if args.phase == "forecast":
            c01.save(sealpath, {"forecasts": forecasts, "source": pins, "data": manifests, "sealed_at": time.time(), "observed_training_results": 0})
        c01.save(HERE / "source_manifest.json", {"executable": pins, "data": manifests, "forecast_seal_sha256": c01.sha(sealpath), "preregistered": True})


if __name__ == "__main__":
    main()
