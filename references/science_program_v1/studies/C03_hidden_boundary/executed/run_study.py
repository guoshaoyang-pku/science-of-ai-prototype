#!/usr/bin/env python3
"""Actual SGD of head and optional hidden weights with activation diagnostics."""
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
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("c01_executor", HERE.parent / "C01_activation_scale/executed/run_study.py")
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)


def make_data(config, distribution):
    rng = torch.Generator().manual_seed(config["data"]["dataset_seeds"][distribution])
    xs = []
    for count in [config["data"]["train_rows"], config["data"]["test_rows"]]:
        half = torch.randn(count // 2, config["architecture"]["input_dim"], generator=rng, dtype=torch.float64)
        x = torch.cat([half, -half])
        if distribution == "shifted_gaussian":
            x += config["data"]["shift"]
        xs.append(x)
    targets = {}
    for name in config["data"]["targets"]:
        ys = [x[:, 0] if name == "linear" else x[:, 0] * x[:, 1] for x in xs]
        center, rms = ys[0].mean(), (ys[0] - ys[0].mean()).square().mean().sqrt()
        targets[name] = [(y - center) / rms for y in ys]
    return xs, targets


class Model(nn.Module):
    def __init__(self, config, cell, train_x):
        super().__init__()
        width = config["architecture"]["width"]
        self.dimension, self.width = train_x.shape[1], width
        self.scale, self.activation = cell["scale"], cell["activation"]
        rng = torch.Generator().manual_seed(cell["seed"])
        weight = torch.randn(width, self.dimension, generator=rng, dtype=torch.float64)
        self.weight = nn.Parameter(weight, requires_grad=cell["hidden_mode"] == "learned")
        self.head = nn.Parameter(torch.zeros(width, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        with torch.no_grad():
            raw = self.raw(train_x)
            self.register_buffer("center", raw.mean(0))
            self.register_buffer("rms", (raw - self.center).square().mean().sqrt())

    def raw(self, x):
        z = self.scale * x @ self.weight.T / self.dimension ** .5
        return torch.relu(z) if self.activation == "relu" else torch.nn.functional.silu(z)

    def features(self, x):
        return (self.raw(x) - self.center) / self.rms / self.width ** .5

    def forward(self, x):
        return self.features(x) @ self.head + self.bias


def train(config, cell, xs, ys):
    x, tx = xs
    y, ty = ys
    model = Model(config, cell, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["optimizer"]["lr"])
    initial_weight = model.weight.detach().clone()
    with torch.no_grad():
        initial_phi = torch.cat([model.features(x), torch.ones(len(x), 1, dtype=torch.float64)], 1)
        initial_kernel = initial_phi @ initial_phi.T / len(x)
        eigenvalues, eigenvectors = torch.linalg.eigh(initial_kernel)
        coefficients = eigenvectors.T @ y
    curves, records = [], []
    steps, lr = config["optimizer"]["steps"], config["optimizer"]["lr"]
    initial_prediction = []
    arrays = {"initial_train_features": initial_phi.numpy()}
    for step in range(steps + 1):
        prediction = model(x)
        residual = prediction - y
        loss = residual.square().mean()
        curves.append(float(loss.detach()))
        initial_prediction.append(float((coefficients.square() * (1 - 2 * lr * eigenvalues).pow(2 * step)).sum() / len(x)))
        if step in config["checkpoints"]:
            with torch.no_grad():
                feature = model.features(x)
                raw, opposite = model.raw(x), model.raw(-x)
                odd = (raw - opposite) / 2
                even = (raw + opposite) / 2
                even = even - even.mean(0)
                residual_alignment = float((feature.T @ residual).square().sum() / (len(x) * residual.square().sum()).clamp_min(1e-30))
                target_alignment = float((feature.T @ y).square().sum() / (len(x) * y.square().sum()))
                records.append({"step": step, "train_mse": curves[-1], "test_mse": float((model(tx) - ty).square().mean()),
                                "hidden_displacement": float((model.weight - initial_weight).norm()), "head_norm": float(model.head.norm()),
                                "even_energy": float(even.square().mean()), "odd_energy": float(odd.square().mean()),
                                "target_kernel_energy": target_alignment, "residual_kernel_energy": residual_alignment})
        if not torch.isfinite(loss):
            return {"status": "nonfinite", "cell": cell, "failed_step": step, "records": records, "finite": False}, arrays
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        arrays.update(final_hidden=model.weight.numpy(), final_head=model.head.numpy(), final_train_prediction=model(x).numpy(), final_test_prediction=model(tx).numpy())
    return {"status": "completed", "cell": cell, "finite": True, "records": records, "train_curve": curves,
            "initial_kernel_curve": initial_prediction, "initial_kernel_curve_error": float(np.max(np.abs(np.array(curves) - initial_prediction)))}, arrays


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
                raise ValueError("Changed frozen C03 executable")
            if not target.exists():
                shutil.copy2(source, target)
        count, reused, started = 0, 0, time.monotonic()
        manifests = {}
        total = len(config["data"]["distributions"]) * len(config["data"]["targets"]) * len(config["scales"]) * len(config["seeds"]) * len(config["activations"]) * len(config["hidden_modes"])
        for distribution in config["data"]["distributions"]:
            xs, targets = make_data(config, distribution)
            arrays = {"train_x": xs[0].numpy(), "test_x": xs[1].numpy()}
            for target, ys in targets.items():
                arrays.update({"train_y_"+target: ys[0].numpy(), "test_y_"+target: ys[1].numpy()})
            datapins = {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)} for k, a in arrays.items()}
            datapath = HERE / f"data_{distribution}.npz"
            if not datapath.exists():
                np.savez_compressed(datapath, **arrays)
            else:
                with np.load(datapath) as saved:
                    if any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
                        raise ValueError("Saved C03 data mismatch")
            manifests[distribution] = {"arrays": datapins, "file_sha256": c01.sha(datapath)}
            for seed in config["seeds"]:
                for scale in config["scales"]:
                    for activation in config["activations"]:
                        for mode in config["hidden_modes"]:
                            for target, ys in targets.items():
                                cell = {"distribution": distribution, "seed": seed, "scale": scale, "activation": activation, "hidden_mode": mode, "target": target}
                                label = f"{distribution}_{seed}_{scale}_{activation}_{mode}_{target}"
                                path = HERE / "results" / f"{label}.json"
                                contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
                                if path.exists():
                                    row = json.loads(path.read_text())
                                    if row["contract_sha256"] != contract or row["arrays_sha256"] != c01.sha(path.with_suffix(".npz")):
                                        raise ValueError("Saved C03 success mismatch")
                                    reused += 1
                                else:
                                    t0 = time.monotonic()
                                    row, output = train(config, cell, xs, ys)
                                    path.parent.mkdir(exist_ok=True)
                                    np.savez_compressed(path.with_suffix(".npz"), **output)
                                    row.update(contract_sha256=contract, arrays_sha256=c01.sha(path.with_suffix(".npz")), seconds=time.monotonic()-t0)
                                    c01.save(path, row)
                                count += 1
                                current = {"status": "measurements_complete" if count == total else "running", "pid": os.getpid(), "completed": count, "total": total,
                                           "reused": reused, "seconds": time.monotonic()-started, "updated_at": time.time(), "last_cell": cell, "solver_evaluation": "not_run"}
                                c01.save(HERE / "current.json", current)
                                print(json.dumps(current), flush=True)
        c01.save(HERE / "source_manifest.json", {"executable": pins, "data": manifests, "preregistered": True})


if __name__ == "__main__":
    main()
