#!/usr/bin/env python3
"""Prospective OOD test of target-specific activation channels."""
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
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / "C01_activation_scale"
spec = importlib.util.spec_from_file_location("c01_executor", PREVIOUS / "executed/run_study.py")
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)


def make_data(config, distribution):
    rng = torch.Generator().manual_seed(config["data"]["dataset_seeds"][distribution])
    xs = []
    for count in [config["data"]["train_rows"], config["data"]["test_rows"]]:
        u = 2 * torch.rand(count // 2, config["architecture"]["input_dim"], generator=rng, dtype=torch.float64) - 1
        half = u * 3 ** .5 if distribution == "uniform" else -torch.sign(u) * torch.log1p(-torch.abs(u)) / 2 ** .5
        xs.append(torch.cat([half, -half]))
    targets = {}
    for name in config["data"]["targets"]:
        ys = [x[:, 0] + .5 * x[:, 1] if name == "linear_sum" else x[:, 0] * x[:, 1] for x in xs]
        mean, rms = ys[0].mean(), (ys[0] - ys[0].mean()).square().mean().sqrt()
        targets[name] = [(y - mean) / rms for y in ys]
    return xs, targets


def make_features(xs, seed, scale, activation, intervention, width):
    rng = torch.Generator().manual_seed(seed)
    weight = torch.randn(width, xs[0].shape[1], generator=rng, dtype=torch.float64) / xs[0].shape[1] ** .5
    zs = [scale * x @ weight.T for x in xs]
    fn = torch.relu if activation == "relu" else torch.nn.functional.silu
    odd = [z / 2 for z in zs]
    raw_even = [(fn(z) + fn(-z)) / 2 for z in zs]
    center = raw_even[0].mean(0)
    even = [a - center for a in raw_even]
    original_ratio = even[0].square().mean() / odd[0].square().mean()
    relu_even = zs[0].abs() / 2
    relu_even = relu_even - relu_even.mean(0)
    reference_ratio = relu_even.square().mean() / odd[0].square().mean()
    multiplier = (reference_ratio / original_ratio).sqrt() if intervention == "parity_balanced" else torch.tensor(1., dtype=torch.float64)
    raw = [o + multiplier * e for o, e in zip(odd, even)]
    norm = raw[0].square().mean().sqrt()
    matrices = [a / norm / width ** .5 for a in raw]
    info = {"preactivation_rms": float(zs[0].square().mean().sqrt()), "raw_feature_rms": float(norm),
            "normalized_row_energy": float(matrices[0].square().sum(1).mean()),
            "even_odd_energy_ratio": float(original_ratio), "relu_reference_ratio": float(reference_ratio),
            "even_multiplier": float(multiplier), "balanced_ratio": float(original_ratio * multiplier.square()),
            "label_free_intervention": True}
    return matrices, info


def main():
    torch.set_num_threads(1)
    with (HERE / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((HERE / "preregistration.json").read_text())
        if c01.sha(PREVIOUS / "report.md") != config["development_report_sha256"]:
            raise ValueError("Previous report changed after OOD prediction was sealed")
        pins = {"run_study.py": c01.sha(__file__), "preregistration.json": c01.sha(HERE / "preregistration.json"),
                "c01_executor.py": c01.sha(PREVIOUS / "executed/run_study.py"), "development_report.md": c01.sha(PREVIOUS / "report.md")}
        (HERE / "executed").mkdir(exist_ok=True)
        for name, source in [("run_study.py", Path(__file__)), ("preregistration.json", HERE / "preregistration.json"),
                             ("c01_executor.py", PREVIOUS / "executed/run_study.py"), ("development_report.md", PREVIOUS / "report.md")]:
            target = HERE / "executed" / name
            if target.exists() and c01.sha(target) != pins[name]:
                raise ValueError("Frozen C02 source changed")
            if not target.exists():
                shutil.copy2(source, target)
        count, reused, started = 0, 0, time.monotonic()
        total = len(config["data"]["distributions"]) * len(config["data"]["targets"]) * len(config["scales"]) * len(config["seeds"]) * len(config["activations"]) * len(config["interventions"])
        manifests = {}
        for distribution in config["data"]["distributions"]:
            xs, targets = make_data(config, distribution)
            arrays = {"train_x": xs[0].numpy(), "test_x": xs[1].numpy()}
            for target, ys in targets.items():
                arrays.update({"train_y_" + target: ys[0].numpy(), "test_y_" + target: ys[1].numpy()})
            datapins = {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)} for k, a in arrays.items()}
            datapath = HERE / f"data_{distribution}.npz"
            if not datapath.exists():
                np.savez_compressed(datapath, **arrays)
            else:
                with np.load(datapath) as saved:
                    if any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
                        raise ValueError("Sealed OOD data changed")
            manifests[distribution] = {"arrays": datapins, "file_sha256": c01.sha(datapath)}
            for seed in config["seeds"]:
                for scale in config["scales"]:
                    for activation in config["activations"]:
                        for intervention in config["interventions"]:
                            matrices, info = make_features(xs, seed, scale, activation, intervention, config["architecture"]["width"])
                            for target, ys in targets.items():
                                cell = {"distribution": distribution, "seed": seed, "scale": scale, "activation": activation, "intervention": intervention, "target": target}
                                label = f"{distribution}_{seed}_{scale}_{activation}_{intervention}_{target}"
                                path = HERE / "results" / f"{label}.json"
                                contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
                                if path.exists():
                                    row = json.loads(path.read_text())
                                    if row["contract_sha256"] != contract or row["arrays_sha256"] != c01.sha(path.with_suffix(".npz")):
                                        raise ValueError("Saved OOD success contract mismatch")
                                    reused += 1
                                else:
                                    t0 = time.monotonic()
                                    row, output = c01.run_cell(config, cell, matrices, ys, info)
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
