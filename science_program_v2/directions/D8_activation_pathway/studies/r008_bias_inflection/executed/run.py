from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import brentq
import torch

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cells(config):
    for seed in config["seeds"]:
        for bias in config["bias_labels"]:
            for scale in config["scales"]:
                yield {"seed": seed, "bias_label": bias, "scale": scale}


def label(cell):
    return f"s{cell['seed']}_{cell['bias_label']}_a{cell['scale']:g}"


def make_data(config):
    rng = torch.Generator().manual_seed(config["data"]["dataset_seed"])
    arrays = {}
    for split, n in (("train", config["data"]["train_rows"]), ("test", config["data"]["test_rows"])):
        half = torch.randn(n // 2, config["architecture"]["input_dim"], generator=rng, dtype=torch.float64)
        x = torch.cat([half, -half])
        arrays[f"{split}_x"] = x.numpy()
        arrays[f"{split}_y"] = torch.stack([x[:, 0], x[:, 0].square()], dim=1).numpy()
    mean = arrays["train_y"].mean(axis=0)
    rms = np.sqrt(np.mean((arrays["train_y"] - mean) ** 2, axis=0))
    for split in ("train", "test"):
        arrays[f"{split}_y"] = (arrays[f"{split}_y"] - mean) / rms
    arrays.update(target_mean=mean, target_rms=rms)
    return arrays


def bias_values(config):
    root = brentq(lambda b: b * np.tanh(b / 2) - 2, *config["bias_root_bracket"], xtol=1e-14)
    return {"zero": 0.0, "one": 1.0, "inflection": float(root)}


def train(config, data, cell, bias):
    rng = torch.Generator().manual_seed(cell["seed"])
    width, d = config["architecture"]["width"], config["architecture"]["input_dim"]
    weight = torch.randn(width, d, generator=rng, dtype=torch.float64) / d ** .5
    us = [torch.from_numpy(data[f"{split}_x"]) @ weight.T for split in ("train", "test")]
    raw = [torch.nn.functional.silu(bias + cell["scale"] * u) for u in us]
    reflected = torch.nn.functional.silu(bias - cell["scale"] * us[0])
    center = raw[0].mean(0)
    norm = (raw[0] - center).square().mean().sqrt()
    phi, test_phi = [(a - center) / norm / width ** .5 for a in raw]
    odd, even = (raw[0] - reflected) / 2, (raw[0] + reflected) / 2 - center
    y, test_y = [torch.from_numpy(data[f"{split}_y"]) for split in ("train", "test")]
    kernel = phi @ phi.T / len(phi)
    eig, vec = torch.linalg.eigh(kernel)
    coeff = vec.T @ y
    head = torch.zeros(width, y.shape[1], dtype=torch.float64)
    steps, lr = config["optimizer"]["steps"], config["optimizer"]["lr"]
    curve, spectral, outputs, test_outputs = [], [], [], []
    for step in range(steps + 1):
        residual = phi @ head - y
        curve.append(residual.square().mean(0).numpy())
        spectral.append(((coeff.square() * (1 - 2 * lr * eig[:, None]).pow(2 * step)).sum(0) / len(phi)).numpy())
        if step in config["checkpoints"]:
            outputs.append((phi @ head).numpy())
            test_outputs.append((test_phi @ head).numpy())
        if step < steps:
            head -= 2 * lr * phi.T @ residual / len(phi)
    arrays = {"hidden_weight": weight.numpy(), "train_u": us[0].numpy(), "train_features": phi.numpy(),
              "test_features": test_phi.numpy(), "odd_raw": odd.numpy(), "even_centered_raw": even.numpy(),
              "feature_center": center.numpy(), "feature_rms": norm.numpy(), "train_y": data["train_y"],
              "test_y": data["test_y"], "train_curve": np.array(curve), "spectral_curve": np.array(spectral),
              "checkpoint_outputs": np.array(outputs), "checkpoint_test_outputs": np.array(test_outputs),
              "final_head": head.numpy(), "eigenvalues": eig.numpy(), "target_spectral_coefficients": coeff.numpy()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Nonfinite evidence; cell not saved as successful")
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((STUDY / "preregistration.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, check=True, text=True).stdout.strip()
    for name in ["preregistration.json", *config["source_sha256"]]:
        path = STUDY / name
        if name in config["source_sha256"] and sha(path) != config["source_sha256"][name]:
            raise RuntimeError(f"Pinned source mismatch: {name}")
        result = subprocess.run(["git", "show", f"{commit}:{path.relative_to(REPO)}"], cwd=REPO, capture_output=True)
        if result.returncode or result.stdout != path.read_bytes():
            raise RuntimeError("Matching preregistration and sources must be committed before data/training")
    data = make_data(config)
    data_path = STUDY / "executed/data.npz"
    if data_path.exists():
        with np.load(data_path, allow_pickle=False) as z:
            if set(z.files) != set(data) or any(not np.array_equal(z[n], a) for n, a in data.items()):
                raise RuntimeError("Saved data differs")
    else:
        np.savez_compressed(data_path, **data)
    manifest = {"preregistration_sha256": sha(STUDY / "preregistration.json"), "source_sha256": config["source_sha256"],
                "preregistration_commit": commit, "data_file_sha256": sha(data_path),
                "data": {n: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape), "dtype": str(a.dtype)} for n, a in data.items()},
                "numpy_version": np.__version__, "torch_version": torch.__version__, "device": "cpu", "threads": 1}
    manifest_path = STUDY / "executed/source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ("preregistration_sha256", "source_sha256", "data", "data_file_sha256"):
            if previous[field] != manifest[field]:
                raise RuntimeError("Source/data manifest differs")
        manifest = previous
    else:
        save(manifest_path, manifest)
    biases = bias_values(config)
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    new, reused = 0, 0
    started = time.monotonic()
    for cell in cells(config):
        path = results / f"{label(cell)}.json"
        request = {"cell": cell, "bias": biases[cell["bias_label"]], "manifest_sha256": sha(manifest_path)}
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if path.exists():
            row = json.loads(path.read_text())
            if (row["status"] != "success" or row["request"] != request or row["contract_sha256"] != contract
                    or row["arrays_sha256"] != sha(path.with_suffix(".npz"))):
                raise RuntimeError("Saved successful cell hash/contract mismatch")
            reused += 1
        else:
            if path.with_suffix(".npz").exists():
                raise RuntimeError("Orphan arrays require inspection; refusing overwrite")
            cell_started = time.monotonic()
            arrays = train(config, data, cell, request["bias"])
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            save(path, {"status": "success", "cell_id": label(cell), "request": request, "contract_sha256": contract,
                        "arrays_sha256": sha(path.with_suffix(".npz")), "seconds": time.monotonic() - cell_started})
            new += 1
        state = {"completed": new + reused, "new": new, "reused": reused, "planned_cells": config["planned_cells"],
                 "seconds": time.monotonic() - started, "last_cell": label(cell)}
        save(STUDY / "current.json", state)
        print(json.dumps(state), flush=True)
        if args.limit is not None and new >= args.limit:
            break


if __name__ == "__main__":
    main()
