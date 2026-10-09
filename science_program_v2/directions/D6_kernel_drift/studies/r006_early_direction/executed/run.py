from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
PREREG = STUDY / "preregistration.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def committed_contract(config):
    relative = str(PREREG.relative_to(REPO))
    for name, expected in config["source_sha256"].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f"Pinned source differs: {name}")
    history = subprocess.run(["git", "log", "--format=%H", "--", relative],
                             cwd=REPO, capture_output=True, text=True, check=True)
    for commit in history.stdout.splitlines():
        blob = subprocess.run(["git", "show", f"{commit}:{relative}"],
                              cwd=REPO, capture_output=True, check=True).stdout
        if blob != PREREG.read_bytes():
            continue
        for name, expected in config["source_sha256"].items():
            path = str((STUDY / name).relative_to(REPO))
            archived = subprocess.run(["git", "show", f"{commit}:{path}"],
                                      cwd=REPO, capture_output=True, check=True).stdout
            if hashlib.sha256(archived).hexdigest() != expected:
                raise RuntimeError("Preregistration commit does not pin executable sources")
        return commit
    raise RuntimeError("No matching preregistration commit; training is forbidden")


def make_data(config, function):
    contract = config["data_contract"]
    rng = np.random.default_rng(contract["data_seed"])
    x = rng.normal(size=(contract["train_n"], contract["input_dim"]))
    if function == "product":
        raw = np.tanh(x[:, 0] * x[:, 1])
    elif function == "mixed_sine":
        raw = np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2])
    else:
        raise ValueError(function)
    y = (raw - raw.mean()) / raw.std(ddof=0)
    return x, y, np.array([raw.mean(), raw.std(ddof=0)])


def jacobian(model, inputs):
    parameters = list(model.parameters())
    rows = []
    for value in model(inputs).reshape(-1):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in gradients]).detach())
    return torch.stack(rows).numpy()


def execute(config, function, width, seed, data):
    x, y, normalization = data
    inputs = torch.tensor(x, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)).double()
    recipe = config["recipe"]
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe["lr"], momentum=0.0, weight_decay=0.0)
    checkpoints = config["checkpoints"]
    js, kernels, parameters, residuals, losses = [], [], [], [], []
    for step in range(recipe["steps"] + 1):
        prediction = model(inputs).reshape(-1)
        residual = prediction - targets
        loss = residual.square().mean() / 2
        losses.append(loss.item())
        if step in checkpoints:
            j = jacobian(model, inputs)
            js.append(j)
            kernels.append(j @ j.T / len(x))
            parameters.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy())
            residuals.append(residual.detach().numpy())
        if step == recipe["steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    kernels = np.stack(kernels)
    residuals = np.stack(residuals)
    eigenvalues = np.linalg.eigvalsh(kernels)
    traces = np.trace(kernels, axis1=1, axis2=2)
    scales = traces / traces[0]
    matched = scales[:, None, None] * kernels[0]
    largest = max(eigenvalues[:, -1].max(), (scales * eigenvalues[0, -1]).max())
    probe_lr = config["probe"]["stability_fraction"] / largest
    fixed = residuals[0]
    fixed_ratios, moving_ratios, target_ratios, permutations, probe_curves = [], [], [], [], []
    for k, matched_k, r in zip(kernels, matched, residuals):
        fixed_ratios.append((fixed @ k @ fixed) / (fixed @ matched_k @ fixed))
        moving_ratios.append((r @ k @ r) / (r @ matched_k @ r))
        target_ratios.append((y @ k @ y) / (y @ matched_k @ y))
        permutation_ratios = []
        for pseed in config["probe"]["permutation_seeds"]:
            perm = np.random.default_rng(pseed).permutation(r)
            permutation_ratios.append((perm @ k @ perm) / (perm @ matched_k @ perm))
        permutations.append(permutation_ratios)
        curves = []
        for candidate in [kernels[0], matched_k, k]:
            current = r.copy()
            curve = [np.mean(current ** 2) / 2]
            for _ in range(config["probe"]["steps"]):
                current = current - probe_lr * candidate @ current
                curve.append(np.mean(current ** 2) / 2)
            curves.append(curve)
        probe_curves.append(curves)
    return {"train_x": x, "train_y": y, "target_normalization": normalization,
            "checkpoints": np.array(checkpoints), "jacobians": np.stack(js),
            "kernels": kernels, "parameters": np.stack(parameters), "residuals": residuals,
            "nonlinear_loss": np.array(losses), "eigenvalues": eigenvalues,
            "trace_scales": scales, "fixed_r0_ratios": np.array(fixed_ratios),
            "moving_rt_ratios": np.array(moving_ratios), "fixed_target_ratios": np.array(target_ratios),
            "permutation_ratios": np.array(permutations), "probe_lr": np.array(probe_lr),
            "probe_loss": np.array(probe_curves)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(PREREG.read_text())
    commit = committed_contract(config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    new_cells = 0
    for function in config["functions"]:
        data = make_data(config, function)
        inputs_hash = {name: hashlib.sha256(array.tobytes()).hexdigest()
                       for name, array in zip(["train_x", "train_y"], data[:2])}
        for width in config["widths"]:
            for seed in config["seeds"]:
                cell = f"{function}_w{width}_s{seed}"
                path = STUDY / "results" / f"{cell}.json"
                request = {"function": function, "width": width, "seed": seed,
                           "preregistration_sha256": sha(PREREG),
                           "source_sha256": config["source_sha256"], "inputs_sha256": inputs_hash}
                if path.exists():
                    previous = json.loads(path.read_text())
                    if (previous["request"] != request or previous["status"] != "success"
                            or previous["contract_sha256"] != digest(request)
                            or previous["arrays_sha256"] != sha(path.with_suffix(".npz"))):
                        raise RuntimeError(f"Saved cell hash/contract differs: {cell}")
                    for name in ["preregistration.json", *config["source_sha256"]]:
                        relative = str((STUDY / name).relative_to(REPO))
                        archived = subprocess.run(
                            ["git", "show", f"{previous['preregistration_commit']}:{relative}"],
                            cwd=REPO, capture_output=True, check=True).stdout
                        if hashlib.sha256(archived).hexdigest() != sha(STUDY / name):
                            raise RuntimeError(f"Saved cell commit differs: {cell}")
                    print(json.dumps({"cell": cell, "resumed": True}), flush=True)
                    continue
                if new_cells >= args.max_new_cells:
                    return
                if time.monotonic() - started >= config["execution"]["max_seconds"]:
                    raise RuntimeError("Round compute budget exhausted; resume remaining cells")
                if path.with_suffix(".npz").exists():
                    raise RuntimeError(f"Orphan arrays require audit: {cell}")
                cell_started = time.monotonic()
                try:
                    arrays = execute(config, function, width, seed, data)
                    if not all(np.isfinite(value).all() for value in arrays.values()):
                        raise RuntimeError("Nonfinite cell result")
                except Exception as error:
                    save(path.with_name(f"{cell}.failure.json"),
                         {"request": request, "status": "failed", "error": repr(error)})
                    raise
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                temporary = path.with_suffix(".npz.tmp")
                temporary.write_bytes(stream.getvalue())
                temporary.replace(path.with_suffix(".npz"))
                save(path, {"cell_id": cell, "request": request, "status": "success",
                            "contract_sha256": digest(request), "preregistration_commit": commit,
                            "arrays_sha256": sha(path.with_suffix(".npz")),
                            "seconds": time.monotonic() - cell_started,
                            "environment": {"python": sys.version, "numpy": np.__version__,
                                            "torch": torch.__version__, "device": "cpu",
                                            "threads": 1}, "pid": os.getpid()})
                new_cells += 1
                print(json.dumps({"cell": cell, "success": True,
                                  "seconds": time.monotonic() - cell_started}), flush=True)


if __name__ == "__main__":
    main()
