"""Portable archived MLP interventions; atomic, content checked per-seed results."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write(chr(10))
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_pin(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "shape": list(a.shape), "dtype": str(a.dtype)}


def activation(name):
    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
            "leaky_relu": lambda: nn.LeakyReLU(.01)}[name]()


class Block(nn.Module):
    def __init__(self, width, norm, act, residual):
        super().__init__()
        self.norm = nn.LayerNorm(width) if norm else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = activation(act)
        self.residual = residual

    def forward(self, x):
        h = self.act(self.linear(self.norm(x)))
        return x + h if self.residual else h


class Model(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        width, depth = recipe["width"], recipe["depth"]
        norms = recipe.get("layer_norm", [False] * depth)
        if len(norms) != depth:
            raise ValueError("LN placement must match depth")
        act = recipe.get("activation", "gelu")
        self.net = nn.Sequential(
            nn.Linear(recipe["input_dim"], width), activation(act),
            *[Block(width, n, act, recipe.get("residual", False)) for n in norms],
            nn.Linear(width, 1),
        )
        scale = recipe.get("weight_scale", 1.0)
        if scale != 1:
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.mul_(scale)
                        module.bias.mul_(scale)

    def forward(self, x):
        return self.net(x)


def build_optimizer(model, recipe):
    name = recipe["optimizer"]
    values = {"lr": recipe["lr"], "weight_decay": recipe.get("weight_decay", 1e-4)}
    if name == "SGD":
        values["momentum"] = recipe.get("momentum", 0.0)
    elif name == "Adam":
        values["betas"] = tuple(recipe.get("betas", [.9, .999]))
    else:
        raise ValueError("Archived process recorder supports SGD and Adam")
    return getattr(torch.optim, name)(model.parameters(), **values)


@contextlib.contextmanager
def executable(study, optimizer):
    study = Path(study)
    directory = study / "executed" / optimizer
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "sources/e17/executed" / optimizer / "matched_bias"
    marker_start = "    head = model.net[-1]" + chr(10)
    marker_end = "    optimizer = build_optimizer(model)" + chr(10)
    pins = {}
    for name in ["model.py", "loss.py", "optimizer.py", "train.py"]:
        raw = (source / name).read_bytes()
        if name == "train.py":
            text = raw.decode()
            left, right = text.index(marker_start), text.index(marker_end)
            insert = """    head = model.net[-1]
    mode = process["intervention"]
    if mode == "matched_bias":
        with torch.no_grad():
            head.bias.add_(process["mean"])
    elif mode == "frozen_hidden":
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(f"net.{len(model.net)-1}."))
    elif mode == "frozen_head":
        for p in head.parameters():
            p.requires_grad_(False)
    elif mode == "frozen_head_weight":
        head.weight.requires_grad_(False)
    elif mode == "baseline":
        pass
    else:
        raise ValueError("Unknown explicit intervention")
"""
            raw = (text[:left] + insert + text[right:]).encode()
        target = directory / name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Execution source changed for an existing study")
        target.write_bytes(raw)
        pins[name] = {"sha256": sha(target), "bytes": len(raw),
                      "source_sha256": sha(source / name)}
    pins["experiment.py"] = {"sha256": sha(Path(__file__)),
                             "path": "../../experiment.py"}
    prior = {n: sys.modules.pop(n, None) for n in ["model", "loss", "optimizer"]}
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("science_archived_train", directory / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, pins
    finally:
        sys.path.remove(str(directory))
        for name, old in prior.items():
            sys.modules.pop(name, None)
            if old is not None:
                sys.modules[name] = old


def run_cell(study, cell, tensors):
    """cell contains id, seed, mean, intervention and the complete recipe."""
    study = Path(study)
    path = study / "results" / (cell["id"] + ".json")
    recipe = cell["recipe"]
    request = {"cell": cell, "inputs": {n: tensor_pin(t) for n, t in
               zip(["train_x", "train_y", "test_x", "test_y"], tensors)}}
    with executable(study, recipe["optimizer"]) as (module, pins):
        request["executable"] = pins
        contract = digest(request)
        if path.exists():
            row = json.loads(path.read_text())
            if row["contract_sha256"] != contract or sha(path.with_suffix(".npz")) != row["arrays_sha256"]:
                raise ValueError("Existing success has a different contract or damaged arrays")
            return row, True
        module.Model = lambda: Model(recipe)
        module.build_optimizer = lambda model: build_optimizer(model, recipe)
        process = {"version": "regression_mse_v1",
                   "steps": sorted(set([1, 8, 32, 128, recipe["steps"]]) & set(range(1, recipe["steps"] + 1))),
                   "max_eval_samples": max(t.shape[0] for t in tensors),
                   "intervention": cell["intervention"], "mean": float(cell["mean"])}
        t0 = time.monotonic()
        result = module.train_and_eval(*tensors, steps=recipe["steps"],
                                      batch_size=recipe["batch_size"], seed=cell["seed"],
                                      fail_threshold=1e12, process=process)
        diagnostics = result["process_diagnostics"]
        arrays = {k: v.detach().cpu().numpy() for k, v in diagnostics["evaluation"].items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path.with_suffix(".npz"))
        row = {**cell, "contract_sha256": contract, "contract": request,
               "arrays_sha256": sha(path.with_suffix(".npz")),
               "test_mse": result["final_test_mse"], "curve": result["step_metrics"],
               "failed": result["failed"], "process": diagnostics["record"],
               "seconds": time.monotonic() - t0, "saved_at": time.time()}
        if cell["intervention"].startswith("frozen_head"):
            for step in row["process"]["steps"]:
                for p in step["parameters"]:
                    frozen = p["role"] == "head" and (cell["intervention"] == "frozen_head" or p["name"].endswith("weight"))
                    if frozen and p["displacement_norm"] != 0:
                        raise AssertionError("Frozen parameter moved")
        save(path, row)
        return row, False
