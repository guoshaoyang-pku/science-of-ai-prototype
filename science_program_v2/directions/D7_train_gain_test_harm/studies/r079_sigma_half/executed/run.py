from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


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
    for relative, expected in config["reference_files_sha256"].items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f"Frozen reference differs: {relative}")
    names = ["preregistration.json", *config["source_sha256"]]
    for name, expected in config["source_sha256"].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f"Pinned source differs: {name}")
    relative = str((STUDY / names[0]).relative_to(REPO))
    history = subprocess.run(["git", "log", "--format=%H", "--", relative],
                             cwd=REPO, capture_output=True, text=True, check=True)
    for commit in history.stdout.splitlines():
        matches = True
        for name in names:
            relative = str((STUDY / name).relative_to(REPO))
            blob = subprocess.run(["git", "show", f"{commit}:{relative}"],
                                  cwd=REPO, capture_output=True, check=True).stdout
            matches = matches and blob == (STUDY / name).read_bytes()
        if matches:
            return commit
    raise RuntimeError("No matching preregistration commit; training is forbidden")


def make_data(config, function, sigma):
    contract = config["data_contract"]
    rng = np.random.default_rng(contract["data_seed"])
    x = rng.normal(size=(contract["train_n"], contract["input_dim"]))
    test_x = rng.normal(size=(contract["test_n"], contract["input_dim"]))
    def target(inputs):
        if function == "mixed_sine":
            return np.sin(1.7 * inputs[:, 0]) + .4 * np.sin(inputs[:, 1] * inputs[:, 2])
        if function == "radial":
            return np.exp(-np.square(inputs).sum(axis=1) / contract["input_dim"])
        raise ValueError(function)
    raw = target(x)
    center, scale = raw.mean(), raw.std(ddof=0)
    clean_y = (raw - center) / scale
    test_y = (target(test_x) - center) / scale
    epsilon = np.random.default_rng(contract["noise_seed"]).normal(size=len(x))
    return {"train_x": x, "clean_train_y": clean_y, "train_y": clean_y + sigma * epsilon,
            "test_x": test_x, "test_y": test_y, "noise_base": epsilon,
            "target_normalization": np.array([center, scale])}


def jacobian(model, inputs):
    parameters = list(model.parameters())
    rows = []
    for output in model(inputs).reshape(-1):
        gradients = torch.autograd.grad(output, parameters, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in gradients]).detach())
    return torch.stack(rows)


def execute(config, width, seed, data):
    x = torch.tensor(data["train_x"], dtype=torch.float64)
    y = torch.tensor(data["train_y"], dtype=torch.float64)
    test_x = torch.tensor(data["test_x"], dtype=torch.float64)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)).double()
    f0, f0_test = model(x).detach().reshape(-1), model(test_x).detach().reshape(-1)
    j, j_test = jacobian(model, x), jacobian(model, test_x)
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    delta = torch.nn.Parameter(torch.zeros(j.shape[1], dtype=torch.float64))
    recipe = config["recipe"]
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe["lr"], momentum=0.0, weight_decay=0.0)
    tangent_optimizer = torch.optim.SGD([delta], lr=recipe["lr"], momentum=0.0, weight_decay=0.0)
    checkpoints = config["checkpoints"]
    train_outputs, test_outputs, losses = [], [], []
    for step in range(recipe["steps"] + 1):
        true_train = model(x).reshape(-1)
        tangent_train = f0 + j @ delta
        true_loss = (true_train - y).square().mean() / 2
        tangent_loss = (tangent_train - y).square().mean() / 2
        if not torch.isfinite(true_loss) or not torch.isfinite(tangent_loss):
            raise RuntimeError("Nonfinite loss")
        losses.append([true_loss.item(), tangent_loss.item()])
        if step in checkpoints:
            with torch.no_grad():
                train_outputs.append(torch.stack([true_train.detach(), tangent_train.detach()]).numpy())
                test_outputs.append(torch.stack([model(test_x).reshape(-1), f0_test + j_test @ delta]).numpy())
        if step == recipe["steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        true_loss.backward()
        optimizer.step()
        tangent_optimizer.zero_grad(set_to_none=True)
        tangent_loss.backward()
        tangent_optimizer.step()
    return {**data, "checkpoints": np.array(checkpoints), "train_outputs": np.stack(train_outputs),
            "test_outputs": np.stack(test_outputs), "train_loss": np.array(losses),
            "initial_jacobian_train": j.numpy(), "initial_jacobian_test": j_test.numpy(),
            "initial_parameters": theta0.numpy(),
            "final_parameters": torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy(),
            "final_tangent_delta": delta.detach().numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-cells", type=int, default=12)
    args = parser.parse_args()
    config = json.loads((STUDY / "preregistration.json").read_text())
    commit = committed_contract(config)
    receipt_path = STUDY / "executed/pre_execution_audit.json"
    receipt = {"preregistration_commit": commit,
               "preregistration_sha256": sha(STUDY / "preregistration.json"),
               "source_sha256": config["source_sha256"],
               "reference_hashes_checked": len(config["reference_files_sha256"])}
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if any(prior[k] != v for k, v in receipt.items()):
            raise RuntimeError("Pre-execution receipt differs")
    else:
        receipt["verified_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        receipt["commit_timestamp"] = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", commit], cwd=REPO, text=True).strip()
        save(receipt_path, receipt)
    expected_cells = {f"{fn}_w{w}_s{s}_sigma{sigma:g}" for fn in config["functions"]
                      for w in config["widths"] for s in config["seeds"]
                      for sigma in config["noise_sigmas"]}
    for artifact in (STUDY / "results").glob("*"):
        if artifact.suffix not in {".json", ".npz"} or artifact.stem not in expected_cells:
            raise RuntimeError(f"Unexpected/orphan/failure artifact requires audit: {artifact}")
        if not artifact.with_suffix(".json").exists() or not artifact.with_suffix(".npz").exists():
            raise RuntimeError(f"Incomplete saved cell requires audit: {artifact}")
    for path in sorted((STUDY / "results").glob("*.json")):
        prior = json.loads(path.read_text())
        cell = path.stem
        fn, width, seed, sigma = (prior["request"][n] for n in ["function", "width", "seed", "sigma"])
        data = make_data(config, fn, sigma)
        request = {"function": fn, "width": width, "seed": seed, "sigma": sigma,
                   "preregistration_sha256": sha(STUDY / "preregistration.json"),
                   "source_sha256": config["source_sha256"],
                   "inputs_sha256": {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in data.items()}}
        if (prior["cell_id"] != cell or prior["request"] != request or prior["contract_sha256"] != digest(request)
                or prior["status"] != "success"
                or prior["arrays_sha256"] != sha(path.with_suffix(".npz"))):
            raise RuntimeError(f"Saved cell hash/contract differs: {cell}")
        for name in ["preregistration.json", *config["source_sha256"]]:
            relative = str((STUDY / name).relative_to(REPO))
            blob = subprocess.run(["git", "show", f"{prior['preregistration_commit']}:{relative}"],
                                  cwd=REPO, capture_output=True, check=True).stdout
            if blob != (STUDY / name).read_bytes():
                raise RuntimeError(f"Saved commit differs: {cell}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started, new_cells = time.monotonic(), 0
    for function in config["functions"]:
        for width in config["widths"]:
            for seed in config["seeds"]:
                for sigma in config["noise_sigmas"]:
                    data = make_data(config, function, sigma)
                    cell = f"{function}_w{width}_s{seed}_sigma{sigma:g}"
                    path = STUDY / "results" / f"{cell}.json"
                    request = {"function": function, "width": width, "seed": seed, "sigma": sigma,
                               "preregistration_sha256": sha(STUDY / "preregistration.json"),
                               "source_sha256": config["source_sha256"],
                               "inputs_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in data.items()}}
                    if path.exists():
                        print(json.dumps({"cell": cell, "resumed": True}), flush=True)
                        continue
                    if new_cells >= args.max_new_cells:
                        return
                    if time.monotonic() - started >= config["execution"]["max_seconds"]:
                        raise RuntimeError("Round compute budget exhausted")
                    if path.with_suffix(".npz").exists():
                        raise RuntimeError(f"Orphan arrays require audit: {cell}")
                    cell_started = time.monotonic()
                    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
                    try:
                        arrays = execute(config, width, seed, data)
                        if not all(np.isfinite(a).all() for a in arrays.values()):
                            raise RuntimeError("Nonfinite result")
                    except Exception as error:
                        save(path.with_name(f"{cell}.failure.json"),
                             {"request": request, "status": "failed", "error": repr(error)})
                        raise
                    stream = io.BytesIO()
                    np.savez_compressed(stream, **arrays)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".npz.tmp")
                    temporary.write_bytes(stream.getvalue())
                    temporary.replace(path.with_suffix(".npz"))
                    save(path, {"cell_id": cell, "status": "success", "request": request,
                                "contract_sha256": digest(request), "preregistration_commit": commit,
                                "arrays_sha256": sha(path.with_suffix(".npz")),
                                "started_at": started_at,
                                "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                                "seconds": time.monotonic() - cell_started,
                                "environment": {"python": sys.version, "numpy": np.__version__,
                                                "torch": torch.__version__, "device": "cpu", "threads": 1}})
                    new_cells += 1
                    print(json.dumps({"cell": cell, "success": True}), flush=True)


if __name__ == "__main__":
    main()
