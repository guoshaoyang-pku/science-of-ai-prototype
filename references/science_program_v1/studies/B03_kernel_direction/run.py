from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROGRAM = HERE.parents[1]
DIRECTION = PROGRAM / "directions/B_effective_time_capacity"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    temporary.replace(path)


def make_data(condition):
    rng = np.random.default_rng(condition["data_seed"])
    d = condition["input_dim"]
    if condition["distribution"] == "gaussian":
        train_x = rng.normal(size=(condition["train_samples"], d))
        test_x = rng.normal(size=(condition["test_samples"], d))
    else:
        train_x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(condition["train_samples"], d))
        test_x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(condition["test_samples"], d))
    def target(x):
        if condition["function"] == "product":
            return np.tanh(x[:, 0] * x[:, 1])
        if condition["function"] == "mixed_sine":
            return np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2])
        if condition["function"] == "triple_product":
            return np.tanh(x[:, 0] * x[:, 1] * x[:, 2])
        if condition["function"] == "soft_bump":
            return np.exp(-.7 * ((x[:, :3] - .3) ** 2).sum(axis=1)) + .2 * np.tanh(x[:, 3])
        raise ValueError("Unknown target")
    y_train, y_test = target(train_x), target(test_x)
    mean, scale = y_train.mean(), y_train.std()
    return tuple(torch.tensor(value, dtype=torch.float64) for value in (train_x, (y_train - mean) / scale, test_x, (y_test - mean) / scale))


def jacobian(model, inputs):
    parameters = list(model.parameters())
    rows = []
    for value in model(inputs).reshape(-1):
        rows.append(torch.cat([gradient.reshape(-1) for gradient in torch.autograd.grad(value, parameters, retain_graph=True)]).detach())
    return torch.stack(rows)


def probe(kernel, cross_kernel, train_residual, test_residual, eta, steps):
    residual = train_residual.copy()
    residual_test = test_residual.copy()
    train_loss = [np.mean(residual ** 2) / 2]
    test_loss = [np.mean(residual_test ** 2) / 2]
    for _ in range(steps):
        residual_test = residual_test - eta * cross_kernel @ residual
        residual = residual - eta * kernel @ residual
        train_loss.append(np.mean(residual ** 2) / 2)
        test_loss.append(np.mean(residual_test ** 2) / 2)
    return np.array(train_loss), np.array(test_loss)


def execute(condition, width, seed, registration, Model):
    train_x, train_y, test_x, test_y = make_data(condition)
    torch.manual_seed(seed)
    model = Model({"input_dim": condition["input_dim"], "width": width, "depth": 1,
                   "activation": "silu", "residual": False, "layer_norm": [False]}).double()
    initial_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    j0, jt0 = jacobian(model, train_x), jacobian(model, test_x)
    optimizer = torch.optim.SGD(model.parameters(), lr=registration["pretrain_lr"], momentum=0.0)
    true_train, true_test = [], []
    for step in range(registration["pretrain_steps"] + 1):
        pred = model(train_x).reshape(-1)
        loss = (pred - train_y).square().mean() / 2
        true_train.append(float(loss.detach().item()))
        with torch.no_grad():
            true_test.append(float((model(test_x).reshape(-1) - test_y).square().mean().item() / 2))
        if step == registration["pretrain_steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    j1, jt1 = jacobian(model, train_x), jacobian(model, test_x)
    with torch.no_grad():
        train_residual = (model(train_x).reshape(-1) - train_y).numpy()
        test_residual = (model(test_x).reshape(-1) - test_y).numpy()
    k0 = (j0 @ j0.T / len(train_x)).numpy()
    k1 = (j1 @ j1.T / len(train_x)).numpy()
    cross0 = (jt0 @ j0.T / len(train_x)).numpy()
    cross1 = (jt1 @ j1.T / len(train_x)).numpy()
    scale = np.trace(k1) / np.trace(k0)
    kernels = {"initial": (k0, cross0), "initial_equal_trace": (scale * k0, scale * cross0), "learned": (k1, cross1)}
    largest = max(np.linalg.eigvalsh(kernel)[-1] for kernel, _ in kernels.values())
    eta = registration["probe_stability_fraction"] / largest
    result = {"train_x": train_x.numpy(), "train_y": train_y.numpy(), "test_x": test_x.numpy(), "test_y": test_y.numpy(),
              "initial_parameters": initial_parameters.numpy(),
              "learned_parameters": torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).numpy(),
              "initial_jacobian": j0.numpy(), "learned_jacobian": j1.numpy(),
              "initial_test_jacobian": jt0.numpy(), "learned_test_jacobian": jt1.numpy(),
              "initial_kernel": k0, "learned_kernel": k1, "scale": np.array(scale), "probe_lr": np.array(eta),
              "pretrain_loss": np.array(true_train), "pretrain_test_loss": np.array(true_test),
              "probe_train_residual": train_residual, "probe_test_residual": test_residual}
    for name, (kernel, cross) in kernels.items():
        train_loss, test_loss = probe(kernel, cross, train_residual, test_residual, eta, registration["probe_steps"])
        result[name + "_probe_train_loss"] = train_loss
        result[name + "_probe_test_loss"] = test_loss
    rayleigh = lambda kernel, residual: float(residual @ kernel @ residual / (residual @ residual))
    observed_ratio = rayleigh(k1, train_residual) / rayleigh(scale * k0, train_residual)
    permutation_ratios = []
    for permutation_seed in registration["permutation_seeds"]:
        residual = np.random.default_rng(permutation_seed).permutation(train_residual)
        permutation_ratios.append(rayleigh(k1, residual) / rayleigh(scale * k0, residual))
    result["rayleigh_ratios"] = np.array([observed_ratio, *permutation_ratios])
    result["trace_match_error"] = np.array(abs(np.trace(scale * k0) - np.trace(k1)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    args = parser.parse_args()
    if not json.loads((PROGRAM / "state/current.json").read_text()).get("first_chain_complete"):
        raise SystemExit("First chain incomplete")
    registration = json.loads((HERE / "preregistration.json").read_text())
    if sha(Path(__file__)) != registration["run_source_sha256"] or sha(HERE / "executed/experiment.py") != registration["model_source_sha256"]:
        raise SystemExit("Source hash differs from preregistration")
    if args.phase == "ood" and not (HERE / "ood_predictions.json").exists():
        raise SystemExit("Save OOD prediction first")
    spec = importlib.util.spec_from_file_location("B03_archived_model", HERE / "executed/experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lock = (HERE / "worker.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    total = len(registration[args.phase + "_conditions"]) * len(registration[args.phase + "_widths"]) * len(registration["seeds"])
    completed = resumed = 0
    started = time.time()
    output = HERE / "results" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    for condition in registration[args.phase + "_conditions"]:
        for width in registration[args.phase + "_widths"]:
            for seed in registration["seeds"]:
                request = {"condition": condition, "width": width, "seed": seed,
                           "preregistration_sha256": sha(HERE / "preregistration.json"),
                           "execution_sha256": registration["run_source_sha256"],
                           "model_sha256": registration["model_source_sha256"]}
                cell_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
                path = output / (cell_id + ".json")
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous["request"] != request or previous["arrays_sha256"] != sha(path.with_suffix(".npz")):
                        raise RuntimeError("Saved evidence hash differs")
                    completed += 1
                    resumed += 1
                    continue
                process = {"pid": os.getpid(), "source_sha256": sha(Path(__file__)), "started_at": started,
                           "phase": args.phase, "current_cell": cell_id, "completed": completed, "total": total,
                           "resumed_successes": resumed, "updated_at": time.time(), "status": "running"}
                save(HERE / "current.json", process)
                direction = json.loads((DIRECTION / "state.json").read_text())
                direction.update({"status": "B03_running", "process": process, "last_completed": f"B03 {args.phase} {completed}/{total}",
                                  "next_action": "Compare equal-trace kernels and target vs permutation probes; keep failed predictions"})
                save(DIRECTION / "state.json", direction)
                cell_started = time.time()
                arrays = execute(condition, width, seed, registration, module.Model)
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise RuntimeError("Nonfinite result")
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                temporary = path.with_suffix(".npz.tmp")
                temporary.write_bytes(stream.getvalue())
                temporary.replace(path.with_suffix(".npz"))
                metadata = {"cell_id": cell_id, "request": request, "status": "success",
                            "arrays_sha256": sha(path.with_suffix(".npz")), "seconds": time.time() - cell_started,
                            "saved_at": time.time()}
                save(path, metadata)
                completed += 1
                print(json.dumps({"cell": cell_id, "condition": condition["name"], "width": width, "seed": seed,
                                  "completed": completed, "total": total, "seconds": metadata["seconds"],
                                  "target_rayleigh_ratio": float(arrays["rayleigh_ratios"][0]),
                                  "permutation_ratio_mean": float(arrays["rayleigh_ratios"][1:].mean())}), flush=True)
    save(HERE / "current.json", {"status": "completed", "phase": args.phase, "completed": completed,
                                    "total": total, "resumed_successes": resumed, "process": None,
                                    "elapsed_seconds": time.time() - started, "updated_at": time.time()})


if __name__ == "__main__":
    main()
