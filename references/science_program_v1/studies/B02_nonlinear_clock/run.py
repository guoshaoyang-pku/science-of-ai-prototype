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


def data(condition):
    rng = np.random.default_rng(condition["data_seed"])
    dimension = condition["input_dim"]
    train_x = rng.normal(size=(condition["train_samples"], dimension))
    test_x = rng.normal(size=(condition["test_samples"], dimension))
    def target(x):
        if condition["function"] == "smooth_sum":
            return np.tanh(x[:, 0]) + .5 * np.tanh(x[:, 1])
        if condition["function"] == "product":
            return np.tanh(x[:, 0] * x[:, 1])
        if condition["function"] == "mixed_sine":
            return np.sin(1.7 * x[:, 0]) + .4 * np.sin(x[:, 1] * x[:, 2])
        if condition["function"] == "radial":
            return np.exp(-np.square(x).sum(axis=1) / dimension)
        raise ValueError("Unknown target function")
    train_y, test_y = target(train_x), target(test_x)
    center, scale = train_y.mean(), train_y.std()
    train_y, test_y = (train_y - center) / scale, (test_y - center) / scale
    return tuple(torch.tensor(value, dtype=torch.float64) for value in (train_x, train_y, test_x, test_y))


def jacobian(model, inputs):
    parameters = list(model.parameters())
    rows = []
    for prediction in model(inputs).reshape(-1):
        gradients = torch.autograd.grad(prediction, parameters, retain_graph=True)
        rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]).detach())
    return torch.stack(rows)


def make_optimizer(parameters, recipe):
    if recipe["optimizer"] == "SGD":
        return torch.optim.SGD(parameters, lr=recipe["lr"], momentum=recipe.get("momentum", 0.0))
    return torch.optim.Adam(parameters, lr=recipe["lr"], betas=tuple(recipe["betas"]), eps=recipe["eps"])


def execute(condition, width, seed, recipe, steps, Model):
    train_x, train_y, test_x, test_y = data(condition)
    torch.manual_seed(seed)
    model = Model({"input_dim": condition["input_dim"], "width": width, "depth": 1,
                   "activation": "silu", "residual": False, "layer_norm": [False]}).double()
    initial_train = model(train_x).detach().reshape(-1)
    initial_test = model(test_x).detach().reshape(-1)
    j_train, j_test = jacobian(model, train_x), jacobian(model, test_x)
    parameters = list(model.parameters())
    theta0 = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    kernel = j_train @ j_train.T / len(train_x)
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    residual_modes = eigenvectors.T @ (initial_train - train_y)
    tangent = torch.nn.Parameter(torch.zeros(j_train.shape[1], dtype=torch.float64))
    optimizer = make_optimizer(parameters, recipe)
    tangent_optimizer = make_optimizer([tangent], recipe)
    checkpoints = sorted(set([0, 1, 8, 32, 128, steps]))
    training_losses, tangent_training_losses = [], []
    records = []
    snapshots = []
    predicted_modes = [residual_modes.numpy().copy()] if recipe["optimizer"] == "SGD" else []
    for step in range(steps + 1):
        true_train = model(train_x).reshape(-1)
        linear_train = initial_train + j_train @ tangent
        true_loss = (true_train - train_y).square().mean() / 2
        linear_loss = (linear_train - train_y).square().mean() / 2
        training_losses.append(float(true_loss.detach().item()))
        tangent_training_losses.append(float(linear_loss.detach().item()))
        if step in checkpoints or step % 8 == 0:
            with torch.no_grad():
                true_test = model(test_x).reshape(-1)
                linear_test = initial_test + j_test @ tangent
                theta = torch.cat([parameter.reshape(-1) for parameter in parameters])
                linear_at_theta = initial_train + j_train @ (theta - theta0)
                motion = true_train.detach() - initial_train
                residual = true_train.detach() - train_y
                records.append([step, training_losses[-1], tangent_training_losses[-1],
                                float((true_test - test_y).square().mean().item() / 2),
                                float((linear_test - test_y).square().mean().item() / 2),
                                float((true_train.detach() - linear_at_theta).norm().item()),
                                float(motion.norm().item()), float((theta - theta0).norm().item()),
                                float(residual @ kernel @ residual / residual.square().sum())])
        if step in checkpoints:
            current_j = jacobian(model, train_x)
            current_kernel = current_j @ current_j.T / len(train_x)
            snapshots.append([step, float((current_j - j_train).norm().item() / j_train.norm().item()),
                              float((current_kernel - kernel).norm().item() / kernel.norm().item()),
                              float(torch.linalg.eigvalsh(current_kernel)[-1].item())])
        if step == steps:
            break
        if not torch.isfinite(true_loss) or not torch.isfinite(linear_loss):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        true_loss.backward()
        optimizer.step()
        tangent_optimizer.zero_grad(set_to_none=True)
        linear_loss.backward()
        tangent_optimizer.step()
        if predicted_modes:
            eta, momentum = recipe["lr"], recipe.get("momentum", 0.0)
            lam = eigenvalues.numpy()
            prediction = (1 - eta * lam) * predicted_modes[-1] if step == 0 else (1 + momentum - eta * lam) * predicted_modes[-1] - momentum * predicted_modes[-2]
            predicted_modes.append(prediction)
    result = {"train_half_mse": np.array(training_losses),
              "tangent_train_half_mse": np.array(tangent_training_losses),
              "records": np.array(records), "kernel_snapshots": np.array(snapshots),
              "initial_kernel_eigenvalues": eigenvalues.numpy(),
              "initial_residual_modes": residual_modes.numpy(),
              "initial_jacobian_train": j_train.numpy(), "initial_jacobian_test": j_test.numpy(),
              "train_x": train_x.numpy(), "train_y": train_y.numpy(),
              "test_x": test_x.numpy(), "test_y": test_y.numpy(),
              "initial_train_output": initial_train.numpy(), "initial_test_output": initial_test.numpy(),
              "final_true_train_output": true_train.detach().numpy(),
              "final_tangent_train_output": linear_train.detach().numpy(),
              "final_true_test_output": true_test.detach().numpy(),
              "final_tangent_test_output": linear_test.detach().numpy(),
              "initial_parameters": theta0.numpy(),
              "parameter_shapes": np.array([parameter.numel() for parameter in parameters])}
    if predicted_modes:
        result["predicted_tangent_train_half_mse"] = np.square(np.array(predicted_modes)).sum(axis=1) / (2 * len(train_x))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if not json.loads((PROGRAM / "state/current.json").read_text()).get("first_chain_complete"):
        raise SystemExit("First chain incomplete")
    registration = json.loads((HERE / "preregistration.json").read_text())
    if sha(Path(__file__)) != registration["run_source_sha256"] or sha(HERE / "executed/experiment.py") != registration["model_source_sha256"]:
        raise SystemExit("Execution pin differs from preregistration")
    if args.phase == "ood" and not (HERE / "ood_predictions.json").exists():
        raise SystemExit("Must save predictions before sealed OOD training")
    spec = importlib.util.spec_from_file_location("B02_archived_experiment", HERE / "executed/experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lock = (HERE / "worker.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    conditions = registration[args.phase + "_conditions"]
    recipes = registration["recipes"]
    total = len(conditions) * len(recipes) * len(registration["widths"]) * len(registration["seeds"])
    completed = resumed = new_cells = 0
    started = time.time()
    output = HERE / "results" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        for width in registration["widths"]:
            for seed in registration["seeds"]:
                for recipe in recipes:
                    request = {"condition": condition, "width": width, "seed": seed, "recipe": recipe,
                               "steps": registration["steps"], "preregistration_sha256": sha(HERE / "preregistration.json"),
                               "run_source_sha256": registration["run_source_sha256"],
                               "model_source_sha256": registration["model_source_sha256"]}
                    cell_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
                    path = output / (cell_id + ".json")
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if previous["request"] != request or previous["arrays_sha256"] != sha(path.with_suffix(".npz")):
                            raise RuntimeError("Existing success contract mismatch")
                        if previous["status"] != "success":
                            raise RuntimeError("Previous failure needs inspection")
                        completed += 1
                        resumed += 1
                        continue
                    identity = {"pid": os.getpid(), "started_at": started, "source_sha256": sha(Path(__file__)),
                                "phase": args.phase, "current_cell": cell_id, "condition": condition["name"],
                                "completed": completed, "resumed_successes": resumed, "total": total, "status": "running",
                                "elapsed_seconds": time.time() - started, "updated_at": time.time()}
                    save(HERE / "current.json", identity)
                    direction = json.loads((DIRECTION / "state.json").read_text())
                    direction.update({"status": "B02_running", "process": identity, "last_completed": f"B02 {args.phase} {completed}/{total}",
                                      "next_action": "Compare true and tangent MLP curves, then save OOD predictions before execution"})
                    save(DIRECTION / "state.json", direction)
                    cell_started = time.time()
                    arrays = execute(condition, width, seed, recipe, registration["steps"], module.Model)
                    if not all(np.isfinite(value).all() for value in arrays.values()):
                        raise RuntimeError("Nonfinite experiment result")
                    stream = io.BytesIO()
                    np.savez_compressed(stream, **arrays)
                    temporary = path.with_suffix(".npz.tmp")
                    temporary.write_bytes(stream.getvalue())
                    temporary.replace(path.with_suffix(".npz"))
                    metadata = {"cell_id": cell_id, "request": request, "status": "success",
                                "arrays_sha256": sha(path.with_suffix(".npz")), "seconds": time.time() - cell_started,
                                "saved_at": time.time(), "final_train_half_mse": float(arrays["train_half_mse"][-1]),
                                "final_tangent_train_half_mse": float(arrays["tangent_train_half_mse"][-1])}
                    save(path, metadata)
                    completed += 1
                    new_cells += 1
                    print(json.dumps({"cell": cell_id, "condition": condition["name"], "width": width, "seed": seed,
                                      "recipe": recipe["name"], "completed": completed, "total": total,
                                      "seconds": metadata["seconds"], "true_loss": metadata["final_train_half_mse"],
                                      "tangent_loss": metadata["final_tangent_train_half_mse"]}), flush=True)
                    if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                        identity.update({"status": "checkpointed", "completed": completed, "process": None})
                        save(HERE / "current.json", identity)
                        return
    save(HERE / "current.json", {"phase": args.phase, "status": "completed", "completed": completed,
                                    "total": total, "resumed_successes": resumed, "elapsed_seconds": time.time() - started,
                                    "updated_at": time.time(), "process": None})


if __name__ == "__main__":
    main()
