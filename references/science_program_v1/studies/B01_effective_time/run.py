from __future__ import annotations

import argparse
import fcntl
import hashlib
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + chr(10))
    temporary.replace(path)


def design(condition, coordinate):
    dimension = condition["dimension"]
    if condition["spectrum"] == "geometric":
        eigenvalues = np.geomspace(1.0, condition["minimum_eigenvalue"], dimension)
    else:
        eigenvalues = np.array(condition["eigenvalues"], dtype=np.float64)
    weights = np.zeros(dimension)
    weights[0 if condition["alignment"] == "fast" else -1] = 1.0
    if coordinate == "diagonal":
        rotation = np.eye(dimension)
    else:
        rng = np.random.default_rng(int(coordinate.split("_")[-1]))
        rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    x = np.diag(np.sqrt(dimension * eigenvalues)) @ rotation.T
    target = rotation @ (weights / np.sqrt(eigenvalues))
    return eigenvalues, weights, rotation, x, target


def execute(condition, coordinate, recipe, steps):
    eigenvalues, weights, rotation, x, target = design(condition, coordinate)
    features = torch.from_numpy(x)
    target_parameter = torch.from_numpy(target)
    targets = features @ target_parameter
    parameter = torch.nn.Parameter(torch.zeros_like(target_parameter))
    eta = recipe["lr"]
    momentum = recipe.get("momentum", 0.0)
    if recipe["optimizer"] == "SGD":
        optimizer = torch.optim.SGD([parameter], lr=eta, momentum=momentum)
    else:
        optimizer = torch.optim.Adam([parameter], lr=eta, betas=tuple(recipe["betas"]), eps=recipe["eps"])
    coefficients = [-weights.copy()]
    predictions = [-weights.copy()] if recipe["optimizer"] == "SGD" else []
    losses = [float(torch.mean(targets.square()).item() / 2)]
    step_norms, gradient_norms = [], []
    for step in range(1, steps + 1):
        previous = parameter.detach().clone()
        loss = torch.mean((features @ parameter - targets).square()) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norms.append(float(parameter.grad.norm().item()))
        optimizer.step()
        value = parameter.detach().numpy()
        coefficients.append(np.sqrt(eigenvalues) * (rotation.T @ (value - target)))
        losses.append(float(torch.mean((features @ parameter - targets).square()).item() / 2))
        delta = parameter.detach().numpy() - previous.numpy()
        step_norms.append(float(np.linalg.norm(np.sqrt(eigenvalues) * (rotation.T @ delta))))
        if predictions:
            if step == 1:
                next_error = (1 - eta * eigenvalues) * predictions[-1]
            else:
                next_error = (1 + momentum - eta * eigenvalues) * predictions[-1] - momentum * predictions[-2]
            predictions.append(next_error)
        if not np.isfinite(losses[-1]):
            break
    trajectory = np.array(coefficients)
    indices = np.arange(len(losses))
    nominal_time = eta * indices / (1 - momentum)
    startup_time = eta * (indices / (1 - momentum) - momentum * (1 - momentum ** indices) / (1 - momentum) ** 2)
    initial_gradient = -(rotation @ (np.sqrt(eigenvalues) * weights))
    if recipe["optimizer"] == "Adam":
        first_parameter_step = -eta * initial_gradient / (np.abs(initial_gradient) + recipe["eps"])
    else:
        first_parameter_step = -eta * initial_gradient
    predicted_first_mode = -weights + np.sqrt(eigenvalues) * (rotation.T @ first_parameter_step)
    arrays = {"mode_errors": trajectory, "loss": np.array(losses),
              "nominal_time": nominal_time, "startup_time": startup_time,
              "function_step_norm": np.array(step_norms), "gradient_norm": np.array(gradient_norms),
              "eigenvalues": eigenvalues, "target_modes": weights, "rotation": rotation,
              "features": x, "targets": targets.numpy(),
              "predicted_first_mode": predicted_first_mode}
    if predictions:
        arrays["predicted_mode_errors"] = np.array(predictions)
    return arrays


def state_update(phase, completed, total, started_at, status, resumed):
    now = time.time()
    value = {"study": "B01_effective_time", "phase": phase, "status": status,
             "updated_at": now, "started_at": started_at, "completed": completed,
             "total": total, "resumed_successes": resumed, "elapsed_seconds": now - started_at,
             "process": {"pid": os.getpid(), "source_sha256": digest(Path(__file__))} if status == "running" else None}
    save_json(HERE / "current.json", value)
    direction = json.loads((DIRECTION / "state.json").read_text())
    direction.update({"status": status, "updated_at": now, "last_completed": f"{phase}: {completed}/{total} saved cells",
                      "process": value["process"],
                      "next_action": "Analyze exact recurrence and coordinate effects; use saved success cells on resume"})
    direction.setdefault("studies", {})["B01"] = value
    save_json(DIRECTION / "state.json", direction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    gate = json.loads((PROGRAM / "state/current.json").read_text())
    if not gate.get("first_chain_complete"):
        raise SystemExit("First chain is not complete; new B training is not authorized yet")
    preregistration = json.loads((HERE / "preregistration.json").read_text())
    source_sha256 = digest(Path(__file__))
    if source_sha256 != preregistration["execution_source_sha256"]:
        raise SystemExit("Execution source changed after preregistration")
    if args.phase == "ood" and not (HERE / "development_analysis.json").exists():
        raise SystemExit("Save development analysis before executing sealed OOD")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = HERE / "results" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    lock = (HERE / "worker.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    conditions = preregistration[args.phase + "_conditions"]
    total = len(conditions) * len(preregistration["coordinates"]) * len(preregistration["recipes"])
    started_at = time.time()
    completed = resumed = new_cells = 0
    state_update(args.phase, completed, total, started_at, "running", resumed)
    for condition in conditions:
        for coordinate in preregistration["coordinates"]:
            for recipe in preregistration["recipes"]:
                request = {"condition": condition, "coordinate": coordinate, "recipe": recipe,
                           "steps": preregistration["steps"], "source_sha256": source_sha256,
                           "preregistration_sha256": digest(HERE / "preregistration.json")}
                cell_id = hashlib.sha256(canonical(request)).hexdigest()[:16]
                metadata_path = output / (cell_id + ".json")
                arrays_path = output / (cell_id + ".npz")
                if metadata_path.exists():
                    saved = json.loads(metadata_path.read_text())
                    if saved["request"] != request or saved["arrays_sha256"] != digest(arrays_path):
                        raise RuntimeError(f"Saved cell pin mismatch: {cell_id}")
                    if saved["status"] != "success":
                        raise RuntimeError(f"Cell previously failed; inspect rather than rerun: {cell_id}")
                    completed += 1
                    resumed += 1
                    continue
                cell_started = time.time()
                arrays = execute(condition, coordinate, recipe, preregistration["steps"])
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                temporary = arrays_path.with_suffix(".npz.tmp")
                temporary.write_bytes(stream.getvalue())
                temporary.replace(arrays_path)
                finite = bool(np.isfinite(arrays["loss"]).all())
                status = "success" if finite and len(arrays["loss"]) == preregistration["steps"] + 1 else "failed"
                summary = {"cell_id": cell_id, "request": request, "status": status,
                           "started_at": cell_started, "finished_at": time.time(),
                           "seconds": time.time() - cell_started, "arrays_sha256": digest(arrays_path),
                           "initial_half_mse": float(arrays["loss"][0]), "final_half_mse": float(arrays["loss"][-1]),
                           "first_step_prediction_max_error": float(np.max(np.abs(arrays["mode_errors"][1] - arrays["predicted_first_mode"]))) }
                if "predicted_mode_errors" in arrays:
                    summary["recurrence_max_error"] = float(np.max(np.abs(arrays["mode_errors"] - arrays["predicted_mode_errors"])))
                save_json(metadata_path, summary)
                completed += 1
                new_cells += 1
                state_update(args.phase, completed, total, started_at, "running", resumed)
                print(json.dumps({"cell": cell_id, "completed": completed, "total": total,
                                  "final_loss": summary["final_half_mse"], "seconds": summary["seconds"]}), flush=True)
                if status != "success":
                    state_update(args.phase, completed, total, started_at, "failed", resumed)
                    raise SystemExit("Nonfinite or truncated result retained")
                if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                    state_update(args.phase, completed, total, started_at, "checkpointed", resumed)
                    return
    state_update(args.phase, completed, total, started_at, "completed", resumed)


if __name__ == "__main__":
    main()
