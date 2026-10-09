from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROGRAM = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(study, phase):
    for path in sorted((study / "results" / phase).glob("*.json")):
        metadata = json.loads(path.read_text())
        arrays_path = path.with_suffix(".npz")
        assert sha(arrays_path) == metadata["arrays_sha256"]
        assert metadata["status"] == "success"
        with np.load(arrays_path) as value:
            arrays = {key: value[key] for key in value.files}
        assert all(np.isfinite(value).all() for value in arrays.values())
        yield metadata, arrays


def loss_curve(kernel, residual, eta, momentum, steps):
    current = residual.copy()
    previous = residual.copy()
    values = [np.mean(current ** 2) / 2]
    for step in range(steps):
        projected = np.einsum("ij,j->i", kernel, current)
        next_value = current - eta * projected if step == 0 else (1 + momentum) * current - eta * projected - momentum * previous
        previous, current = current, next_value
        values.append(np.mean(current ** 2) / 2)
    return np.array(values)


def main():
    summary = {"B01": {}, "B02": {}, "B03": {}}
    studies = {
        "B01": PROGRAM / "studies/B01_effective_time",
        "B02": PROGRAM / "studies/B02_nonlinear_clock",
        "B03": PROGRAM / "studies/B03_kernel_direction",
    }
    for name, study in studies.items():
        registration = json.loads((study / "preregistration.json").read_text())
        source_pin = registration.get("run_source_sha256", registration.get("execution_source_sha256"))
        assert sha(study / "run.py") == source_pin
        if name != "B01":
            assert sha(study / "executed/experiment.py") == registration["model_source_sha256"]
        for phase in ("development", "ood"):
            count, max_error = 0, 0.0
            starts, finishes = [], []
            prediction_time = registration["created_at"] if name == "B01" else json.loads((study / "ood_predictions.json").read_text())["created_at"]
            for metadata, arrays in evidence(study, phase):
                count += 1
                start = metadata.get("started_at", metadata.get("saved_at", 0) - metadata["seconds"])
                finish = metadata.get("finished_at", metadata.get("saved_at", 0))
                starts.append(start)
                finishes.append(finish)
                if phase == "ood":
                    assert start > prediction_time
                if name == "B01":
                    request = metadata["request"]
                    rotation, eigenvalues = arrays["rotation"], arrays["eigenvalues"]
                    assert np.max(np.abs(np.einsum("ki,kj->ij", rotation, rotation) - np.eye(len(rotation)))) < 1e-12
                    spectrum = np.linalg.eigvalsh(np.einsum("ki,kj->ij", arrays["features"], arrays["features"]) / len(rotation))
                    assert np.max(np.abs(spectrum - np.sort(eigenvalues))) < 1e-12
                    if request["recipe"]["optimizer"] == "SGD":
                        expected = loss_curve(np.diag(eigenvalues), -arrays["target_modes"], request["recipe"]["lr"],
                                              request["recipe"].get("momentum", 0), request["steps"])
                        error = float(np.max(np.abs(expected * len(eigenvalues) - arrays["loss"])))
                        max_error = max(max_error, error)
                        assert error < 1e-9
                if name == "B02":
                    j0 = arrays["initial_jacobian_train"]
                    kernel = np.einsum("ip,jp->ij", j0, j0) / len(j0)
                    assert np.max(np.abs(np.linalg.eigvalsh(kernel) - arrays["initial_kernel_eigenvalues"])) < 1e-9
                    residual = arrays["initial_train_output"] - arrays["train_y"]
                    assert abs(np.mean(residual ** 2) / 2 - arrays["train_half_mse"][0]) < 1e-12
                    recipe = metadata["request"]["recipe"]
                    if recipe["optimizer"] == "SGD":
                        expected = loss_curve(kernel, residual, recipe["lr"], recipe["momentum"], metadata["request"]["steps"])
                        error = float(np.max(np.abs(expected - arrays["tangent_train_half_mse"])))
                        max_error = max(max_error, error)
                        assert error < 1e-9
                if name == "B03":
                    j0, j1 = arrays["initial_jacobian"], arrays["learned_jacobian"]
                    k0 = np.einsum("ip,jp->ij", j0, j0) / len(j0)
                    k1 = np.einsum("ip,jp->ij", j1, j1) / len(j1)
                    assert np.max(np.abs(k0 - arrays["initial_kernel"])) < 1e-9
                    assert np.max(np.abs(k1 - arrays["learned_kernel"])) < 1e-9
                    scale = float(np.trace(k1) / np.trace(k0))
                    assert abs(scale - float(arrays["scale"])) < 1e-12
                    residual = arrays["probe_train_residual"]
                    eta = float(arrays["probe_lr"])
                    for label, kernel in (("initial", k0), ("initial_equal_trace", scale * k0), ("learned", k1)):
                        expected = loss_curve(kernel, residual, eta, 0, registration["probe_steps"])
                        error = float(np.max(np.abs(expected - arrays[label + "_probe_train_loss"])))
                        max_error = max(max_error, error)
                        assert error < 1e-9
            if name == "B01":
                expected = len(registration[phase + "_conditions"]) * len(registration["recipes"]) * len(registration["coordinates"])
            elif name == "B02":
                expected = len(registration[phase + "_conditions"]) * len(registration["recipes"]) * len(registration["widths"]) * len(registration["seeds"])
            else:
                expected = len(registration[phase + "_conditions"]) * len(registration[phase + "_widths"]) * len(registration["seeds"])
            assert count == expected
            summary[name][phase] = {"saved_cells": count, "max_independent_loss_prediction_error": max_error,
                                   "first_cell_started_at": min(starts), "last_cell_finished_at": max(finishes),
                                   "evidence_elapsed_seconds": max(finishes) - min(starts),
                                   "prediction_saved_at": prediction_time if phase == "ood" else registration["created_at"]}
    result = {"status": "pass", "training_executed": False, "model_calls": 0, "studies": summary}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
