from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load_cells(phase):
    cells = []
    for path in sorted((HERE / "results" / phase).glob("*.json")):
        metadata = json.loads(path.read_text())
        arrays_path = path.with_suffix(".npz")
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != metadata["arrays_sha256"]:
            raise RuntimeError(f"Evidence hash differs: {path.name}")
        with np.load(arrays_path) as data:
            arrays = {key: data[key] for key in data.files}
        cells.append((metadata, arrays))
    return cells


def saturation(loss, fraction=0.01):
    below = np.maximum.accumulate(loss[::-1])[::-1] <= fraction * loss[0]
    crossings = np.where(below)[0]
    return int(crossings[0]) if len(crossings) else None


def analyze(phase):
    preregistration = json.loads((HERE / "preregistration.json").read_text())
    cells = load_cells(phase)
    expected = len(preregistration[phase + "_conditions"]) * len(preregistration["coordinates"]) * len(preregistration["recipes"])
    if len(cells) != expected or any(cell[0]["status"] != "success" for cell in cells):
        raise RuntimeError(f"Study incomplete: {len(cells)}/{expected}")
    rows, groups = [], {}
    for metadata, arrays in cells:
        request = metadata["request"]
        condition = request["condition"]
        recipe = request["recipe"]
        eigenvalues, target = arrays["eigenvalues"], arrays["target_modes"]
        nominal_modes = -np.exp(-arrays["nominal_time"][:, None] * eigenvalues[None]) * target[None]
        startup_modes = -np.exp(-arrays["startup_time"][:, None] * eigenvalues[None]) * target[None]
        nominal_loss = np.square(nominal_modes).sum(axis=1) / 2
        startup_loss = np.square(startup_modes).sum(axis=1) / 2
        row = {"cell_id": metadata["cell_id"], "condition": condition["name"],
               "coordinate": request["coordinate"], "recipe": recipe["name"],
               "optimizer": recipe["optimizer"], "dimension": condition["dimension"],
               "alignment": condition["alignment"], "initial_loss": float(arrays["loss"][0]),
               "final_loss": float(arrays["loss"][-1]), "minimum_loss": float(arrays["loss"].min()),
               "saturation_step": saturation(arrays["loss"]),
               "nominal_loss_rmse": float(np.sqrt(np.mean(np.square(arrays["loss"] - nominal_loss)))),
               "startup_loss_rmse": float(np.sqrt(np.mean(np.square(arrays["loss"] - startup_loss)))),
               "first_target_mode_decrease": float(1.0 - abs(arrays["mode_errors"][1, np.argmax(target)])),
               "first_step_prediction_error": metadata["first_step_prediction_max_error"],
               "recurrence_error": metadata.get("recurrence_max_error"), "seconds": metadata["seconds"]}
        if "predicted_mode_errors" in arrays:
            predicted_loss = np.square(arrays["predicted_mode_errors"]).sum(axis=1) / 2
            row["recurrence_loss_rmse"] = float(np.sqrt(np.mean(np.square(arrays["loss"] - predicted_loss))))
        rows.append(row)
        groups.setdefault((condition["name"], recipe["name"]), {})[request["coordinate"]] = (row, arrays)
    rotation_checks = []
    for (condition, recipe), coordinates in sorted(groups.items()):
        diagonal_row, diagonal = coordinates["diagonal"]
        for name, (row, arrays) in coordinates.items():
            if name == "diagonal":
                continue
            rotation_checks.append({"condition": condition, "recipe": recipe, "optimizer": row["optimizer"],
                                    "coordinate": name,
                                    "loss_trajectory_max_difference": float(np.max(np.abs(arrays["loss"] - diagonal["loss"]))),
                                    "final_loss_difference": row["final_loss"] - diagonal_row["final_loss"],
                                    "first_target_mode_decrease_ratio": row["first_target_mode_decrease"] / diagonal_row["first_target_mode_decrease"]})
    sgd_rows = [row for row in rows if row["optimizer"] == "SGD"]
    sgd_rotations = [row for row in rotation_checks if row["optimizer"] == "SGD"]
    long_startup = [row for row in rows if row["recipe"] == "momentum_long_startup" and row["alignment"] == "fast"]
    predictions = {
        "P1": {"max_mode_error": max(row["recurrence_error"] for row in sgd_rows),
                 "pass": all(row["recurrence_error"] < 1e-10 for row in sgd_rows)},
        "P2": {"max_loss_rotation_difference": max(row["loss_trajectory_max_difference"] for row in sgd_rotations),
                 "pass": all(row["loss_trajectory_max_difference"] < 1e-10 for row in sgd_rotations)},
        "P3": {"max_first_step_error": max(row["first_step_prediction_error"] for row in rows if row["optimizer"] == "Adam"),
                 "pass": all(row["first_step_prediction_error"] < 1e-10 for row in rows if row["optimizer"] == "Adam")},
        "P6": {"pass": all(row["recurrence_loss_rmse"] < row["nominal_loss_rmse"] for row in long_startup),
                 "nominal_rmse_range": [min(row["nominal_loss_rmse"] for row in long_startup), max(row["nominal_loss_rmse"] for row in long_startup)]}}
    small = [row for row in rows if row["recipe"] in ("sgd_small", "momentum_small")]
    predictions["P5"] = {"pass": all((row["saturation_step"] is not None) == (row["alignment"] == "fast") for row in small),
                            "unit_count": len(small)}
    if phase == "ood":
        checks = [row for row in rotation_checks if row["recipe"] == "adam" and row["condition"] == "geometric_slow_d16"]
        predictions["P4"] = {"pass": all(row["first_target_mode_decrease_ratio"] > 1 for row in checks),
                                "ratios": [row["first_target_mode_decrease_ratio"] for row in checks]}
    fieldnames = list(rows[0])
    if "recurrence_loss_rmse" not in fieldnames:
        fieldnames.append("recurrence_loss_rmse")
    with (HERE / (phase + "_cells.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"phase": phase, "cell_count": len(rows), "predictions": predictions,
               "execution_seconds": sum(row["seconds"] for row in rows),
               "source_sha256": preregistration["execution_source_sha256"],
               "preregistration_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
               "rotation_checks": rotation_checks, "cells": rows}
    (HERE / (phase + "_analysis.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10))
    plot(phase, groups)
    print(json.dumps({"phase": phase, "n": len(rows), "predictions": predictions,
                      "training_seconds": summary["execution_seconds"]}, ensure_ascii=False, indent=2))


def plot(phase, groups):
    dimension = "d8" if phase == "development" else "d16"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), layout="constrained")
    for alignment in ("fast", "slow"):
        condition = f"geometric_{alignment}_{dimension}"
        for recipe in ("sgd_small", "momentum_small"):
            row, arrays = groups[(condition, recipe)]["diagonal"]
            axes[0].semilogy(arrays["nominal_time"], np.maximum(arrays["loss"] / arrays["loss"][0], 1e-12),
                             label=f"{alignment} / {recipe}")
    condition = f"geometric_fast_{dimension}"
    row, arrays = groups[(condition, "momentum_long_startup")]["diagonal"]
    step = np.arange(161)
    axes[1].plot(step, arrays["loss"][:161], label="measured", linewidth=2)
    axes[1].plot(step, np.square(arrays["predicted_mode_errors"][:161]).sum(axis=1) / 2, label="exact recurrence", linestyle="--")
    lam = arrays["eigenvalues"][0]
    axes[1].plot(step, arrays["loss"][0] * np.exp(-2 * lam * arrays["nominal_time"][:161]), label="nominal U", linestyle=":")
    condition = f"geometric_slow_{dimension}"
    for coordinate, (row, arrays) in groups[(condition, "adam")].items():
        axes[2].semilogy(np.arange(len(arrays["loss"])), np.maximum(arrays["loss"], 1e-12), label=coordinate)
    axes[0].set(xlabel="nominal U", ylabel="relative half MSE", title="Target alignment changes fit time")
    axes[1].set(xlabel="step", ylabel="half MSE", title="Momentum startup and oscillation")
    axes[2].set(xlabel="step", ylabel="half MSE", title="Adam depends on parameter coordinates")
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.legend(fontsize=6.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / (phase + "_dynamics.png"), dpi=180)
    fig.savefig(HERE / (phase + "_dynamics.svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    analyze(parser.parse_args().phase)
