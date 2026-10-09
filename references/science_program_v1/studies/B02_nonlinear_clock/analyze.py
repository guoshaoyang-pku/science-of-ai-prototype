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
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def read(phase):
    cells = []
    for path in sorted((HERE / "results" / phase).glob("*.json")):
        metadata = json.loads(path.read_text())
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != metadata["arrays_sha256"]:
            raise RuntimeError("Saved evidence hash differs")
        with np.load(path.with_suffix(".npz")) as value:
            arrays = {key: value[key] for key in value.files}
        cells.append((metadata, arrays))
    return cells


def analysis(phase):
    registration = json.loads((HERE / "preregistration.json").read_text())
    cells = read(phase)
    expected = len(registration[phase + "_conditions"]) * len(registration["seeds"]) * len(registration["widths"]) * len(registration["recipes"])
    if len(cells) != expected:
        raise RuntimeError(f"Incomplete study: {len(cells)}/{expected}")
    rows, groups = [], {}
    for metadata, arrays in cells:
        request = metadata["request"]
        true = arrays["train_half_mse"]
        tangent = arrays["tangent_train_half_mse"]
        initial = true[0]
        eigenvalues, modes = arrays["initial_kernel_eigenvalues"], arrays["initial_residual_modes"]
        rayleigh = float(np.sum(eigenvalues * modes ** 2) / np.sum(modes ** 2))
        record = arrays["records"]
        kernel = arrays["kernel_snapshots"]
        row = {"cell_id": metadata["cell_id"], "condition": request["condition"]["name"],
               "width": request["width"], "seed": request["seed"], "recipe": request["recipe"]["name"],
               "optimizer": request["recipe"]["optimizer"], "initial_loss": float(initial),
               "first_step_relative_error": float(abs(true[1] - tangent[1]) / initial),
               "final_true_train_loss": float(true[-1]), "final_tangent_train_loss": float(tangent[-1]),
               "final_true_test_loss": float(record[-1, 3]), "final_tangent_test_loss": float(record[-1, 4]),
               "final_train_difference_relative_to_initial": float((tangent[-1] - true[-1]) / initial),
               "trajectory_relative_rmse": float(np.sqrt(np.mean((true - tangent) ** 2)) / initial),
               "linearization_defect_relative_motion": float(record[-1, 5] / max(record[-1, 6], 1e-15)),
               "final_jacobian_relative_drift": float(kernel[-1, 1]),
               "final_kernel_relative_drift": float(kernel[-1, 2]),
               "initial_max_eigenvalue": float(eigenvalues[-1]), "initial_target_rayleigh": rayleigh,
               "initial_effective_rank": float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)),
               "kernel_trace": float(np.sum(eigenvalues)), "parameter_count": int(sum(arrays["parameter_shapes"])),
               "recurrence_error": None, "seconds": metadata["seconds"]}
        if "predicted_tangent_train_half_mse" in arrays:
            row["recurrence_error"] = float(np.max(np.abs(arrays["predicted_tangent_train_half_mse"] - tangent)))
        rows.append(row)
        groups.setdefault((row["condition"], row["width"], row["recipe"]), []).append(row)
    summaries = []
    metrics = [name for name in rows[0] if name not in ("cell_id", "condition", "width", "seed", "recipe", "optimizer", "recurrence_error")]
    for (condition, width, recipe), values in sorted(groups.items()):
        means = {metric: float(np.mean([value[metric] for value in values])) for metric in metrics}
        errors = {metric: float(np.std([value[metric] for value in values], ddof=1) / np.sqrt(len(values))) for metric in metrics}
        summaries.append({"condition": condition, "width": width, "recipe": recipe, "n_seeds": len(values), "mean": means, "se": errors})
    predictions = {
        "B02_P1": {"pass": all(row["recurrence_error"] < 1e-9 for row in rows if row["recurrence_error"] is not None),
                     "max_loss_error": max(row["recurrence_error"] for row in rows if row["recurrence_error"] is not None)},
        "B02_P2": {"pass": float(np.mean([row["first_step_relative_error"] <= .001 for row in rows if row["optimizer"] == "SGD"])) >= .9,
                     "fraction_within_threshold": float(np.mean([row["first_step_relative_error"] <= .001 for row in rows if row["optimizer"] == "SGD"])),
                     "max_first_step_relative_error": max(row["first_step_relative_error"] for row in rows if row["optimizer"] == "SGD")},
        "B02_P3": {"pass": any(abs(row["mean"]["final_train_difference_relative_to_initial"]) > .05 for row in summaries),
                     "units_above_threshold": sum(abs(row["mean"]["final_train_difference_relative_to_initial"]) > .05 for row in summaries),
                     "total_units": len(summaries)}}
    drift = [row["final_kernel_relative_drift"] for row in rows]
    defects = [row["trajectory_relative_rmse"] for row in rows]
    correlation = spearmanr(drift, defects)
    result = {"phase": phase, "paired_cells": len(rows), "trained_models": 2 * len(rows),
              "execution_seconds": sum(row["seconds"] for row in rows), "predictions": predictions,
              "kernel_drift_vs_curve_error_spearman": {"rho": float(correlation.statistic), "p_descriptive": float(correlation.pvalue),
                                                         "scope": "post hoc across cells, not independent evidence"},
              "groups": summaries, "cells": rows}
    if phase == "ood":
        predictions_saved = json.loads((HERE / "ood_predictions.json").read_text())
        checks = []
        lookup = {(row["condition"], row["width"], row["recipe"]): row for row in summaries}
        for prediction in predictions_saved["predictions"]:
            group = lookup[(prediction["condition"], prediction["width"], prediction["recipe"])]
            observed = group["mean"]["final_train_difference_relative_to_initial"]
            success = observed >= prediction["minimum_gain_relative_initial"]
            checks.append({**prediction, "observed_gain_relative_initial": observed, "pass": success,
                           "observed_true_test_loss": group["mean"]["final_true_test_loss"],
                           "observed_tangent_test_loss": group["mean"]["final_tangent_test_loss"]})
        result["sealed_ood_checks"] = checks
    with (HERE / (phase + "_cells.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / (phase + "_analysis.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    plot(phase, cells)
    print(json.dumps({key: value for key, value in result.items() if key not in ("groups", "cells")}, ensure_ascii=False, indent=2))
    print("Group means: true/tangent train; true/tangent test; kernel drift")
    for row in summaries:
        means = row["mean"]
        print(row["condition"], row["width"], row["recipe"],
              *[round(means[k], 6) for k in ("final_true_train_loss", "final_tangent_train_loss",
                                           "final_true_test_loss", "final_tangent_test_loss", "final_kernel_relative_drift")])


def plot(phase, cells):
    conditions = sorted({metadata["request"]["condition"]["name"] for metadata, _ in cells})
    fig, axes = plt.subplots(len(conditions), 2, figsize=(9.0, 3.4 * len(conditions)), layout="constrained", squeeze=False)
    for index, condition in enumerate(conditions):
        for width in (8, 32):
            matched = [arrays for metadata, arrays in cells if metadata["request"]["condition"]["name"] == condition
                       and metadata["request"]["width"] == width and metadata["request"]["recipe"]["name"] == "sgd_momentum"]
            for field, style in (("train_half_mse", "-"), ("tangent_train_half_mse", "--")):
                values = np.stack([arrays[field] for arrays in matched])
                mean = values.mean(axis=0)
                axes[index, 0].semilogy(np.arange(len(mean)), mean, linestyle=style, label=f"w{width} {'true' if style == '-' else 'tangent'}")
        for recipe in ("sgd_plain", "sgd_momentum", "adam"):
            matched = [arrays for metadata, arrays in cells if metadata["request"]["condition"]["name"] == condition
                       and metadata["request"]["width"] == 32 and metadata["request"]["recipe"]["name"] == recipe]
            for column, style in ((3, "-"), (4, "--")):
                values = np.stack([arrays["records"][:, column] for arrays in matched])
                axes[index, 1].semilogy(matched[0]["records"][:, 0], values.mean(axis=0), linestyle=style, label=f"{recipe} {'true' if column == 3 else 'tangent'}")
        axes[index, 0].set(title=f"{condition}: train / momentum SGD", xlabel="step", ylabel="half MSE")
        axes[index, 1].set(title=f"{condition}: test / width32", xlabel="step", ylabel="half MSE")
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.title.set_fontsize(10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / (phase + "_curves.png"), dpi=180)
    fig.savefig(HERE / (phase + "_curves.svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    analysis(parser.parse_args().phase)
