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


def read(phase):
    cells = []
    for path in sorted((HERE / "results" / phase).glob("*.json")):
        metadata = json.loads(path.read_text())
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != metadata["arrays_sha256"]:
            raise RuntimeError("Saved evidence hash differs")
        with np.load(path.with_suffix(".npz")) as value:
            cells.append((metadata, {key: value[key] for key in value.files}))
    return cells


def main(phase):
    registration = json.loads((HERE / "preregistration.json").read_text())
    cells = read(phase)
    expected = len(registration[phase + "_conditions"]) * len(registration[phase + "_widths"]) * len(registration["seeds"])
    if len(cells) != expected:
        raise RuntimeError(f"Incomplete study: {len(cells)}/{expected}")
    rows, groups = [], {}
    for metadata, arrays in cells:
        request = metadata["request"]
        ratios = arrays["rayleigh_ratios"]
        row = {"cell_id": metadata["cell_id"], "condition": request["condition"]["name"],
               "width": request["width"], "seed": request["seed"], "trace_scale": float(arrays["scale"]),
               "trace_match_error": float(arrays["trace_match_error"]),
               "target_rayleigh_ratio": float(ratios[0]),
               "permutation_ratio_mean": float(ratios[1:].mean()),
               "target_above_permutation_fraction": float(np.mean(ratios[0] > ratios[1:])),
               "target_to_permutation_mean": float(ratios[0] / ratios[1:].mean()),
               "learned_probe_final_relative_loss": float(arrays["learned_probe_train_loss"][-1] / arrays["learned_probe_train_loss"][0]),
               "equaltrace_initial_probe_final_relative_loss": float(arrays["initial_equal_trace_probe_train_loss"][-1] / arrays["initial_equal_trace_probe_train_loss"][0]),
               "unscaled_initial_probe_final_relative_loss": float(arrays["initial_probe_train_loss"][-1] / arrays["initial_probe_train_loss"][0]),
               "learned_probe_final_test_loss": float(arrays["learned_probe_test_loss"][-1]),
               "equaltrace_initial_probe_final_test_loss": float(arrays["initial_equal_trace_probe_test_loss"][-1]),
               "nonlinear_pretrain_final_loss": float(arrays["pretrain_loss"][-1]),
               "seconds": metadata["seconds"]}
        row["equal_trace_probe_gain_relative_initial_residual"] = row["equaltrace_initial_probe_final_relative_loss"] - row["learned_probe_final_relative_loss"]
        rows.append(row)
        groups.setdefault((row["condition"], row["width"]), []).append(row)
    summaries = []
    metrics = [name for name in rows[0] if name not in ("cell_id", "condition", "width", "seed")]
    for (condition, width), values in sorted(groups.items()):
        means = {metric: float(np.mean([value[metric] for value in values])) for metric in metrics}
        errors = {metric: float(np.std([value[metric] for value in values], ddof=1) / np.sqrt(len(values))) for metric in metrics}
        summaries.append({"condition": condition, "width": width, "n_seeds": len(values), "mean": means, "se": errors})
    wider = max(registration[phase + "_widths"])
    wide = [group for group in summaries if group["width"] == wider]
    predictions = {
        "B03_P1": {"pass": any(abs(group["mean"]["equal_trace_probe_gain_relative_initial_residual"]) > .05 for group in summaries),
                     "group_gains": [group["mean"]["equal_trace_probe_gain_relative_initial_residual"] for group in summaries]},
        "B03_P2": {"pass": all(group["mean"]["target_rayleigh_ratio"] > 1 for group in wide),
                     "ratios": [group["mean"]["target_rayleigh_ratio"] for group in wide],
                     "scope": "development claim only; OOD values descriptive unless sealed"},
        "B03_P3": {"pass": all(group["mean"]["target_to_permutation_mean"] > 1 for group in wide),
                     "ratios": [group["mean"]["target_to_permutation_mean"] for group in wide],
                     "scope": "development claim only; OOD values descriptive unless sealed"}}
    summary = {"phase": phase, "pretrained_models": len(cells), "linear_probe_trajectories": 3 * len(cells),
               "execution_seconds": sum(row["seconds"] for row in rows), "predictions": predictions,
               "trace_match_max_error": max(row["trace_match_error"] for row in rows),
               "groups": summaries, "cells": rows}
    if phase == "ood":
        saved = json.loads((HERE / "ood_predictions.json").read_text())
        lookup = {(group["condition"], group["width"]): group for group in summaries}
        checks = []
        for prediction in saved["predictions"]:
            observed = lookup[(prediction["condition"], prediction["width"])]
            quantity = observed["mean"][prediction["quantity"]]
            success = quantity > prediction["threshold"] if prediction["direction"] == "greater" else quantity < prediction["threshold"]
            checks.append({**prediction, "observed": quantity, "pass": success})
        summary["sealed_ood_checks"] = checks
    with (HERE / (phase + "_cells.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / (phase + "_analysis.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10))
    plot(phase, cells)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("groups", "cells")}, ensure_ascii=False, indent=2))
    for group in summaries:
        print(group["condition"], group["width"], "targetRayleigh", round(group["mean"]["target_rayleigh_ratio"], 5),
              "target/permutation", round(group["mean"]["target_to_permutation_mean"], 5),
              "probeGain", round(group["mean"]["equal_trace_probe_gain_relative_initial_residual"], 5))


def plot(phase, cells):
    conditions = sorted({metadata["request"]["condition"]["name"] for metadata, _ in cells})
    widths = sorted({metadata["request"]["width"] for metadata, _ in cells})
    fig, axes = plt.subplots(len(conditions), len(widths), figsize=(8.8, 3.3 * len(conditions)), layout="constrained", squeeze=False)
    for row, condition in enumerate(conditions):
        for column, width in enumerate(widths):
            matched = [arrays for metadata, arrays in cells if metadata["request"]["condition"]["name"] == condition and metadata["request"]["width"] == width]
            for kernel in ("initial", "initial_equal_trace", "learned"):
                values = np.stack([arrays[kernel + "_probe_train_loss"] / arrays[kernel + "_probe_train_loss"][0] for arrays in matched])
                axes[row, column].plot(np.arange(values.shape[1]), values.mean(axis=0), label=kernel.replace("_", " "))
            axes[row, column].set(title=f"{condition} / w{width}", xlabel="probe step", ylabel="relative half MSE")
            axes[row, column].legend(fontsize=7)
            axes[row, column].title.set_fontsize(10)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / (phase + "_kernel_probes.png"), dpi=180)
    fig.savefig(HERE / (phase + "_kernel_probes.svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "ood"], required=True)
    main(parser.parse_args().phase)
