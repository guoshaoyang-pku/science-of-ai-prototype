#!/usr/bin/env python3
"""Recompute prediction verdicts and plots from all saved C01 cells."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for path in sorted((HERE / "results").glob("*.json")):
        row = json.loads(path.read_text())
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != row["arrays_sha256"]:
            raise ValueError("Saved array fingerprint mismatch")
        rows.append(row)
    if len(rows) != 216:
        raise ValueError("C01 needs all216 preregistered cells")
    lookup = {(r["cell"]["seed"], r["cell"]["scale"], r["cell"]["activation"], r["cell"]["control"], r["cell"]["target"]): r for r in rows}
    seeds = sorted({r["cell"]["seed"] for r in rows})
    groups = []
    for control in ["none", "preactivation_ln", "linear_skip"]:
        for scale in [.05, .2, 1.]:
            for target in ["linear", "quadratic"]:
                for activation in ["relu", "silu"]:
                    matched = [lookup[s, scale, activation, control, target] for s in seeds]
                    summary = {"control": control, "scale": scale, "target": target, "activation": activation}
                    for name in ["initial_target_kernel_energy", "initial_progress", "spectral_max_error"]:
                        values = [r[name] for r in matched]
                        summary[name] = {"median": float(np.median(values)), "min": float(min(values)), "max": float(max(values))}
                    for name in ["train_mse", "test_mse"]:
                        values = [r["records"][-1][name] for r in matched]
                        summary["final_" + name] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "median": float(np.median(values))}
                    groups.append(summary)
    relu_diffs = {c: [] for c in ["none", "preactivation_ln", "linear_skip"]}
    for control in relu_diffs:
        for seed in seeds:
            for target in ["linear", "quadratic"]:
                base = lookup[seed, 1., "relu", control, target]
                for scale in [.05, .2]:
                    row = lookup[seed, scale, "relu", control, target]
                    delta = max(np.max(np.abs(np.array(base["train_curve"]) - row["train_curve"])),
                                np.max(np.abs(np.array([r["test_mse"] for r in base["records"]]) - [r["test_mse"] for r in row["records"]])))
                    relu_diffs[control].append(float(delta))
    scale_energy = {control: [lookup[s, .2, "silu", control, "linear"]["features"]["even_odd_energy_ratio"] /
                              lookup[s, .05, "silu", control, "linear"]["features"]["even_odd_energy_ratio"] for s in seeds] for control in ["none", "linear_skip"]}
    ratios = {}
    for target in ["linear", "quadratic"]:
        ratios[target] = {name: [lookup[s, .05, "silu", "none", target][name] / lookup[s, .05, "relu", "none", target][name] for s in seeds]
                          for name in ["initial_target_kernel_energy", "initial_progress"]}
    ln_delta = [abs(lookup[s, .05, "silu", "preactivation_ln", t]["initial_target_kernel_energy"] /
                    lookup[s, 1., "silu", "preactivation_ln", t]["initial_target_kernel_energy"] - 1) for s in seeds for t in ["linear", "quadratic"]]
    skip_ratio = [lookup[s, .05, "silu", "linear_skip", "quadratic"]["initial_target_kernel_energy"] /
                  lookup[s, .05, "silu", "none", "quadratic"]["initial_target_kernel_energy"] for s in seeds]
    p1_pass = max(relu_diffs["none"] + relu_diffs["linear_skip"]) < 1e-8 and max(relu_diffs["preactivation_ln"]) < 1e-4
    verdicts = {
      "P1": {"pass": p1_pass, "max_curve_difference_by_control": {k: max(v) for k, v in relu_diffs.items()}, "note": "Finite-epsilon LN can violate exact homogeneity; failure retained"},
      "P2": {"pass": all(10 < np.median(v) < 20 for v in scale_energy.values()), "even_odd_ratio_factor": scale_energy},
      "P3": {"pass": all(np.median(ratios["quadratic"][n]) < .1 and np.median(ratios["linear"][n]) >= .5 for n in ratios["linear"]), "paired_silu_over_relu": ratios},
      "P4": {"pass": max(ln_delta) < .01 and max(skip_ratio) < 2, "max_ln_energy_relative_difference": max(ln_delta), "skip_over_none_quadratic_energy": skip_ratio},
      "P5": {"pass": max(r["spectral_max_error"] for r in rows) < 1e-8, "max_absolute_error": max(r["spectral_max_error"] for r in rows)}}
    save = {"cells": len(rows), "failed": sum(not r["finite"] for r in rows), "groups": groups, "verdicts": verdicts,
            "training_seconds": sum(r["seconds"] for r in rows), "statistical_unit": "two target functions / recipes, not216 independent datasets",
            "preregistration_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "summary.json").write_text(json.dumps(save, ensure_ascii=False, indent=2) + chr(10))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.3))
    for target, ax in zip(["linear", "quadratic"], axes[:2]):
        for activation, color in [("relu", "#2463a8"), ("silu", "#bc4d20")]:
            for scale, style in [(.05, "-"), (.2, "--"), (1., ":")]:
                selected = [lookup[s, scale, activation, "none", target] for s in seeds]
                points = [p["step"] for p in selected[0]["records"]]
                mean = np.mean([[p["test_mse"] for p in r["records"]] for r in selected], axis=0)
                ax.plot(points, mean, style, color=color, label=f"{activation} scale{scale}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set(yscale="log", xlabel="Head GD steps", ylabel="Test MSE", title=target.capitalize() + " target")
    axes[1].legend(frameon=False, fontsize=7)
    for activation, color in [("relu", "#2463a8"), ("silu", "#bc4d20")]:
        scales = [.05, .2, 1.]
        energy = [np.median([lookup[s, a, activation, "none", "linear"]["features"]["even_odd_energy_ratio"] for s in seeds]) for a in scales]
        axes[2].plot(scales, energy, "o-", color=color, label=activation)
    axes[2].set(xscale="log", yscale="log", xlabel="Preactivation weight scale", ylabel="Even / odd squared feature energy", title="Equal total feature RMS")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "activation_target_channels.svg")
    fig.savefig(HERE / "activation_target_channels.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cells": len(rows), "verdicts": verdicts, "training_seconds": save["training_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
