#!/usr/bin/env python3
"""Audit sealed OOD predictions and plot the intervention contrast."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / "preregistration.json").read_text())
    rows = []
    for path in sorted((HERE / "results").glob("*.json")):
        row = json.loads(path.read_text())
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != row["arrays_sha256"]:
            raise ValueError("OOD result arrays mismatch")
        rows.append(row)
    if len(rows) != 192:
        raise ValueError("Missing preregistered OOD cells")
    lookup = {(r["cell"]["distribution"], r["cell"]["seed"], r["cell"]["scale"], r["cell"]["activation"],
               r["cell"]["intervention"], r["cell"]["target"]): r for r in rows}
    seeds = config["seeds"]
    summary, verdicts = [], {}
    transfer, intervention, curve_wins, scale_error = {}, {}, {}, {}
    for distribution in config["data"]["distributions"]:
        transfer[distribution], intervention[distribution] = {}, {}
        for target in config["data"]["targets"]:
            transfer[distribution][target] = [lookup[distribution, s, .03, "silu", "plain", target]["initial_target_kernel_energy"] /
                                              lookup[distribution, s, .03, "relu", "plain", target]["initial_target_kernel_energy"] for s in seeds]
            scale_error[distribution + "_" + target] = [abs(lookup[distribution, s, .03, "silu", "parity_balanced", target]["initial_target_kernel_energy"] /
                                                           lookup[distribution, s, .1, "silu", "parity_balanced", target]["initial_target_kernel_energy"] - 1) for s in seeds]
            for scale in config["scales"]:
                for activation in config["activations"]:
                    for control in config["interventions"]:
                        selected = [lookup[distribution, s, scale, activation, control, target] for s in seeds]
                        values = [r["records"][-1]["test_mse"] for r in selected]
                        summary.append({"distribution": distribution, "target": target, "scale": scale, "activation": activation, "intervention": control,
                                        "mean_initial_kernel_energy": float(np.mean([r["initial_target_kernel_energy"] for r in selected])),
                                        "mean_final_test_mse": float(np.mean(values)), "sd_final_test_mse": float(np.std(values, ddof=1))})
        for reference in ["plain_silu", "plain_relu"]:
            act = "silu" if reference == "plain_silu" else "relu"
            intervention[distribution][reference] = [lookup[distribution, s, .03, "silu", "parity_balanced", "interaction"]["initial_target_kernel_energy"] /
                                                    lookup[distribution, s, .03, act, "plain", "interaction"]["initial_target_kernel_energy"] for s in seeds]
        diffs = [lookup[distribution, s, .03, "silu", "parity_balanced", "interaction"]["records"][-1]["test_mse"] -
                 lookup[distribution, s, .03, "silu", "plain", "interaction"]["records"][-1]["test_mse"] for s in seeds]
        curve_wins[distribution] = {"seed_differences": diffs, "wins": sum(d < 0 for d in diffs), "mean_difference": float(np.mean(diffs))}
    verdicts["O1"] = {"pass": all(max(v["interaction"]) < .1 and min(v["linear_sum"]) > .5 for v in transfer.values()), "paired_plain_silu_over_relu": transfer}
    verdicts["O2"] = {"pass": all(min(v["plain_silu"]) > 20 and min(v["plain_relu"]) > .5 for v in intervention.values()), "balanced_silu_energy_ratios": intervention}
    verdicts["O3"] = {"pass": all(v["wins"] >= 3 and v["mean_difference"] < 0 for v in curve_wins.values()), "balanced_minus_plain_test_mse": curve_wins}
    verdicts["O4"] = {"pass": all(max(v) < (1e-8 if key.endswith("linear_sum") else .1) for key, v in scale_error.items()), "relative_kernel_energy_error": scale_error}
    verdicts["O5"] = {"pass": max(r["spectral_max_error"] for r in rows) < 1e-8, "max_absolute_error": max(r["spectral_max_error"] for r in rows)}
    output = {"cells": len(rows), "failed": sum(not r["finite"] for r in rows), "groups": summary, "verdicts": verdicts,
              "training_seconds": sum(r["seconds"] for r in rows), "research_units": "two distributions × two targets; four within-condition seeds",
              "prediction_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
              "development_report_sha256": config["development_report_sha256"],
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "solver_evaluation": "not_run"}
    (HERE / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for ax, distribution in zip(axes, config["data"]["distributions"]):
        for activation, intervention, color, label in [("relu", "plain", "#2463a8", "ReLU"), ("silu", "plain", "#bc4d20", "SiLU"),
                                                      ("silu", "parity_balanced", "#398352", "SiLU even balanced")]:
            selected = [lookup[distribution, s, .03, activation, intervention, "interaction"] for s in seeds]
            steps = [p["step"] for p in selected[0]["records"]]
            curves = np.array([[p["test_mse"] for p in r["records"]] for r in selected])
            ax.plot(steps, curves.mean(0), color=color, label=label)
            ax.fill_between(steps, curves.min(0), curves.max(0), color=color, alpha=.12)
        ax.set_xscale("symlog", linthresh=1)
        ax.set(yscale="log", xlabel="Head GD steps", ylabel="Test MSE", title=distribution.capitalize() + "; unseen interaction")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "parity_ood.svg")
    fig.savefig(HERE / "parity_ood.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cells": len(rows), "verdicts": verdicts, "seconds": output["training_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
