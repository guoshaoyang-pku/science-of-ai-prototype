#!/usr/bin/env python3
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
        r = json.loads(path.read_text())
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != r["arrays_sha256"]:
            raise ValueError("C03 array hash mismatch")
        rows.append(r)
    if len(rows) != 128 or not all(r["finite"] for r in rows):
        raise ValueError("All128finite results required for this analysis")
    lookup = {(r["cell"]["distribution"], r["cell"]["seed"], r["cell"]["scale"], r["cell"]["activation"], r["cell"]["hidden_mode"], r["cell"]["target"]): r for r in rows}
    seeds = [311, 312, 313, 314]
    early = {}
    for mode in ["frozen", "learned"]:
        early[mode] = {}
        for target in ["linear", "interaction"]:
            early[mode][target] = []
            for seed in seeds:
                relu = lookup["symmetric_gaussian", seed, .03, "relu", mode, target]["train_curve"]
                silu = lookup["symmetric_gaussian", seed, .03, "silu", mode, target]["train_curve"]
                early[mode][target].append((silu[0] - silu[1]) / (relu[0] - relu[1]))
    train_diffs, time128_diffs, energy_growth = [], [], []
    for seed in seeds:
        frozen = lookup["symmetric_gaussian", seed, .03, "silu", "frozen", "interaction"]
        learned = lookup["symmetric_gaussian", seed, .03, "silu", "learned", "interaction"]
        relu = lookup["symmetric_gaussian", seed, .03, "relu", "learned", "interaction"]
        train_diffs.append(learned["train_curve"][-1] - frozen["train_curve"][-1])
        time128_diffs.append(learned["train_curve"][128] - relu["train_curve"][128])
        energy_growth.append({"even_energy_ratio": learned["records"][-1]["even_energy"] / learned["records"][0]["even_energy"],
                              "target_kernel_energy_ratio": learned["records"][-1]["target_kernel_energy"] / learned["records"][0]["target_kernel_energy"]})
    frozen_error = max(r["initial_kernel_curve_error"] for r in rows if r["cell"]["hidden_mode"] == "frozen")
    learned_error = max(r["initial_kernel_curve_error"] for r in rows if r["cell"]["hidden_mode"] == "learned")
    verdicts = {"H1": {"pass": all(max(v["interaction"]) < .1 and min(v["linear"]) > .5 for v in early.values()), "step1_silu_over_relu": early},
                "H2": {"pass": bool(sum(d < 0 for d in train_diffs) >= 3 and np.median(time128_diffs) > 0),
                       "learned_minus_frozen_final_train_mse": train_diffs, "silu_minus_relu_step128_train_mse": time128_diffs},
                "H3": {"pass": frozen_error < 1e-8 and learned_error > 1e-3, "frozen_max_error": frozen_error, "learned_max_error": learned_error},
                "H4": {"pass": sum(max(v.values()) > 1.2 for v in energy_growth) >= 3, "learned_channel_growth": energy_growth}}
    groups = []
    for distribution in ["symmetric_gaussian", "shifted_gaussian"]:
        for target in ["linear", "interaction"]:
            for scale in [.03, .3]:
                for activation in ["relu", "silu"]:
                    for mode in ["frozen", "learned"]:
                        selected = [lookup[distribution, s, scale, activation, mode, target] for s in seeds]
                        row = {"distribution": distribution, "target": target, "scale": scale, "activation": activation, "hidden_mode": mode}
                        for name in ["train_mse", "test_mse", "hidden_displacement", "even_energy", "target_kernel_energy"]:
                            values = [r["records"][-1][name] for r in selected]
                            row[name] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
                        groups.append(row)
    output = {"cells": len(rows), "failed": 0, "groups": groups, "verdicts": verdicts, "training_seconds": sum(r["seconds"] for r in rows),
              "prediction_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "solver_evaluation": "not_run"}
    (HERE / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for ax, scale in zip(axes, [.03, .3]):
        for activation, color in [("relu", "#2463a8"), ("silu", "#bc4d20")]:
            for mode, style in [("frozen", "--"), ("learned", "-")]:
                selected = [lookup["symmetric_gaussian", s, scale, activation, mode, "interaction"] for s in seeds]
                steps = [r["step"] for r in selected[0]["records"]]
                ax.plot(steps, np.mean([[r["test_mse"] for r in item["records"]] for item in selected], 0), style, color=color, label=f"{activation} {mode}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set(yscale="log", xlabel="SGD steps", ylabel="Test MSE", title=f"Interaction; scale{scale}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "hidden_boundary.svg")
    fig.savefig(HERE / "hidden_boundary.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cells": len(rows), "verdicts": verdicts, "seconds": output["training_seconds"]}))


if __name__ == "__main__":
    main()
