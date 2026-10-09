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
            raise ValueError("C04 results mismatch")
        rows.append(r)
    if len(rows) != 64 or not all(r["finite"] for r in rows):
        raise ValueError("All64preregistered cells required")
    lookup = {(r["cell"]["n"], r["cell"]["seed"], r["cell"]["scale"], r["cell"]["activation"], r["cell"]["noise_sd"]): r for r in rows}
    seeds = [411, 412, 413, 414]
    groups = []
    for n in [32, 128]:
        for sigma in [0., 1.]:
            for activation in ["relu", "silu"]:
                for scale in [.03, .3]:
                    selected = [lookup[n, s, scale, activation, sigma] for s in seeds]
                    minima = [min(r["records"], key=lambda p: p["expected_test_clean_mse"]) for r in selected]
                    row = {"n": n, "noise_sd": sigma, "activation": activation, "scale": scale,
                           "minimum_steps": [r["step"] for r in minima],
                           "mean_expected_risk_minimum": float(np.mean([r["expected_test_clean_mse"] for r in minima]))}
                    for key in ["train_mse", "test_clean_mse", "expected_test_clean_mse", "expected_noise_energy", "signal_bias_mse"]:
                        values = [r["records"][-1][key] for r in selected]
                        row["final_" + key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
                    groups.append(row)
    relu = [lookup[32, s, .03, "relu", 1.] for s in seeds]
    small_silu = [lookup[32, s, .03, "silu", 1.] for s in seeds]
    later = []
    for r in relu:
        best = min(r["records"], key=lambda p: p["expected_test_clean_mse"])
        last = r["records"][-1]
        later.append({"minimum_step": best["step"], "risk_increase": last["expected_test_clean_mse"]-best["expected_test_clean_mse"],
                      "noise_growth": last["expected_noise_energy"]-best["expected_noise_energy"],
                      "bias_reduction": best["signal_bias_mse"]-last["signal_bias_mse"]})
    noise_mono = max(np.diff([p["expected_noise_energy"] for p in r["records"]]).min() * -1 for r in rows)
    decomposition = max(p["decomposition_error"] for r in rows for p in r["records"])
    risk32 = np.mean([r["records"][-1]["expected_test_clean_mse"] for r in relu])
    risk128 = np.mean([lookup[128, s, .03, "relu", 1.]["records"][-1]["expected_test_clean_mse"] for s in seeds])
    small_steps = [min(r["records"], key=lambda p:p["expected_test_clean_mse"])["step"] for r in small_silu]
    relu_steps = [v["minimum_step"] for v in later]
    verdicts = {
      "N1": {"pass": max(r["max_train_increase"] for r in rows) < 1e-10 and sum(v["risk_increase"] >= .05 and v["minimum_step"] < 16384 for v in later) >= 3,
             "max_train_increase": max(r["max_train_increase"] for r in rows), "relu_n32_late_risk": later},
      "N2": {"pass": bool(decomposition < 1e-8 and noise_mono < 1e-10 and np.mean([v["noise_growth"]-v["bias_reduction"] for v in later]) > 0),
             "max_decomposition_error": decomposition, "max_expected_noise_decrease": noise_mono},
      "N3": {"pass": bool(risk128 < risk32), "relu_final_expected_risk_n32": float(risk32), "relu_final_expected_risk_n128": float(risk128)},
      "N4": {"pass": bool(np.median(small_steps) > np.median(relu_steps) and np.mean([r["records"][-1]["expected_noise_energy"] for r in small_silu]) <
                           np.mean([r["records"][-1]["expected_noise_energy"] for r in relu])), "relu_minimum_steps": relu_steps, "small_silu_minimum_steps": small_steps}}
    output = {"cells": len(rows), "groups": groups, "verdicts": verdicts, "training_seconds": sum(r["seconds"] for r in rows),
              "prediction_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "solver_evaluation": "not_run"}
    (HERE / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for activation, scale, color, label in [("relu", .03, "#2463a8", "ReLU"), ("silu", .03, "#bc4d20", "SiLU scale0.03"), ("silu", .3, "#398352", "SiLU scale0.3")]:
        selected = [lookup[32, s, scale, activation, 1.] for s in seeds]
        steps = [p["step"] for p in selected[0]["records"]]
        for ax, name in zip(axes[:2], ["train_mse", "expected_test_clean_mse"]):
            ax.plot(steps, np.mean([[p[name] for p in r["records"]] for r in selected], 0), color=color, label=label)
    selected = relu
    steps = [p["step"] for p in selected[0]["records"]]
    for name, color, label in [("signal_bias_mse", "#2463a8", "Signal fitting error"), ("expected_noise_energy", "#bc4d20", "Expected noise fitting")]:
        axes[2].plot(steps, np.mean([[p[name] for p in r["records"]] for r in selected], 0), color=color, label=label)
    for ax, title in zip(axes, ["Train noisy labels", "Expected clean-test risk", "ReLU bias / variance"]):
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlabel="GD steps", ylabel="MSE", title=title)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "noise_u_curve.svg")
    fig.savefig(HERE / "noise_u_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cells":len(rows),"verdicts":verdicts,"seconds":output["training_seconds"]}))


if __name__ == "__main__":
    main()
