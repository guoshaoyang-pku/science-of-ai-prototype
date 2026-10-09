#!/usr/bin/env python3
"""Read saved D2 curves; rebuild matched val and draw measured/mechanism figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#62676c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Arial"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 9, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.8, "savefig.dpi": 260,
    "axes.grid": True, "grid.alpha": .12, "axes.unicode_minus": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, directory, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def spectral_functions(arrays, variance, eta):
    eigenvalues = np.maximum(arrays["eigenvalues"], 0.)
    nonzero = eigenvalues > 1e-12
    q = 1 - 2 * eta * eigenvalues
    logq = np.log(np.clip(q, 1e-300, 1.))
    gram = arrays["basis_gram"]
    target = arrays["basis_target"]
    projection = arrays["signal_projection"]
    diagonal = np.diag(gram)

    def factors(times):
        times = np.atleast_1d(times)
        result = 2 * eta * np.broadcast_to(times[:, None], (len(times), len(q))).copy()
        result[:, nonzero] = -np.expm1(times[:, None] * logq[nonzero]) / eigenvalues[nonzero]
        return result

    def derivatives(t):
        factor = factors([t])[0]
        first = np.full_like(eigenvalues, 2 * eta)
        first[nonzero] = -logq[nonzero] * np.exp(logq[nonzero] * t) / eigenvalues[nonzero]
        second = logq * first
        second[~nonzero] = 0.
        coefficient = factor * projection
        coefficient_first = first * projection
        coefficient_second = second * projection
        residual = np.einsum("ij,j->i", gram, coefficient) - target
        bias_first = 2 * np.einsum("i,i->", residual, coefficient_first)
        noise_first = 2 * variance * np.einsum("i,i->", diagonal, factor * first)
        curvature = (2 * np.einsum("i,ij,j->", coefficient_first, gram, coefficient_first)
                     + 2 * np.einsum("i,i->", residual, coefficient_second)
                     + 2 * variance * np.einsum("i,i->", diagonal, first**2 + factor * second))
        return float(bias_first), float(noise_first), float(curvature)

    def risk_increment(center, delta):
        initial = factors([center])[0]
        difference = factors([center + delta])[0] - initial
        c0, dc = initial * projection, difference * projection
        residual = np.einsum("ij,j->i", gram, c0) - target
        return float(2 * np.einsum("i,i->", residual, dc) + np.einsum("i,ij,j->", dc, gram, dc)
                     + variance * np.einsum("i,i->", diagonal, 2 * initial * difference + difference**2))

    return factors, derivatives, risk_increment, q


def analyze(source, out, figdir):
    legacy_summary = json.loads((source / "summary.json").read_text())
    receipt = json.loads((source / "results/receipt.json").read_text())
    saved = {row["label"]: row for row in legacy_summary["cells"]}
    rows, curves = [], {}
    checks = {"risk_identity_max_error": 0., "reconstructed_train_max_error": 0.,
              "final_direct_val_max_error": 0., "legacy_minimum_max_error": 0.}
    eta = .3
    for npz in sorted((source / "results").glob("*.npz")):
        metadata = json.loads(npz.with_suffix(".json").read_text())
        if sha(npz) != metadata["arrays_sha256"] or sha(npz.with_suffix(".json")) != receipt["cells"][npz.stem]:
            raise ValueError(f"Saved cell hash mismatch: {npz.stem}")
        data_path = source / metadata["data_file"]
        if sha(data_path) != metadata["data_sha256"]:
            raise ValueError(f"Saved data hash mismatch: {npz.stem}")
        with np.load(npz) as archive, np.load(data_path) as data_archive:
            arrays = {key: archive[key] for key in archive.files}
            data = {key: data_archive[key] for key in data_archive.files}
        cell = metadata["cell"]
        variance, n, seed = cell["noise_variance"], cell["n"], cell["seed"]
        risk, bias, train = arrays["expected_risk"], arrays["signal_bias"], arrays["train_mse"]
        noise = variance * arrays["variance_unit"]
        tstar = int(risk.argmin())
        if tstar != saved[npz.stem]["t_star"]:
            raise ValueError(f"Legacy oracle mismatch: {npz.stem}")
        checks["legacy_minimum_max_error"] = max(checks["legacy_minimum_max_error"], abs(risk[tstar] - saved[npz.stem]["minimum_expected_risk"]))
        checks["risk_identity_max_error"] = max(checks["risk_identity_max_error"], float(np.max(abs(risk - (bias + noise)))))
        factors, derivatives, increment, q = spectral_functions(arrays, variance, eta)
        matched_projection = np.einsum("ji,j->i", arrays["eigenvectors"], data["train_y"] + variance**.5 * data["epsilon"])
        times = np.arange(len(risk))
        coefficient = factors(times) * matched_projection
        val = (np.einsum("ti,ij,tj->t", coefficient, arrays["basis_gram"], coefficient, optimize=True)
               - 2 * np.einsum("ti,i->t", coefficient, arrays["basis_target"]) + float(arrays["audit_target_square"]))
        reconstructed_train = np.mean(q[None, :] ** (2 * times[:, None]) * matched_projection[None, :]**2, axis=1)
        expected_train = np.mean(q[None, :] ** (2 * times[:, None]) * (arrays["signal_projection"][None, :]**2 + variance), axis=1)
        direct_final_val = float(np.mean((np.einsum("ij,j->i", data["audit_features"], arrays["final_heads"][:, 0]) - data["audit_y"])**2))
        if not all(np.isfinite(array).all() for array in (val, reconstructed_train, expected_train)):
            raise ValueError(f"Nonfinite reconstructed curve: {npz.stem}")
        checks["reconstructed_train_max_error"] = max(checks["reconstructed_train_max_error"], float(np.max(abs(reconstructed_train - train))))
        checks["final_direct_val_max_error"] = max(checks["final_direct_val_max_error"], abs(direct_final_val - val[-1]))
        valstar = int(val.argmin())
        continuous = brentq(lambda t: sum(derivatives(t)[:2]), max(tstar - 1, 0), tstar + 1)
        curvature = derivatives(continuous)[2]
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError(f"Invalid minimum curvature: {npz.stem}")
        second_diff = np.diff(risk[tstar:], n=2)
        turns = np.where((second_diff[:-1] > 0) & (second_diff[1:] < 0))[0]
        inflection = int(tstar + turns[0] + 1) if len(turns) else None
        quadratic = {f"{fraction:g}": increment(continuous, fraction * continuous) / (.5 * curvature * (fraction * continuous)**2) for fraction in (.001, .01, .1, 1.)}
        row = {
            "label": npz.stem, "n": n, "seed": seed, "noise_variance": variance,
            "source_npz": str(npz), "source_npz_sha256": metadata["arrays_sha256"],
            "source_data": str(data_path), "source_data_sha256": metadata["data_sha256"],
            "tstar_expected": tstar, "tstar_actual_val": valstar, "tstar_continuous": continuous,
            "train_at_expected_rise": float(train[tstar]), "train_fraction_at_expected_rise": float(train[tstar] / train[0]),
            "train_fraction_at_actual_val_min": float(train[valstar] / train[0]),
            "expected_train_fraction_at_expected_rise": float(expected_train[tstar] / expected_train[0]),
            "minimum_expected_risk": float(risk[tstar]), "minimum_actual_val": float(val[valstar]),
            "noise_fraction_of_risk_at_minimum": float(noise[tstar] / risk[tstar]),
            "noise_accumulation_fraction": {f"{fraction:g}": float(noise[round(fraction * tstar)] / noise[tstar]) for fraction in (.25, .5, .75)},
            "bias_slope_at_continuous_minimum": derivatives(continuous)[0],
            "noise_slope_at_continuous_minimum": derivatives(continuous)[1],
            "curvature_at_continuous_minimum": curvature,
            "first_post_rise_curvature_turn_step": inflection,
            "quadratic_ratio_by_relative_offset": quadratic,
        }
        rows.append(row)
        if variance == 1. and n == 64:
            curves[seed] = dict(arrays=arrays, train=train, val=val, risk=risk, bias=bias, noise=noise,
                                tstar=tstar, continuous=continuous, inflection=inflection, valstar=valstar,
                                factors=factors, derivatives=derivatives, increment=increment)
    if checks["reconstructed_train_max_error"] > 1e-8 or checks["final_direct_val_max_error"] > 1e-8 or checks["legacy_minimum_max_error"] > 1e-12:
        raise ValueError(f"Numerical verification failed: {checks}")
    def spread(values):
        return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}
    summary = {
        "analysis": "curve_analysis_20261009", "domain": "post_hoc_development", "training_runs": 0,
        "source_summary": str(source / "summary.json"), "source_summary_sha256": sha(source / "summary.json"),
        "source_script_sha256": sha(Path(__file__)), "cells": rows, "n_cells": len(rows), "checks": checks,
        "definitions": {"val": "same saved noisy-training realization evaluated on independent synthetic clean audit labels; not benchmark test",
                        "expected_risk": "conditional expectation of clean audit MSE over iid train-label noise, with inputs/features fixed",
                        "rise": "first global argmin of expected_risk on saved integer grid 0..16384",
                        "very_low_train_probe": "post hoc descriptive threshold train/current initial train < 0.10; not preregistered",
                        "overfit_accumulation": "sigma^2 times variance_unit; positive noise damage can accumulate while net risk still decreases",
                        "curvature_turn": "first saved-grid positive-to-negative second-difference crossing after expected-risk minimum"},
        "aggregate": {
            "train_fraction_at_expected_rise": spread([r["train_fraction_at_expected_rise"] for r in rows]),
            "train_fraction_at_actual_val_min": spread([r["train_fraction_at_actual_val_min"] for r in rows]),
            "expected_train_fraction_at_expected_rise": spread([r["expected_train_fraction_at_expected_rise"] for r in rows]),
            "actual_val_min_to_expected_min_step_ratio": spread([r["tstar_actual_val"] / r["tstar_expected"] for r in rows]),
            "train_fraction_lt_0_10_at_expected_rise": sum(r["train_fraction_at_expected_rise"] < .1 for r in rows),
            "train_fraction_lt_0_10_at_actual_val_min": sum(r["train_fraction_at_actual_val_min"] < .1 for r in rows),
            "noise_fraction_of_risk_at_minimum": spread([r["noise_fraction_of_risk_at_minimum"] for r in rows]),
            "noise_accumulation_fraction_at_half_tstar": spread([r["noise_accumulation_fraction"]["0.5"] for r in rows]),
            "bias_descending_and_noise_ascending_at_rise": sum(r["bias_slope_at_continuous_minimum"] < 0 < r["noise_slope_at_continuous_minimum"] for r in rows),
            "curvature_turn_step_ratio": spread([r["first_post_rise_curvature_turn_step"] / r["tstar_expected"] for r in rows if r["first_post_rise_curvature_turn_step"] is not None]),
            "local_quadratic_ratio_at_0_001_tstar": spread([r["quadratic_ratio_by_relative_offset"]["0.001"] for r in rows]),
            "local_quadratic_ratio_at_0_01_tstar": spread([r["quadratic_ratio_by_relative_offset"]["0.01"] for r in rows]),
        },
        "boundaries": ["One saved task only: Gaussian inputs, mixed target x0+0.5*x0*x1, frozen ReLU width128, zero linear head, full-batch GD.",
                       "All metrics and examples are development reanalysis; no fresh task evidence and no causal train-only trigger.",
                       "Curvature is from continuous spectral interpolation; its grid crossing location has step-scale resolution."],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot_curves(curves, rows, figdir)
    plot_logic(curves[411], figdir)
    print(json.dumps({"n_cells": len(rows), "checks": checks, "aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))


def plot_curves(curves, rows, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    for ax, seed in zip(axes, (411, 414)):
        c = curves[seed]
        t = np.arange(len(c["risk"]))
        ax.plot(t, c["train"], color=BLUE, label="train · 含噪标签（单次）")
        ax.plot(t, c["val"], color=ORANGE, label="val · 干净标签（同一次训练）")
        ax.plot(t, c["risk"], color=GREEN, linestyle="--", label="干净风险 · 对训练噪声取期望")
        ax.axvline(c["tstar"], color=GREY, lw=1.1, ls=":")
        ax.scatter(c["tstar"], c["risk"][c["tstar"]], color=GREEN, zorder=5, s=25)
        ax.scatter(c["valstar"], c["val"][c["valstar"]], color=ORANGE, zorder=5, s=25)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, len(t) - 1)
        ax.set_xlabel("训练步数（后段为对数刻度）")
        ax.set_ylabel("MSE")
        ax.set_title(f"n=64，σ²=1，seed {seed}\n实际 val 最低点 {c['valstar']} 步；期望风险最低点 {c['tstar']} 步")
        ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, figdir, "legacy_train_val")

    c = curves[411]
    t = np.arange(min(800, len(c["risk"])))
    relative = t / c["tstar"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(relative, c["risk"][t], color=GREY, label="期望风险 R=B+σ²N")
    ax.plot(relative, c["bias"][t], color=BLUE, label="信号误差 B")
    ax.plot(relative, c["noise"][t], color=ORANGE, label="噪声损害 σ²N")
    ax.axvspan(0, 1, color=ORANGE, alpha=.045)
    ax.axvline(1, color=GREY, ls=":", lw=1.1)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, .5)
    ax.set_xlabel("t / t*（期望风险最低点归一化为 1）")
    ax.set_ylabel("干净 audit MSE")
    ax.set_title("噪声损害在最低点之前就积累")
    ax.legend(loc="upper right")
    ax = axes[1]
    derivative_t = np.linspace(.15 * c["tstar"], 5 * c["tstar"], 500)
    derivatives = np.array([c["derivatives"](time) for time in derivative_t])
    ax.plot(derivative_t / c["tstar"], -derivatives[:, 0], color=BLUE, label="信号学习收益 −B′(t)")
    ax.plot(derivative_t / c["tstar"], derivatives[:, 1], color=ORANGE, label="噪声损害速度 σ²N′(t)")
    ax.axvline(c["continuous"] / c["tstar"], color=GREY, ls=":", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlim(.15, 5)
    ax.set_ylim(1e-4, .04)
    ax.set_xlabel("t / t*")
    ax.set_ylabel("每训练一步的边际变化")
    ax.set_title("最低点：学习收益恰好抵消噪声损害")
    ax.legend(loc="upper right")
    save_figure(fig, figdir, "legacy_rise_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    for n, color in ((32, ORANGE), (64, BLUE), (128, GREEN)):
        subset = [row for row in rows if row["n"] == n]
        axes[0].scatter([row["noise_variance"] for row in subset], [row["train_fraction_at_expected_rise"] for row in subset], color=color, s=28, alpha=.7, label=f"n={n}")
    axes[0].axhline(.1, color=GREY, ls="--", lw=1.1, label="低于初始值的 10%（事后描述阈值）")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([.25, .5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("训练标签噪声方差 σ²")
    axes[0].set_ylabel("rise 时 train MSE / 初始 train MSE")
    axes[0].set_title("60 条曲线：rise 时训练误差未接近零")
    axes[0].legend(loc="upper left", fontsize=8)
    ratios = [.25, .5, .75]
    ax = axes[1]
    for index, fraction in enumerate(ratios):
        values = [row["noise_accumulation_fraction"][f"{fraction:g}"] for row in rows]
        jitter = (np.arange(len(values)) % 8 - 3.5) * .015
        ax.scatter(index + jitter, values, s=14, color=ORANGE, alpha=.34, edgecolors="none")
        ax.plot(index, np.median(values), marker="_", ms=24, color=GREY, mew=2.5)
    ax.set_xticks([0, 1, 2], ["¼ t*", "½ t*", "¾ t*"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("当前 σ²N / 最低点处的 σ²N")
    ax.set_xlabel("最低点之前的时刻")
    ax.set_title("半程时，噪声损害已累积约 74%（中位）")
    save_figure(fig, figdir, "legacy_train_threshold_accumulation")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    for ax, seed in zip(axes, (411, 414)):
        c = curves[seed]
        center = c["continuous"]
        offset = np.logspace(-3, np.log10(3.), 350)
        curvature = c["derivatives"](center)[2]
        increments = np.array([c["increment"](center, fraction * center) for fraction in offset])
        coefficient = .5 * curvature * center**2
        ax.loglog(offset, increments / coefficient, color=BLUE, label="期望风险增量（保存谱重建）")
        ax.loglog(offset, offset**2, color=GREY, ls="--", label="最低点局部二次 Δt²")
        ax.axvspan(.1, 1., color=GREEN, alpha=.07)
        at_one = c["increment"](center, center) / coefficient
        finite_range = np.logspace(-1, 0, 100)
        ax.loglog(finite_range, at_one * finite_range**1.55, color=ORANGE, ls=":", label="旧短段经验式 x^1.55")
        ax.set_xlim(.001, 3)
        ax.set_ylim(1e-6, 10.)
        ax.set_xlabel("x = (t − 连续最低点) / 连续最低点")
        ax.set_ylabel("风险增量 / [½R″(t*) · t*²]")
        ax.set_title(f"seed {seed}：最低点附近二次，有限区间可变")
        ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, figdir, "legacy_local_shape")


def plot_logic(c, figdir):
    fig = plt.figure(figsize=(10.8, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.])
    ax = fig.add_subplot(grid[0, :])
    ax.set_axis_off()
    boxes = [(0.03, "训练推进", "不同谱模态逐渐被拟合"),
             (0.34, "信号与噪声同时进入模型", "信号误差 B 下降；噪声损害 σ²N 上升"),
             (0.69, "风险回升", "噪声损害速度超过信号学习收益")]
    for x, title, text in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, .37), .26, .39, boxstyle="round,pad=.016", transform=ax.transAxes, facecolor="#f7f8f9", edgecolor="#cdd0d2"))
        ax.text(x + .13, .63, title, ha="center", va="center", fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(x + .13, .47, text, ha="center", va="center", fontsize=9, transform=ax.transAxes, wrap=True)
    for left, right in ((.29, .325), (.60, .675)):
        ax.annotate("", xy=(right, .57), xytext=(left, .57), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.text(.5, .17, "最低点：−B′(t*) = σ²N′(t*)。此前已有噪声损害，只是被信号收益遮住。", ha="center", fontsize=11, transform=ax.transAxes)
    ax.set_title("D2 机制图像 · 连线表示解析逻辑；下方曲线为真实保存数据（n=64，σ²=1，seed411）", loc="left", fontsize=12, pad=5)
    time = np.arange(0, 501)
    normalized = time / c["tstar"]
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(normalized, c["bias"][time], color=BLUE, label="B：信号误差")
    ax.plot(normalized, c["noise"][time], color=ORANGE, label="σ²N：噪声损害")
    ax.axvline(1, color=GREY, lw=1, ls=":")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, .45)
    ax.set_title("rise 前已经累积")
    ax.set_ylabel("MSE")
    ax.set_xlabel("t / t*")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[1, 1])
    ax.plot(normalized, c["risk"][time], color=GREY)
    ax.scatter([1, c["inflection"] / c["tstar"]], [c["risk"][c["tstar"]], c["risk"][c["inflection"]]], color=[GREEN, ORANGE], zorder=4, s=30)
    ax.annotate("最低点\n一阶导换号", (1, c["risk"][c["tstar"]]), xytext=(.2, .145), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8, color=GREY))
    ax.annotate("数学拐点\n二阶导换号", (c["inflection"] / c["tstar"], c["risk"][c["inflection"]]), xytext=(2.1, .31), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8, color=GREY))
    ax.set_xlim(0, 5)
    ax.set_ylim(.13, .36)
    ax.set_title("最低点 ≠ 数学拐点")
    ax.set_xlabel("t / t*")
    ax.set_ylabel("期望干净风险")
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(normalized, c["train"][time] / c["train"][0], color=BLUE, label="train / 初始 train")
    ax.plot(normalized, c["val"][time], color=ORANGE, label="actual val")
    ax.axvline(1, color=GREY, lw=1, ls=":")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, .75)
    ax.set_title("train 仍降，val 可先回升")
    ax.set_xlabel("t / t*")
    ax.set_ylabel("归一化 train / val MSE")
    ax.legend(fontsize=8)
    save_figure(fig, figdir, "legacy_mechanism_logic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("science_program_v2/directions/D2_overfitting_u_curve/studies/r003_noise_sample_scaling"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--figures", type=Path, default=Path(__file__).resolve().parents[2] / "figs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        analyze(args.source, args.output, args.figures)
