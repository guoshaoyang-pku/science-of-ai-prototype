#!/usr/bin/env python3
"""Render D2 web figures from saved results; never run a new experiment.

First render: python render_charts.py --run /path/to/d2_sparse_cross_20261028
Portable redraw: python render_charts.py --plot-data research/d2_web_20261028/plot_data.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/模糊方向/figs"
HERE = Path(__file__).resolve().parent
COL = {"full": "#233443", "diagonal": "#bf6146", "change05": "#087f78",
       "magnitude05": "#bd8733", "random": "#8c99aa"}


def configure():
    candidates = ["/System/Library/Fonts/STHeiti Light.ttc", "/Library/Fonts/Arial Unicode.ttf"]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "sans-serif"]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.edgecolor": "#b4bcc4", "axes.labelcolor": "#394957",
                        "text.color": "#233443", "xtick.color": "#596977",
                        "ytick.color": "#596977", "axes.unicode_minus": False,
                        "svg.fonttype": "path", "svg.hashsalt": "d2-web-20261028"})


def load_data(run: Path):
    summary = json.loads((run / "summary.json").read_text())
    focus = json.loads((run / "focus_case.json").read_text())
    counters = {}
    with (run / "rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            item = counters.setdefault(row["method"], {"count": 0, "time_pass": 0, "risk_pass": 0})
            item["count"] += 1
            item["time_pass"] += int(row["reference_interior"] and row["time_pass"])
            item["risk_pass"] += int(row["risk_pass"])
    methods = {}
    for name, value in summary["methods"].items():
        overall = value["overall"]
        for key, number in counters[name].items():
            assert overall[key] == number, (name, key, overall[key], number)
        methods[name] = {key: overall[key] for key in ["count", "interior", "time_pass", "time_rate", "risk_pass", "risk_rate"]}
        methods[name]["budget_percent"] = 100 if name == "full" else 0 if name == "diagonal" else 5
        methods[name]["passing_recipes"] = sum(bool(r["stopping_pass"]) for r in value["recipes"])
    curves = np.load(run / "focus_curves.npz")
    # Display sampling only: preserve all early integer points and both highlighted steps.
    indexes = np.unique(np.r_[np.arange(min(257, len(curves["t"]))),
                              np.linspace(257, len(curves["t"]) - 1, 700).astype(int),
                              np.flatnonzero(np.isin(curves["t"], [1, 255]))])
    data = {
        "study": summary["study"], "domain": "development",
        "source_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                          for name in ["summary.json", "rows.jsonl", "focus_case.json", "focus_curves.npz"]},
        "methods": methods, "retained_fraction": summary["retained_fraction"],
        "focus": {k: focus[k] for k in ["condition", "rank", "pair_count", "retained_pairs", "rows"]},
        "focus_curves": {k: curves[k][indexes].tolist() for k in ["t", "full", "diagonal", "change05", "magnitude05"]},
        "render_sampling": "All first 257 saved grid entries plus 700 uniformly indexed later entries; steps 1 and 255 retained. Display only, no minima recomputed.",
        "frontier_scope": "Observed points at nominal budget ceilings 0, 5 and 100 percent only; dotted line is a visual guide, not a measured continuous frontier. Budget is retained pairs, not runtime.",
    }
    (HERE / "plot_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"d2_20261028_{stem}.{ext}", dpi=180, facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            svg = OUT / f"d2_20261028_{stem}.{ext}"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def pareto(data):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7))
    fig.subplots_adjust(left=.075, right=.965, top=.77, bottom=.28, wspace=.22)
    fig.text(.075, .94, "少量关键交叉项，保住大部分停止决策", size=22, weight="bold")
    fig.text(.075, .885, "观测 Pareto 对比：横轴越小保留越少，纵轴越高效果越好", size=13, color="#596977")
    methods = data["methods"]
    for ax, field, title, den in zip(axes, ["time_rate", "risk_rate"],
                                   ["停止时间在两倍以内", "实际额外风险不超过初值的 1%"], [3137, 3240]):
        rate = lambda name: 100 * methods[name][field]
        ax.plot([0, 5, 100], [rate("diagonal"), rate("change05"), 100],
                color="#b5c3c1", ls=(0, (3, 4)), lw=1.6, zorder=1)
        # Large hollow markers denote the observed non-dominated choices.
        ax.scatter([0, 5, 100], [rate("diagonal"), rate("change05"), 100],
                   s=210, facecolors="none", edgecolors="#d2dfdc", lw=2, zorder=2)
        ax.scatter([0], [rate("diagonal")], s=70, color=COL["diagonal"], zorder=5)
        ax.scatter([100], [100], s=75, color=COL["full"], marker="s", zorder=5)
        ax.scatter([5], [rate("magnitude05")], s=160, facecolors="none",
                   edgecolors=COL["magnitude05"], marker="D", lw=1.7, zorder=5)
        ax.scatter([5], [rate("change05")], s=65, color=COL["change05"], zorder=6)
        random_rates = [rate(name) for name in methods if name.startswith("random")]
        ax.scatter(np.full(10, 5), random_rates, s=23, color=COL["random"], alpha=.8, zorder=4)
        ax.set(xlim=(-4, 106), ylim=(70, 103), xticks=[0, 5, 25, 50, 75, 100],
               yticks=[70, 80, 90, 100], xlabel="交叉项保留预算上限（%）")
        ax.set_title(title, loc="left", pad=12, weight="bold")
        ax.set_ylabel(f"通过率（%，分母 {den}）")
        ax.grid(axis="y", color="#e9edf0", lw=.8)
        ax.annotate(f"按变化选：{rate('change05'):.1f}%\n按大小选：{rate('magnitude05'):.1f}%",
                    xy=(5, rate("change05")), xytext=(22, 93.2), ha="left", va="center",
                    fontsize=11.5, color=COL["change05"],
                    arrowprops={"arrowstyle": "-", "color": COL["change05"], "lw": .8})
        random_y = 80.8 if field == "time_rate" else 81.5
        ax.annotate(f"随机 10 次\n{min(random_rates):.1f}%–{max(random_rates):.1f}%",
                    xy=(5, np.mean(random_rates)), xytext=(24, random_y), fontsize=11.5,
                    color="#68788a", va="center", arrowprops={"arrowstyle": "-", "color": "#9caab8", "lw": .8})
        ax.annotate(f"全删 {rate('diagonal'):.1f}%", xy=(0, rate("diagonal")),
                    xytext=(3, -24 if field == "time_rate" else 12), textcoords="offset points",
                    fontsize=11, color=COL["diagonal"])
        ax.annotate("完整参照\n100%", xy=(100, 100), xytext=(-5, -24),
                    textcoords="offset points", ha="right", va="top", fontsize=11, color=COL["full"])
    handles = [Line2D([], [], marker="o", color=COL["change05"], ls="", label="按变化选 5%"),
               Line2D([], [], marker="D", markerfacecolor="none", color=COL["magnitude05"], ls="", label="按大小选 5%"),
               Line2D([], [], color="#b5c3c1", ls="--", label="观测非支配点连线，仅作阅读引导")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.068, .13), ncol=3, frameon=False, fontsize=11)
    fig.text(.075, .095, "只测了 0%、5%、100% 三档，未测中间比例；5% 严格向下取整，实际保留可能更少。", size=10.5, color="#596977")
    fig.text(.075, .052, "同一批已看过的数据；随机 10 次不是 10 份新数据。完整参照定义了评价基准；图不表示计算加速。", size=10.5, color="#596977")
    save(fig, "pareto")


def focus_chart(data):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), gridspec_kw={"width_ratios": [1.32, 1]})
    fig.subplots_adjust(left=.075, right=.965, top=.77, bottom=.31, wspace=.25)
    fig.text(.075, .94, "为什么从第 1 步，改成第 255 步停止？", size=22, weight="bold")
    fig.text(.075, .885, "一维正弦 · 有隐藏偏置 · n = 32 · 噪声方差 = 1 · seed = 6103 · 保留 23 / 465 对", size=12.5, color="#596977")
    c = {k: np.asarray(v) for k, v in data["focus_curves"].items()}
    rows = {r["method"]: r for r in data["focus"]["rows"]}
    ax = axes[0]
    labels = {"full": "完整风险（实际参照）", "diagonal": "全删后的估计", "change05": "按变化保留后的估计", "magnitude05": "按大小保留后的估计"}
    for method, style, width in [("full", "-", 3.4), ("diagonal", "-", 2.4), ("change05", "--", 2.2), ("magnitude05", ":", 2.0)]:
        ax.plot(c["t"], c[method], style, color=COL[method], lw=width, label=labels[method])
    ax.set_xscale("symlog", linthresh=1)
    ax.set(xlim=(0, 262144), ylim=(0, 2.4), xticks=[0, 1, 16, 255, 4096, 262144],
           xticklabels=["0", "1", "16", "255", "4096", "262144"],
           xlabel="训练步数（1 步以后为对数刻度）", ylabel="风险 / 风险估计（越低越好）")
    ax.set_title("删项改变了对谷底的判断", loc="left", pad=12, weight="bold")
    ax.grid(axis="y", color="#e9edf0")
    for method in ["diagonal", "change05"]:
        row = rows[method]
        ax.scatter(row["t_selected"], row["estimated_risk_at_selected"], s=65, color=COL[method], zorder=7, edgecolor="white")
    ax.annotate("全删：选第 1 步", xy=(1, rows["diagonal"]["estimated_risk_at_selected"]),
                xytext=(2.5, 1.65), fontsize=11, color=COL["diagonal"],
                arrowprops={"arrowstyle": "-", "color": COL["diagonal"], "lw": .9})
    ax.annotate("两种选择都回到第 255 步\n与完整参照几乎重合", xy=(255, rows["change05"]["estimated_risk_at_selected"]),
                xytext=(40, .65), fontsize=10.5, color=COL["change05"],
                arrowprops={"arrowstyle": "-", "color": COL["change05"], "lw": .9})
    ax = axes[1]
    names = ["全删", "随机保留\n10 次均相同", "按变化选", "按大小选", "完整参照"]
    ids = ["diagonal", "random05_00", "change05", "magnitude05", "full"]
    values = [rows[m]["full_risk_at_selected"] for m in ids]
    colors = [COL["diagonal"], COL["random"], COL["change05"], COL["magnitude05"], COL["full"]]
    bars = ax.barh(np.arange(5), values, height=.55, color=colors)
    ax.invert_yaxis()
    ax.set(yticks=np.arange(5), yticklabels=names, xlim=(0, 1.06), xlabel="按建议停止时的实际风险（越低越好）")
    ax.set_title("停止决策的实际代价", loc="left", pad=12, weight="bold")
    ax.grid(axis="x", color="#e9edf0")
    ax.set_axisbelow(True)
    for bar, value, method in zip(bars, values, ids):
        ax.text(value + .025, bar.get_y() + bar.get_height() / 2, f"{value:.3f}\n第 {rows[method]['t_selected']} 步", va="center", fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.068, .13), ncol=2, frameon=False, fontsize=11)
    fig.text(.075, .075, "左图展示各方法的估计，右图统一用完整风险评价。改善来自找回抵消作用，并未改变训练本身。", size=10.5, color="#596977")
    fig.text(.075, .035, "这是已知失败例子的修复，不是新任务盲测；所有方法选点都使用原注册的完整搜索窗口。", size=10.5, color="#596977")
    save(fig, "focus")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--plot-data", type=Path)
    args = parser.parse_args()
    if args.plot_data:
        data = json.loads(args.plot_data.read_text())
    elif args.run:
        data = load_data(args.run)
    else:
        parser.error("Provide --run or --plot-data")
    configure()
    pareto(data)
    focus_chart(data)
    print("Rendered saved evidence only: d2_20261028_pareto and d2_20261028_focus (PNG + SVG).")


if __name__ == "__main__":
    main()
