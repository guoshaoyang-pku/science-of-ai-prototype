#!/usr/bin/env python3
"""Plot RL run metrics from a trainer log_history.json: reward, seqlen, clip.

Usage:
  python plot_run_metrics.py --log-history log_history.json \
      --out-dir docs/reports/figs --prefix myrun \
      --eval "61:0.33" "91:0.32" "144:0.47"        # optional held-out probes

Writes <prefix>_reward / _seqlen / _clip as .svg (for the panel) and .png (2x).
Figures carry no title: the panel heading states the finding. Style: panel_style.py.
Requires matplotlib only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_style import C, FIG_HALF, FIG_WIDE, apply, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def roll(xs, w=10):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-history", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--eval", nargs="*", default=[],
                    help="held-out probes as step:acc, e.g. 61:0.33 144:0.47")
    a = ap.parse_args()
    apply()

    H = json.load(open(a.log_history))
    rows = [d for d in H if "reward" in d and "completions/mean_length" in d]
    step = [d["step"] for d in rows]
    rew = [d["reward"] for d in rows]
    meanlen = [d["completions/mean_length"] for d in rows]
    trunc = [d.get("completions/clipped_ratio", 0) for d in rows]
    low = [d.get("clip_ratio/low_mean", 0) for d in rows]
    high = [d.get("clip_ratio/high_mean", 0) for d in rows]
    ent = [d.get("entropy", 0) for d in rows]
    kl = [d.get("kl", 0) for d in rows]
    evals = [tuple(x.split(":")) for x in a.eval]
    os.makedirs(a.out_dir, exist_ok=True)
    stem = lambda name: os.path.join(a.out_dir, f"{a.prefix}_{name}")

    # reward + held-out probes
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(step, rew, color=C["soft"], lw=0.8, label="每步 reward")
    ax.plot(step, roll(rew), color=C["orange"], label="reward（10 步滑动平均）")
    ax.axhline(0, color=C["mid"], lw=0.6, ls=":")
    for s, acc in evals:
        ax.scatter([int(s)], [float(acc)], marker="o", s=28, color=C["blue"], zorder=5)
        ax.annotate(f"独立测试 {float(acc):.2f}", (int(s), float(acc)),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=C["blue"])
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend(loc="lower right")
    save(fig, stem("reward"))

    # completion length (log) + truncation ratio
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(step, meanlen, color=C["orange"], label="回答长度均值")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("回答长度（token，对数轴）")
    ax2 = ax.twinx()
    ax2.plot(step, trunc, color=C["blue"], lw=1.2, label="截断比例")
    ax2.set_ylabel("截断比例", color=C["blue"])
    ax2.tick_params(axis="y", colors=C["blue"])
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    save(fig, stem("seqlen"))

    # clip ratios | entropy + KL
    fig, axes = plt.subplots(1, 2, figsize=(FIG_HALF[0] * 2, FIG_HALF[1]))
    ax = axes[0]
    ax.plot(step, low, color=C["blue"], lw=1.2, label="clip ratio（下界）")
    ax.plot(step, high, color=C["orange"], lw=1.2, label="clip ratio（上界）")
    ax.set_xlabel("step")
    ax.set_ylabel("比例（对数轴）")
    ax.set_yscale("log")
    ax.legend()
    ax = axes[1]
    ax.plot(step, ent, color=C["green"], lw=1.2, label="entropy")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy", color=C["green"])
    ax2 = ax.twinx()
    ax2.plot(step, kl, color=C["mid"], lw=1.0, ls="--", label="KL")
    ax2.set_ylabel("KL", color=C["muted"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    save(fig, stem("clip"))
    print("figures written:", a.prefix, "->", a.out_dir)


if __name__ == "__main__":
    main()
