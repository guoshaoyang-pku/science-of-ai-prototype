#!/usr/bin/env python3
"""Generate the two data charts for the SAI Chinese panel, in report-to-human style."""
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# panel_style.py ships with the report-to-human skill; override its location with PANEL_STYLE_DIR.
sys.path.insert(0, os.environ.get(
    "PANEL_STYLE_DIR", str(Path.home() / "Desktop/workdir/Ideas/report-to-human/skills/report-to-human/scripts")))
import panel_style as ps

ps.apply()

# ---------- chart 1: equivalent compute ----------
models = ["GPT-5.6 Sol", "Claude Opus 5", "DeepSeek V4 Flash", "DeepSeek V4 Pro", "GPT-5.6 Luna",
          "Gemini 3.1 Pro", "Claude Sonnet 5", "GPT-4o", "DeepSeek R1", "Gemini 2.5 Pro"]
rho = [8.5, 7.0, 4.5, 4.0, 3.5, 3.0, 2.5, 1.5, 1.5, 1.0]
acc = [77.6, 76.0, 70.4, 68.3, 67.8, 66.8, 65.4, 60.2, 59.4, 55.8]
# Public builds anonymize model identities (Frontier-1 strongest ... Frontier-10 weakest), as in the public figures.
if os.environ.get("SAI_PUBLIC"):
    models = [f"Frontier-{i}" for i in range(1, len(models) + 1)]

fig, ax = plt.subplots(figsize=(7.2, 3.9))
y = list(range(len(models)))[::-1]
ax.barh(y, rho, color=ps.C["mid"], edgecolor=ps.C["mid"], height=0.62, zorder=3)
for yi, r, a in zip(y, rho, acc):
    ax.text(r + 0.18, yi, f"{r}%（闭卷 {a}%）", va="center", fontsize=8.5, color=ps.C["muted"])
ax.axvline(10, color=ps.C["orange"], lw=1.8, ls="--", zorder=4)
ax.text(9.6, 2.2, "只用 10% 预算试探训练：\n选对 79.4%，超过全部闭卷模型",
        color=ps.C["orange"], fontsize=9, ha="right", va="center")
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=9)
ax.set_xlim(0, 12.5)
ax.set_xlabel("等价算力 ρ*：花每个候选训练预算的百分之几做试探，就能追平该模型的闭卷准确率")
ax.grid(axis="x")
ps.save(fig, "figs/equiv_compute")

# ---------- chart 2: paired KB lifts ----------
settings = ["混合题型\n150 题次", "架构题\n240 题次", "置信度前40\n+试用槽", "PUCT 前40", "PUCT 前20"]
wo = [0.660, 0.454, 0.458, 0.492, 0.450]
wi = [0.720, 0.596, 0.617, 0.525, 0.417]
lift = ["+6.0", "+14.2", "+15.8", "+3.3", "−3.3"]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
import numpy as np
x = np.arange(len(settings))
w = 0.34
b1 = ax.bar(x - w / 2, wo, w, color=ps.C["soft"], edgecolor=ps.C["mid"], lw=0.8, label="不带知识库", zorder=3)
b2 = ax.bar(x + w / 2, wi, w, color=ps.C["orange"], edgecolor=ps.C["orange"], label="带知识库", zorder=3)
for xi, a, b, lf in zip(x, wo, wi, lift):
    top = max(a, b)
    col = ps.C["orange"] if not lf.startswith("−") else ps.C["blue"]
    ax.text(xi, top + 0.02, f"{lf} 点", ha="center", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=9)
ax.set_ylim(0, 0.78)
ax.set_ylabel("准确率（同题同种子配对）")
ax.legend(loc="upper right", ncol=2)
ps.save(fig, "figs/kb_lift")
print("charts done")
