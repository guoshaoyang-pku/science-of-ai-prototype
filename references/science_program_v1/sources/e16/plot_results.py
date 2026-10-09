#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
data = json.loads((REPO / "analysis.json").read_text())
datasets = list(dict.fromkeys(c["dataset"] for c in data["conditions"]))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True, sharey=True)
for ax, dataset in zip(axes, datasets):
    cells = sorted((c for c in data["conditions"] if c["dataset"] == dataset and c["condition"]),
                   key=lambda c: c["target_mean"])
    means = [c["delta"]["mean"] for c in cells]
    ax.errorbar([c["target_mean"] for c in cells], means,
                yerr=[[c["delta"]["mean"] - c["delta"]["ci95"][0] for c in cells],
                      [c["delta"]["ci95"][1] - c["delta"]["mean"] for c in cells]],
                fmt="o", capsize=4, color="#226a97")
    ax.axhline(0, color="#555", linewidth=.8)
    ax.set_title(dataset.split("/")[-1])
    ax.set_xticks([-3, 0, 3])
    ax.set_xlabel("Training-label mean (only offset changes)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("SGD MSE - Adam MSE")
fig.suptitle("Same-function offset intervention: 10 paired seeds, 95% intervals", fontsize=12)
fig.text(.5, .005, "Positive: Adam wins; negative: SGD wins. Exact recipe, three fixed functions; no benchmark score evaluation.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
fig.savefig(REPO / "offset_intervention.png", dpi=180)
fig.savefig(REPO / "offset_intervention.svg")
