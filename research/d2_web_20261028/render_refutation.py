#!/usr/bin/env python3
"""Plot the saved D2 scaling-law failures; no fitting or new experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from render_charts import configure

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "evidence/d2_scaling_20261009/studies/n_sigma_tasks/summary.json"
OUT = ROOT / "docs/模糊方向/figs"
HERE = Path(__file__).resolve().parent


def main():
    data = json.loads(SOURCE.read_text())
    task_names = {"linear": "线性", "quadratic": "单变量二次", "additive": "加性二次",
                  "interaction": "线性＋交互", "product": "纯交互", "sine": "正弦", "vector": "双输出"}
    protocols = {"d1_homogeneous": "1维·无偏置", "d1_biased": "1维·有偏置",
                 "d4_homogeneous": "4维·无偏置", "d4_biased": "4维·有偏置"}
    rows = []
    for task in data["tasks"]:
        result = task["models"]["linear"]["expanded"]
        rows.append({"protocol": task["protocol"], "task": task["task"],
                     "label": protocols[task["protocol"]] + "｜" + task_names[task["task"]],
                     "within2": result["within2"], "total": result["total"],
                     "percent": 100 * result["within2"] / result["total"]})
    assert len(rows) == 12 and all(r["total"] == 225 for r in rows)
    assert sum(r["within2"] for r in rows) == 1232
    assert all(r["percent"] < 80 for r in rows)
    (HERE / "refutation_plot_data.json").write_text(json.dumps({
        "kind": "saved_result_display_only", "new_experiments": False,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_fields": "tasks[].models.linear.expanded.{within2,total}",
        "threshold_percent": 80, "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n")

    configure()
    fig, ax = plt.subplots(figsize=(13.6, 9.1))
    fig.subplots_adjust(left=.255, right=.965, bottom=.23, top=.79)
    fig.text(.045, .935, "扩大范围后，12组都未达到原公式的通过门槛", fontsize=23, weight="bold")
    fig.text(.045, .885, "原主张：最佳训练量 U* ≈ c × 样本数 n / 标签噪声方差 σ²", fontsize=14, color="#394957")
    fig.text(.045, .843, "每组都单独校准了系数 c，再到扩大后的范围检验；不是直接套用同一个系数。", fontsize=12, color="#596977")
    ys = np.arange(len(rows))
    ax.barh(ys, [r["percent"] for r in rows], height=.59, color="#bf6146", zorder=2)
    ax.set_yticks(ys, [r["label"] for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("停止时间预测落在参照的一半到两倍以内的比例", labelpad=12)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#e9edf0", zorder=0)
    ax.axvline(80, ls=(0, (4, 3)), lw=1.7, color="#394957", zorder=3)
    ax.text(80, 1.02, "原通过门槛：80%", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color="#394957", fontsize=11.5)
    for y, row in zip(ys, rows):
        ax.text(row["percent"] + 1.2, y, f'{row["within2"]}/{row["total"]} · {row["percent"]:.1f}%',
                va="center", ha="left", fontsize=11.5, color="#713c2f", zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    fig.text(.045, .125, "扩域合计：1232 / 2700（45.6%）；12 / 12 组均未过门槛。", fontsize=14, weight="bold", color="#713c2f")
    fig.text(.045, .083, "原小范围的 57/60 来自另一组留出样本量检验，不能与这里当作同批条件直接比较。", fontsize=11.5, color="#596977")
    fig.text(.045, .047, "固定特征合成回归；边界最低点在本缩放检验中计失败。此图重绘已保存结果，没有新增实验。", fontsize=11.5, color="#596977")
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = OUT / f"d2_20261029_refutation.{extension}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "within2": 1232, "total": 2700,
                      "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
