"""Plot the registered D1 adaptive-optimizer endpoint errors from the saved run."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT.parent / "AIQ_KB_DATA"
DATA = Path(os.environ.get("AIQ_KB_DATA_ROOT", DEFAULT_DATA))
SUMMARY = DATA / "runs" / "d1_adaptive_optimizers_r1_utf8" / "directions" / "D1_u_effective_time" / "studies" / "r001_adaptive_optimizer_u" / "summary.json"
OUT = Path(__file__).with_name("D1_adaptive_U.svg")

COLORS = {"Adam": "#277da1", "AdamW": "#f28e2b", "RMSprop": "#7b61a8", "Adagrad": "#3a9d6f"}
LABELS = {"Adam": "Adam", "AdamW": "AdamW", "RMSprop": "RMSprop", "Adagrad": "Adagrad"}
QS = ["0.9", "0.1", "0.01"]
WIDTH, HEIGHT = 1000, 440
LEFT, TOP, PLOT_W, PLOT_H = 100, 68, 830, 275
YMAX = 150


def y(value: float) -> float:
    return TOP + PLOT_H * (1 - value / YMAX)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    methods = list(summary["by_optimizer"])
    group_x = [250, 500, 750]
    bar_w, gap = 22, 7
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">D1 自适应优化器 U 端点预测误差</title>',
        '<desc id="desc">Adam、AdamW、RMSprop 和 Adagrad 在 q=0.9、0.1、0.01 下的中位相对绝对端点误差。误差随阈值变严格而增大。</desc>',
        '<style>.axis{font:12px sans-serif;fill:#334155}.small{font:11px sans-serif;fill:#64748b}.value{font:10px sans-serif;fill:#334155}.grid{stroke:#e5eaf0}.bar{rx:3}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="100" y="30" class="axis" font-weight="600">中位相对绝对端点误差（%）；按 q 分组</text>',
    ]
    for tick in range(0, YMAX + 1, 25):
        yy = y(tick)
        parts.append(f'<line x1="{LEFT}" y1="{yy:.1f}" x2="{LEFT + PLOT_W}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{LEFT - 12}" y="{yy + 4:.1f}" class="small" text-anchor="end">{tick}</text>')
    yy = y(5)
    parts.append(f'<line x1="{LEFT}" y1="{yy:.1f}" x2="{LEFT + PLOT_W}" y2="{yy:.1f}" stroke="#bf4c3b" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{LEFT + PLOT_W - 4}" y="{yy - 5:.1f}" fill="#a33c30" font-size="11" text-anchor="end">5% 准则</text>')
    for gi, q in enumerate(QS):
        center = group_x[gi]
        parts.append(f'<text x="{center}" y="{TOP + PLOT_H + 28}" class="axis" text-anchor="middle">q={q}</text>')
        start = center - (len(methods) * bar_w + (len(methods) - 1) * gap) / 2
        for mi, method in enumerate(methods):
            stats = summary["by_optimizer"][method]["by_q"][q]
            value = 100 * stats["median_relative_error"]
            x = start + mi * (bar_w + gap)
            yy = y(min(YMAX, value))
            if value == 0:
                parts.append(f'<circle cx="{x + bar_w / 2:.1f}" cy="{y(0):.1f}" r="3" fill="{COLORS[method]}"/>')
                label = "0"
            else:
                parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w}" height="{y(0) - yy:.1f}" fill="{COLORS[method]}" class="bar"/>')
                label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(TOP - 5, yy - 4):.1f}" class="value" text-anchor="middle">{label}</text>')
    legend_x = 290
    for i, method in enumerate(methods):
        x = legend_x + i * 150
        parts.append(f'<rect x="{x}" y="390" width="12" height="12" fill="{COLORS[method]}" rx="2"/>')
        parts.append(f'<text x="{x + 18}" y="400" class="small">{LABELS[method]}</text>')
    parts.append('</svg>')
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.name} from {SUMMARY}")


if __name__ == "__main__":
    main()
