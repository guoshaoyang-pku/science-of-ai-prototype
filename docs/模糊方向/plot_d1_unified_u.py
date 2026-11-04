"""Build one U-scale comparison for all six D1 optimizers.

Adaptive loss traces are read from the completed, preregistered run. SGD and
momentum traces are reconstructed from the exact scalar quadratic recurrence
for the same h and G0 conditions; no optimizer run is launched here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIQ_KB_DATA_ROOT", ROOT.parent / "AIQ_KB_DATA"))
STUDY = DATA / "runs" / "d1_adaptive_optimizers_r1_utf8" / "directions" / "D1_u_effective_time" / "studies" / "r001_adaptive_optimizer_u"
OUT_DIR = Path(__file__).parent

ETA, MU, G0, T_MAX = 0.1, 0.9, 0.003, 40000
H_VALUES = [0.0001, 0.001, 0.005]
METHODS = ["SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad"]
COLORS = {"SGD": "#263238", "Momentum": "#d95f02", "Adam": "#277da1",
          "AdamW": "#f28e2b", "RMSprop": "#7b61a8", "Adagrad": "#3a9d6f"}
EPS = 1e-8
BETAS = (0.9, 0.999)
RMS_BETA, ADAMW_OMEGA = 0.99, 0.01
WIDTH, HEIGHT = 1280, 760
PLOT_X, PLOT_Y, PLOT_W, PLOT_H = 130, 150, 1050, 470
X_MIN, X_MAX = 1e-4, 1e5
Y_MIN, Y_MAX = 1e-8, 10


def u_step_factor(t: int, mu: float) -> float:
    if mu == 0:
        return float(t)
    return t / (1 - mu) - mu * (1 - mu**t) / (1 - mu) ** 2


def sgd_steps(h: float, q: float) -> int:
    return math.ceil(math.log(q) / (2 * math.log1p(-h)))


def adaptive_u(method: str, h: float) -> list[float]:
    lam = h / ETA
    theta0 = G0 / lam
    theta = theta0
    m = v = acc = 0.0
    values = [0.0]
    for t in range(1, T_MAX + 1):
        if method in ("Adam", "AdamW"):
            beta1, beta2 = BETAS
            m = beta1 * m + (1 - beta1) * G0
            v = beta2 * v + (1 - beta2) * G0 * G0
            mhat = m / (1 - beta1**t)
            vhat = v / (1 - beta2**t)
            step = ETA * mhat / (math.sqrt(vhat) + EPS)
            if method == "AdamW":
                theta *= 1 - ETA * ADAMW_OMEGA
                theta -= step
                value = abs(theta - theta0)  # includes decoupled shrinkage
            else:
                value = ETA * t * G0 / (G0 + EPS)
        elif method == "RMSprop":
            v = RMS_BETA * v + (1 - RMS_BETA) * G0 * G0
            step = ETA * G0 / (math.sqrt(v) + EPS)
            theta -= step
            value = abs(theta - theta0)
        else:
            acc += G0 * G0
            step = ETA * G0 / (math.sqrt(acc) + EPS)
            theta -= step
            value = abs(theta - theta0)
        values.append(value)
    return values


def scalar_quadratic_curve(h: float, steps: list[int], momentum: float) -> dict[int, float]:
    """Exact normalized loss for scalar quadratic GD or zero-start momentum."""
    prev = curr = 1.0  # r(-1)=r(0)=1, matching the registered D1 condition
    wanted = set(steps)
    out = {0: 1.0} if 0 in wanted else {}
    for t in range(1, max(steps) + 1):
        nxt = (1 + momentum - h) * curr - momentum * prev
        prev, curr = curr, nxt
        if t in wanted:
            out[t] = curr * curr
    return out


def sustained_endpoint(h: float, q: float, momentum: float) -> int:
    prev = curr = 1.0
    losses = [1.0]
    for _ in range(1, T_MAX + 1):
        nxt = (1 + momentum - h) * curr - momentum * prev
        prev, curr = curr, nxt
        losses.append(curr * curr)
    last_failure = max((t for t, loss in enumerate(losses) if loss > q), default=-1)
    return last_failure + 1


def verify_saved_endpoints() -> None:
    summary_path = ROOT / "science_program_v2" / "directions" / "D1_u_effective_time" / "studies" / "r001_momentum_startup" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for h in H_VALUES:
        row = next(r for r in summary["aggregate"]
                   if r["endpoint"] == "sustained_to_T" and r["threshold"] == 0.01
                   and r["g0"] == G0 and abs(r["h"] - h) < 1e-12)
        assert sustained_endpoint(h, 0.01, 0.0) == row["t_plain_seed_range"][0]
        assert sustained_endpoint(h, 0.01, MU) == row["t_momentum_seed_range"][0]


def sx(x: float, left: float, width: float) -> float:
    x = max(X_MIN, min(X_MAX, x))
    return left + width * math.log10(x / X_MIN) / math.log10(X_MAX / X_MIN)


def sy(y: float, top: float, height: float) -> float:
    y = max(Y_MIN, min(Y_MAX, y))
    return top + height * math.log10(Y_MAX / y) / math.log10(Y_MAX / Y_MIN)


def main() -> None:
    verify_saved_endpoints()
    summary = json.loads((STUDY / "summary.json").read_text(encoding="utf-8"))
    written = []
    for h in H_VALUES:
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
            f'<title id="title">D1 六种优化器的统一 U 尺度轨迹，h={h:g}</title>',
            f'<desc id="desc">固定 G0={G0}，曲率 h={h:g}，叠画 SGD、动量 SGD、Adam、AdamW、RMSprop、Adagrad。所有曲线以 SGD 达到 q 等于 0.01 的 U 归一化。圆点是 q 等于 0.01 的持续端点。</desc>',
            '<style>.title{font:600 17px sans-serif;fill:#172033}.axis{font:14px sans-serif;fill:#334155}.tick{font:12px sans-serif;fill:#64748b}.grid{stroke:#e5eaf0;stroke-width:1}.threshold{stroke:#aa4638;stroke-dasharray:6 5;stroke-width:1.5}.legend{font:13px sans-serif;fill:#475569}.note{font:12px sans-serif;fill:#64748b}</style>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="70" y="38" class="title">六种优化器的 U 尺度损失轨迹：h={h:g}</text>',
            '<text x="70" y="68" class="note">固定 G₀=0.003、η=0.1。横轴 U(t)/U_SGD(q=0.01)，纵轴 L/L₀；若 U 能统一进度，各色曲线应重合。</text>',
        ]
        for i, method in enumerate(METHODS):
            col, row = i % 3, i // 3
            x, y = 735 + col * 165, 34 + row * 28
            parts.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{COLORS[method]}" stroke-width="3.2"/>')
            label = {"Momentum": "动量 SGD"}.get(method, method)
            parts.append(f'<text x="{x+36}" y="{y+5}" class="legend">{label}</text>')

        x0, y0, pw, ph = PLOT_X, PLOT_Y, PLOT_W, PLOT_H
        u_ref = ETA * G0 * sgd_steps(h, 0.01)
        for val in [1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            yy = sy(val, y0, ph)
            parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x0+pw}" y2="{yy:.2f}" class="grid"/>')
            parts.append(f'<text x="{x0-12}" y="{yy+4:.2f}" class="tick" text-anchor="end">{val:g}</text>')
        for val in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]:
            xx = sx(val, x0, pw)
            parts.append(f'<line x1="{xx:.2f}" y1="{y0}" x2="{xx:.2f}" y2="{y0+ph}" class="grid"/>')
            parts.append(f'<text x="{xx:.2f}" y="{y0+ph+22}" class="tick" text-anchor="middle">{val:g}</text>')
        for q in [0.1, 0.01]:
            yy = sy(q, y0, ph)
            parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x0+pw}" y2="{yy:.2f}" class="threshold"/>')
            parts.append(f'<text x="{x0+pw+9}" y="{yy+4:.2f}" class="tick" fill="#aa4638">q={q:g}</text>')

        adaptive_samples = {}
        for method in ["Adam", "AdamW", "RMSprop", "Adagrad"]:
            curve_file = STUDY / "results" / method / f"h_{h:g}_g_{G0:g}.json"
            adaptive_samples[method] = json.loads(curve_file.read_text(encoding="utf-8"))
        steps = sorted({sample["step"] for rows in adaptive_samples.values() for sample in rows})
        scalar_curves = {"SGD": scalar_quadratic_curve(h, steps, 0.0),
                         "Momentum": scalar_quadratic_curve(h, steps, MU)}
        u_coordinates, full_u = {}, {}
        for method in METHODS:
            if method in ("SGD", "Momentum"):
                u_coordinates[method] = {t: G0 * ETA * u_step_factor(t, MU if method == "Momentum" else 0.0)
                                         for t in steps}
                samples = [{"step": t, "loss_ratio": scalar_curves[method][t]} for t in steps]
            else:
                u_all = adaptive_u(method, h)
                full_u[method] = u_all
                u_coordinates[method] = {t: u_all[t] for t in steps}
                samples = adaptive_samples[method]
            coords = []
            for sample in samples:
                t = sample["step"]
                if t == 0:
                    continue
                x = sx(u_coordinates[method][t] / u_ref, x0, pw)
                y = sy(sample["loss_ratio"], y0, ph)
                coords.append(f"{x:.2f},{y:.2f}")
            label = {"Momentum": "动量 SGD"}.get(method, method)
            parts.append(f'<path d="M{" L".join(coords)}" fill="none" stroke="{COLORS[method]}" stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round"><title>{label} 已保存/重建的归一化损失轨迹</title></path>')

        for method in METHODS:
            if method == "SGD":
                endpoint = sustained_endpoint(h, 0.01, 0.0)
            elif method == "Momentum":
                endpoint = sustained_endpoint(h, 0.01, MU)
            else:
                cell = next(c for c in summary["cells"] if c["optimizer"] == method
                            and abs(c["h"]-h) < 1e-12 and c["g0"] == G0 and c["q"] == 0.01)
                endpoint = cell["t_sustained_observed"]
            if endpoint is not None:
                if method == "SGD":
                    u_endpoint = G0 * ETA * endpoint
                elif method == "Momentum":
                    u_endpoint = G0 * ETA * u_step_factor(endpoint, MU)
                else:
                    u_endpoint = full_u[method][endpoint]
                xx = sx(u_endpoint / u_ref, x0, pw)
                yy = sy(0.01, y0, ph)
                parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5.2" fill="{COLORS[method]}" stroke="white" stroke-width="1.8"><title>{method} 持续达到 q=0.01 的端点，t={endpoint}</title></circle>')

        parts.append(f'<text x="{x0+pw/2}" y="{y0+ph+58}" class="axis" text-anchor="middle">U(t) / U_SGD(q=0.01)（对数刻度）</text>')
        parts.append(f'<text transform="translate(30 {y0+ph/2}) rotate(-90)" class="axis" text-anchor="middle">归一化损失 L/L₀（对数刻度）</text>')
        parts.append('<text x="70" y="735" class="note">U(t) 按恒定初始梯度公式标定，AdamW 计入解耦衰减；圆点表示持续端点，悬停曲线或端点可查看说明。</text>')
        parts.append('</svg>')
        out = OUT_DIR / f"D1_unified_U_h_{h:g}.svg"
        out.write_text("\n".join(parts) + "\n", encoding="utf-8")
        written.append(out.name)
    print(f"Wrote {', '.join(written)}; reconstructed SGD/momentum endpoints match saved summaries.")


if __name__ == "__main__":
    main()
