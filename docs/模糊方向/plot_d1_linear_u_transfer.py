"""Render saved D1 no-ReLU trajectories; this script never trains a model."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
RUN_ROOT = (REPO_ROOT.parent / "AIQ_KB_DATA" / "runs" / "d1_no_relu_20261105"
            / "directions" / "D1_u_effective_time" / "studies" / "r001_no_relu")
ARCHITECTURES = {
    "width64_depth2": "宽 64 · 2 个隐藏层 · 无 ReLU",
    "width128_depth2": "宽 128 · 2 个隐藏层 · 无 ReLU",
    "width128_depth4": "宽 128 · 4 个隐藏层 · 无 ReLU",
}
SEEDS = (11, 23, 37)
OPTIMIZERS = ("SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad")
COLORS = {
    "SGD": "#333D4B", "Momentum": "#C48800", "Adam": "#0072B2",
    "AdamW": "#B4559B", "RMSprop": "#008B68", "Adagrad": "#D55E00",
}
SEED_DASHES = {11: "solid", 23: "dash", 37: "dot"}
SEED_MARKERS = {11: "circle", 23: "square", 37: "triangle-up"}
PLOTLY_CDN = "https://cdn.plot.ly/plotly-4.1.1.min.js"


def decimal_log_ticks(lower: float, upper: float) -> tuple[list[float], list[str]]:
    """Use decimal labels at familiar values, including when the floor is nonzero."""
    values = []
    for exponent in range(math.floor(math.log10(lower)) - 1,
                          math.ceil(math.log10(upper)) + 1):
        for multiplier in (1, 2, 3, 5, 7):
            value = multiplier * 10.0 ** exponent
            if lower <= value <= upper:
                values.append(value)
    if len(values) < 3:
        values = np.geomspace(lower, upper, 5).tolist()
    labels = [format(value, ".3g") if value < 1e-4 else
              np.format_float_positional(value, precision=4, trim="-")
              for value in values]
    return values, labels


def make_interactive_plot(architecture: str, summary: dict,
                          run_root: Path = RUN_ROOT,
                          output_dir: Path = OUTPUT_DIR) -> Path:
    records = {(record["optimizer"], record["seed"]): record
               for record in summary["cell_records"]
               if record["architecture"] == architecture}
    runs = {}
    all_end_u = []
    useful_u_endpoints = []
    all_positive_losses = []
    trajectory_sources = {}
    for method in OPTIMIZERS:
        for seed in SEEDS:
            record = records[(method, seed)]
            if record.get("metrics_file"):
                path = run_root / record["metrics_file"]
                metrics = json.loads(path.read_text(encoding="utf-8"))
                u = np.array(metrics["U_initial_gradient"], dtype=float)
                loss = np.array(metrics["train_loss_ratio"], dtype=float)
                steps = np.array(metrics["steps"], dtype=int)
                expected_steps = np.arange(2001)
                source_kind = "every-step metrics"
            else:
                path = run_root / record.get(
                    "curve_file", f"results/{architecture}/seed_{seed}/{method}.json")
                rows = json.loads(path.read_text(encoding="utf-8"))
                u = np.array([row["U_initial_gradient"] for row in rows], dtype=float)
                loss = np.array([row["train_loss_ratio"] for row in rows], dtype=float)
                steps = np.array([row["step"] for row in rows], dtype=int)
                expected_steps = np.arange(0, 2001, 5)
                source_kind = "five-step legacy fallback"
            if (not np.array_equal(steps, expected_steps)
                    or u.shape != steps.shape or loss.shape != steps.shape):
                raise ValueError(f"Incomplete saved trajectory: {path}")
            trajectory_sources[f"{method}/seed_{seed}"] = {
                "file": path.relative_to(run_root).as_posix(),
                "kind": source_kind, "points": len(steps),
            }
            floor = float(record["affine_floor_loss_ratio"])
            if (not np.all(np.isfinite(u)) or not np.all(np.isfinite(loss))
                    or not np.all(np.diff(u) > 0) or np.min(loss) < 0
                    or not np.isfinite(floor) or not 0 <= floor <= 1):
                raise ValueError(f"Invalid trajectory or affine floor: {path}")
            target_loss = floor + 0.01 * (1.0 - floor)
            crossings = np.flatnonzero(loss <= target_loss)
            if crossings.size:
                useful_u_endpoints.append(float(u[crossings[0]]))
            all_end_u.append(float(u[-1]))
            all_positive_losses.extend(loss[loss > 0].tolist())
            runs[(method, seed)] = (u.tolist(), loss.tolist(), steps.tolist(), floor)

    maximum_u = max(all_end_u)
    shared_u_end = min(all_end_u)
    useful_u_maximum = max(useful_u_endpoints, default=maximum_u)
    focus_x_max = useful_u_maximum * 1.04
    full_x_max = maximum_u * 1.04
    floors = [run[3] for run in runs.values()]
    positive_floors = [floor for floor in floors if floor > 0]
    lower_loss = min(positive_floors) * 0.9 if positive_floors else min(all_positive_losses) * 0.9
    focus_peak = max(value for u, loss, _, _ in runs.values()
                     for x, value in zip(u, loss) if x <= focus_x_max)
    focus_upper_loss = max(1.05, focus_peak * 1.05)
    full_lower_loss = min(lower_loss, min(all_positive_losses) * 0.9)
    full_upper_loss = max(1.05, max(all_positive_losses) * 1.05)
    focus_y_range = [math.log10(lower_loss), math.log10(focus_upper_loss)]
    full_y_range = [math.log10(full_lower_loss), math.log10(full_upper_loss)]
    focus_ticks, focus_tick_labels = decimal_log_ticks(lower_loss, focus_upper_loss)
    full_ticks, full_tick_labels = decimal_log_ticks(full_lower_loss, full_upper_loss)

    traces = []
    for method in OPTIMIZERS:
        for seed in SEEDS:
            u, loss, steps, floor = runs[(method, seed)]
            color = COLORS[method]
            visible = seed == SEEDS[0]
            shared = {"name": method, "legendgroup": method,
                      "visible": visible, "type": "scatter"}
            traces.append({
                **shared, "mode": "lines", "x": u, "y": loss,
                "showlegend": seed == SEEDS[0],
                "meta": {"seed": seed, "optimizer": method, "kind": "curve"},
                "line": {"color": color, "dash": SEED_DASHES[seed], "width": 2.5},
                "customdata": [[step, seed] for step in steps],
                "hovertemplate": (f"{method}<br>seed=%{{customdata[1]}}"
                                  "<br>step=%{customdata[0]}<br>U=%{x:.6g}"
                                  "<br>L(t)/L(0)=%{y:.6g}<extra></extra>"),
            })
            traces.append({
                **shared, "mode": "markers", "x": [u[-1]], "y": [loss[-1]],
                "showlegend": False,
                "meta": {"seed": seed, "optimizer": method, "kind": "step2000"},
                "marker": {"color": color, "symbol": SEED_MARKERS[seed], "size": 10,
                           "line": {"color": "white", "width": 1.4}},
                "customdata": [[2000, seed]],
                "hovertemplate": (f"{method}<br>seed=%{{customdata[1]}}"
                                  "<br><b>step=2000</b><br>U=%{x:.6g}"
                                  "<br>L(t)/L(0)=%{y:.6g}<extra></extra>"),
            })
            traces.append({
                **shared, "mode": "markers", "x": [shared_u_end],
                "y": [float(np.interp(shared_u_end, u, loss))], "showlegend": False,
                "meta": {"seed": seed, "optimizer": method, "kind": "shared_u"},
                "marker": {"color": color, "symbol": "diamond-open", "size": 10,
                           "line": {"color": color, "width": 2}},
                "customdata": [[seed, float(np.interp(shared_u_end, u, steps))]],
                "hovertemplate": (f"{method}<br>seed=%{{customdata[0]}}"
                                  "<br><b>共同 U 比较点</b><br>step≈%{customdata[1]:.1f}"
                                  "<br>U=%{x:.6g}<br>L(t)/L(0)=%{y:.6g}<extra></extra>"),
            })
    for seed in SEEDS:
        seed_floors = [runs[(method, seed)][3] for method in OPTIMIZERS]
        if not np.allclose(seed_floors, seed_floors[0], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Optimizer pairing changed the initial loss: {architecture}, seed={seed}")
        floor = seed_floors[0]
        traces.append({
            "type": "scatter", "mode": "lines", "x": [0, full_x_max], "y": [floor, floor],
            "name": "仿射 OLS 地板", "showlegend": False, "visible": seed == SEEDS[0],
            "meta": {"seed": seed, "kind": "affine_floor"},
            "line": {"color": "#94A3B8", "dash": "dash", "width": 1},
            "hovertemplate": (f"seed={seed}<br>仿射 OLS 地板<br>L_affine*/L(0)={floor:.6g}"
                              "<br>来源：同一训练集含截距的最小二乘解<extra></extra>"),
        })

    layout = {
        "autosize": True, "height": 680,
        "margin": {"l": 84, "r": 24, "t": 20, "b": 112},
        "paper_bgcolor": "white", "plot_bgcolor": "white",
        "hovermode": "closest", "dragmode": "zoom",
        "xaxis": {"title": {"text": "有效时间 U(t)"}, "type": "linear",
                  "range": [0, focus_x_max], "gridcolor": "#E4E8EC", "zeroline": False,
                  "showline": True, "linecolor": "#64748B", "ticks": "outside", "ticklen": 5},
        "yaxis": {"title": {"text": "训练损失比  L(t) / L(0)"}, "type": "log",
                  "range": focus_y_range, "tickmode": "array", "tickvals": focus_ticks,
                  "ticktext": focus_tick_labels, "ticks": "outside", "ticklen": 5,
                  "tickcolor": "#64748B", "gridcolor": "#E4E8EC", "zeroline": False,
                  "showline": True, "linecolor": "#64748B"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22,
                   "groupclick": "togglegroup", "font": {"size": 12}},
        "font": {"family": "Arial, Microsoft YaHei, sans-serif", "size": 13, "color": "#334155"},
    }
    provenance = {
        "study": summary.get("study"), "preregistration_commit": summary.get("preregistration_commit"),
        "execution_source_sha256": summary.get("execution_source_sha256"),
        "architecture": architecture, "focus_rule": "first L/L0 <= floor + 0.01 * (1-floor), maximum U over reached trajectories",
        "focus_u_end": useful_u_maximum, "shared_u": shared_u_end,
        "training_cells": len(runs), "focus_threshold_reached": len(useful_u_endpoints),
        "trajectory_sources": trajectory_sources,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"D1_linear_u_{architecture}_interactive.html"
    output.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(ARCHITECTURES[architecture])} · D1 U 轨迹</title>
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
<style>
html,body{{margin:0;padding:0;background:#fff;font-family:Arial,"Microsoft YaHei",sans-serif;color:#334155}}
.controls{{display:flex;align-items:center;gap:.6rem;padding:.35rem .75rem .1rem;font-size:14px}}
.embed-mode .controls{{display:none}}
.controls label{{color:#475569}}
.controls select,.controls button{{font:inherit;padding:.25rem .45rem;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#334155}}
.controls button{{cursor:pointer}}
#chart{{width:100%;height:680px}}
@media(max-width:700px){{#chart{{height:620px}}.controls{{font-size:13px}}}}
</style></head><body>
<div class="controls"><label for="seed-filter">seed</label>
<select id="seed-filter" aria-label="选择显示的 seed">
<option value="11" selected>11</option><option value="23">23</option>
<option value="37">37</option><option value="all">全部</option></select>
<button id="u-range-toggle" type="button">查看完整坐标范围</button></div>
<div id="chart" role="img" aria-label="无 ReLU 网络各优化器的 U 和训练损失曲线"></div>
<script id="provenance" type="application/json">{json.dumps(provenance, ensure_ascii=False)}</script>
<script>
const traces = {json.dumps(traces, ensure_ascii=False, separators=(",", ":"))};
const layout = {json.dumps(layout, ensure_ascii=False, separators=(",", ":"))};
const focusXRange = [0, {focus_x_max!r}];
const fullXRange = [0, {full_x_max!r}];
const focusYRange = {json.dumps(focus_y_range)};
const fullYRange = {json.dumps(full_y_range)};
const focusYTicks = {json.dumps(focus_ticks)};
const focusYLabels = {json.dumps(focus_tick_labels)};
const fullYTicks = {json.dumps(full_ticks)};
const fullYLabels = {json.dumps(full_tick_labels)};
const chart = document.getElementById("chart");
const embedMode = new URLSearchParams(window.location.search).get("embed") === "1";
if (embedMode) {{
  document.body.classList.add("embed-mode");
  layout.dragmode = false;
  layout.legend = Object.assign({{}}, layout.legend, {{itemclick:false, itemdoubleclick:false}});
}}
Plotly.newPlot(chart, traces, layout, {{responsive:true, displaylogo:false,
  displayModeBar:!embedMode, scrollZoom:!embedMode, doubleClick:embedMode ? false : "reset",
  modeBarButtonsToRemove:["select2d","lasso2d"]}});
document.getElementById("seed-filter").addEventListener("change", function() {{
  const selected = this.value;
  Plotly.restyle(chart, {{visible:traces.map(trace => selected === "all" || trace.meta.seed === Number(selected))}});
}});
document.getElementById("u-range-toggle").addEventListener("click", function() {{
  const showingFull = this.dataset.full === "true";
  Plotly.relayout(chart, {{"xaxis.range": showingFull ? focusXRange : fullXRange,
    "yaxis.range": showingFull ? focusYRange : fullYRange,
    "yaxis.tickvals": showingFull ? focusYTicks : fullYTicks,
    "yaxis.ticktext": showingFull ? focusYLabels : fullYLabels}});
  this.dataset.full = String(!showingFull);
  this.textContent = showingFull ? "查看完整坐标范围" : "回到重点区间";
}});
</script></body></html>
""", encoding="utf-8", newline="\n")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    source = args.run_root / "run.py"
    if source.exists() and summary.get("execution_source_sha256"):
        if hashlib.sha256(source.read_bytes()).hexdigest() != summary["execution_source_sha256"]:
            raise ValueError("Saved execution source does not match the summary hash")
    for architecture in ARCHITECTURES:
        make_interactive_plot(architecture, summary, args.run_root, args.output_dir)


if __name__ == "__main__":
    main()
