"""Render saved D1 MLP evidence; never train a model."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
RUN_ROOT = (REPO_ROOT.parent / "AIQ_KB_DATA" / "runs" / "d1_nn_u_transfer_r1"
            / "directions" / "D1_u_effective_time" / "studies" / "r001_nn_u_transfer")
ARCHITECTURES = {
    "width64_depth2": ("宽 64 · 2 个隐藏层", "16 → 64 → 64 → 1"),
    "width128_depth2": ("宽 128 · 2 个隐藏层", "16 → 128 → 128 → 1"),
    "width128_depth4": ("宽 128 · 4 个隐藏层", "16 → 128 → 128 → 128 → 128 → 1"),
}
SEEDS = (11, 23, 37)
OPTIMIZERS = ("SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad")
STYLES = {
    "SGD": "#333D4B", "Momentum": "#C48800", "Adam": "#0072B2",
    "AdamW": "#B4559B", "RMSprop": "#008B68", "Adagrad": "#D55E00",
}
SEED_DASHES = {11: "solid", 23: "dash", 37: "dot"}
SEED_MARKERS = {11: "circle", 23: "square", 37: "triangle-up"}
PLOTLY_CDN = "https://cdn.plot.ly/plotly-4.1.1.min.js"


def read_curve(architecture, seed, optimizer):
    path = RUN_ROOT / "results" / architecture / f"seed_{seed}" / f"{optimizer}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_interactive_plot(architecture: str) -> Path:
    runs = {}
    all_end_u = []
    all_positive_loss = []
    useful_u_endpoints = []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            rows = read_curve(architecture, seed, optimizer)
            assert len(rows) == 401 and rows[0]["step"] == 0 and rows[-1]["step"] == 2000
            u = [float(row["U_initial_gradient"]) for row in rows]
            loss = [float(row["train_loss_ratio"]) for row in rows]
            steps = [int(row["step"]) for row in rows]
            assert all(b > a for a, b in zip(u, u[1:]))
            assert all(np.isfinite(u)) and all(np.isfinite(loss)) and min(loss) >= 0
            runs[(optimizer, seed)] = (u, loss, steps)
            all_end_u.append(u[-1])
            all_positive_loss.extend(value for value in loss if value > 0)
            threshold_crossings = [i for i, value in enumerate(loss) if value <= 0.01]
            if threshold_crossings:
                useful_u_endpoints.append(u[threshold_crossings[0]])

    shared_u_end = min(all_end_u)
    maximum_u = max(all_end_u)
    useful_u_maximum = max(useful_u_endpoints)
    minimum_positive_loss = min(all_positive_loss)
    traces = []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            u, loss, steps = runs[(optimizer, seed)]
            color = STYLES[optimizer]
            custom = [[step, seed] for step in steps]
            visible = seed == SEEDS[0]
            traces.append({
                "type": "scatter", "mode": "lines", "x": u, "y": loss,
                "name": optimizer, "legendgroup": optimizer,
                "showlegend": seed == SEEDS[0], "visible": visible,
                "meta": {"seed": seed, "optimizer": optimizer, "kind": "curve"},
                "line": {"color": color, "dash": SEED_DASHES[seed], "width": 2.5},
                "customdata": custom,
                "hovertemplate": (
                    f"{optimizer}<br>seed=%{{customdata[1]}}"
                    "<br>step=%{customdata[0]}<br>U=%{x:.6g}"
                    "<br>L(t)/L(0)=%{y:.5g}<extra></extra>"),
            })
            traces.append({
                "type": "scatter", "mode": "markers", "x": [u[-1]], "y": [loss[-1]],
                "name": optimizer, "legendgroup": optimizer, "showlegend": False,
                "visible": visible,
                "meta": {"seed": seed, "optimizer": optimizer, "kind": "step2000"},
                "marker": {"color": color, "symbol": SEED_MARKERS[seed], "size": 10,
                           "line": {"color": "white", "width": 1.4}},
                "customdata": [[2000, seed]],
                "hovertemplate": (
                    f"{optimizer}<br>seed=%{{customdata[1]}}"
                    "<br><b>step=2000</b><br>U=%{x:.6g}"
                    "<br>L(t)/L(0)=%{y:.5g}<extra></extra>"),
            })
            common_loss = float(np.interp(shared_u_end, u, loss))
            common_step = float(np.interp(shared_u_end, u, steps))
            traces.append({
                "type": "scatter", "mode": "markers", "x": [shared_u_end], "y": [common_loss],
                "name": optimizer, "legendgroup": optimizer, "showlegend": False,
                "visible": visible,
                "meta": {"seed": seed, "optimizer": optimizer, "kind": "shared_u"},
                "marker": {"color": color, "symbol": "diamond-open", "size": 10,
                           "line": {"color": color, "width": 2}},
                "customdata": [[seed, common_step]],
                "hovertemplate": (
                    f"{optimizer}<br>seed=%{{customdata[0]}}"
                    "<br><b>共同 U 比较点</b><br>step≈%{customdata[1]:.1f}"
                    "<br>U=%{x:.6g}<br>L(t)/L(0)=%{y:.5g}<extra></extra>"),
            })

    data_json = json.dumps(traces, ensure_ascii=False, separators=(",", ":"))
    layout_json = json.dumps({
        "autosize": True,
        "height": 680,
        "margin": {"l": 78, "r": 24, "t": 20, "b": 112},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "closest",
        "dragmode": "zoom",
        "xaxis": {"title": {"text": "有效时间 U(t)"}, "type": "linear",
                  "range": [0, useful_u_maximum * 1.04], "gridcolor": "#E4E8EC",
                  "zeroline": False, "showline": True, "linecolor": "#64748B"},
        "yaxis": {"title": {"text": "训练损失比  L(t) / L(0)"}, "type": "log",
                  "range": [-2, 0.06],
                  "tickmode": "array", "tickvals": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                  "ticktext": ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"],
                  "ticks": "outside", "ticklen": 5, "tickcolor": "#64748B",
                  "exponentformat": "power", "showexponent": "all",
                  "gridcolor": "#E4E8EC", "zeroline": False,
                  "showline": True, "linecolor": "#64748B"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22,
                   "groupclick": "togglegroup", "font": {"size": 12}},
        "font": {"family": "Arial, Microsoft YaHei, sans-serif", "size": 13,
                 "color": "#334155"},
    }, ensure_ascii=False, separators=(",", ":"))
    document_title = html.escape(ARCHITECTURES[architecture][0])
    output = OUTPUT_DIR / f"D1_nn_u_{architecture}_interactive.html"
    output.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{document_title} · D1 U 轨迹</title>
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
<style>
html,body{{margin:0;padding:0;background:#fff;font-family:Arial,"Microsoft YaHei",sans-serif;color:#334155}}
.controls{{display:flex;align-items:center;gap:.6rem;padding:.35rem .75rem .1rem;font-size:14px}}
.embed-mode .controls{{display:none}}
.controls label{{color:#475569}}
.controls select{{font:inherit;padding:.25rem .45rem;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#334155}}
.controls button{{font:inherit;padding:.25rem .5rem;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#334155;cursor:pointer}}
#chart{{width:100%;height:680px}}
@media(max-width:700px){{#chart{{height:620px}}.controls{{font-size:13px}}}}
</style></head><body>
<div class="controls"><label for="seed-filter">seed</label>
<select id="seed-filter" aria-label="选择显示的 seed">
<option value="11" selected>11</option><option value="23">23</option>
<option value="37">37</option><option value="all">全部</option></select>
<button id="u-range-toggle" type="button">查看完整坐标范围</button></div>
<div id="chart" role="img" aria-label="各优化器的 U 和训练损失交互曲线"></div>
<script>
const traces = {data_json};
const layout = {layout_json};
const usefulUMax = {useful_u_maximum * 1.04!r};
const fullUMax = {maximum_u * 1.04!r};
const fullYMin = {float(np.log10(minimum_positive_loss))!r};
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
  const visible = traces.map(trace => selected === "all" || trace.meta.seed === Number(selected));
  Plotly.restyle(chart, {{visible:visible}});
}});
document.getElementById("u-range-toggle").addEventListener("click", function() {{
  const showingFull = this.dataset.full === "true";
  const nextMaximum = showingFull ? usefulUMax : fullUMax;
  Plotly.relayout(chart, {{"xaxis.range": [0, nextMaximum],
    "yaxis.range": showingFull ? [-2, 0.06] : [fullYMin, 0.06],
    "yaxis.tickmode": showingFull ? "array" : "linear", "yaxis.dtick": 4}});
  this.dataset.full = String(!showingFull);
  this.textContent = showingFull ? "查看完整坐标范围" : "回到重点区间";
}});
</script></body></html>
""", encoding="utf-8", newline="\n")
    print(output.name)
    return output


def setting_html(architecture, summary):
    records = [r for r in summary["cell_records"] if r["architecture"] == architecture]
    nparams = records[0]["parameter_count"]
    ref = {r["seed"]: r for r in records if r["optimizer"] == "SGD"}
    ref_text = "；".join(
        f"seed {s}：首次达到 q=0.5 为 "
        + (f"{ref[s]['train_endpoints']['0.5']['first']} 步"
           if ref[s]["train_endpoints"]["0.5"]["first"] is not None else "未达到（2000 步截尾）")
        + f"，末端损失比 {ref[s]['final_train_loss_ratio']:.3f}" for s in SEEDS)
    settings = [
        ("student 网络", f"{ARCHITECTURES[architecture][1]}；共 {nparams:,} 个可训练参数。深度指隐藏层数；每个隐藏层为 Linear + ReLU，最后线性输出标量；各 Linear 含 bias。所有层同时训练，无 dropout、归一化层、残差或参数冻结。"),
        ("输入与目标", "训练/验证输入各 256×16，独立标准正态采样。teacher 实际为 16 → 32 → 1：ReLU(XW₁)W₂，bias 为零；W₁ 元素按 N(0,2/16)、W₂ 按 N(0,1/32) 采样，无额外标签噪声。"),
        ("标准化与损失", "用训练目标均值和总体标准差（unbiased=False）做中心化和缩放，验证目标用同一变换。L(t)=mean[(fθ(x)−y)²]；图中为 L(t)/L(0)，不减最优损失。验证集未用于调参、选点或作本图纵轴。"),
        ("seed 与初始化", "seed 为 11、23、37。数据生成器 seed=100000+s；初始化 seed=200000+1000a+s，a=0/1/2 对应三种结构的顺序。student 使用 PyTorch nn.Linear 默认初始化。同一结构、同一 seed 的六种优化器共享数据、初始权重和初始损失。"),
        ("训练协议", "CPU、float64、单线程、PyTorch deterministic algorithms；全批量 256 个样本，不洗牌，无随机层。所有优化器学习率恒为 0.001，无学习率调度，统一训练 2000 次更新。未搜索各优化器学习率。"),
        ("SGD 参考结果", ref_text),
        ("曲线保存与汇总", "每 5 步保存一次，包含 t=0 和 t=2000，共 401 个点。交互图保留每条轨迹直到第 2000 步的全部数据；实心符号标出 t=2000，空心菱形标出所有 seed 和优化器共同达到的最大 U。按 seed 筛选或切换完整 U 范围时，端点标记仍对应原始第 2000 步。"),
        ("显示与分辨率", "x=U_method,s(t)=‖θᶜ_method,s(t)−θ₀,s‖₂，直接使用原值和线性刻度；六种优化器（含标准 SGD）共用同一结构下的 U 坐标范围。训练损失比 L(t)/L(0) 使用对数纵轴。默认重点视窗 y≥0.01；x 轴右端设为能达到 L(t)/L(0)≤0.01 的轨迹中首次达到该阈值的最大 U。阈值对应相对初始损失至少下降 99%，聚焦主要下降区段。全部原始轨迹和 t=2000 标记均保留，按钮可切换完整坐标范围。"),
    ]
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in settings)
    optimizer_rows = (
        "<tr><td>SGD</td><td>momentum=0，weight_decay=0</td></tr>"
        "<tr><td>Momentum SGD</td><td>momentum=0.9，dampening=0，Nesterov=false，weight_decay=0</td></tr>"
        "<tr><td>Adam</td><td>β₁=0.9，β₂=0.999，ε=10⁻⁸，偏差修正，weight_decay=0</td></tr>"
        "<tr><td>AdamW</td><td>同 Adam 的 β 和 ε；解耦 weight_decay=0.01，作用于全部参数（含 bias）</td></tr>"
        "<tr><td>RMSprop</td><td>α=0.99，ε=10⁻⁸，momentum=0，centered=false，不作偏差修正，weight_decay=0</td></tr>"
        "<tr><td>Adagrad</td><td>初始累积量=0，ε=10⁻⁸，lr_decay=0，weight_decay=0</td></tr>"
    )
    endpoint_rows = []
    for seed in SEEDS:
        for o in OPTIMIZERS[1:]:
            c = next(c for c in summary["endpoint_cells"] if c["architecture"] == architecture
                     and c["seed"] == seed and c["optimizer"] == o and c["q"] == .5)
            def number(key):
                return "—" if c[key] is None else str(c[key])
            error = "不可比较" if c["relative_absolute_error"] is None else f"{100*c['relative_absolute_error']:.2f}%"
            endpoint_rows.append(
                f"<tr><td>{seed}</td><td>{o}</td><td>{number('sgd_reference_step')}</td>"
                f"<td>{number('predicted_step')}</td><td>{number('observed_first_step')}</td>"
                f"<td>{number('observed_sustained_step')}</td><td>{error}</td></tr>")
    return f"""<details class="experiment-setting nn-setting">
<summary>完整 setting · {ARCHITECTURES[architecture][0]} · 数据、优化器、U 与逐 seed 端点</summary>
<div class="table-wrap"><table><tbody>{rows}</tbody></table></div>
<p><strong>六种优化器的完整参数：</strong>各状态从零开始。除 AdamW 外均不衰减参数。</p>
<div class="table-wrap"><table><thead><tr><th>优化器</th><th>设置（所有学习率均为 0.001）</th></tr></thead><tbody>{optimizer_rows}</tbody></table></div>
<p><strong>U 的计算：</strong>对初始参数 θ₀ 计算完整训练集梯度 g₀；固定 g=g₀，用相应优化器的同一更新规则和零初始状态模拟参数 θᶜ(t)，定义 U(t)=‖θᶜ(t)−θ₀‖₂。SGD 为 ηt‖g₀‖₂；Momentum 包含零启动累积项；Adam 包含偏差修正；AdamW 的总位移包含独立衰减；RMSprop 不做偏差修正；Adagrad 从零平方累积量开始。本节的 U 是参数位移的范数；前文的累计步长系数须先作用于 g₀，再聚合成位移，AdamW 还要计入对 θ₀ 的衰减项。这个标量丢弃方向信息，不等于真实轨迹的累计路程。</p>
<p><strong>预测与判据：</strong>q∈{{0.9,0.5,0.1,0.01}}。同一 seed 的 SGD 首次达到 L(t)/L(0)≤q 时取得目标 U；其他优化器首次达到该 U 的整数步为 t_pred，其真实损失首次达到 q 的步数为 t_obs。误差 E=|t_pred−t_obs|/t_obs，E≤5% 通过；主检验 q=0.5。真实端点、预测端点或参考端点在 2000 步内未达到时保留为空；不补成 2000 步。持续端点是此后直到第 2000 步均满足阈值的首步，仅作辅助记录。</p>
<p><strong>q=0.5 的逐 seed 精确端点：</strong>下表来自逐步阈值记录，不从五步采样曲线反推；“持续”与主指标“首次”分别列出。</p>
<div class="table-wrap"><table><thead><tr><th>seed</th><th>优化器</th><th>SGD 首次</th><th>预测步数</th><th>实测首次</th><th>实测持续</th><th>首次步数误差</th></tr></thead><tbody>{''.join(endpoint_rows)}</tbody></table></div>
<p><strong>记录差异：</strong>预注册文字把 teacher 写成“两隐藏层”，冻结源码实际为一个宽 32 的隐藏层；本页按执行实现描述，并保留这项协议文字与实现的差异。预注册端点主指标为“首次达到”；持续端点是另存的辅助指标。交互图展示未归一化 U，端点匹配依据逐步记录。深层结构在该阈值缺少 SGD 参考端点。</p>
<p><strong>证据与范围：</strong>预注册 commit <code>7b4921a</code>，结果 commit <code>1fdabc0</code>；PyTorch {html.escape(summary['torch_version'])}、NumPy {html.escape(summary['numpy_version'])}。产物目录 <code>AIQ_KB_DATA/runs/d1_nn_u_transfer_r1/directions/D1_u_effective_time/studies/r001_nn_u_transfer/</code>；<code>preregistration.json</code> 为协议，<code>run.py</code> 为执行实现，<code>summary.json</code> 为逐步端点摘要，<code>results/{architecture}/seed_*/</code> 为五步采样轨迹。本图使用 18 个训练单元；本轮合计 54 个单元、216 个优化器×阈值记录。结论只覆盖该合成回归、这些结构和固定参数，不构成真实任务结果或优化器排名。</p>
</details>"""


def update_page(summary):
    page = OUTPUT_DIR / "D1_u_effective_time.html"
    text = page.read_text(encoding="utf-8")
    first_chart = text.index('<figure class="experiment-chart u-scale-chart nn-chart" aria-labelledby="d1-nn-u-')
    start = text.rfind("<p>", 0, first_chart)
    end = text.index("<p><strong>结论：</strong>", start)
    blocks = ['<p>图中保留所有原始轨迹和第 2000 步端点。默认重点视窗显示损失比 0.01 以上，并将横轴聚焦到能达到该损失比的轨迹首次到达阈值的最大 U。达到 0.01 表示损失较初始值下降至少 99%；之后的参数位移仍按定义可计算，但靠近零损失时不再作为主要比较区段。阈值只控制显示范围，不删数据；点击“查看完整坐标范围”可查看完整 U 和损失范围。实心端点表示 t=2000，空心菱形表示所有 seed 和优化器共同达到的最大 U。悬停可查看采样点，下拉菜单可切换 seed。</p>\n']
    for i, arch in enumerate(ARCHITECTURES):
        title = ARCHITECTURES[arch][0]
        filename = f"D1_nn_u_{arch}_interactive.html"
        blocks.append(f"""<figure class="experiment-chart u-scale-chart nn-chart" aria-labelledby="d1-nn-u-{arch}">
<figcaption id="d1-nn-u-{arch}"><strong>{title}</strong><a href="{filename}" target="_blank" rel="noopener">查看完整页面 ↗</a></figcaption>
<iframe class="nn-interactive-frame" src="{filename}?embed=1" title="{title}的 U 轨迹图（悬停查看数值）" loading="lazy"></iframe>
{setting_html(arch, summary)}
</figure>
""")
    text = text[:start] + "".join(blocks) + text[end:]
    text = text.replace("<td>未测</td>", "<td>学习特征 MLP 的迁移检验见第4节；逐坐标位移取欧氏范数后得到标量 U。</td>")
    start = text.index('<p class="chart-note">表中数值')
    end = text.index("</p>", start) + 4
    text = text[:start] + '<p class="chart-note">表中数值为三个 seed 的首次阈值步数相对误差中位数，括号为逐 seed 通过数。E=|t_pred−t_obs|/t_obs，E≤5% 通过；t_obs 按首次达到阈值计算。持续达到阈值是独立辅助指标，逐 seed 值列于每图下的完整 setting。</p>' + text[end:]
    old_setting = '<details class="experiment-setting"><summary>展开查看实验设定'
    if old_setting in text:
        start = text.index(old_setting)
        end = text.index("</details>", start) + len("</details>")
        text = text[:start] + text[end:]
    css = """.nn-chart{width:min(1320px,calc(100vw - 40px));max-width:none;box-sizing:border-box;margin-left:50%;transform:translateX(-50%);font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif}
.nn-interactive-frame{display:block;width:100%;height:740px;border:0;background:#fff}.nn-setting{max-width:1100px;margin-left:auto;margin-right:auto}
.nn-setting table{font-size:.93rem}.nn-setting th{vertical-align:top;min-width:7em}.nn-setting td{line-height:1.75}.nn-chart a{overflow-wrap:anywhere}
@media(max-width:800px){.nn-interactive-frame{height:700px}}
"""
    if ".nn-chart{" in text:
        css_start = text.index(".nn-chart{")
        css_end = text.index("</style>", css_start)
        text = text[:css_start] + css + text[css_end:]
    else:
        text = text.replace("</style>", css + "</style>", 1)
    page.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-page", action="store_true")
    args = parser.parse_args()
    summary = json.loads((RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((RUN_ROOT / "run.py").read_bytes()).hexdigest() == summary["execution_source_sha256"]
    for architecture in ARCHITECTURES:
        make_interactive_plot(architecture)
    if args.update_page:
        update_page(summary)


if __name__ == "__main__":
    main()
