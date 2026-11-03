"""Render saved D1 MLP evidence; never train a model."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullFormatter
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
    "SGD": ("#333D4B", "-", "o"), "Momentum": ("#C48800", "--", "s"),
    "Adam": ("#0072B2", "-", "^"), "AdamW": ("#B4559B", "--", "v"),
    "RMSprop": ("#008B68", "-.", "D"), "Adagrad": ("#D55E00", ":", "P"),
}
FLOOR = 1e-4
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 12,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "svg.hashsalt": "d1-nn-u-detailed-v2",
})


def read_curve(architecture, seed, optimizer):
    path = RUN_ROOT / "results" / architecture / f"seed_{seed}" / f"{optimizer}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_curve(architecture, seed, optimizer):
    denominator = read_curve(architecture, seed, "SGD")[-1]["U_initial_gradient"]
    curve = read_curve(architecture, seed, optimizer)
    x = np.asarray([row["U_initial_gradient"] / denominator for row in curve])
    y = np.asarray([row["train_loss_ratio"] for row in curve])
    assert len(curve) == 401 and curve[-1]["step"] == 2000
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(np.diff(x) > 0) and np.all(y >= 0)
    # Omit U=0 from log-x display, instead of assigning it a positive coordinate.
    return x[1:], y[1:]


def aggregate(runs):
    lo, hi = max(x[0] for x, _ in runs), min(x[-1] for x, _ in runs)
    grid = np.unique(np.concatenate(
        [np.geomspace(lo, hi, 450)] + [x[(x >= lo) & (x <= hi)] for x, _ in runs]))
    aligned = np.vstack([np.interp(grid, x, y) for x, y in runs])
    return grid, np.median(aligned, axis=0), aligned.min(axis=0), aligned.max(axis=0)


def line(ax, x, y, optimizer, linewidth=2.2, floor_mark=True):
    color, linestyle, symbol = STYLES[optimizer]
    ax.plot(x, np.where(y >= FLOOR, y, np.nan), color=color, linestyle=linestyle,
            linewidth=linewidth, marker=symbol, markevery=max(1, len(x) // 14),
            markersize=4.4, markerfacecolor="white", markeredgewidth=1.1)
    below = np.flatnonzero(y < FLOOR)
    if len(below) and floor_mark:
        ax.scatter(x[below[0]], FLOOR, marker="v", s=34, color=color, zorder=5, clip_on=False)


def decorate(ax, overview=True):
    ax.set_xscale("log")
    if overview:
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 1.12)
        ax.set_ylabel("训练损失比  L(t) / L(0)（对数）", fontsize=12)
    else:
        ax.set_ylim(.25, 1.02)
        ax.set_yticks([.3, .5, .7, .9, 1.0])
        ax.set_ylabel("训练损失比（线性）", fontsize=12)
    for q in ((.9, .5, .1, .01) if overview else (.9, .5)):
        ax.axhline(q, color="#9F5156", linewidth=1.25 if q == .5 else .8,
                   linestyle=(0, (5, 4)), alpha=.8, zorder=0)
        ax.text(.99, q, f" q={q:g} ", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=10, color="#8E454B",
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": .4})
    ax.axvline(1, color="#77818D", linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    ax.set_xlabel("归一化 U：U(t) / U_SGD(2000)（对数）", fontsize=12, labelpad=10)
    ax.grid(which="major", color="#E4E8EC", linewidth=.7)
    ax.tick_params(labelsize=11)


def handles():
    return [Line2D([], [], color=STYLES[o][0], linestyle=STYLES[o][1],
                   marker=STYLES[o][2], markerfacecolor="white",
                   linewidth=2.3, markersize=5.5, label=o) for o in OPTIMIZERS]


def save(fig, filename, preview_dir):
    path = OUTPUT_DIR / filename
    fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    # Keep generated text portable across Windows and Linux checkouts.
    svg = path.read_text(encoding="utf-8")
    path.write_text("\n".join(part.rstrip() for part in svg.splitlines()) + "\n",
                    encoding="utf-8", newline="\n")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / path.with_suffix(".png").name, dpi=120, facecolor="white")
    plt.close(fig)
    print(path.name)


def make_figures(architecture, summary, preview_dir):
    display_name = ARCHITECTURES[architecture][0]
    curves = {o: [normalized_curve(architecture, seed, o) for seed in SEEDS]
              for o in OPTIMIZERS}
    aggregates = {o: aggregate(runs) for o, runs in curves.items()}
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 8.0),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.075, right=.975, top=.80, bottom=.25, wspace=.26)
    fig.text(.075, .94, display_name + "：U 标定后的损失轨迹", fontsize=20, weight="bold")
    count = next(r["parameter_count"] for r in summary["cell_records"]
                 if r["architecture"] == architecture)
    fig.text(.075, .89, f"{count:,} 个参数 · 六种优化器 · seed 11 / 23 / 37 · 同一初始模型配对",
             fontsize=12, color="#52606D")
    near = []
    for optimizer, (grid, median, low, high) in aggregates.items():
        near.append(grid[np.argmin(np.abs(median - .5))])
        for ax in axes:
            ax.fill_between(grid, np.maximum(low, FLOOR), np.maximum(high, FLOOR),
                            where=high >= FLOOR, color=STYLES[optimizer][0], alpha=.09, linewidth=0)
            line(ax, grid, median, optimizer, floor_mark=ax is axes[0])
    decorate(axes[0])
    decorate(axes[1], overview=False)
    all_max = max(max(x[-1] for x, _ in runs) for runs in curves.values())
    axes[0].set_xlim(.002, all_max * 1.18)
    axes[1].set_xlim(min(min(near), 1) * .45, max(max(near), 1) * 1.8)
    left, right = axes[1].get_xlim()
    axes[1].set_xticks([x for x in [.1, .2, .5, 1, 2, 5, 10, 20, 50] if left <= x <= right])
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_title("全程概览", loc="left", fontsize=14, pad=14)
    axes[1].set_title("q=0.5 附近放大", loc="left", fontsize=14, pad=14)
    fig.legend(handles=handles(), ncol=6, loc="lower center", bbox_to_anchor=(.53, .13),
               frameon=False, fontsize=12, handlelength=2.6, columnspacing=1.3)
    fig.text(.075, .095, "粗线：三个 seed 的逐点中位数；浅带：三个 seed 的最小值—最大值范围。",
             fontsize=11, color="#52606D")
    fig.text(.075, .057, "仅在共同采样范围内沿 U 线性插值；不外推。竖虚线 x=1 为 SGD 第 2000 步的 U。",
             fontsize=10.5, color="#52606D")
    fig.text(.075, .023, "曲线每 5 步保存一次；t=0 不显示在对数横轴上。▼ 表示低于 10⁻⁴，未画成平坦收敛段。",
             fontsize=10.5, color="#52606D")
    save(fig, f"D1_nn_u_{architecture}.svg", preview_dir)
    detail, axs = plt.subplots(3, 1, figsize=(13.8, 16.8))
    detail.subplots_adjust(left=.10, right=.97, top=.88, bottom=.055, hspace=.39)
    detail.text(.10, .97, display_name + "：逐 seed 保存轨迹", fontsize=21, weight="bold")
    detail.text(.10, .94, "每条线连接 400 个正 U 采样点；图形符号仅稀疏显示采样位置。", fontsize=11, color="#52606D")
    for i, (seed, ax) in enumerate(zip(SEEDS, axs)):
        for o in OPTIMIZERS:
            x, y = curves[o][i]
            line(ax, x, y, o, linewidth=1.9)
        decorate(ax)
        ax.set_xlim(.002, all_max * 1.18)
        ax.set_title(f"seed = {seed}", loc="left", fontsize=14, pad=10)
    detail.legend(handles=handles(), ncol=6, loc="upper center", bbox_to_anchor=(.53, .925),
                  frameon=False, fontsize=11, handlelength=2.6, columnspacing=1.3)
    save(detail, f"D1_nn_u_{architecture}_seeds.svg", preview_dir)


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
        ("曲线保存与汇总", "每 5 步保存一次，包含 t=0 和 t=2000，共 401 个点。t=0 因 U=0 不显示在对数横轴；主图仅在三个 seed 共同的正 U 范围内按 U 线性插值后取逐点中位数与最小—最大范围，不平滑、不外推；右图仅放大坐标。逐 seed 图直接连接保存的点。"),
        ("显示与分辨率", "x=U_method,s(t)/U_SGD,s(2000)，按同一结构和 seed 归一化，x=1 是 SGD 预算末端。主图左侧双对数、右侧纵轴线性；低于 10⁻⁴ 的部分留空并以向下三角标记。若前 5 步已穿过阈值，采样曲线不能恢复这一段；精确端点取自逐步计算后保存的 summary。"),
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
<p><strong>记录差异：</strong>预注册文字把 teacher 写成“两隐藏层”，冻结源码实际为一个宽 32 的隐藏层；本页按执行实现描述，并保留这项协议文字与实现的差异。预注册端点主指标为“首次达到”；持续端点是另存的辅助指标。图中用 SGD 第 2000 步 U 归一化属于展示后处理；预注册提到的 q=0.5 端点匹配仍用于原误差指标，深层结构在该阈值缺少参考端点。</p>
<p><strong>证据与范围：</strong>预注册 commit <code>7b4921a</code>，结果 commit <code>1fdabc0</code>；PyTorch {html.escape(summary['torch_version'])}、NumPy {html.escape(summary['numpy_version'])}。产物目录 <code>AIQ_KB_DATA/runs/d1_nn_u_transfer_r1/directions/D1_u_effective_time/studies/r001_nn_u_transfer/</code>；<code>preregistration.json</code> 为协议，<code>run.py</code> 为执行实现，<code>summary.json</code> 为逐步端点摘要，<code>results/{architecture}/seed_*/</code> 为五步采样轨迹。本图使用 18 个训练单元；本轮合计 54 个单元、216 个优化器×阈值记录。结论只覆盖该合成回归、这些结构和固定参数，不构成真实任务结果或优化器排名。</p>
</details>"""


def update_page(summary):
    page = OUTPUT_DIR / "D1_u_effective_time.html"
    text = page.read_text(encoding="utf-8")
    start = text.index("<p>每张图左侧") if "<p>每张图左侧" in text else text.index("<p>下面三张图")
    end = text.index("<p><strong>结论：</strong>", start)
    blocks = ['<p>每张图左侧展示完整 U 尺度轨迹，右侧放大 q=0.5 附近；不同颜色、线型和符号区分优化器。粗线为三个配对 seed 的逐点中位数，浅带为三个 seed 的最小—最大范围。每图下方可展开逐 seed 原始采样图和完整 setting。横轴按同一结构、同一 seed 的 SGD 第 2000 步 U 归一化；图用于观察对齐程度，数值判定仍来自逐步端点记录。</p>\n']
    for i, arch in enumerate(ARCHITECTURES):
        title = ARCHITECTURES[arch][0]
        finding = ("SGD 与 Momentum 在 q=0.5 附近接近，自适应优化器未普遍对齐。"
                   if i < 2 else "SGD 在 2000 步内未到 q=0.5，端点匹配缺少参考。")
        filename, seed_file = f"D1_nn_u_{arch}.svg", f"D1_nn_u_{arch}_seeds.svg"
        blocks.append(f"""<figure class="experiment-chart u-scale-chart nn-chart" aria-labelledby="d1-nn-u-{arch}">
<figcaption id="d1-nn-u-{arch}"><strong>{title}</strong><span>{finding}</span><a href="{filename}" target="_blank" rel="noopener">打开大图 ↗</a></figcaption>
<a class="nn-image-link" href="{filename}" target="_blank" rel="noopener"><img src="{filename}" alt="{title}的U尺度损失全程图和q等于0.5附近放大图" loading="lazy"></a>
<p class="nn-view-hint">窄屏可左右滑动图像，或点击“打开大图”缩放查看。</p>
<p class="chart-note">浅带是三个 seed 的观测范围，不是置信区间。Adam 与 AdamW 高度接近时，以蓝色实线/上三角和紫色虚线/下三角辨认。曲线每 5 步保存一次；RMSprop 可能在首个正 U 采样点之前达到阈值，放大图不能恢复缺失的逐步轨迹。精确首次与持续端点见下方 setting。</p>
<details class="experiment-setting nn-seeds"><summary>展开逐 seed 保存轨迹（11、23、37）</summary>
<p>三行分别对应三个独立 seed，不作跨 seed 插值或汇总。只连接保存的点；符号稀疏标示采样位置，末端为第 2000 步。</p>
<p><a href="{seed_file}" target="_blank" rel="noopener">打开逐 seed 原尺寸图 ↗</a></p>
<a href="{seed_file}" target="_blank" rel="noopener"><img src="{seed_file}" alt="{title}的三个seed分别绘制的六优化器轨迹" loading="lazy"></a>
</details>
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
.nn-chart img{min-width:0;width:100%;height:auto}.nn-chart .nn-image-link{display:block;margin:.5rem -1rem}
.nn-setting{max-width:1100px;margin-left:auto;margin-right:auto}.nn-setting table{font-size:.93rem}.nn-setting th{vertical-align:top;min-width:7em}.nn-setting td{line-height:1.75}.nn-seeds img{margin-top:1rem}.nn-chart a{overflow-wrap:anywhere}.nn-view-hint{display:none;font-size:.85rem;color:#64748b}
@media(max-width:800px){.nn-chart .nn-image-link{overflow-x:auto}.nn-chart .nn-image-link img{min-width:980px}.nn-seeds{overflow-x:auto}.nn-seeds img{min-width:900px}.nn-view-hint{display:block}}
"""
    if ".nn-chart img" in text:
        css_start = text.index(".nn-chart{") if ".nn-chart{" in text else text.index(".nn-chart img")
        css_end = text.index("</style>", css_start)
        text = text[:css_start] + css + text[css_end:]
    else:
        text = text.replace("</style>", css + "</style>", 1)
    page.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--update-page", action="store_true")
    args = parser.parse_args()
    summary = json.loads((RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((RUN_ROOT / "run.py").read_bytes()).hexdigest() == summary["execution_source_sha256"]
    for architecture in ARCHITECTURES:
        make_figures(architecture, summary, args.preview_dir)
    if args.update_page:
        update_page(summary)


if __name__ == "__main__":
    main()
