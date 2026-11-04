"""Publish only saved no-ReLU results; retain the original ReLU figures."""
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
DOCS = ROOT / "docs" / "模糊方向"
ARCHS = {"width64_depth2": "宽 64 · 2 个隐藏层", "width128_depth2": "宽 128 · 2 个隐藏层", "width128_depth4": "宽 128 · 4 个隐藏层"}
METHODS = ["Momentum", "Adam", "AdamW", "RMSprop", "Adagrad"]


def load_plot_helper():
    spec = importlib.util.spec_from_file_location("original_d1_plot_helper", DOCS / "plot_d1_nn_u_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(summary):
    counts = {"通过": 0, "未通过": 0, "不可比较": 0}
    for method in METHODS:
        for arch in ARCHS:
            group = summary["primary_by_optimizer_architecture_q"][method][arch]["0.5"]
            key = "不可比较" if group["uncensored_comparisons"] < 3 else "通过" if group["median_relative_error"] <= .05 else "未通过"
            counts[key] += 1
    return counts


def group_label(summary, method, arch):
    g = summary["primary_by_optimizer_architecture_q"][method][arch]["0.5"]
    if g["median_relative_error"] is None:
        return "不可比较（0/3）"
    return f"{100*g['median_relative_error']:.2f}%（≤5%：{g['within_5pct']}/{g['uncensored_comparisons']}）" + (f"；可比较 {g['uncensored_comparisons']}/3" if g["uncensored_comparisons"] < 3 else "")


def settings(arch, summary, helper):
    value = helper.setting_html(arch, summary)
    value = value.replace("每个隐藏层为 Linear + ReLU，最后线性输出标量", "每个隐藏层的 ReLU 替换为 Identity，层间无非线性激活，最后线性输出标量")
    value = value.replace("d1_nn_u_transfer_r1/directions/D1_u_effective_time/studies/r001_nn_u_transfer", "d1_no_relu_20261105/directions/D1_u_effective_time/studies/r001_no_relu")
    value = value.replace("预注册 commit <code>7b4921a</code>，结果 commit <code>1fdabc0</code>", f"预注册 commit <code>{summary['preregistration_commit'][:7]}</code>；结果文件 SHA-256 <code>{hashlib.sha256((SOURCE/'summary.json').read_bytes()).hexdigest()[:12]}</code>")
    display = "主页面默认显示 seed=11，仅悬停可读点值；另外两个 seed、缩放和完整范围切换在完整页面中提供。横轴直接使用未归一化的 U，纵轴为原始 L(t)/L(0) 的对数刻度。对每条轨迹，按仿射下限 r*=L*affine/L0 定义重点阈值 r*+0.01(1−r*)；默认横轴右端取已达到此阈值的轨迹首次到达时的最大 U。该阈值对应消除 99% 的可消除初始损失，只控制视窗。所有原始点和 t=2000 标记都保留，可切换完整范围。虚线为同一训练集的最优仿射损失比下限。"
    value = re.sub(r"(<th>显示与分辨率</th><td>).*?(</td>)", lambda m: m[1]+display+m[2], value, flags=re.S)
    sampling = "每步在 metrics/ 保存一次，含 t=0 和 t=2000，共 2001 个点；另在 results/ 保留每五步采样的 401 点曲线。新交互图使用完整 2001 点，包含 RMSprop 的第 1 步及后续反弹；实心符号标出 t=2000，空心菱形标出所有轨迹共同达到的最大 U。"
    value = re.sub(r"(<th>曲线保存与汇总</th><td>).*?(</td>)", lambda m: m[1]+sampling+m[2], value, flags=re.S)
    note = "<p><strong>配对与模型范围：</strong>teacher 保留原来的单 ReLU 隐藏层；student 仅把 ReLU 改成 Identity，初始 Linear 张量与原模型逐项一致，但函数输出、初始损失及初始梯度随之改变，U 必须重新计算。所有层一起训练；模型对输入是仿射的，对各层参数联合并非二次损失。低于仿射最优残差的阈值属于模型类不可达到，不记为数值预测失败。</p>"
    value = re.sub(r"<p><strong>记录差异：</strong>.*?</p>", note, value, flags=re.S)
    floors = [r for r in summary["cell_records"] if r["architecture"] == arch and r["optimizer"] == "SGD"]
    floor_text = "；".join(f"seed {r['seed']}：L*affine={r['affine_ols_training_loss']:.4f}，L*affine/L0={r['affine_floor_loss_ratio']:.3f}" for r in floors)
    value = value.replace("</details>", f"<p><strong>仿射最小二乘下限：</strong>{floor_text}。由训练输入附加常数列后直接最小二乘求得；只作表达下限诊断，不用验证集选点，不对主损失比减残差。</p></details>")
    value = value.replace("为五步采样轨迹。", "为五步采样轨迹，metrics/ 为全部 2001 个逐步记录，可复算阈值端点。")
    return value


def main():
    summary = json.loads((SOURCE / "summary.json").read_text())
    previous = json.loads((SOURCE / "relu_reference_summary.json").read_text())
    helper = load_plot_helper()
    preserved_assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DOCS.glob("D1_nn_u_*") if p.is_file()}
    page = DOCS / "D1_u_effective_time.html"
    text = page.read_text(encoding="utf-8")
    start = text.index("<h2>4.")
    end = text.index('<div class="stamp">', start)
    archive = DOCS / "D1_nn_relu_u_transfer.html"
    if not archive.exists():
        head = text[:text.index("<body>")]
        head = re.sub(r"<title>.*?</title>", "<title>D1 原 ReLU 实验记录</title>", head)
        extra_css = "\n".join(re.findall(r"<style>.*?</style>", text[text.index("<body>"):start], re.S))
        archive.write_text(head + '<body><main><p><a href="D1_u_effective_time.html">返回 D1</a></p><h1>原 ReLU 实验记录</h1>' + extra_css + text[start:end] + '</main></body></html>\n', encoding="utf-8")
    counts = classify(summary)
    old_counts = classify(previous)
    records = summary["cell_records"]
    floor_ratios = [r["affine_floor_loss_ratio"] for r in records]
    fragments = [f'''<h2>4. 神经网络实验</h2>
<p>在固定的非线性回归任务上，训练三种宽深的无激活多层网络，比较六种优化器的损失曲线。横轴为 U，纵轴为 L(t)/L(0)，完整设置见各图下方。</p>
<p>在本设定下，U 仍未统一所有优化器：以 q=0.5、相对步数误差 ≤5% 为标准，15 个“优化器 × 结构”汇总格中 {counts['通过']} 个通过、{counts['未通过']} 个未通过。</p>
<p class="chart-note">虚线为最优仿射损失比，实心点为第 2000 步，空心菱形为共同 U 位置。</p>
''']
    for arch, title in ARCHS.items():
        filename = f"D1_linear_u_{arch}_interactive.html"
        fragments.append(f'''<figure class="experiment-chart u-scale-chart nn-chart" aria-labelledby="d1-linear-u-{arch}">
<figcaption id="d1-linear-u-{arch}"><strong>{title}</strong><a href="{filename}" target="_blank" rel="noopener">查看完整页面 ↗</a></figcaption>
<iframe class="nn-interactive-frame" src="{filename}?embed=1" title="{title}无激活网络的 U 曲线" loading="lazy"></iframe>
{settings(arch, summary, helper)}
</figure>''')
    rows = "".join("<tr><td>"+title+"</td>"+"".join("<td>"+group_label(summary,m,arch)+"</td>" for m in METHODS)+"</tr>" for arch,title in ARCHS.items())
    fragments.append('<details class="experiment-setting"><summary>端点误差与适用范围</summary>')
    fragments.append('<p><strong>q=0.5 的成立程度：</strong>表中为三个 seed 的相对步数误差中位数，括号为误差 ≤5% 的个数/可比较个数。汇总格需三个 seed 均可比较且中位误差 ≤5% 才计为通过；持续达到阈值另作辅助记录。</p><div class="table-wrap"><table><thead><tr><th>无激活结构</th>'+''.join('<th>'+m+'</th>' for m in METHODS)+'</tr></thead><tbody>'+rows+'</tbody></table></div>')
    fragments.append('<p><strong>首次达到的局限：</strong>RMSprop 的三个汇总格中位误差为 0%，但 9 个配对中有 8 个是预测与实测都在第 1 步跨过 q=0.5，整数步记录的分辨率有限。这不代表整条曲线重合；深 4 结构的三个 seed 都在第 1 步首次跨过阈值，却分别直到第 1495、1741、1646 步才持续保持在阈值内。后期反弹保留在完整范围中，不能因重点视窗截取而忽略。</p>')
    secondary_rows = []
    for q in [.9,.5,.1,.01]:
        cells = [c for c in summary["endpoint_cells"] if c["optimizer"] != "SGD" and c["q"] == q]
        comparable = [c for c in cells if c["relative_absolute_error"] is not None]
        lower_bound_impossible = sum(c["class_unattainable_threshold"] for c in cells)
        secondary_rows.append(f"<tr><td>{q:g}</td><td>{len(comparable)}/45</td><td>{sum(c['within_5pct'] for c in comparable)}/{len(comparable)}</td><td>{lower_bound_impossible}/45</td></tr>")
    fragments.append(f'''<p><strong>下限与截尾：</strong>仿射最优训练损失比的范围为 {min(floor_ratios):.3f}–{max(floor_ratios):.3f}。低于各自下限的 q 在该模型类中不可达到；其余未达到端点仅表示本次 2000 步预算截尾，不用 2000 补齐，也不将缺失端点记成误差超过 5%。</p>
<div class="table-wrap"><table><thead><tr><th>损失比 q</th><th>可比较配对</th><th>误差 ≤5%</th><th>q 低于仿射下限</th></tr></thead><tbody>{''.join(secondary_rows)}</tbody></table></div>
<p><strong>方法与范围：</strong>54 个新训练单元，3 种结构 × 3 个 seed × 6 种优化器；单一学习率 0.001、float64、全批量合成回归。本结果只检验这组标量 U 标定，不构成优化器排名、真实任务结论或 ReLU 失效机理的因果证明。</p>
</details>
''')
    comparison_rows = ''.join('<tr><td>'+title+'</td><td>'+m+'</td><td>'+group_label(previous,m,a)+'</td><td>'+group_label(summary,m,a)+'</td></tr>' for a,title in ARCHS.items() for m in METHODS)
    fragments.append(f'''<details class="experiment-setting"><summary>原 ReLU 结果与配对对照（原图保留）</summary>
<p>原 ReLU：{old_counts['通过']} 格通过、{old_counts['未通过']} 格未通过、{old_counts['不可比较']} 格不可比较；无激活：{counts['通过']} 格通过、{counts['未通过']} 格未通过、{counts['不可比较']} 格不可比较。激活改变初始函数和梯度，因此 U 与 L0 都按各自模型重新计算；同一行比较相同数据和相同初始权重下的误差表现，不假定初始预测相同。</p>
<div class="table-wrap"><table><thead><tr><th>结构</th><th>优化器</th><th>原 ReLU</th><th>无激活</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p><a href="D1_nn_relu_u_transfer.html" target="_blank" rel="noopener">打开原 ReLU 图与完整记录 ↗</a>。原 SVG、交互 HTML 和原训练产物保留；本页默认展示无激活对照。</p></details>
<p class="chart-note">预注册 <code>{summary['preregistration_commit'][:7]}</code>；源码、协议、结果摘要与复算记录见 <a href="../../research/d1_no_relu_20261105/README.md">研究目录</a>。全部逐步产物保存于 <code>AIQ_KB_DATA/runs/d1_no_relu_20261105/</code>。</p>
''')
    prefix = text[:start].replace("学习特征 MLP 的迁移检验见第4节", "多层网络（含无激活对照）的迁移检验见第4节")
    page.write_text(prefix+'\n'.join(fragments)+'\n'+text[end:],encoding="utf-8",newline="\n")
    assert preserved_assets == {name: hashlib.sha256((DOCS/name).read_bytes()).hexdigest() for name in preserved_assets}
    (SOURCE/"preserved_relu_figures.json").write_text(json.dumps(preserved_assets,indent=2)+"\n")
    print(json.dumps({"page":str(page),"archive":str(archive),"primary_counts":counts},ensure_ascii=False))


if __name__ == "__main__":
    main()
