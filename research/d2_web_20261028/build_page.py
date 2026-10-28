"""Publish a readable D2 synthesis from saved, hash-verified results; no experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'docs/模糊方向'
OUT = WEB / 'd2_evidence_20261028'
REPO = 'https://github.com/guoshaoyang-pku/science-of-ai-prototype'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    exports = []
    for study, prefix, names in [
        ('d2_diagonal_20261027', 'diagonal_', ['summary.json', 'numerical_verification.json', 'independent_audit.json']),
        ('d2_sparse_cross_20261028', '', ['summary.json', 'numerical_verification.json', 'independent_audit.json', 'focus_case.json', 'focus_curves.npz']),
    ]:
        manifest = json.loads((ROOT / 'research' / study / 'result_manifest.json').read_text())
        pins = {a['path']: a['sha256'] for a in manifest['artifacts']}
        for name in names:
            src = args.data_root / 'runs' / study / name
            digest = sha(src)
            assert digest == pins[name], f'changed saved artifact: {study}/{name}'
            dest = OUT / (prefix + name)
            shutil.copyfile(src, dest)
            exports.append({'source': f'runs/{study}/{name}', 'published': dest.name, 'sha256': digest, 'byte_identical': True})
    (OUT / 'export_manifest.json').write_text(json.dumps({'kind': 'read_only_publication_export', 'new_experiments': False, 'files': exports}, ensure_ascii=False, indent=2)+'\n')
    data = json.loads((OUT / 'summary.json').read_text())
    methods = data['methods']
    def rate(n, d): return f'{n}/{d} <small>({n/d:.1%})</small>'
    main_rows = []
    for key, label, budget in [('diagonal','全部删除','0%'),('change05','按变化保留','≤5%'),('magnitude05','按大小保留','≤5%'),('full','完整参照','100%')]:
        m = methods[key]; o=m['overall']
        main_rows.append(f'<tr><th scope="row">{label}</th><td>{budget}</td><td>{rate(o["time_pass"],o["interior"])}</td><td>{rate(o["risk_pass"],o["count"])}</td><td>{m["recipes_passing_both"]}/12</td></tr>')
    randoms = [methods[f'random05_{i:02}'] for i in range(10)]
    random_row = '<tr><th scope="row">随机保留（10 次）</th><td>≤5%</td><td>75.5%–76.5%</td><td>85.0%–86.3%</td><td>5–6/12</td></tr>'
    main_rows.insert(3,random_row)
    random_rows=''.join(f'<tr><td>{i+1:02}</td><td>{m["overall"]["time_pass"]}/3137</td><td>{m["overall"]["risk_pass"]}/3240</td><td>{m["recipes_passing_both"]}/12</td></tr>' for i,m in enumerate(randoms))
    task_names={'linear':'线性','quadratic':'单变量二次','additive':'加性二次','interaction':'线性＋交互','product':'纯交互','sine':'正弦','vector':'双输出'}
    protocols={'d1_homogeneous':'1 维 · 无偏置','d1_biased':'1 维 · 有偏置','d4_homogeneous':'4 维 · 无偏置','d4_biased':'4 维 · 有偏置'}
    shapes={'linear':r'$a$', 'quadratic':r'$a+0.5(a^2-1)$','additive':r'$a+0.5(b^2-1)$','interaction':r'$a+0.5ab$','product':r'$ab$','vector':r'$(a,\ a+0.5ab)$'}
    task_rows=[]
    for r in methods['change05']['recipes']:
        shape=shapes.get(r['task'],r'$\sin(2a)$' if r['protocol'].startswith('d1') else r'$\sin a+0.5\sin(2b)$')
        task_rows.append(f'<tr><th scope="row">{protocols[r["protocol"]]}<br>{task_names[r["task"]]}</th><td>{shape}</td><td>{r["time_pass"]}/{r["interior"]}</td><td>{r["risk_pass"]}/{r["count"]}</td></tr>')
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>D2：训练到什么时候该停？</title>
<meta name="description" content="从训练为何可能越久越差讲起，逐步解释停止时间、交叉项、实验方法与结果。">
<link rel="stylesheet" href="style.css">
<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']],displayMath:[['\\[','\\]'],['$$','$$']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
.mobile-hint{display:none}main{max-width:1040px;padding-top:36px}body{font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:16px;line-height:1.85}h1{font-size:clamp(25px,4vw,35px);max-width:830px}h2{margin-top:40px;font-size:22px}p{max-width:900px}.eyebrow{color:#7b6048;font-size:13px;letter-spacing:.08em}.lead{font-size:18px}.note,figcaption,small{font-size:13px;color:#625f58}.toc{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:14px;margin:22px 0}.callout{background:#eeeee3;border-left:3px solid #788c5d;padding:12px 16px;border-radius:0 8px 8px 0}.warning{background:#f8eee7;border-left-color:#bf7454}table{min-width:640px;font-size:14px}th{color:#363a36;font-weight:600;background:#eeeee6}tbody th{background:transparent;font-weight:500}td small{display:block}figure{margin:24px 0}figure img{display:block;width:100%;border-radius:9px;border:1px solid #e5e1d5;background:#faf9f5}figcaption{margin-top:9px;line-height:1.7}details{margin:16px 0;border:1px solid #ddd9cc;border-radius:9px;padding:12px 16px}summary{font-weight:600;cursor:pointer;line-height:1.6}details[open]>summary{margin-bottom:14px}summary:focus-visible,a:focus-visible{outline:3px solid #6a9bcc;outline-offset:4px}.section-index{color:#8b887d;font-size:14px;margin-right:9px}.refs{font-size:14px}section{scroll-margin-top:20px}.stamp{color:#625f58;margin-top:35px}.formula-table td:nth-child(2){min-width:270px}@media(max-width:620px){.mobile-hint{display:block}main{padding:24px 16px 60px}h2{font-size:20px}.lead{font-size:17px}details{padding:11px}.nav{line-height:1.8}}
</style></head><body><main>
<nav class="nav" aria-label="站点导航"><a href="index.html">← 研究总览</a> · <a href="D1_u_effective_time.html">D1：训练时间</a> · <a href="D2_文献深度确认.html">文献对照</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/edit/main/docs/%E6%A8%A1%E7%B3%8A%E6%96%B9%E5%90%91/D2_overfitting_u_curve.html">编辑此页</a></nav>
<header><p class="eyebrow">D2 · 训练科学 · 更新于 2026-10-28</p><h1>训练到什么时候该停？</h1>
<p class="lead">模型训练得越久，在新数据上的表现就一定越好吗？不一定。它可能一边学到有用的规律，一边也记住训练数据中的偶然误差；继续训练，有时反而会让新数据上的误差变大。</p>
<p>这就是这里讨论的<strong>过拟合</strong>。D2 想研究的是：<strong>能不能预测一个合适的停止时间，让模型既学到了规律，又没有因为继续训练而损失太多效果？</strong></p>
<p class="note">下面先介绍这个问题，再解释我们怎样简化计算，最后看具体例子和总体结果。本研究使用可精确计算的简化模型，距离实际神经网络的早停工具还有差距。</p></header>
<p class="note mobile-hint">手机上表格可左右滑动，点击图片可查看高清图。</p>
<nav class="toc" aria-label="页内目录"><a href="#question">为什么要停</a><a href="#formulas">交叉项是什么</a><a href="#method">怎样做实验</a><a href="#example">看一个例子</a><a href="#results">总体效果</a><a href="#limits">失败与边界</a><a href="#evidence">证据</a></nav>
<section id="question"><h2><span class="section-index">01</span>先看问题：我们想找到误差曲线的低处</h2>
<p>训练数据用来学习，另外一批没有参加训练的数据用来检查学得好不好，这里叫<strong>验证数据</strong>。把验证误差随训练步数画出来，有些曲线先下降、后回升：停得太早，规律还没学好；停得太晚，效果已经变差。我们要找的是观察时间范围内，误差最低的位置。</p>
<p>一个直觉是：训练样本越多，可以学得久一些；答案中的随机误差（<strong>标签噪声</strong>）越大，可能越该早点停。原 D2 先尝试只用这两个量预测停止时间，但扩大任务范围后，这个简单公式不够用了。</p>
<p>于是问题变成：<strong>除了样本数和噪声，还需要保留哪些学习过程的信息，才能判断何时停止？</strong></p>
</section>
<section id="formulas"><h2><span class="section-index">02</span>为什么要考虑“交叉项”？</h2>
<p>在我们的简化模型里，预测可以拆成若干个部分，它们分别沿不同的“学习方向”逐渐变化。这些部分相加以后，可能相互增强，也可能相互抵消。因此，计算总误差时，不能只把每一部分单独记账。</p>
<p>两部分预测分别是 $a$ 和 $b$，平方展开时有 $(a+b)^2=a^2+b^2+2ab$。其中的 $2ab$ 就是<strong>交叉项</strong>，记录两部分合在一起产生的影响。比如两部分分别贡献 2 和 −1，合起来是 1；忽略它们的抵消，误差就可能算错。</p>
<p>交叉项很多，我们想知道：<strong>能否只保留其中少量重要的项，仍然把停止时间预测好？</strong> 下表把完整计算和简化计算放在一起，后面的实验就比较它们。</p>
<p>下文把用于评价的平均验证误差称为<strong>风险</strong>：固定输入与特征，对训练答案中的随机噪声取平均。误差用预测与答案之差的平方计算，数值越小越好。$t_*$ 是观察窗口内的最低点估计：它告诉我们何时停止较好，不保证是第一次局部回升。</p>
<div class="table-wrap"><table class="formula-table"><caption class="note">固定隐藏特征，只训练输出层；$t$ 为步数，$U=\eta t$。</caption><thead><tr><th scope="col">想回答什么</th><th scope="col">公式</th><th scope="col">直观含义与结论</th></tr></thead><tbody>
<tr><th scope="row">样本多，能否按比例晚停？</th><td>$U_*\approx c\,n/\sigma^2$</td><td>旧经验式，在较窄范围有用；扩大范围后，即使每个任务重新拟合系数也不够。完整计数留在后面的历史结果中。</td></tr>
<tr><th scope="row">误差来自哪里？</th><td>$R(t)=B(t)+\sigma^2N(t)$</td><td>无训练噪声时的误差，加上学习噪声带来的代价。这是本模型中的精确分解。</td></tr>
<tr><th scope="row">最低点怎样出现？</th><td>$B'(t_*)+\sigma^2N'(t_*)=0$</td><td>连续插值的光滑内部最低点处，变化速度相抵；不要求两部分的数值相等，也不保证只有一个谷底。</td></tr>
<tr><th scope="row">各方向能否分开算？</th><td>$R=R_{\rm diag}+\sum_{i&lt;j}C_{ij}$</td><td>还要算相互增强或抵消。$C_{ij}$ 就是两个学习方向的交叉项；全部删掉后只剩 $R_{\rm diag}$。</td></tr>
<tr><th scope="row">能否只留少量交叉项？</th><td>$\widehat R_S=R_{\rm diag}+\sum_{(i,j)\in S}C_{ij}$</td><td>本轮检验的近似。$S$ 是保留名单；其余对角项、目标匹配项和噪声项都保持完整。</td></tr>
</tbody></table></div>
<p class="note">$n$：训练样本数；$\sigma^2$：标签噪声方差；$B$：无训练噪声时的验证误差；$N$：单位噪声方差的响应；撇号：随步数的变化率；$c$：拟合系数。最低点公式用于离散 GD 的连续谱插值。</p>
<details><summary>交叉项到底是什么？一个算得出来的小例子</summary><p>两个学习方向分别贡献 $a$ 和 $b$，目标是 $y$，平方误差为 $(a+b-y)^2=a^2+b^2+2ab-2y(a+b)+y^2$。其中 $2ab$ 就是交叉项。</p><p>例如 $a=2,b=-1,y=1$，完整误差为 0；删掉 $2ab=-4$ 后，估计变成 4。方向之间的抵消被漏算了。这个手算例子只是解释代数；下面的第 1 步与第 255 步来自保存的实验数据。</p></details>
</section>
<section id="method"><h2><span class="section-index">03</span>实验怎么做，12 组分别是什么？</h2>
<p>沿用已有的 <strong>12 组任务与特征 × 10 档样本量 × 9 档噪声 × 3 个主种子 = 3240 条曲线</strong>。模型先把输入变成 128 个固定特征，训练只调整这些特征怎样组合成答案，也就是只训练输出层。我们复用已有学习过程，比较不同误差计算方式；本轮没有重新训练网络。</p>
<ol><li><strong>先核验数据、再预注册。</strong> 重验旧数据与产物哈希，冻结选择规则与通过门槛，然后计算删项后的风险曲线。</li><li><strong>每组按同样的预算选项。</strong> 有效方向数为 $r$，可选对数 $M=r(r-1)/2$，保留 $K=\lfloor0.05M\rfloor$ 对；同一任务与数据的名单在九档噪声间共享。</li><li><strong>先用近似选时间，再用完整风险评分。</strong> $\Delta=[R(\hat t)-R(t_*)]/R(0)$。$\Delta\le1\%$ 才算额外误差小，不能拿近似模型自己的低估当成成功。</li></ol>
<div class="table-wrap"><table><thead><tr><th>选择方法</th><th>评分或规则</th><th>通俗解释</th></tr></thead><tbody>
<tr><th scope="row">按变化（主方案）</th><td>$s_{ij}=\max_{t\in\mathcal G}|(t+1)C'_{ij}(t)|$</td><td>优先保留沿训练进程变化显著的项</td></tr>
<tr><th scope="row">按大小（对照）</th><td>$s_{ij}=|C_{ij}(262144)|$</td><td>优先保留窗口末绝对贡献大的项</td></tr>
<tr><th scope="row">随机（对照）</th><td>等预算、不放回抽取；固定十次</td><td>检验有针对性的选择是否比运气更好</td></tr>
</tbody></table></div>
<p><strong>怎样算预测得好？我们分开看两件事。</strong> 一是停止步数是否接近完整计算的最低点；二是按建议停止时，实际多付出了多少误差。平坦的谷底可能允许很多相近效果的停止时间，所以这两项不能互相替代。</p>
<div class="table-wrap"><table><thead><tr><th>评价指标</th><th>通过标准</th><th>一个说明标准的例子</th></tr></thead><tbody>
<tr><th scope="row">时间是否接近</th><td>预测步数在参照的一半到两倍之间</td><td>若参照是第 100 步，预测第 50–200 步都算通过。</td></tr>
<tr><th scope="row">停止后的误差是否够小</th><td>比最低误差多出的部分，不超过训练初始误差的 1%</td><td>若初始误差是 1，最低误差是 0.2，建议停止时的误差不超过 0.21 就通过。</td></tr>
</tbody></table></div><p class="note">表中数字仅用于解释评价规则，不是新实验结果。落在观察窗口两端的最低点不计算时间倍数，但仍评价实际误差。</p>
<details><summary>展开 12 组任务、目标形状及本轮逐组结果</summary><p>下面是<strong>归一化前的目标形状</strong>；实际每个输出通道按总体均方根归一化为 1。$a,b$ 是前两个输入坐标。四维目标不一定用到全部四个坐标；“有偏置”指 ReLU 内的阈值偏移。目标中的 $ab$ 与误差展开中的谱交叉项不是同一个对象。</p><div class="table-wrap"><table><thead><tr><th>任务与特征设置</th><th>目标形状</th><th>时间通过</th><th>额外误差通过</th></tr></thead><tbody>@@TASK_ROWS@@</tbody></table></div><p class="note">本表为按变化保留最多 5% 的结果。每组时间分母只含内部最低点；风险分母均为 270。</p></details>
<details><summary>展开计算约定与精确谱公式</summary><p>输入为 1 或 4 维标准高斯；样本量 16–1024、噪声方差 1/32–8、独立干净验证样本 2048 个；输出层零初始化，MSE、学习率 $\eta=0.3$，无动量和权重衰减。120 个数据与特征抽样被任务和噪声条件复用，3240 条不是独立重复。</p><p>在训练协方差的特征方向中，$f_i(t)=[1-(1-2\eta\lambda_i)^t]/\lambda_i$；$\lambda_i$ 为有效特征值。$g_i$ 为训练目标投影，$H_{ij}$ 为验证特征在这组方向中的内积，$h_{ij}=H_{ij}\langle g_i g_j\rangle$，尖括号表示对输出通道平均。于是 $C_{ij}=2h_{ij}f_i f_j$。未保留项置零，系数不重新拟合。</p><p>评分格点 $\mathcal G$ 为整数 0–16 加 17–262144 的 128 个取整对数格点。找最低点则使用整数 0–256 加 4096 个取整对数格点，并在附近精化；没有穷举所有整数极值。全部 103 个窗口边界最低点保留在风险评价中，只退出时间倍数分母。</p><p>60/360 个任务与数据组合因低秩而保留 0 对；21/120 个数据与特征抽样有很接近的特征值，方向选择的稳健性还没有验证。变化和大小评分仍使用干净目标、验证几何，并检查全部交叉项。</p></details>
</section>
<section id="example"><h2><span class="section-index">04</span>为什么会把第 255 步错认成第 1 步？</h2>
<p>这个例子是一维正弦任务，有隐藏偏置，$n=32,\sigma^2=1$，seed=6103。完整风险在第 255 步更低；全删交叉项却建议第 1 步就停。31 个有效方向共有 465 对交叉项，补回 23 对后，变化法和大小法都选回第 255 步。</p>
<figure><a href="figs/d2_20261028_focus.svg" aria-label="打开高清重点例子图"><img src="figs/d2_20261028_focus.png" alt="同一训练过程的风险估计对比：全删交叉项漏掉晚期谷底；保留23对后恢复第255步的低风险。" loading="lazy" width="1600" height="800"></a><figcaption>改变的是风险估计，不是重新训练模型。同一步的真实风险对所有方法都一样；额外误差始终由完整风险评价。</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>同一时间的记账</th><th>第 1 步</th><th>第 255 步</th><th>怎么理解</th></tr></thead><tbody>
<tr><th scope="row">删掉全部交叉项的估计</th><td>0.858</td><td>1.021</td><td>误以为继续训练更差</td></tr>
<tr><th scope="row">完整交叉项的总贡献</th><td>−0.003</td><td>−0.769</td><td>晚期有很大的抵消作用</td></tr>
<tr><th scope="row">完整风险</th><td>0.855</td><td>0.252</td><td>加回抵消项，晚期明显更好</td></tr>
<tr><th scope="row">按变化保留 23 对后的估计</th><td>0.858 − 0.003 ≈ 0.855</td><td>1.021 − 0.769 ≈ 0.252</td><td>少数项近似恢复了净贡献</td></tr>
</tbody></table></div>
<p>全删和十次随机保留都选第 1 步，实际风险约 0.855；两种有针对性的选择都选第 255 步，实际风险约 0.252。修复的是一个已知开发例子，不能把它当作新任务盲测。表内短数经过舍入，完整精度见<a href="d2_evidence_20261028/focus_case.json">逐项记录</a>。</p>
</section>
<section id="results"><h2><span class="section-index">05</span>放回全部实验，效果怎么样？</h2>
<p>前面的例子说明了交叉项怎样影响判断。再看全部条件：按变化选择最多 5% 的交叉项，<strong>时间预测通过率为 97.0%，额外误差通过率为 98.8%</strong>。前者只评价 3137 个内部最低点，后者评价全部 3240 个条件；它们不是模型答题的正确率。</p>
<p>下面的图比较“保留多少交叉项”和“有多少条件通过上述标准”。横轴越小，保留的交叉项越少；纵轴越高，停止效果越好。<strong>Pareto 前沿</strong>指已测方法中，没有另一方法在“保留不更多、效果不更差”的同时，至少有一项更好的点。</p>
<figure><a href="figs/d2_20261028_pareto.svg" aria-label="打开高清 Pareto 对比图"><img src="figs/d2_20261028_pareto.png" alt="实测 Pareto 对比：0%全删、最多5%按变化或大小及十次随机选择、100%完整参照；纵轴分别为时间和低额外风险通过率。" width="1600" height="800"></a><figcaption>只测了 0%、5%、100% 三档预算；中间比例没有实验，不能据此推断连续前沿。5% 为上限，实际向下取整。完整参照的 100% 通过率来自评价定义；横轴不代表运行时间或训练成本。</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th scope="col">方法</th><th scope="col">交叉项预算</th><th scope="col">时间在两倍以内</th><th scope="col">额外误差 ≤ 初值 1%</th><th scope="col">达标任务组</th></tr></thead><tbody>@@MAIN_ROWS@@</tbody></table></div>
<p><strong>变化法与大小法基本打平。</strong> 两者都通过 12 组，两个总体指标各只差 1 条，不能据此认定变化评分更优。随机法十次均明显较差；这些是同一批数据上的十份名单，不是十次独立数据实验。</p>
<div class="callout"><strong>12/12 是什么意思？</strong> 每组有 270 个条件，必须同时满足：内部最低点中至少 80% 的时间预测在两倍以内；全部条件中至少 90% 的额外误差不超过初值的 1%。12/12 表示每组都过了这两个门槛，<strong>不表示 3240 条全部正确，也不表示覆盖了 12 类现实任务。</strong></div>
<details><summary>查看十次随机选择的全部结果</summary><div class="table-wrap"><table><thead><tr><th>随机名单</th><th>时间通过</th><th>额外误差通过</th><th>达标组数</th></tr></thead><tbody>@@RANDOM_ROWS@@</tbody></table></div></details>
</section>
<section id="limits"><h2><span class="section-index">06</span>哪些失败仍然存在？</h2>
<div class="callout warning"><strong>12/12 达标，不等于没有反例。</strong> 主方案仍有 94/3137 条时间未达标、40/3240 条额外误差未达标。最差一条把第 862 步选成第 32 步，实际风险从最低的 0.118 变成 0.255，多出初始风险的 12.8%。</div>
<p>这个反例是四维无偏置的加性二次任务，$n=24,\sigma^2=1/32$，seed=6103。另有 252/3240 条近似无噪声误差为负、20/3240 条近似总风险为负；负误差是近似失效，不能当作更好效果。</p>
<details><summary>旧 D2 的哪些结论仍然有用？</summary><div class="table-wrap"><table><thead><tr><th>问题</th><th>保存的证据</th><th>应怎样理解</th></tr></thead><tbody>
<tr><th scope="row">能否只用样本数／噪声预测？</th><td>旧留出 n：57/60；扩展验证：1232/2700</td><td>每任务拟合系数仍有 12/12 组未过 80% 时间门槛；这与新稀疏方法的两项门槛不同。</td></tr>
<tr><th scope="row">训练误差必须先接近零吗？</th><td>最低点处训练误差中位仍为初值的 33%；仅 766/3137 ≤ 10%</td><td>验证误差回升不要求先把训练误差压得很低。</td></tr>
<tr><th scope="row">回升前就会学到噪声吗？</th><td>半个最低点时间处，噪声代价已达最低点时的中位 77%</td><td>总误差仍下降，不代表噪声尚未造成损害。</td></tr>
<tr><th scope="row">最低点就是曲率拐点吗？</th><td>旧 60 条曲线的后续曲率换号约在最低点的 1.2–8.1 倍时间</td><td>斜率为零和曲率换号是两件事；局部谷底通常近似二次形状。</td></tr>
<tr><th scope="row">旧尾部公式与归一化修正呢？</th><td>1.55 次方尾部外推 78/78 失败；旧 r113 归一化后系数极差增至 4.68 倍</td><td>有限区间拟合不能当成普适规律；部分种子变好不能覆盖总体失败。</td></tr>
</tbody></table></div><p><a href="d2_evidence_20261009/report.md">2026-10-09 完整历史报告</a>保留原始细节。</p></details>
</section>
<section id="next"><h2><span class="section-index">07</span>现在能说到哪里，下一步验证什么？</h2>
<p><strong>当前支持：这批固定特征回归条件中的停止决策，通常可以用少量有针对性的交叉项保留。</strong> 这不是“任意复杂任务只需 5%”，也不是“训练省了 95%”：选择时仍检查了所有交叉项。</p>
<div class="table-wrap"><table><thead><tr><th>下一步问题（尚未运行）</th><th>为什么要测</th><th>通过后才能增加的结论</th></tr></thead><tbody>
<tr><th scope="row">更多有效方向、交叉作用更分散</th><td>直接挑战“少数大项足够”的条件</td><td>对指定复杂任务仍能压缩，并能描述失败边界</td></tr>
<tr><th scope="row">只用实际可获得的带噪标签</th><td>当前方法依赖干净目标和验证几何</td><td>不依赖理想信息，也能选出有用项</td></tr>
<tr><th scope="row">单次轨迹与会学习特征的网络</th><td>噪声平均曲线不等于一次训练，学习方向也可能变化</td><td>才有机会成为实际训练的早停工具</td></tr>
</tbody></table></div>
<p class="note">分类、交叉熵、Adam、小批量和端到端特征学习均未在本研究中验证。旧 60 个条件里，单次轨迹最低点与噪声平均最低点之比为 0.24–86.72，不能直接混用。</p>
</section>
<section id="evidence"><h2><span class="section-index">08</span>数据与复核入口</h2>
<p class="refs">本页只整理已保存结果，没有新增实验。两轮预注册均早于计算；本次发布前再次按结果清单校验导出文件。完整曲线重建最大归一化差为 7.8×10⁻¹¹；4320 个掩码已核对，另有 96 次独立特征空间核验。</p>
<div class="table-wrap"><table><thead><tr><th>证据</th><th>直接查看</th></tr></thead><tbody>
<tr><th scope="row">2026-10-09 原 D2</th><td><a href="d2_evidence_20261009/report.md">历史报告</a> · <a href="d2_evidence_20261009/summary.json">完整汇总</a></td></tr>
<tr><th scope="row">2026-10-27 对角消融</th><td><a href="d2_evidence_20261028/diagonal_summary.json">完整汇总</a> · <a href="d2_evidence_20261028/diagonal_independent_audit.json">独立代理审计</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/d2796a484712fc95268c2400a6d85d5cf121163c/research/d2_diagonal_20261027">预注册与源码</a></td></tr>
<tr><th scope="row">2026-10-28 保留 5%</th><td><a href="d2_evidence_20261028/summary.json">完整汇总（含逐组／逐种子）</a> · <a href="d2_evidence_20261028/independent_audit.json">独立代理审计</a> · <a href="d2_evidence_20261028/numerical_verification.json">数值核验</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/7dc745d2c4e21acb98ef4fa0a4534c2e5e707eda/research/d2_sparse_cross_20261028">预注册与源码</a></td></tr>
<tr><th scope="row">第 1 步／255 步例子</th><td><a href="d2_evidence_20261028/focus_case.json">逐项计算</a> · <a href="d2_evidence_20261028/focus_curves.npz">保存的曲线数组</a></td></tr>
<tr><th scope="row">网页与图表来源</th><td><a href="d2_evidence_20261028/export_manifest.json">原件哈希与发布清单</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/main/research/d2_web_20261028">页面与绘图源码</a></td></tr>
</tbody></table></div>
</section><footer class="stamp">更新于 2026-10-28 · 固定特征回归 · 已观察数据上的研究结果 · 完整精度保留在 JSON 证据中</footer>
</main></body></html>'''
    rendered=template.replace('@@MAIN_ROWS@@',''.join(main_rows)).replace('@@RANDOM_ROWS@@',random_rows).replace('@@TASK_ROWS@@',''.join(task_rows))
    assert '@@' not in rendered
    (WEB / 'D2_overfitting_u_curve.html').write_text(rendered)
    print(json.dumps({'page':'docs/模糊方向/D2_overfitting_u_curve.html','exported_files':len(exports),'all_export_hashes_verified':True},ensure_ascii=False))

if __name__=='__main__': main()
