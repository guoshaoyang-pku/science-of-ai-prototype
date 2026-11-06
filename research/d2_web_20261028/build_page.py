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
    # Six-round evidence is exported separately, without executing any study.
    continuous_dir = WEB / 'd2_evidence_20261106'
    continuous_manifest = json.loads((continuous_dir / 'export_manifest.json').read_text())
    for round_entry in continuous_manifest['rounds']:
        assert sha(ROOT / round_entry['result_manifest']) == round_entry['result_manifest_sha256']
        for artifact in round_entry['artifacts']:
            if artifact['published']:
                assert sha(continuous_dir / artifact['published']) == artifact['sha256']
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
<title>D2：为什么训练更好，泛化却变差？</title>
<meta name="description" content="从最初的停止时间公式，到扩域反例、交叉项消融、5%保留，再到噪声、表达能力、误差抵消和学习偏好，按研究顺序讲清楚每一步。">
<link rel="stylesheet" href="style.css">
<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']],displayMath:[['\\[','\\]'],['$$','$$']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
.mobile-hint{display:none}main{max-width:1040px;padding-top:36px}.bridge{border-left:3px solid #b0aea5;padding:8px 14px;background:#f0efe9}body{font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:16px;line-height:1.85}h1{font-size:clamp(25px,4vw,35px);max-width:830px}h2{margin-top:40px;font-size:22px}p{max-width:900px}.eyebrow{color:#7b6048;font-size:13px;letter-spacing:.08em}.lead{font-size:18px}.note,figcaption,small{font-size:13px;color:#625f58}.toc{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:14px;margin:22px 0}.callout{background:#eeeee3;border-left:3px solid #788c5d;padding:12px 16px;border-radius:0 8px 8px 0}.warning{background:#f8eee7;border-left-color:#bf7454}table{min-width:640px;font-size:14px}th{color:#363a36;font-weight:600;background:#eeeee6}tbody th{background:transparent;font-weight:500}td small{display:block}figure{margin:24px 0}figure img{display:block;width:100%;border-radius:9px;border:1px solid #e5e1d5;background:#faf9f5}figcaption{margin-top:9px;line-height:1.7}details{margin:16px 0;border:1px solid #ddd9cc;border-radius:9px;padding:12px 16px}summary{font-weight:600;cursor:pointer;line-height:1.6}details[open]>summary{margin-bottom:14px}summary:focus-visible,a:focus-visible{outline:3px solid #6a9bcc;outline-offset:4px}.section-index{color:#8b887d;font-size:14px;margin-right:9px}.refs{font-size:14px}section{scroll-margin-top:20px}.stamp{color:#625f58;margin-top:35px}.formula-table td:nth-child(2){min-width:270px}@media(max-width:620px){.mobile-hint{display:block}main{padding:24px 16px 60px}h2{font-size:20px}.lead{font-size:17px}details{padding:11px}.nav{line-height:1.8}}
</style></head><body><main>
<nav class="nav" aria-label="站点导航"><a href="index.html">← 研究总览</a> · <a href="D1_u_effective_time.html">D1：训练时间</a> · <a href="D2_文献深度确认.html">文献对照</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/edit/main/docs/%E6%A8%A1%E7%B3%8A%E6%96%B9%E5%90%91/D2_overfitting_u_curve.html">编辑此页</a></nav>
<header><p class="eyebrow">D2 · 泛化为什么变弱 · 证据更新于 2026-11-06</p><h1>D2：从“什么时候该停”，到“为什么越学越差”</h1>
<p class="lead">模型在练习题上越来越熟练，换一批题却可能做得更差。这就是 D2 要解释的现象：训练误差继续下降，泛化误差却开始回升。</p>
<p>这项研究最初想找一个简单公式：知道样本数和标签噪声，能不能估计训练到哪一步该停？后来公式遇到了反例，我们开始把误差拆开，再逐步排查泛化变差的原因。</p>
<div class="callout"><strong>这是一份从头往下读的研究记录。</strong> 每一步都交代：当时的问题、怎样验证、结果如何、为什么继续下一步。最初的公式、失败的尝试、5% 交叉项实验，以及最近的学习偏好分析，都在同一条主线上。</div>
<p class="note">目前的结论仍限于小规模、固定特征的合成回归。本文整理已有实验，没有新开训练；拟议的同速对照继续暂缓。</p></header>
<p class="note mobile-hint">手机上表格可左右滑动，点击图片可查看高清图。</p>
<nav class="toc" aria-label="页内目录"><a href="#question">先懂问题</a><a href="#beginning">1 最初公式</a><a href="#status">2 扩大检验</a><a href="#formulas">3 拆开误差</a><a href="#method">4 删项与 5%</a><a href="#study">5 回到原因</a><a href="#progress">6 排查噪声</a><a href="#validation">7 核对真实回升</a><a href="#capacity">8 排查表达能力</a><a href="#cancellation">9 误差抵消</a><a href="#preference">10 学习偏好</a><a href="#next">当前结论</a><a href="#evidence">原始证据</a></nav>
<section id="question"><h2><span class="section-index">01</span>先看全貌：学会训练题，为什么不等于学会规律？</h2>
<p><strong>训练误差</strong>衡量模型对用来学习的样本做得怎样；<strong>泛化误差</strong>关心它遇到新输入时是否仍然可靠。这里以预测和正确答案之差的平方平均值（MSE）记分，越低越好。验证集是用一批新输入估计泛化误差；在已知分布的当前实验里，还可以直接计算总体平均，减少验证样本波动的干扰。</p>
<p>训练标签可能混有随机错误，叫<strong>标签噪声</strong>。模型继续降低训练误差时，既可能更接近真实规律，也可能更贴近这些错误。因此，训练进步与泛化改善不是同一件事。噪声是候选原因之一，不能未经检查就把每一次回升都归因于它。</p>

<p>停止时间是这个现象的一种测量。本文寻找观察窗口内的最低误差位置，记为 $t_*$；它是格点搜索与局部精化的估计，<strong>不保证等于第一次局部回升的位置</strong>。</p>
</section>
<section id="beginning"><h2>1 · 最初：能不能用样本数和噪声，估计该停在哪一步？</h2>
<p><strong>当时的问题：</strong>模型在训练题上越来越好，新题上的误差却先降后升，形成一个谷底。如果能提前知道谷底在哪里，就不用一路训练到变差。</p>
<figure><a href="figs/d2_20261009_legacy_train_val.svg"><img src="figs/d2_20261009_legacy_train_val.png" alt="最早保存的两个训练例子：蓝线训练误差持续下降，橙线验证误差先降后升，绿色虚线为对训练标签噪声平均后的风险。" loading="lazy"></a><figcaption>最早保存的两条实际训练轨迹。蓝线是训练题误差，橙线是新题误差；绿色虚线是对训练标签噪声平均后的风险。两种谷底并不一定在同一步，后来的简单公式讨论的是平均风险的最低点。</figcaption></figure>
<p><strong>最初的猜想：</strong>样本越多，可以学得更久；标签中的随机错误越多，就应该早点停。于是用下面的比例关系概括：</p>
<div class="table-wrap"><table><thead><tr><th>写法</th><th>通俗解释</th></tr></thead><tbody>
<tr><th scope="row">$U_*\approx0.28\,n/\sigma^2$</th><td>有效训练时间，大致随样本数增加、随噪声方差减少。0.28 是这批旧数据拟合出来的系数。</td></tr>
<tr><th scope="row">$U_*=\eta t_*$</th><td>$t_*$ 是停止步数，$\eta$ 是学习率；比较时必须分清“步数”和“学习率 × 步数”。</td></tr>
<tr><th scope="row">$n$ 与 $\sigma^2$</th><td>$n$ 是训练样本数量；$\sigma^2$ 是标签随机错误的方差，越大表示噪声越强。</td></tr>
</tbody></table></div>
<p><strong>怎么验证：</strong>先看原来的四维线性＋交互目标，固定 128 个 ReLU 特征，只训练输出层；样本数为 32、64、128，噪声方差为 0.25–4，学习率为 0.3。旧 4 个种子、3 档样本数、5 档噪声，一共 60 个条件。每次留出一个样本量，用其他样本量拟合后再预测它。</p>
<p><strong>结果：</strong>60 个预测中有 57 个落在实际最低点的一半到两倍之间。例如实际第 100 步最低，预测第 50–200 步都算通过；这不是精确命中。</p>
<p class="bridge"><strong>为什么做下一步：</strong>小范围里看起来不错，究竟是普遍规律，还是只适合这一个任务和范围？所以接下来扩大样本数、噪声范围，并更换目标任务。</p>
</section>
<section id="status"><h2>2 · 扩大检验：简单比例公式失效了</h2>
<p><strong>怎么做：</strong>把样本数扩大到 16–1024，噪声方差扩大到 1/32–8，加入线性、二次、交互、正弦等目标，共 12 组任务与特征设置。每组先用 45 个条件校准，再预测另外 225 个条件，共 2700 个扩展条件。</p>
<p><strong>结果：</strong>即使每组都允许重新拟合自己的系数，也只有 1232/2700（45.6%）落在两倍以内，12 组都没达到事先规定的 80% 通过率。因此，<strong>“扩大范围仍只需换个系数”的说法被推翻了</strong>；原小范围的 57/60 仍然保留。</p>
<div class="table-wrap"><table><thead><tr><th>具体改了什么</th><th>看到什么</th><th>能说明什么</th></tr></thead><tbody>
<tr><th scope="row">只增加样本数的一个例子</th><td>同一任务、同一噪声：样本数 128→256→512；最低点 139→173→209 步。</td><td>样本翻倍，停止时间没有翻倍。</td></tr>
<tr><th scope="row">在原样本／噪声范围内换任务</th><td>四维交互任务的校准通过 45/45；四维线性只有 28/45。</td><td>任务本身也影响公式；这两个数是校准结果，不能当成新的独立验证。</td></tr>
<tr><th scope="row">单独扩大噪声范围</th><td>大扫描同时扩了样本和噪声范围。</td><td>不能把全部失败单独归给噪声扩域；尚缺完整的单因素归因。</td></tr>
</tbody></table></div>
<p>同一轮还检查了几个容易误解的说法。它们的失败也保留在这里：</p>
<div class="table-wrap"><table><thead><tr><th>接受检验的说法</th><th>已有反证</th><th>现在的判定</th></tr></thead><tbody>
<tr><th scope="row">扩大范围后，每任务只需换一个系数，仍可用 $n/\sigma^2$ 预测</th><td>扩域 1232/2700；12/12 组未过门槛</td><td><strong>原扩域主张被推翻</strong>；原窄范围经验拟合保留。</td></tr>
<tr><th scope="row">约 1.55 次方可以描述整个后期回升</th><td>外推到窗口末，78/78 超出两倍误差</td><td><strong>尾部外推被推翻</strong>；原相对距离 0.1–1 内的短段描述保留。</td></tr>
<tr><th scope="row">至少 90% 的回升点应先把训练误差压到初值 10% 以下</th><td>仅 766/3137（24.4%）满足；中位仍为初值 33%</td><td><strong>这条必要条件预测被推翻</strong>；训练误差还不很低时，泛化就可能恶化。</td></tr>
<tr><th scope="row">旧归一化修正能让跨样本量的系数聚合极差减半</th><td>旧 r113 的极差反而增至 4.68 倍</td><td><strong>该修正预测失败</strong>；不能用部分种子变好覆盖总体失败。</td></tr>
</tbody></table></div>
<figure><a href="figs/d2_20261029_refutation.svg" aria-label="打开原公式扩域反证图"><img src="figs/d2_20261029_refutation.png" alt="12组任务各自校准系数后，原样本数除以噪声方差公式的扩域通过率均低于80%门槛。" loading="lazy"></a><figcaption>这是原公式的扩域检验。每组用 45 个条件校准系数，再评价其余 225 个条件；边界最低点按失败计。它与后面 5% 消融的评价对象和分母不同。</figcaption></figure>

<p class="note">上述“被推翻”都对应明确的检验范围。尚未测试的学习特征、分类、Adam 等场景，标为未验证，不伪装成已经做过失败实验。</p>
</section>
<section id="formulas"><h2>3 · 换个思路：不只看样本数，把误差拆开算</h2>
<p class="bridge"><strong>为什么转向这里：</strong>一个比例公式丢掉了任务和学习过程的细节。我们转而看完整误差由哪些部分组成，哪些在改善，哪些在变差。</p>
<p>本研究的模型先把输入变成固定的 ReLU 特征，只学习最后如何组合这些特征。隐藏特征不随训练变化，平方损失下的梯度更新因此可以解析求解。<strong>固定特征是关键前提，小模型则让完整计算容易实现。</strong> 这些结果不能直接搬到训练中持续改变特征的网络上。</p>
<p>下面把对训练噪声取平均后的干净验证误差称为<strong>风险 $R$</strong>。它可以拆成：没有训练标签噪声时本来就有的误差 $B$，加上学习标签噪声带来的平均代价 $\sigma^2N$。</p>
<div class="table-wrap"><table class="formula-table"><thead><tr><th scope="col">想回答什么</th><th scope="col">公式</th><th scope="col">人话解释与适用条件</th></tr></thead><tbody>
<tr><th scope="row">误差由什么组成？</th><td>$R(t)=B(t)+\sigma^2N(t)$</td><td>固定输入、特征和零均值独立同方差训练噪声下的精确分解；它是解析事实，不是新发现。</td></tr>
<tr><th scope="row">为什么还会继续改善？</th><td>$B'(t)+\sigma^2N'(t)&lt;0$</td><td>若无噪声部分改善得比噪声代价增加得快，总误差仍下降。</td></tr>
<tr><th scope="row">何时转为恶化？</th><td>$B'(t)+\sigma^2N'(t)&gt;0$</td><td>总变化率为正。也可能是 $B$ 自己在增加，不能总说成“只因拟合噪声”。</td></tr>
<tr><th scope="row">最低点的条件是什么？</th><td>$B'(t_*)+\sigma^2N'(t_*)=0$</td><td>连续谱插值的光滑内部最低点满足斜率为零；不要求 $B=\sigma^2N$，也不保证只有一个谷底。</td></tr>
</tbody></table></div>
<p class="note">$B$：无训练噪声时的验证误差，不保证始终下降；$N$：单位噪声方差的响应；撇号：随步数的变化率。连续插值的导数条件不等于整数步最低点的精确等式。</p>
<p><strong>已有测量支持的具体认识：</strong>在 3137 个内部最低点条件中，到一半最低点时间时，噪声代价已达到最低点时该项的中位 77%（范围 35%–100%）。所以“总误差尚未回升”不等于“还没开始学进噪声”。这反驳一种直觉解释，不把它冒充为新增的原预注册主张。</p>
<details><summary>为什么可以不逐步训练，直接算出风险？</summary><p>每个谱方向的参数偏离最终拟合解的量满足 $e_i(t+1)=(1-2\eta\lambda_i)e_i(t)$；$\lambda_i$ 是固定训练特征协方差的特征值。因此可以直接写出第 $t$ 步的参数，而不必逐步迭代。</p><p>这个固定因子作用于<strong>参数在该方向上的偏差</strong>，不是说验证误差每一步都按固定比例下降。验证风险还要结合目标、验证输入上的几何结构和噪声来算。当前完整风险和 5% 选择使用干净目标与验证几何，尚不是只看带噪训练数据就能部署的早停方法。</p><p>单次训练也不等于噪声平均：旧 60 个条件里，单次干净验证最低点与期望风险最低点之比为 0.24–86.72。平均曲线的最低点不能直接当成每次训练的最佳步数。</p></details>
</section>
<section id="method"><h2>4 · 先全删，再补回 5%：交叉项到底有多重要？</h2>
<p>简单公式失败后，我们研究了一个更窄的问题：<strong>已知完整风险表达式，能否用少量项近似保留停止决策？</strong> 这与解释“为什么模型实际泛化变弱”有关，但不是同一个实验目标。</p>
<p>两个学习方向分别贡献 $a$ 和 $b$，目标为 $y$，平方误差展开为 $(a+b-y)^2=a^2+b^2+2ab-2y(a+b)+y^2$。其中 $2ab$ 是交叉项，记录两部分相互增强或抵消。例如 $a=2,b=-1,y=1$，完整误差为 0；漏掉交叉项 −4，估计会变成 4。这只是手算示意。</p>
<div class="callout warning"><strong>删项改的是计算公式，没有改变真实训练。</strong> 所以“删项后停止时间算错”说明近似漏掉了重要信息，不能据此说交叉项导致了过拟合，更不能说保留交叉项改善了模型本身的泛化。</div>
<div class="table-wrap"><table class="formula-table"><thead><tr><th>计算方式</th><th>公式</th><th>改变了什么</th></tr></thead><tbody>
<tr><th scope="row">完整风险</th><td>$R=R_{\rm diag}+\sum_{i&lt;j}C_{ij}$</td><td>保留所有方向之间的配合。</td></tr>
<tr><th scope="row">只留对角项</th><td>$\widehat R=R_{\rm diag}$</td><td>只删除谱二次项中的交叉项；目标匹配项等仍保留。</td></tr>
<tr><th scope="row">保留至多 5%</th><td>$\widehat R_S=R_{\rm diag}+\sum_{(i,j)\in S}C_{ij}$</td><td>$S$ 是保留名单；不重新拟合系数，不删除模型参数。</td></tr>
</tbody></table></div>
<p>沿用已有的 <strong>12 组任务与特征 × 10 档样本量 × 9 档噪声 × 3 个主种子 = 3240 条曲线</strong>。模型先把输入变成 128 个固定特征，训练只调整这些特征怎样组合成答案，也就是只训练输出层。我们复用已有学习过程，比较不同误差计算方式；本轮没有重新训练网络。</p>
<h3>第一轮：先把交叉项全部删掉</h3>
<p><strong>问题：</strong>各方向之间的相互影响能不能忽略？我们保留其他项，只删掉谱二次项中的交叉项，再看建议停止时间是否还可靠。结果：12 组中只有 6 组达到时间和额外误差的双重门槛；下面的正弦例子甚至从第 255 步错选到第 1 步。</p>
<p class="bridge"><strong>这引出下一问：</strong>全删不行，那是否只补回少量重要的项，就能恢复大部分停止决策？</p>
<h3>第二轮：只保留至多 5%，与随机和按大小选择比较</h3>
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
<section id="example"><h3>一个贯穿两轮的例子：第 255 步为什么会被错认成第 1 步？</h3>
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
<section id="results"><h3>补回 5% 后：从 6/12 组达标，提高到 12/12</h3>
<p><strong>“所有交叉项都可以忽略”的注册检验失败：</strong>对角近似只有 6/12 组同时过关。补回少量有针对性的项后，在同一批已看过的数据中改善明显。再看全部条件：按变化选择最多 5% 的交叉项，<strong>时间预测通过率为 97.0%，额外误差通过率为 98.8%</strong>。前者只评价 3137 个内部最低点，后者评价全部 3240 个条件；它们不是模型答题的正确率。</p>
<p>下面的图比较“保留多少交叉项”和“有多少条件通过上述标准”。横轴越小，保留的交叉项越少；纵轴越高，停止效果越好。<strong>Pareto 前沿</strong>指已测方法中，没有另一方法在“保留不更多、效果不更差”的同时，至少有一项更好的点。</p>
<figure><a href="figs/d2_20261028_pareto.svg" aria-label="打开高清 Pareto 对比图"><img src="figs/d2_20261028_pareto.png" alt="实测 Pareto 对比：0%全删、最多5%按变化或大小及十次随机选择、100%完整参照；纵轴分别为时间和低额外风险通过率。" width="1600" height="800"></a><figcaption>只测了 0%、5%、100% 三档预算；中间比例没有实验，不能据此推断连续前沿。5% 为上限，实际向下取整。完整参照的 100% 通过率来自评价定义；横轴不代表运行时间或训练成本。</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th scope="col">方法</th><th scope="col">交叉项预算</th><th scope="col">时间在两倍以内</th><th scope="col">额外误差 ≤ 初值 1%</th><th scope="col">达标任务组</th></tr></thead><tbody>@@MAIN_ROWS@@</tbody></table></div>
<p><strong>变化法与大小法基本打平。</strong> 两者都通过 12 组，两个总体指标各只差 1 条，不能据此认定变化评分更优。随机法十次均明显较差；这些是同一批数据上的十份名单，不是十次独立数据实验。</p>
<div class="callout"><strong>这次 12/12 是什么意思？</strong> 每组有 270 个条件，必须同时满足：内部最低点中至少 80% 的时间预测在两倍以内；全部条件中至少 90% 的额外误差不超过初值的 1%。12/12 表示每组都过了这两个门槛，<strong>不表示 3240 条全部正确，也不表示覆盖了 12 类现实任务。</strong> 这是风险表达式压缩的检验；原 $n/\sigma^2$ 公式的扩域失败没有因此改变。</div>
<details><summary>查看十次随机选择的全部结果</summary><div class="table-wrap"><table><thead><tr><th>随机名单</th><th>时间通过</th><th>额外误差通过</th><th>达标组数</th></tr></thead><tbody>@@RANDOM_ROWS@@</tbody></table></div></details>
</section>
<section id="limits"><h3>这一步解决了什么，又没解决什么？</h3>
<div class="callout warning"><strong>12/12 达标，不等于没有反例。</strong> 主方案仍有 94/3137 条时间未达标、40/3240 条额外误差未达标。最差一条把第 862 步选成第 32 步，实际风险从最低的 0.118 变成 0.255，多出初始风险的 12.8%。</div>
<p>这个反例是四维无偏置的加性二次任务，$n=24,\sigma^2=1/32$，seed=6103。另有 252/3240 条近似无噪声误差为负、20/3240 条近似总风险为负；负误差是近似失效，不能当作更好效果。</p>
<div class="table-wrap"><table><thead><tr><th>说法</th><th>证据状态</th><th>为什么要这样标记</th></tr></thead><tbody>
<tr><th scope="row">5% 足以准确处理每一条曲线</th><td><strong>有反例</strong></td><td>94 条时间失败、40 条额外误差失败；但原注册要求是每组达到比例门槛，并非逐条全对。</td></tr>
<tr><th scope="row">5% 近似总是一个有效的风险函数</th><td><strong>有反例</strong></td><td>出现负的平方误差估计。停止决策通过与整条风险函数合理，是两个不同要求。</td></tr>
<tr><th scope="row">按变化选择明显优于按大小选择</th><td><strong>没有优势证据</strong></td><td>时间通过 3043 对 3042；低额外误差通过 3200 对 3199。两者基本打平。</td></tr>
<tr><th scope="row">已经实现训练前、只依赖训练数据的实际预测</th><td><strong>未建立</strong></td><td>当前依赖干净目标与验证几何；解析可算不等于真实问题中信息可得。</td></tr>
<tr><th scope="row">只留 5% 就节省 95% 计算</th><td><strong>未建立</strong></td><td>选项时仍读取所有交叉项；没有端到端算力收益测量。</td></tr>
<tr><th scope="row">已找到泛化恶化的普适原因</th><td><strong>未建立</strong></td><td>当时的 5% 消融只改风险表达式。后面的“排查表达能力”实验虽改变了训练目标，仍没有建立普适原因；固定特征结果也未覆盖端到端网络。</td></tr>
</tbody></table></div>
<p class="note">“未建立”表示没有足够证据，不等于已做实验并证伪。未覆盖分类、交叉熵、Adam、小批量、学习特征；已有 12 组均属合成回归与已观察的开发数据。</p>
</section>
@@CONTINUOUS_SECTIONS@@
<section id="next"><h2>走到现在：有了哪些认识，还差哪一步？</h2>
<div class="table-wrap"><table><thead><tr><th>证据层次</th><th>当前能说什么</th></tr></thead><tbody>
<tr><th scope="row">解析事实</th><td>固定特征下，不同训练方向有不同恢复速度；风险可以精确拆解；训练看不见的方向在零初始化下不更新。</td></tr>
<tr><th scope="row">本次实测</th><td>特定无噪声、可表示目标仍会回升；9 组设置都先恢复线性一半幅度；后期有用信号改善可能被其他成分增加盖过。</td></tr>
<tr><th scope="row">待检验解释</th><td>如果改变学习快慢，泛化回升是否随之改变？观察到先后关系，尚不能回答这个因果问题。</td></tr>
<tr><th scope="row">暂缓，未启动</th><td>原梯度下降与“各可学习方向同速”的对照；按当前用户要求先更新页面。同速实验未预注册、未运行；追加新样本对照也未启动。</td></tr>
<tr><th scope="row">尚未建立</th><td>更复杂任务、可学习特征或大模型上的规律；不依赖干净答案与已知总体分布的停止预测器；通用的 5% 保留保证。</td></tr>
</tbody></table></div>
<p>当前结论只适用于这些固定特征、平方损失、全批量训练的合成回归开发条件。新种子只能检验重复性；要谈更广泛适用，必须事先指定新条件，不能把旧例子反复通过当成泛化。</p>
</section>
<section id="evidence"><h2><span class="section-index">09</span>证据与已有研究：哪些来自论文，哪些来自本实验</h2>
<p>固定特征下解析求解学习过程、按谱方向分析早停，都有已有理论。交叉项来自平方展开，本身不是新发现；历史实验做了删项与停止决策对照，最新六轮则做噪声、目标表达能力、误差组成与学习先后的核验。以下核对的是相关章节，不是完整的新颖性审查。</p>
<div class="table-wrap"><table><thead><tr><th>原始来源</th><th>与 D2 的关系</th><th>不能据此声称什么</th></tr></thead><tbody>
<tr><th scope="row"><a href="https://proceedings.mlr.press/v89/ali19a/ali19a.pdf">Ali、Kolter、Tibshirani，2019</a></th><td>Lemma 1、式（6）、3.1 节：固定特征最小二乘梯度流的解析路径与谱分解。</td><td>论文主要分析连续时间；D2 离散步数实现及其特定数字需要自身证据。</td></tr>
<tr><th scope="row"><a href="https://arxiv.org/abs/2110.11805">Bodin、Macris，2021</a></th><td>随机特征模型在高维渐近设置下的训练与泛化误差时间路径。</td><td>不能把其特定假设下的结果当成 D2 所有有限样本任务的保证。</td></tr>
<tr><th scope="row"><a href="https://www.jmlr.org/papers/volume15/raskutti14a/raskutti14a.pdf">Raskutti、Wainwright、Yu，2014</a></th><td>3.1 节、式（4）–（6）：基于数据谱和噪声水平的早停规则，无需独立验证集。</td><td>统计误差保证不等于准确命中每条曲线的最佳整数步。</td></tr>
</tbody></table></div>
<p class="note">固定 5% 预算与本轮变化评分是实验设计选择，没有“5% 必然足够”的定理。当前检索未找到完全相同的实验，不等于确认首次提出；旧“文献深度确认”页只是摘要级历史对照。</p>
@@CONTINUOUS_EVIDENCE@@
<p class="refs">本次页面更新没有新开实验。历史两轮消融及最新六轮各有早于计算的预注册；发布前按结果清单校验保存产物，完整精度、失败和来源留在证据中。以下数值核验概述针对历史消融：完整曲线重建最大归一化差为 7.8×10⁻¹¹；4320 个掩码已核对，另有 96 次独立特征空间核验。</p>
<div class="table-wrap"><table><thead><tr><th>证据</th><th>直接查看</th></tr></thead><tbody>
<tr><th scope="row">2026-10-09 原 D2</th><td><a href="d2_evidence_20261009/report.md">历史报告</a> · <a href="d2_evidence_20261009/summary.json">完整汇总</a></td></tr>
<tr><th scope="row">2026-10-27 对角消融</th><td><a href="d2_evidence_20261028/diagonal_summary.json">完整汇总</a> · <a href="d2_evidence_20261028/diagonal_independent_audit.json">独立代理审计</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/d2796a484712fc95268c2400a6d85d5cf121163c/research/d2_diagonal_20261027">预注册与源码</a></td></tr>
<tr><th scope="row">2026-10-28 保留 5%</th><td><a href="d2_evidence_20261028/summary.json">完整汇总（含逐组／逐种子）</a> · <a href="d2_evidence_20261028/independent_audit.json">独立代理审计</a> · <a href="d2_evidence_20261028/numerical_verification.json">数值核验</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/7dc745d2c4e21acb98ef4fa0a4534c2e5e707eda/research/d2_sparse_cross_20261028">预注册与源码</a></td></tr>
<tr><th scope="row">第 1 步／255 步例子</th><td><a href="d2_evidence_20261028/focus_case.json">逐项计算</a> · <a href="d2_evidence_20261028/focus_curves.npz">保存的曲线数组</a></td></tr>
<tr><th scope="row">网页与图表来源</th><td><a href="d2_evidence_20261028/export_manifest.json">原件哈希与发布清单</a> · <a href="https://github.com/guoshaoyang-pku/science-of-ai-prototype/tree/main/research/d2_web_20261028">页面与绘图源码</a></td></tr>
</tbody></table></div>
</section><footer class="stamp">更新于 2026-11-06 · 六轮机制研究已审计 · 原广义预测主张仍被推翻 · 固定特征回归 · 已观察数据上的研究结果 · 完整精度保留在 JSON 证据中</footer>
</main></body></html>'''
    template=template.replace('@@CONTINUOUS_SECTIONS@@', (Path(__file__).with_name('continuous_sections.html')).read_text())
    template=template.replace('@@CONTINUOUS_EVIDENCE@@', (Path(__file__).with_name('continuous_evidence.html')).read_text())
    import re
    template=re.sub(r'<span class="section-index">\d+</span>', '', template)
    rendered=template.replace('@@MAIN_ROWS@@',''.join(main_rows)).replace('@@RANDOM_ROWS@@',random_rows).replace('@@TASK_ROWS@@',''.join(task_rows))
    assert '@@' not in rendered
    (WEB / 'D2_overfitting_u_curve.html').write_text(rendered)
    print(json.dumps({'page':'docs/模糊方向/D2_overfitting_u_curve.html','exported_files':len(exports),'all_export_hashes_verified':True},ensure_ascii=False))

if __name__=='__main__': main()
