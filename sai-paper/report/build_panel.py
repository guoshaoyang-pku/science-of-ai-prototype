#!/usr/bin/env python3
"""Assemble the single-file Chinese panel for the SAI paper's core figures."""
import os
import re
from pathlib import Path

HERE = Path(__file__).parent
# Public builds anonymize model identities, matching gen_charts.py and the public figures.
STRONGEST = "Frontier-1" if os.environ.get("SAI_PUBLIC") else "GPT-5.6 Sol"


def inline_svg(name, cls="chart"):
    t = (HERE / "figs" / name).read_text(encoding="utf-8")
    t = re.sub(r"<\?xml[^>]*\?>", "", t)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<svg([^>]*?)width=\"[^\"]*\"([^>]*?)>", r"<svg\1\2>", t, count=1)
    t = re.sub(r"<svg([^>]*?)height=\"[^\"]*\"([^>]*?)>", r"<svg\1\2>", t, count=1)
    t = t.replace("<svg ", f'<svg class="{cls}" ', 1)
    return t.strip()


AXES = inline_svg("diagram_axes.svg")
LADDER = inline_svg("diagram_ladder.svg")
LOOP = inline_svg("diagram_loop.svg")
MAP = inline_svg("diagram_map.svg")
EQUIV = inline_svg("equiv_compute.svg")
LIFT = inline_svg("kb_lift.svg")

HEROES = """
<table>
<tr><th>系统（具体工作）</th><th>M0 知识提分</th><th>M1 发现规律</th><th>M2 出人意料</th><th>M3 提升前沿</th><th>M4 解释全部</th><th>M5 机器验证的科学</th><th>M6 论文级科学</th></tr>
<tr><td>智能体记忆（ExpeL、Voyager、AWM、SkillRL、Dynamic Cheatsheet）</td><td>过</td><td>部分</td><td>未过</td><td>未过</td><td>未过</td><td>未过</td><td>未过</td></tr>
<tr><td>程序搜索（FunSearch、AlphaEvolve、OpenEvolve）</td><td>—</td><td>部分</td><td>过†</td><td>未过</td><td>未过</td><td>未过</td><td>未过</td></tr>
<tr><td>AI Scientist v1/v2、Agentic Scientist</td><td>未过</td><td>部分</td><td>部分</td><td>未过</td><td>未过</td><td>未过</td><td>部分*</td></tr>
<tr><td>Zochi / Locus</td><td>未过</td><td>部分</td><td>部分</td><td>未过</td><td>未过</td><td>未过</td><td>部分*</td></tr>
<tr><td>AI co-scientist</td><td>未过</td><td>部分</td><td>过†</td><td>未过</td><td>未过</td><td>未过</td><td>未过</td></tr>
<tr><td>Robin（生物医学）</td><td>未过</td><td>部分</td><td>过†</td><td>未过</td><td>未过</td><td>未过</td><td>未过</td></tr>
<tr><td>他域定律发现机（Eureqa、AI Feynman）</td><td>—</td><td>过†</td><td>过†</td><td>—</td><td>—</td><td>部分†</td><td>未过</td></tr>
<tr><td><b>SAI-min（本工作，ML 域内）</b></td><td><b>过</b></td><td><b>过</b></td><td><b>过</b></td><td>未过</td><td>未过</td><td>部分</td><td>未过</td></tr>
</table>
<p style="font:13px/1.7 var(--head);color:var(--muted)">「过」= 通过且有可复查证据；「部分」= 部分通过；「未过」= 未通过；— = 不适用。† 在该系统的非 ML 本域内。* 由同行评审授予（workshop / 主会接收），没有机器验证的证据链：这是社会学意义上的 M6，不是预注册预测意义上的 M6。SAI-min 的证据：M0 配对提升 +14.2 / +15.8 点；M1 有 34 条假说在新题上验证、9 条被证伪；M2 有与前沿模型先验矛盾且验证成立的发现；M5 为预注册机制检验并记录证伪、经人工审计，尚未达到「确认的新理论」。</p>
"""

WORKS_ROWS = [
    ("Kaplan 2020（Scaling Laws for Neural Language Models）", "经验定律", "L2", "loss 与参数量、数据量的幂律关系，拟合并外推"),
    ("Hoffmann 2022（Chinchilla）", "经验定律", "L2", "计算最优的参数量–数据量分配关系"),
    ("Muennighoff 2023（数据受限缩放）", "经验定律", "L2", "重复数据下的缩放律修正"),
    ("Yang 2022（μP / Tensor Programs 系列）", "经验定律", "L2", "超参随宽度零样本迁移"),
    ("Zhang 2017 / Belkin 2019（双下降）", "经验定律", "L2", "插值峰与双下降曲线"),
    ("Power 2022（grokking）", "经验定律", "L2", "延迟泛化的相变行为"),
    ("Gao 2022（奖励过优化）", "经验定律", "L2", "代理奖励缺口与 KL（新旧策略散度）开方的函数关系"),
    ("Hu 2025（RL 计算缩放的艺术）", "经验定律", "L2", "sigmoid 计算–表现曲线，40 万 GPU 小时"),
    ("本工作（有效时间 U 与过拟合起点）", "经验定律", "L2", "有效时间 U（学习率累积和）坍缩训练曲线；过拟合起点 U_c（临界累积更新量）正比于样本数/噪声方差，盲测后收窄"),
    ("Jacot 2018（NTK）", "学习理论", "L3", "宽极限下的收敛与泛化理论"),
    ("Chizat 2019（lazy / rich）", "学习理论", "L3", "懒训练与富训练的二分"),
    ("Bahri 2023（缩放律的统计力学）", "学习理论", "L3", "从第一性原理解释缩放指数"),
    ("Agarwal 2020（策略梯度理论）", "学习理论", "L3", "最优性、近似与分布偏移分析"),
    ("Watkins 1992（Q-learning）", "学习理论", "L3", "带假设的几乎必然收敛"),
    ("Hyperball 2026 及其批评", "优化器线程", "L3", "从尺度不变性推导范数与角向步长，在理想化假设内成立；批评文给出竞争解释"),
    ("Olsson 2022（induction heads）", "可解释性", "L2", "in-context 学习的电路机制（证据零散）"),
    ("Elhage 2022（叠加）", "可解释性", "L1–L2", "特征存储的几何"),
    ("Bricken 2023 / Templeton 2024（单义性）", "可解释性", "L1–L2", "字典学习特征，直至前沿规模"),
    ("Nanda 2023（grokking 机制）", "可解释性", "L3", "傅里叶电路解释 + 干预预测（限于所研究的模运算任务）"),
    ("Shao 2024（GRPO / DeepSeekMath）", "经验 ML", "L1", "组相对优势在指定奖励下稳定训练"),
    ("Zheng 2025（GSPO）", "经验 ML", "L1", "序列级裁剪稳定 MoE 的 RL 训练"),
    ("Dao 2022（FlashAttention）", "系统", "L1", "IO 感知精确注意力的吞吐关系（声明者域内）"),
    ("Hu 2021（LoRA）", "经验 ML", "L1", "低秩适配在声明者任务上匹配全参微调"),
    ("Shinn 2023（Reflexion）", "智能体记忆", "L1", "言语反思提升后续任务成功率"),
    ("Wang 2023（Voyager）", "智能体记忆", "L1", "可执行技能库的自动积累"),
    ("Zhao 2023（ExpeL）", "智能体记忆", "L1", "跨任务洞察，带增删改投票"),
    ("Wang 2024（Agent Workflow Memory）", "智能体记忆", "L1", "归纳工作流跨任务复用"),
    ("Jiang 2025（G-Memory）", "智能体记忆", "L1", "多智能体层级记忆追踪"),
    ("Suzgun 2025（Dynamic Cheatsheet）", "智能体记忆", "L1", "测试时自适应记忆"),
    ("Lu 2024（AI Scientist）", "AI-scientist", "L1", "端到端生成 workshop 级稿件；产出为评审稿件"),
    ("Yamada 2025（AI Scientist v2）", "AI-scientist", "L1", "agentic 树搜索，workshop 接收；产出为评审稿件"),
    ("Intology 2025（Zochi / Locus）", "AI-scientist", "L1", "ACL 主会接收；RE-Bench 超人类专家；产出为评审稿件"),
    ("Gottweis 2025（co-scientist）", "AI-scientist", "L2（候选）", "给人类实验者的排序假说，尚未系统扩展检验"),
    ("Schmidgall 2025（Agent Laboratory）", "AI-scientist", "L1", "人机协作研究流水线"),
    ("Ghareeb 2025（Robin）", "AI-scientist", "L2（候选）", "生物医学发现战役；产出为待检假说"),
    ("Nam 2025: MLE-STAR（带搜索与定向精炼的 ML 工程智能体）", "AI-scientist", "L1", "在搜索与定向精炼下完成 ML 工程任务"),
    ("Romera-Paredes 2024（FunSearch）", "程序搜索", "L1", "LLM 进化搜索产出新数学构造；产出为可执行工件"),
    ("DeepMind 2025（AlphaEvolve）", "程序搜索", "L1", "编码智能体做算法发现；产出为可执行工件"),
    ("Fawzi 2022（AlphaTensor）", "程序搜索", "L1", "矩阵乘法算法；产出为可执行工件"),
    ("Langley 1987（BACON）", "他域发现机", "L2", "从数据表再发现经验定律"),
    ("King 2004 / 2009（Adam 机器人科学家）", "他域发现机", "L2–L3", "假说–实验闭环的机器人科学家"),
    ("Schmidt 2009（Eureqa）", "他域发现机", "L2", "从实验时序蒸馏符号定律"),
    ("Udrescu 2020（AI Feynman）", "他域发现机", "L2", "物理启发的符号回归"),
    ("Merchant 2023（GNoME）", "他域发现机", "L2", "220 万稳定晶体"),
    ("Jumper 2021（AlphaFold）", "他域发现机", "L3", "蛋白质结构预测（限于该域；本清单未拆具体主张）"),
    ("Bernat 2026（Modelpedia）", "策展", "L1", "从 ICLR 24–25 抽取千余条发现并编目"),
    ("验证缺口调查 2026", "策展", "—", "元研究审计：主张比代码更难验证，35 个系统七维审计"),
    ("Chan 2024（MLE-bench）", "基准", "—", "Kaggle 奖牌；测量仪器，测 L1 任务表现"),
    ("Aronovich 2024（RE-Bench）", "基准", "—", "限时 ML 工程，对人类专家；测量仪器"),
    ("Fajman 2025（PaperBench）", "基准", "—", "复现论文结果；测量仪器"),
    ("METR 2025（长任务视野）", "基准", "—", "可完成任务的时间视野；测量仪器"),
    ("ArchitectureIQ 2026", "基准", "—", "判断真实训练 run 的胜负；测量仪器，最接近知识"),
    ("本工作（SAI-min）", "本工作", "L1→L2", "免训练闭环；+15.8 点；程序验收；预注册证伪"),
]

works_table = "\n".join(
    f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in WORKS_ROWS
)

HTML = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML 的产出大多停在「知识」层：缺的是可验证的定律与科学</title>
<style>
  :root {{
    --dark: #141413; --light: #faf9f5; --mid: #b0aea5; --soft: #e8e6dc;
    --muted: #5e5d59; --orange: #d97757; --blue: #6a9bcc; --green: #788c5d;
    --head: "Poppins", "PingFang SC", "Noto Sans CJK SC", "Hiragino Sans GB", Arial, sans-serif;
    --body: "Lora", "Noto Serif CJK SC", "Songti SC", Georgia, serif;
    --mono: "SF Mono", Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--light); color: var(--dark); font: 17px/1.8 var(--body); }}
  main {{ max-width: 760px; margin: 0 auto; padding: 72px 24px 96px; }}
  h1, h2, h3 {{ font-family: var(--head); font-weight: 600; line-height: 1.35; letter-spacing: -0.01em; }}
  h1 {{ font-size: 30px; margin: 0 0 16px; }}
  h2 {{ font-size: 21px; margin: 60px 0 12px; }}
  p {{ margin: 0 0 16px; }}
  .lede {{ font-size: 19px; }}
  a {{ color: inherit; text-decoration: underline; text-decoration-color: var(--orange); text-underline-offset: 3px; }}
  code, .id {{ font-family: var(--mono); font-size: 0.86em; background: var(--soft); padding: 1px 5px; border-radius: 4px; }}
  .settings {{ display: flex; flex-wrap: wrap; gap: 8px 20px; list-style: none; padding: 0; margin: 24px 0 0;
              font: 14px/1.6 var(--head); color: var(--muted); }}
  .settings li {{ display: inline-flex; align-items: center; gap: 6px; }}
  .settings svg {{ width: 15px; height: 15px; stroke: var(--dark); fill: none; stroke-width: 1.6;
                  stroke-linecap: round; stroke-linejoin: round; }}
  .settings b {{ color: var(--dark); font-weight: 500; }}
  figure {{ margin: 28px 0; padding: 20px; background: #fff; border: 1px solid var(--soft); border-radius: 10px; }}
  figure img, figure svg.chart {{ display: block; width: 100%; height: auto; }}
  figcaption {{ margin-top: 12px; font: 14px/1.6 var(--head); color: var(--muted); }}
  .warn {{ border-left: 3px solid var(--orange); padding: 4px 0 4px 16px; margin: 24px 0; }}
  .warn strong {{ font-family: var(--head); }}
  details {{ margin: 16px 0; }}
  summary {{ cursor: pointer; font: 15px var(--head); color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font: 13.5px/1.6 var(--head); margin: 16px 0; }}
  th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--soft); vertical-align: top; }}
  th {{ font-weight: 500; color: var(--muted); }}
  footer {{ margin-top: 80px; font: 13px var(--head); color: var(--mid); }}
</style>
</head>
<body>
<main>

<h1>ML 的产出大多停在「知识」层：缺的不是更大的模型，而是可验证的定律与科学</h1>
<p class="lede">我们用误差、代价、不适用测度三根轴衡量一条主张，按边界怎样得知把主张分成知识、定律、科学三层，并把一批代表性工作定位到这张地图上；一个零训练的原型（SAI-min，即 Science of AI 最小闭环：解题–发现–总结）已经从知识爬到了候选定律。</p>
<ul class="settings">
  <li><svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/></svg>测试床 <b>ArchitectureIQ</b>（判断多个真实训练配置谁赢的题库，500 题）</li>
  <li><svg viewBox="0 0 24 24"><path d="M4 19l5-7 4 4 7-10"/></svg>原型 <b>SAI-min</b>（解题–发现–总结三段循环，零训练）</li>
  <li><svg viewBox="0 0 24 24"><path d="M6 3v18M18 3v18M6 12h12"/></svg>旧稿审计 <b>RL 经典 51 篇</b>：43 知识 / 3 定律 / 5 科学（旧门槛）</li>
  <li><svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>引用 <b>108 条</b>，全部程序化核验</li>
  <li><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg><b>2026-10-28</b></li>
</ul>

<h2>一条主张好在哪里：误差、代价、不适用测度三个方向都要减小</h2>
<p>SAI（Science of AI）研究 AI 系统在给定条件下会怎样，目标是以更小的误差、更低的代价、在更大的条件范围内预知结果。先固定一份合同：条件空间 Ω（任务 × 架构 × 超参 × 数据 × 预算的全部组合，一个元素是一次完整的训练设定）、Ω 上的权重 μ（关心哪些设定、各占多少）、要预测的量和容忍度 ε（答案允许偏离真值多少）。一条主张对每个设定要么给出答案，要么声明不适用。</p>
<p>误差 e 是它作答的设定里超出 ε 的比例；代价 c 是得到答案所花的计算与试探，以直接把这次训练跑完为 1；不适用测度 u 是它不作答的设定在 μ 下的占比。三者都在 0 到 1 之间。直接训练位于 (e≈种子噪声, c=1, u=0)：精确、处处可用、最贵。科学的方向是从这里走向原点：用更少的试探得到同样的答案。原点是「万物理论」式的理想方向，不是已证的数学结论。</p>
<figure>
  {AXES}
  <figcaption>按用户手绘三轴图重绘。坐标都相对于合同 (Ω, μ, ε)；三个点的位置是示意，不是测量值。</figcaption>
</figure>
<p>边界已知很值钱。主张若说清自己在哪里不适用，那部分可以回退到直接训练：u 按单价 1 换成代价，坐标变为 ((1−u)e, (1−u)c+u, 0)。不知道边界的主张在未知位置出错，使用者挑不出该回退的条件。</p>
<p class="warn"><strong>三点限制。</strong>u 依赖 μ：换一种权重，同一主张的 u 会变，两条主张只能在同一份合同下比较。有限实验只给估计和上界：100 个独立设定零失败，95% 单侧上限仍约 3%；未测的设定记为未知，不算成立。u=0 表示在 μ 下几乎处处适用，不等于逐点无例外。</p>

<h2>三层的区别在于边界怎样得知：知识靠观察，定律靠系统扩展，科学靠从原理推出</h2>
<p>分类的单位是一条主张，不是论文。知识是在一个或少数具体设定下观察到的现象，外延尚未展开，未测设定记为未知。定律是沿声明的条件轴系统扩展、尽量解除约束后仍保留的稳定关系，边界由观测到的失败定位，预算用完不等于到头。科学是从可独立检验的前提推出关系为何成立、边界在哪里，边界就是前提失效的地方；单条明确的条件推导即可，不要求同时推出多条定律。</p>
<p>另外两件事分开记录。Scope 是适用范围有多大，即上一节的 u：窄域推导可以是科学，广域经验关系仍是定律。证据协议是坐标有多可信：预注册、盲测、程序裁决、独立复现提高可信度，但不改变层次。科学还要分开两问：推导在假设内是否正确；前提在真实系统中是否成立，后者要像定律一样用实验检验。锚点：GRPO、GSPO 是知识，scaling law 是定律，Hyperball 按主贡献是科学。</p>
<figure>
  {LADDER}
  <figcaption>左：三层的判据、晋升与降级。右栏：层次、Scope、证据协议分开记录；一篇论文可以同时含三层的主张。</figcaption>
</figure>

<h2>知识的生产是一个五算子闭环，四条纪律保证主张可以被证伪</h2>
<p>任何生产知识的系统，都可以写成五个算子在「实验宇宙」（任务 × 架构 × 超参 × 数据 × 预算的全空间）上的循环：解题、发现、核验、总结、出题。现有系统的差别，在于实现了哪几个算子、核验有多硬。四条纪律让循环诚实：预测先预注册再跑实验；核验只用注册后生成的新题；封存测试集看一次就降级；证伪记录永不删除。</p>
<figure>
  {LOOP}
  <figcaption>五算子闭环与知识库的状态流转。虚线是知识库的读与写。</figcaption>
</figure>

<h2>现有工作的重心在知识层：旧稿审计的 51 篇 RL 论文里 43 篇是知识、3 篇是定律、5 篇是科学</h2>
<p>把十个家族放进「谁生产 × 产出在哪层」的平面：经验 ML 与系统论文、智能体记忆、AI-scientist 系统都停在知识层；缩放律一族是少数定律岛；NTK 与统计力学是脱离前沿实践的科学飞地；FunSearch、AlphaEvolve 有最强的核验却不沉淀主张；Eureqa、GNoME、AlphaFold 证明机器在他域能爬到定律和科学，因为那里的 oracle（判定对错的裁判）便宜。图中所列基准都只测第一层的任务表现。51 篇是论文草稿的旧样本，按旧门槛分类，不代表全领域比例。</p>
<figure>
  {MAP}
  <figcaption>气泡大小示意产出体量；蓝色 = 机器可验证，灰色 = 评审或成功率验证，橙色 = 本工作。箭头是 SAI-min 的爬升轨迹。</figcaption>
</figure>

<h2>在 ArchitectureIQ 闭卷协议下，每个前沿模型的知识都不值试探训练预算的十分之一</h2>
<p>等价算力 ρ*（知识值多少：花每个候选训练预算的百分之几做「各训一段、选当前领先者」的试探，就能追平该模型的闭卷准确率）给出统一标价。只用 10% 预算的试探选对 79.4%，超过全部闭卷前沿模型；最强的 {STRONGEST} 也只值 8.5%。在胜负到训练后期才分的难题上，试探掉到 44–53%，而所有模型只有 41–53%：知识最值钱的地方，正是模型最没知识的地方。</p>
<figure>
  {EQUIV}
  <figcaption>横条：各模型的等价算力 ρ*（括号内为闭卷准确率）。橙色虚线：10% 预算的无知识试探基线。</figcaption>
</figure>

<h2>经核验的知识最多提 15.8 点；按探索奖励排序只取前 20 条时反而扣 3.3 点</h2>
<p>SAI-min 的三段循环在配对臂上把准确率从 0.454 提到 0.596（架构题，+14.2 点）；按置信度（被引用后答对答错的拉普拉斯平滑比率）排序取前 40 条并留 10 个新主张试用槽，+15.8 点。按 PUCT（一种给未验证主张加探索奖励的排序公式）排序时，结果取决于上屏条数：取前 40 条仍提 3.3 点（0.492→0.525）；取前 20 条（旧管线默认）扣 3.3 点（0.450→0.417），低于不带知识库，因为探索项让未验证的新主张持续轮换上屏。76 条新假说里 34 条在后续新题上验证、9 条被证伪淘汰。上屏规则应按核验后的正确率排序，新主张只给有限的试用槽。</p>
<figure>
  {LIFT}
  <figcaption>同题同种子配对：不带 / 带知识库的准确率。橙色为带知识库（主结果）。</figcaption>
</figure>

<h2>对着里程碑梯子打分：还没有系统通过「机器验证的科学生产」</h2>
<p>里程碑梯子有七级：M0 = 知识提分、M1 = 发现规律、M2 = 出人意料、M3 = 提升前沿模型、M4 = 解释全部题目、M5 = 机器验证的科学、M6 = 论文级科学。记忆系统过 M0；发现机在本域过 M1–M2；AI-scientist 靠评审摸到 M6 但跳过验证梯；M5 在 ML 域内至今空缺。SAI-min 过 M0–M2，M5 partial：它的机制假说被自己的盲测杀死过，这恰恰是它与「产稿系统」的区别。</p>
{HEROES}

<details>
<summary>具体工作支撑清单（{len(WORKS_ROWS)} 条）</summary>
<table>
<tr><th>工作</th><th>家族</th><th>层级</th><th>它提供什么（一句话）</th></tr>
{works_table}
</table>
</details>

<footer>定义全文：<code>sai-paper/SAI_definition.md</code> · 来源：论文草稿 <code>sai-paper/main.pdf</code> · ArchitectureIQ 与 知识库循环原型站点 · 周会与交接文档 · 2026-10-28</footer>
</main>
</body>
</html>
"""

(HERE / "sai-core-figures.html").write_text(HTML, encoding="utf-8")
print("panel written:", len(HTML), "bytes")
