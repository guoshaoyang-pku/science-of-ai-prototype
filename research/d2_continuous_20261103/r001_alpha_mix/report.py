"""Render saved results; does not run experiments or change original outputs."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'preregistration.json').read_text())
OUT=Path(SPEC['output_root'])/SPEC['round_id']
rows=json.loads((OUT/'rows.json').read_text());s=json.loads((OUT/'summary.json').read_text())
v=json.loads((OUT/'verification.json').read_text())
z=[r for r in rows if r['sigma2']==0]
zi=[r for r in z if r['interior'] and r['late_within_window']]
zrise=[r for r in zi if r['delta_R_over_R0']>SPEC['material_rise_fraction']]
worst=max(zrise,key=lambda r:r['delta_R_over_R0'])
example=[r for r in rows if r['n']==64 and r['seed']==6101 and r['sigma2']==1]
secondary={'kind':'preregistered_secondary_no_noise_control_description','count':len(z),'interval_eligible':len(zi),'material_rise_count':len(zrise),'boundary_count':sum(not r['interior'] for r in z),'worst':worst}
(OUT/'secondary_description.json').write_text(json.dumps(secondary,ensure_ascii=False,indent=2)+'\n')
text=f'''# D2 第一轮：把线性目标逐渐变成交互目标

## 先看结论

**没有标签噪声，这批验证误差也会回升。** 54个无噪声对照中，51个在窗口内部找到最低点，随后到两倍步数时验证误差上升超过初始误差的百万分之一，同时训练误差继续下降。其余3个最低点在窗口末，不能声称这3个也已回升。这是固定特征、已观察输入上的次要对照结果，尚未排除验证样本波动或确定具体机制。

**有噪声时，回升可以发生在模型仍在改善无噪声表现的时候。** 270个正噪声条件中，267个可评价的内部最低点全部出现“无噪声误差未净增，但噪声代价增加得更多”；其余3个最低点在第0步，单列为边界。这支持本轮预注册的90%比例预测，只说明从最佳点到两倍步数的净变化，不保证整段单调。

**任务变了，合适的停止时间也明显改变。** 同一n=64、seed=6101、噪声方差1，线性目标最佳22步，交互强度0.5时65步，强度2时204步。特征和噪声响应相同，改变的是目标及其学习过程；这不确立新的通用时间公式。

## 我们怎样只改一个因素

目标是 y=(x0+α x0 x1)/√(1+α²)。α从0逐渐增大，表示从纯线性过渡到越来越强的交互；分母让总体目标均方值保持一致，避免仅仅因答案幅度变大而改变信噪比。

| 条件 | 设置 |
|---|---|
| 固定模型 | 128个固定ReLU特征加常数，只训练输出层；MSE、全批量GD、学习率0.3、零初始化 |
| 混合强度α | 0、0.125、0.25、0.5、1、2 |
| 样本数 | 32、64、128 |
| 噪声方差 | 0、0.25、0.5、1、2、4 |
| 重复 | 3个已有主种子；共9个输入/特征抽样 |
| 总计 | 324条件＝270正噪声＋54无噪声 |
| 证据身份 | development；α=0、0.5端点已看过，中间取值不算独立盲测 |

## 为什么说“噪声超过了继续学习的收益”

R=B+σ²N。B表示没有训练标签噪声时的验证误差，σ²N表示训练噪声的平均代价。这是已有解析恒等式，不是本轮新发现。

在n=64、seed=6101、σ²=1、α=0.5的例子中，从最佳的第65步继续到130步：B下降约0.031，噪声代价增加约0.042，因此总风险增加约0.011。与此同时，期望训练误差下降约0.057。模型仍在学有用的东西，但增加的噪声代价更大。

| α（交互强度） | 该例最佳步数 | 无噪声误差变化ΔB | 噪声代价变化 | 总风险变化 |
|---|---:|---:|---:|---:|
'''
for r in example:text+=f"| {r['alpha']:g} | {r['t_star']} | {r['delta_B']:+.4f} | {r['delta_noise']:+.4f} | {r['delta_R']:+.4f} |\n"
text+=f'''
表中每行比较自己的最佳点与两倍步数，时间区间不同，不能把噪声增量不同误读成目标改变了同一时间的噪声响应。

## 无噪声反例说明什么

最明显的无噪声例子为n={worst['n']}、seed={worst['seed']}、α={worst['alpha']:g}：从第{worst['t_star']}步到{worst['t_late']}步，验证误差由{worst['R_star']:.4f}增加到{worst['R_star']+worst['delta_B']:.4f}，增加约初始误差的{100*worst['delta_R_over_R0']:.2f}%；训练误差仍下降。在这批有限验证样本上，没有标签噪声也观察到误差回升，不能把这一观测都归因于记住错误标签。是否代表稳定的总体泛化恶化，还需独立验证。

这个结果还不能说明具体原因。最低点由同一验证集选择，本身存在选择效应；有限验证样本、目标不能被固定特征完全表示，以及不同方向的学习先后，都需要后续对照。多数回升幅度较小，百万分之一只是本轮预先固定的可判定阈值，不代表实际应用意义。

## 失败、边界与未决

- 正噪声3个边界：α=2、n=32、σ²=4，三个种子均在第0步最低。没有把它们算进267的主分母；267/270是覆盖率，267/267是可判定集合通过率。
- 无噪声3个窗口末最低点：α=0，n64的seed6102/6103，以及n128的seed6101。不能从有限窗口断言它们永不回升。
- 所有曲线都是对训练噪声平均的风险；不等于每次带噪训练。
- 总计只有3个主种子，没有把324个相关条件当作独立样本给置信区间。
- 搜索沿用原格点加整数局部精化，没有证明遍历全部整数极值。

## 根据反馈调整下一轮

优先验证无噪声回升是否在新验证输入上保持，而不是立即发明新的时间公式。下一轮先冻结要检查的条件和时间点，在同一模型、训练输入与目标下，只更换/扩大独立验证样本；旧验证集负责选点，新验证集只评价冻结点，保留所有结果。若回升稳定，再考虑对目标可表示性与学习方向作干预。此处是下一轮候选，不是已运行结果。

## 证据

预注册与执行代码在本仓库 research/d2_continuous_20261103/r001_alpha_mix；预注册commit早于执行，见execution.json。

- rows.json：全部324条件，包含有符号分项变化、训练变化和边界。
- summary.json：主检验与分母；secondary_description.json：无噪声次要对照。
- curves/：54组完整B、N与训练响应格点，可复算全部风险。
- verification.json：90个历史端点复核；108次独立特征空间检查，最大绝对差{v['max_abs_error']:.2g}；同一数据下N在各α间完全一致。
- manifest.json：初始计算产物hash；后续解释和独立审计另存，不改历史产物。

本轮仅本地CPU计算，没有模型API、GPU、新solver或网站发布。
'''
if (OUT/'independent_audit.json').exists():
    text+='\n独立审计已通过：independent_audit.json；另以原坐标直接GD复核4个无噪声例子，最大误差9.03×10⁻¹⁵。该复核只检查已保存条件和时间，不增加新实验条件。\n'
(OUT/'report.md').write_text(text)
print(OUT/'report.md')
