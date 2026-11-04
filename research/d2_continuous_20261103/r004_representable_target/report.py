"""Render saved r004 results only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'preregistration.json').read_text());out=Path(spec['output_root'])/spec['round_id']
rows=json.loads((out/'rows.json').read_text());summary=json.loads((out/'summary.json').read_text());v=json.loads((out/'verification.json').read_text());execution=json.loads((out/'execution.json').read_text())
e=[r for r in rows if r['status']=='evaluated'];primary=[r for r in e if r['primary_member']];witness=[r for r in primary if r['witness']]
remaining=max(witness,key=lambda r:r['delta']);flipped=min(primary,key=lambda r:r['delta_change'])
for candidate in ['/System/Library/Fonts/STHeiti Light.ttc','/Library/Fonts/Arial Unicode.ttf']:
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate);plt.rcParams['font.family']=font_manager.FontProperties(fname=candidate).get_name();break
plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'d2-r004'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',gridspec_kw={'width_ratios':[.85,1.25]})
ax.barh([1,0],[50,5],color=['#9da7b2','#b54c40'],height=.5)
ax.set_yticks([1,0],['原目标','可表示目标']);ax.set_xlim(0,57)
for y,count in [(1,50),(0,5)]:ax.text(count+1,y,f'{count}/50',va='center')
ax.set_xlabel('原50个回升条件中，仍回升的数量');ax.set_title('45个转为改善，5个仍变差',loc='left',pad=12)
examples=[flipped,remaining]
for i,r in enumerate(examples):
    bx.bar(i-.17,r['original_delta'],width=.32,color='#9da7b2',label='原目标' if i==0 else None)
    bx.bar(i+.17,r['delta'],width=.32,color='#277d80',label='可表示目标' if i==0 else None)
    for offset,value in [(-.17,r['original_delta']),(.17,r['delta'])]:
        bx.annotate(f'{value:+.5f}',(i+offset,value),xytext=(0,5 if value>=0 else -13),textcoords='offset points',ha='center',fontsize=9)
bx.axhline(0,color='#444444',linewidth=.8)
bx.set_xticks([0,1],[f"种子{r['seed']}，α={r['alpha']:g}\n第{r['t_star']}→{r['t_late']}步" for r in examples])
bx.set_ylabel('晚期 − 早期的总体MSE');bx.set_ylim(-.009,.011)
bx.set_title('两个例子：一个转好，一个仍变差\n样本数均为32；正数表示恶化',loc='left',fontsize=11,pad=12)
bx.legend(loc='upper center',ncol=2,fontsize=9,frameon=False)
for a in (ax,bx):a.spines[['top','right']].set_visible(False);a.grid(axis='x' if a is ax else 'y',alpha=.12);a.set_axisbelow(True)
fig.savefig(out/'target_intervention.png',dpi=180);fig.savefig(out/'target_intervention.svg');plt.close(fig)
checks=v['conditions'];direct=v['direct_gd_checks']
text=f'''# D2 第四轮：如果模型能表达正确答案，还会越学越差吗？

## 结论：大多数回升消失，但留下了5个反例

前三轮已经确认：没有标签噪声，训练误差下降时，总体误差也可能上升。这一轮进一步把正确答案换成同一模型**确实能够表示**的版本，其他训练条件和两个评价时刻保持原值。

**原来50个总体误差回升的条件中，45个转为改善，5个仍然变差，而且训练误差继续下降。** 这说明：移除目标中不能表示的部分，对这批条件的表现有很大影响；但“表达不了目标”并不是回升必不可少的条件。我们找到了5个反例，没有因此确定唯一原因或普遍规律。

这里是50个已有开发条件，来自9个训练/特征抽样、3个主种子，不是50种独立任务。另一个原本改善的条件仍改善，3个窗口边界仍不评价；完整总账始终是54个。

## 这次具体改了什么

可以把固定特征想成模型手里的一套积木。原来的答案可能需要它没有的形状。这次先找出：用现有积木能拼出的、总体上最接近原答案的版本。再把训练标签改成这个版本。

这一步叫“总体投影”。它保证存在一组输出权重能完全表示新目标，但不保证模型靠有限训练样本就能找到这组权重。

| 项目 | 本轮设置 |
|---|---|
| 改变 | 训练标签：原目标y → 可表示目标g |
| 固定 | 输入样本、固定特征、零初始化、学习率0.3、全批量MSE、旧的两个时间点 |
| 标签噪声 | 始终为0 |
| 新停止点 | 没有重新搜索 |
| 主比较范围 | 第三轮确认回升的50个条件 |
| 判据 | 至少找到1个充分核验的例子：训练MSE下降，同时总体MSE上升超过预设小门槛 |

投影没有重新放大或缩小。我们另外保存了目标的均方大小，避免把幅度变化隐藏起来。这是借助已知总体信息做的诊断实验，不是现实任务中只看训练数据就能直接使用的方法。

## 结果成立到什么程度

| 观察 | 数量 | 解释 |
|---|---:|---|
| 回升转为改善 | 45/50 | 在原来冻结的两个时刻，新目标下继续训练有益 |
| 仍回升，训练误差仍下降 | 5/50 | 可表示性和无标签噪声都不能保证总体误差一直下降 |
| 回升幅度比原来更大 | 3/50 | 三个都在上述5个之内，不能再加到总数上 |
| 原本改善的辅助条件 | 1/1仍改善 | 未拿来补主预测的成功数 |
| 窗口边界 | 3/54 | 保留，不评价窗口外 |

预注册只预测“至少存在1个”，结果找到5个，支持这一存在性预测。45个转好不等于“90%的泛化恶化已被解释”；两个目标的训练轨迹发生了变化，这里没有计算因果贡献比例。

![可表示目标干预的结果](target_intervention.png)

## 同一个时间区间，两个具体例子

| 条件 | 时间 | 原目标的总体MSE变化 | 换目标后的变化 | 换目标后的训练MSE变化 |
|---|---|---:|---:|---:|
| n={flipped['n']}，seed={flipped['seed']}，α={flipped['alpha']:g} | {flipped['t_star']}→{flipped['t_late']} | {flipped['original_delta']:+.5f} | {flipped['delta']:+.5f} | {flipped['train_delta']:+.6f} |
| n={remaining['n']}，seed={remaining['seed']}，α={remaining['alpha']:g} | {remaining['t_star']}→{remaining['t_late']} | {remaining['original_delta']:+.5f} | {remaining['delta']:+.5f} | {remaining['train_delta']:+.6f} |

第一例由变差转成明显改善。第二例回升小了很多，但没有消失：总体MSE仍从{remaining['early_mse']:.5f}升到{remaining['late_mse']:.5f}，训练MSE却从{remaining['train_early']:.6f}降到{remaining['train_late']:.6f}。

## 为什么这不只是换了评分标准

g是原目标y的总体投影。对同一套特征能表示的任意模型f，都有：

**对原答案的MSE = 对投影答案的MSE + 一个与训练时间无关的常数。**

所以，对同一条训练轨迹，用原答案或投影答案评价，“后期减早期”的风险变化相同。本轮真正改变的是训练标签，进而改变训练过程；并非只删除风险公式中的某些项。

更具体地，Q表示总体特征矩、h表示特征与原答案的联系；投影权重b*满足Qb*=h。新目标为g=φᵀb*，总体误差直接计算为(β−b*)ᵀQ(β−b*)。训练集有自己的矩H=ΦᵀΦ/n，H与Q通常不同。因此，训练题上的改善不自动等于总体改善。这是解析上的可能性，本轮尚未逐项归因剩余5例。

## 留下的5个例子全部列出

α沿用原目标的混合强度编号；投影后的目标是原目标的可表示近似，不必仍是纯线性/交互多项式。

| n | 种子 | α | 时间 | 原总体Δ | 新总体Δ | 新训练Δ | 回升幅度 |
|---:|---:|---:|---|---:|---:|---:|---|
'''
for r in witness:
    text+=f"| {r['n']} | {r['seed']} | {r['alpha']:g} | {r['t_star']}→{r['t_late']} | {r['original_delta']:+.8f} | {r['delta']:+.8f} | {r['train_delta']:+.8f} | {'增大' if r['comparison']=='increased' else '减小'} |\n"
text+='''
其中4个来自原α=0目标，1个来自α=2。这个分组是看到结果后的描述，不是预注册的任务类型规律。所有改善/恶化只指原来的两个时间点；不能说45条以后再也不会回升，或5条整个区间都在变差。

## 下一步：解释剩余5个例子

模型能表达答案，不代表有限训练样本能辨认完整答案：两套不同的答案，在训练样本上可能完全一样，在新输入上却不同。

下一轮候选把目标拆成“训练样本能约束的部分”和“样本上看不出来的部分”，检查剩余5例：继续学习前一部分时，是否破坏了两部分在样本外的误差抵消？如果能约束的部分自身也变差，就需要保留另一种解释。

先核对两种目标在训练标签和完整GD轨迹上的一致性，不能把小特征值随便当作零空间。这个对照改变了总体答案，属于辨识与风险归因，不是修复原任务；直接报有符号变化，不编造贡献百分比。下一轮尚未预注册或执行。

## 来源、核验与限制

'''
text+=f'''- 预注册commit：`{execution['commit']}`，早于实验；计算约{execution['wall_seconds']:.1f}秒，单线程CPU。
- models/：51个条件的可表示目标权重、实际训练标签、两时刻模型及未截断GD对照；teacher_bases_60decimal.json：3个特征空间的高精度投影基。
- rows.json：全部54条件、符号、门槛、原/新风险及训练变化；summary.json保留50主分母、1辅助条件、3边界。
- 51对模型均核对未截断的完整GD矩阵幂，风险差异最大{max(r['full_gram_risk_error'] for r in checks):.2g}；不依赖把小特征值截掉来制造结果。
- 9个α=.5条件在0、1、32、256、2000步直接GD，共45次核对；风险最大差{max(r['risk_error'] for r in direct):.2g}。全部51对总体风险另用60位重算，最大差{max(r['high_precision_error'] for r in checks):.2g}。
- 投影正交残差最大{max(r['orthogonality_residual'] for r in checks):.2g}；R_y=R_g+常数的数值误差最大{max(r['risk_identity_error'] for r in checks):.2g}。没有数值未决或悄悄删除的条件。
- 数学推导见本仓库research/d2_continuous_20261103/r004_representable_target/formulation.md；高斯矩来源沿用第三轮。初始产物由manifest.json固定，后补报告/图/审计另由仓库result_manifest.json固定。
- 交接检查仍因当前worktree的ArchitectureIQ子模块未初始化，在commit检查处失败；该子模块未操作。D2原始输入均另经hash核对；仓库测试与交接状态见result_manifest.json。

没有付费模型/API、GPU、solver或网站发布。结论限定于已知高斯输入下的固定特征开发模型，不覆盖学习特征或真实任务。
'''
if (out/'independent_audit.json').exists():
    assert json.loads((out/'independent_audit.json').read_text())['status']=='passed'
    text+='\n独立审计已通过，详见independent_audit.json。\n'
(out/'report.md').write_text(text);print(out/'report.md')
