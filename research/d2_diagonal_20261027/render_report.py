#!/usr/bin/env python3
"""Descriptive reporting of the completed round; does not change its decisions."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from run import OUT, NEW_INPUT, legacy, write_json
from numerics import base_grid, diagonal_bias


def title(row):
    tasks = {'linear':'线性','additive':'加性二次','interaction':'线性＋交互',
             'product':'纯交互','sine':'正弦','vector':'双输出','quadratic':'二次'}
    protocols = {'d4_homogeneous':'4维','d1_homogeneous':'1维无偏置',
                 'd1_biased':'1维有偏置','d4_biased':'4维有偏置'}
    return protocols[row['protocol']]+' · '+tasks[row['task']]


def ratio(a,b):
    return f'{a}/{b}（{a/b:.0%}）' if b else '无内部最低点'


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    execution = json.loads((OUT/'execution.json').read_text())
    if execution['status'] != 'complete':
        raise RuntimeError('Only completed rounds can be reported')
    rows = [json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines()]
    pin = hashlib.sha256((OUT/'rows.jsonl').read_bytes()).hexdigest()
    if pin != summary['rows_sha256']:
        raise RuntimeError('Result rows changed')
    fonts = Path('/Library/Fonts/Arial Unicode.ttf')
    if fonts.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(fonts))
        plt.rcParams['font.family'] = FontProperties(fname=fonts).get_name()
    plt.rcParams.update({'axes.unicode_minus':False, 'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    figs = OUT/'figs'
    figs.mkdir(exist_ok=True)
    recs=summary['recipes']
    yy=np.arange(len(recs))
    fig,ax=plt.subplots(1,2,figsize=(13,7),sharey=True)
    ax[0].barh(yy,[r['time_rate']*100 for r in recs],color='#287b96',label='删除交叉项')
    ax[1].barh(yy,[r['risk_rate']*100 for r in recs],color='#32946b',label='删除交叉项')
    for i,key,denom,threshold,heading in [(0,'baseline_time_pass','interior',80,'停止时间在两倍以内'),(1,'baseline_risk_pass','count',90,'多付出的误差不超过初值的1%')]:
        ax[i].scatter([100*r[key]/r[denom] for r in recs],yy,color='#a95d2a',marker='D',s=27,label='旧样本数/噪声公式',zorder=3)
        ax[i].axvline(threshold,color='#555',ls='--',lw=1,label=f'预定门槛 {threshold}%')
        ax[i].set(xlim=(0,105),xlabel='通过比例（%）',title=heading)
    ax[0].set_yticks(yy,[title(r) for r in recs]);ax[0].invert_yaxis()
    fig.suptitle('去掉交叉项，停止建议还能用吗？',fontsize=18)
    fig.text(.5,.045,'柱形：删除交叉项；橙色菱形：旧样本数/噪声公式；虚线：预定门槛（左80%，右90%）。',ha='center',fontsize=9)
    fig.text(.5,.015,'左：3137个内部最低点；右：全部3240条。停止效果通过，不代表近似误差一定非负。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(figs/'stopping_quality.png',dpi=180);plt.close(fig)

    selected=[]
    for recipe in recs:
        group=sorted([r for r in rows if (r['protocol'],r['task'])==(recipe['protocol'],recipe['task'])],key=lambda r:(r['regret_over_initial'],r['curve']))
        selected.extend([{'selection':'median_regret_within_recipe','row':group[len(group)//2]},
                         {'selection':'worst_regret_within_recipe','row':group[-1]}])
    examples=OUT/'examples';examples.mkdir(exist_ok=True)
    for i, item in enumerate(selected):
        r=item['row']
        with np.load(NEW_INPUT/r['data'],allow_pickle=False) as z: data={k:z[k] for k in z.files}
        model=legacy.spectral_model(data);terms=legacy.target_terms(model,data,r['task'])
        grid=np.unique(np.r_[base_grid(),r['t_diagonal'],r['t_reference'],r['t_baseline']])
        f,_=legacy.response(model,grid)
        b,n,_,_=legacy.expected_curves(model,terms,grid)
        bd=diagonal_bias(f,terms['h'],terms['linear'],terms['val_y2'])
        name=f'example_{i:02d}.npz'
        np.savez_compressed(examples/name,t=grid,full=b+r['sigma2']*n,
                            diagonal=bd+r['sigma2']*n,bias=b,diagonal_bias=bd,noise=r['sigma2']*n)
        item['arrays']=f'examples/{name}'
    write_json(examples/'index.json',selected)
    worst=max(selected,key=lambda s:s['row']['regret_over_initial'])
    representative=sorted(selected,key=lambda s:s['row']['regret_over_initial'])[len(selected)//2]
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    for j,item in enumerate([representative,worst]):
        r=item['row']
        with np.load(OUT/item['arrays']) as z:
            u=z['t']*.3; actual=z['full']/r['risk_initial'];approx=z['diagonal']/r['risk_initial']
        ax[j,0].plot(u,actual,color='#287b96',label='实际期望风险')
        ax[j,1].plot(u,actual,color='#287b96',label='完整模型')
        ax[j,1].plot(u,approx,color='#c56a37',label='删除交叉项')
        for a in ax[j]:
            a.set_xscale('symlog',linthresh=.3)
            a.axvline(r['t_reference']*.3,color='#32946b',ls='--',label='原最低点')
            a.axvline(r['t_diagonal']*.3,color='#c56a37',ls=':',label='近似建议')
            a.axhline(0,color='#aaa',lw=.7)
            a.set(xlabel='累计训练量 U（对数刻度；保留0）',ylabel='风险 / 初始风险')
        ax[j,0].set_title(f'{title(r)}，n={r["n"]}，噪声={r["sigma2"]:g}，seed={r["seed"]}\n停错的额外风险：初值的 {r["regret_over_initial"]:.2%}')
        ax[j,1].set_title('同一条轨迹：近似误差是否合理')
        ax[j,0].legend(fontsize=8);ax[j,1].legend(fontsize=8)
    fig.suptitle('停止质量与曲线可信度，是两件不同的事',fontsize=18)
    fig.text(.5,.005,'示例在结果揭示后按已登记规则选取，仅作解释；最低点按保存格点与局部精化估计。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(figs/'examples.png',dpi=180);plt.close(fig)
    allr=summary['overall'];inv=json.loads((OUT/'input_verification.json').read_text());ver=json.loads((OUT/'numerical_verification.json').read_text())
    worst_row=max(rows,key=lambda r:r['regret_over_initial'])
    descriptive = {
        'status':'post-result descriptive aggregation; primary decisions unchanged',
        'invalid_bias_but_low_regret':sum(r['invalid_bias'] and r['risk_pass'] for r in rows),
        'invalid_total_but_low_regret':sum(r['invalid_total'] and r['risk_pass'] for r in rows),
        'unique_task_draws_with_invalid_bias':len(set((r['data'],r['task']) for r in rows if r['invalid_bias'])),
        'unique_task_draws':len(set((r['data'],r['task']) for r in rows)),
        'baseline_only_low_regret':sum(r['baseline_risk_pass'] and not r['risk_pass'] for r in rows),
        'diagonal_only_low_regret':sum(r['risk_pass'] and not r['baseline_risk_pass'] for r in rows),
        'recipes_passing_per_seed':{str(s):sum(r['stopping_pass'] for r in summary['recipe_seeds'] if r['seed']==s) for s in (6101,6102,6103)},
        'worst_row':worst_row
    }
    write_json(OUT/'report_descriptive.json',descriptive)
    table=['| 条件 | 时间在两倍内 | 额外风险≤初值1% | 出现负信号误差 | 两项均过门槛 |','|---|---:|---:|---:|---|']
    for r in recs:
        table.append(f'| {title(r)} | {ratio(r["time_pass"],r["interior"])} | {ratio(r["risk_pass"],r["count"])} | {r["invalid_bias"]}/{r["count"]} | {"是" if r["stopping_pass"] else "否"} |')
    seed_lines='\n'.join(f'- seed {r["seed"]}：时间 {ratio(r["time_pass"],r["interior"])}；低额外风险 {ratio(r["risk_pass"],r["count"])}。' for r in summary['seeds'])
    conclusion=('在已有条件中，去掉交叉项仍能满足预定的停止质量要求。' if summary['recipes_passing_both']==12 else '交叉项不能在这些条件中一概省掉：删掉它们后，停止建议没有达到预定要求。')
    md=f'''# D2：把学习方向之间的相互作用删掉，会不会停错？

## 结论

**多数时候，删掉交叉项造成的停止损失不大；但少数条件会严重停错，所以不能把它们普遍省掉。** 额外风险的中位数只有初始风险的 **{allr['regret_over_initial']['median']:.2%}**，最差却达到 **{allr['regret_over_initial']['max']:.0%}**。12个任务与特征条件中，只有 **{summary['recipes_passing_both']}/12** 同时满足预定的两项门槛。

最直观的失败是：模型实际还可以继续学到第 **{worst_row['t_reference']}** 步，近似却建议第 **{worst_row['t_diagonal']}** 步就停。这样得到的误差约为 **{worst_row['full_risk_at_diagonal']:.2f}**，而原最低点约为 **{worst_row['risk_reference']:.2f}**。下面分别说明平均表现、这种严重失败，以及哪些结论还不能推广。

本轮结论只针对已经看过的合成固定特征回归条件，不是新任务上的独立验证。

把所有条件放在一起，近似给出的停止时间有 **{ratio(allr['time_pass'],allr['interior'])}** 落在原最低点时间的两倍以内；按它建议的位置停止，有 **{ratio(allr['risk_pass'],allr['count'])}** 的额外风险不超过初始风险的1%。两项衡量的是不同问题：时间猜得不准，未必会明显损害效果；反过来，曲线看着接近，也可能选错位置。

还有 **{allr['invalid_bias']}/{allr['count']}** 条近似出现负的“无训练噪声误差”，其中 **{allr['invalid_total']}/{allr['count']}** 条连加上噪声后的总风险也为负。误差不应为负；这些保留为模型失效。前一个计数对应 **{descriptive['unique_task_draws_with_invalid_bias']}/{descriptive['unique_task_draws']}** 个任务与数据组合，同一组合的不同噪声设置并非独立重复。

**算出负误差与停错，不是一回事。** 这{allr['invalid_bias']}条中仍有{descriptive['invalid_bias_but_low_regret']}条的停止损失不超过初值1%；反过来，最严重的停错例子，近似误差始终非负。因此不能只靠“曲线没有负数”判断停止建议是否可靠。

![各条件的停止质量](figs/stopping_quality.png)

## 问题怎么定义

固定隐藏特征后，可以把学习拆成许多方向：有的学得快，有的学得慢。不同方向对新样本的误差可能相互抵消，也可能相互增强，这种相互作用记在“交叉项”中。本轮保留每个方向自身的贡献，只删掉信号误差公式里的交叉项，再看它建议何时停止。

这里的风险，是对训练标签噪声取平均后的干净验证误差；“最低点”是原窗口内的格点搜索及局部精化估计，不保证是第一次回升，也不宣称搜索了所有整数极值。初始风险是尚未训练时的风险。额外风险等于“按近似建议停止时的真实风险”减去“原最低点风险”。

## 成立到什么程度

预注册要求：**每个条件**的内部最低点中，至少80%的时间预测落在两倍内；**每个条件**全部曲线中，至少90%的停止建议把额外风险控制在初值的1%以内。只看总体平均不能替代12个条件逐个过关。

{chr(10).join(table)}

旧的样本数/噪声公式，在本轮相同评价口径下，时间通过 **{ratio(allr['baseline_time_pass'],allr['interior'])}**，低额外风险通过 **{ratio(allr['baseline_risk_pass'],allr['count'])}**。这是全3240条的对照，不等同于旧报告只用2700条扩展条件的通过率。

对角近似总体比旧公式好，但并非处处更好：{descriptive['diagonal_only_low_regret']}条只有对角近似达到低损失门槛，另有{descriptive['baseline_only_low_regret']}条只有旧公式达到。这里是结果揭示后的配对描述，不是额外的独立验证。

## 怎么做的

使用原来的120个数据与特征组合、12种任务与特征条件、3240条轨迹；模型、噪声大小、学习率和训练窗口均沿用旧合同。没有重新训练、调用模型或拟合新系数。预测器仍知道干净目标和验证分布，因此不能直接当成只看训练数据的早停工具。

公式层面只把信号二次项中的矩阵 h 换成它的对角部分；线性项和噪声响应完全保留。没有把负误差裁成零。停止点用原整数格点加局部精化选择，再用完整模型计算实际会多损失多少。

这批数据只有3个独立主种子；样本量、任务和噪声设置之间共享数据，不能把3240条当成3240次独立重复：

{seed_lines}

分种子重新描述，分别只有{descriptive['recipes_passing_per_seed']['6101']}、{descriptive['recipes_passing_per_seed']['6102']}、{descriptive['recipes_passing_per_seed']['6103']}个条件达到两项门槛。因此，合并后的6/12不能理解成“六类任务在每次抽样中都稳定成功”。

## 失败与具体代价

按近似停止的额外风险，中位数为初值的 **{allr['regret_over_initial']['median']:.2%}**，90分位为 **{allr['regret_over_initial']['p90']:.2%}**。最差条件是 **{title(worst_row)}、n={worst_row['n']}、噪声方差={worst_row['sigma2']:g}、seed={worst_row['seed']}**：建议在第 **{worst_row['t_diagonal']}** 步停止，原最低点在第 **{worst_row['t_reference']}** 步，额外风险为初值的 **{worst_row['regret_over_initial']:.2%}**。

下图最坏例子的含义很直接：训练初期出现一个浅低谷，继续训练后还会出现更深的低谷。删掉交叉项后，近似模型把前面的浅低谷判成了全窗口最好的位置，于是过早叫停。这里不能把“首次回升”与“全窗口最佳停止点”混为一谈。

全部103个边界最低点都保留；时间倍数统计只对3137个内部最低点有定义，风险统计则包含全部3240条。原先位于内部、近似却选到边界，仍计入时间失败。

![实际停错的代价与近似曲线](figs/examples.png)

## 还不能据此断言什么

这一轮删的是计算公式中的项，没有改变实际训练过程，所以不能直接称为真实训练中的因果干预。它也不证明某一个交叉项是唯一原因。近似依赖所选训练特征基；{summary['near_degenerate_draws']}/120个组合出现相邻活动特征值间隔不超过最大特征值的1e-10，需注意近重复特征值下的基选择问题。

本轮仍是冻结特征、平方损失、全批量梯度下降和噪声平均风险。结论不自动推广到学习特征、分类、Adam或一次具体训练的早停。

后续最值得问的是：能否提前识别哪些方向之间的相互作用必须保留，让简化模型既省信息、又不会漏掉后面更好的低谷？本轮没有执行这项后续实验，也没有证明保留少数交叉项就足够。

## 证据与复核

- 预注册与计算代码冻结提交：`{summary['contract']['commit']}`，早于本轮计算。
- 输入：D2归档36个分片、总归档与逐文件hash核对；本轮再次校验 **{inv['files_checked']}** 个清单文件及 **{inv['receipt_payloads_checked']}** 个回执记录的数组文件。
- 完整模型在全部3240条公共格点上重建核对，最大归一化差 **{ver['maximum_normalized_saved_curve_error']:.2g}**；8个指定条件在原停止点与近似停止点用原特征坐标独立核对，最大绝对差 **{ver['maximum_independent_absolute_error']:.2g}**。
- 本轮计算耗时约 **{summary['compute_seconds']:.0f}秒**；没有模型API或GPU调用。
- [完整精度结果](summary.json)、[逐条件结果](rows.jsonl)、[预注册](preregistration.json)、[输入校验](input_verification.json)、[数值核对](numerical_verification.json)、[执行记录](execution.json)、[补充描述统计](report_descriptive.json)、[解释性示例](examples/index.json)。

图中示例是在全部结果揭示后按登记规则选取，只用于说明，不是额外验证。短数用于阅读，完整精度保留在JSON中。
'''
    (OUT/'report.md').write_text(md)
    print(json.dumps({'report':str(OUT/'report.md'),'figures':str(figs),'examples':len(selected)},ensure_ascii=False))


if __name__=='__main__':
    main()
