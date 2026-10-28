#!/usr/bin/env python3
"""Explain a completed sparse-cross-term round without changing its results."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RANDOM_METHODS = [f'random05_{i:02d}' for i in range(10)]
ORDER = ['full', 'diagonal', 'change05', 'magnitude05'] + RANDOM_METHODS
LABELS = {
    'full': '完整参照',
    'diagonal': '交叉项全删',
    'change05': '按变化评分留5%',
    'magnitude05': '按幅度留5%',
    **{name: f'随机{i + 1:02d}' for i, name in enumerate(RANDOM_METHODS)},
}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(value):
    """Keep arithmetic readable; unrounded values stay in focus_case.json."""
    return '0' if abs(value) < 1e-12 else f'{value:.3f}'


def condition_name(row):
    protocols = {'d1_biased': '一维有偏置', 'd1_homogeneous': '一维无偏置',
                 'd4_biased': '四维有偏置', 'd4_homogeneous': '四维无偏置'}
    tasks = {'linear': '线性', 'quadratic': '二次', 'sine': '正弦',
             'additive': '加性二次', 'interaction': '线性与交互',
             'product': '纯交互', 'vector': '双输出'}
    return protocols[row['protocol']] + '、' + tasks[row['task']]


def rate(numerator, denominator):
    return f'{numerator}/{denominator}（{numerator / denominator:.1%}）'


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('/Library/Fonts/Arial Unicode.ttf')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        'axes.unicode_minus': False, 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white',
    })
    return plt


def focus_plot(plt, out, arrays, focus_rows):
    t = arrays['t']
    random_values = np.stack([arrays[m] for m in RANDOM_METHODS])
    low, high = random_values.min(axis=0), random_values.max(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    colors = {'full': '#216b8b', 'diagonal': '#777777',
              'change05': '#b44a38', 'magnitude05': '#2c956a'}
    for index, ax in enumerate(axes):
        visible = t <= 512 if index == 0 else np.ones(len(t), dtype=bool)
        ax.fill_between(t[visible], low[visible], high[visible],
                        color='#d5a34c', alpha=.25, label='随机10次估计范围')
        for method in ('full', 'diagonal', 'change05', 'magnitude05'):
            ax.plot(t[visible], arrays[method][visible], color=colors[method],
                    ls='--' if method == 'diagonal' else '-',
                    lw=2.1 if method == 'full' else 1.4,
                    label='实际期望风险' if method == 'full' else f'{LABELS[method]}：估计')
        for method, marker in [('change05', 'o'), ('magnitude05', 's')]:
            row = focus_rows[method]
            if index == 1 or row['t_selected'] <= 512:
                ax.scatter([row['t_selected']], [row['full_risk_at_selected']],
                           color=colors[method], marker=marker, s=48, edgecolors='white',
                           zorder=5, label=f'{LABELS[method]}：实际停点')
        for method, style in [('full', ':'), ('diagonal', '--')]:
            selected = focus_rows[method]['t_selected']
            if index == 1 or selected <= 512:
                ax.axvline(selected, color=colors[method], ls=style, lw=.9, alpha=.7)
        ax.axhline(0, color='#999', lw=.7)
        ax.set(xlabel='训练步数', ylabel='误差（MSE）')
        if index == 0:
            ax.set(xlim=(0, 512), title='放大前512步：1步与255步之间发生了什么')
        else:
            ax.set_xscale('symlog', linthresh=1)
            ax.set(xlim=(0, int(t.max())), xlabel='训练步数（对数刻度；保留0）',
                   title='整个窗口：后期近似是否仍然合理')
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle('同一个已知失败：补回5%的交叉项', fontsize=17)
    fig.text(.5, .015, '蓝线是实际期望风险；其他线是模型估计。圆点/方点落在实际风险上。随机阴影是10次范围，不是置信区间。',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(out / 'figs/focus_cross_terms.png', dpi=180)
    plt.close(fig)


def overall_plot(plt, out, summary):
    methods = summary['methods']
    labels = [LABELS[m] for m in ORDER[:4]] + ['随机5%：10次中位与范围']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, key, title in zip(axes, ['time_rate', 'risk_rate'],
                              ['停止时间在两倍以内', '实际额外风险不超过初值1%']):
        values = [100 * methods[m]['overall'][key] for m in ORDER[:4]]
        ax.barh(np.arange(4), values, color=['#216b8b', '#999999', '#b44a38', '#2c956a'])
        random = np.array([100 * methods[m]['overall'][key] for m in RANDOM_METHODS])
        median = float(np.median(random))
        ax.errorbar(median, 4, xerr=[[median - random.min()], [random.max() - median]],
                    fmt='D', color='#a77a28', capsize=5, markersize=6)
        for i, value in enumerate(values):
            ax.text(min(value + 1, 101), i, f'{value:.1f}%', va='center', fontsize=9)
        ax.set(xlim=(0, 110), xlabel='通过比例（%）', title=title)
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(np.arange(5), labels)
    axes[0].invert_yaxis()
    fig.suptitle('全部条件的停止质量', fontsize=17)
    fig.text(.5, .015, '左分母3137，右分母3240。图是总体概览；正式判定仍要求12种条件逐个通过。随机10次复用同一批数据。',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(out / 'figs/overall_quality.png', dpi=180)
    plt.close(fig)


def arithmetic_table(focus, step):
    lines = [f'**第{step}步：**', '',
             '| 方法 | 自身项 + 保留交叉项 + 目标匹配项 + 起点项 + 噪声项 | 模型估计 | 实际风险 |',
             '|---|---|---:|---:|']
    for method in ORDER[:4]:
        part = next(p for p in focus['breakdown'][method] if p['t'] == step)
        values = [part[k] for k in ['self_terms', 'retained_cross', 'target_match',
                                    'target_squared_mean', 'noise']]
        expression = ' + '.join(f'({short(v)})' if v < 0 else short(v) for v in values)
        lines.append(f'| {LABELS[method]} | {expression} | {short(part["estimated_risk"])} | {short(part["full_risk"])} |')
    return '\n'.join(lines)


def main():
    spec = read_json(HERE / 'preregistration.json')
    data_root = Path(os.environ.get('AIQ_KB_DATA_ROOT', spec['data_root'])).resolve()
    out = data_root / spec['run_relative']
    execution = read_json(out / 'execution.json')
    if execution['status'] != 'complete':
        raise RuntimeError('Refusing to report an incomplete run as completed')
    summary = read_json(out / 'summary.json')
    focus = read_json(out / 'focus_case.json')
    inputs = read_json(out / 'input_verification.json')
    verification = read_json(out / 'numerical_verification.json')
    if inputs['status'] != 'verified' or verification['status'] != 'verified':
        raise RuntimeError('Completed report requires successful input and numerical gates')
    if digest(out / 'rows.jsonl') != summary['rows_sha256']:
        raise RuntimeError('Result rows changed')
    rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()]
    if len(rows) != summary['counts']['total_rows_including_references']:
        raise RuntimeError('Result denominator mismatch')
    focus_rows = {r['method']: r for r in focus['rows']}
    if set(focus_rows) != set(ORDER):
        raise RuntimeError('Focus case is missing methods')
    for method in ORDER:
        for part in focus['breakdown'][method]:
            reconstructed = sum(part[k] for k in ['self_terms', 'retained_cross', 'target_match',
                                                  'target_squared_mean', 'noise'])
            if abs(reconstructed - part['estimated_risk']) > 1e-8 * max(1., abs(reconstructed)):
                raise RuntimeError('Focus arithmetic does not match saved estimate')
    with np.load(out / 'focus_curves.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    (out / 'figs').mkdir(exist_ok=True)
    plt = setup_plotting()
    focus_plot(plt, out, arrays, focus_rows)
    overall_plot(plt, out, summary)

    methods = summary['methods']
    primary = methods['change05']
    pooled = primary['overall']
    old = methods['diagonal']
    selected = focus_rows['change05']
    complete = focus_rows['full']
    old_selected = focus_rows['diagonal']
    repaired = focus['repaired']['change05']
    case_verdict = '这个已知失败修好了' if repaired else '这个已知失败还没有同时满足两项修复标准'
    global_verdict = '通过本轮预定要求' if primary['recipes_passing_both'] == 12 else '未通过本轮预定要求'
    primary_rows = [r for r in rows if r['method'] == 'change05']
    worst = max(primary_rows, key=lambda r: r['regret_over_initial'])
    low_risk_only = selected['risk_pass'] and not selected['time_pass']
    nuance = ('它控制住了实际损失，但停止时间仍偏得太远，属于“效果改善、时间未修好”。'
              if low_risk_only else
              '判断修复要同时满足时间误差和实际损失要求，不能只挑其中一项。')
    focus_table = ['| 方法 | 建议停止步数 | 实际额外风险 / 初值 | 修复这个例子 |',
                   '|---|---:|---:|---|']
    global_table = ['| 方法 | 两项都过关的条件 | 时间准确 | 额外风险小 | 负信号 / 负总风险 |',
                    '|---|---:|---:|---:|---:|']
    for method in ORDER:
        r = focus_rows[method]
        focus_table.append(f'| {LABELS[method]} | {r["t_selected"]} | {r["regret_over_initial"]:.2%} | {"是" if focus["repaired"][method] else "否"} |')
        result = methods[method]
        all_rows = result['overall']
        global_table.append(f'| {LABELS[method]} | {result["recipes_passing_both"]}/12 | '
                            f'{rate(all_rows["time_pass"], all_rows["interior"])} | '
                            f'{rate(all_rows["risk_pass"], all_rows["count"])} | '
                            f'{all_rows["invalid_bias"]} / {all_rows["invalid_total"]} |')
    random_time = [methods[m]['overall']['time_rate'] for m in RANDOM_METHODS]
    random_risk = [methods[m]['overall']['risk_rate'] for m in RANDOM_METHODS]
    seed_text = '；'.join(f'{r["seed"]}：时间{r["time_rate"]:.1%}、低损失{r["risk_rate"]:.1%}'
                         for r in primary['seeds'])
    missed = [r for r in primary['recipes'] if not r['stopping_pass']]
    failed_text = '、'.join(f'{r["protocol"]}/{r["task"]}' for r in missed) if missed else '没有条件未达两项门槛'
    failed_sentence = f'未满足两项门槛的条件：{failed_text}。' if missed else '12种条件都通过了停止质量门槛，但这不要求每一条预测都正确。'
    magnitude_focus = focus_rows['magnitude05']
    mag = methods['magnitude05']['overall']
    diagonal_at_255 = next(p for p in focus['breakdown']['diagonal'] if p['t'] == 255)
    full_at_255 = next(p for p in focus['breakdown']['full'] if p['t'] == 255)
    diagonal_at_1 = next(p for p in focus['breakdown']['diagonal'] if p['t'] == 1)
    audit_path = out / 'independent_audit.json'
    audit_note = ('独立代理审计已通过：重新核对4320组选中名单、45360行记录及728组汇总；'
                  '[查看审计记录](independent_audit.json)。' if audit_path.exists()
                  and read_json(audit_path).get('status') == 'passed' else '')
    report = f'''# D2：只补回5%的相互作用，能修好“太早停”吗？

## 结论

**{case_verdict}；放到全部条件上，主方案{global_verdict}。** 原来建议第{old_selected['t_selected']}步就停、完整模型最低点在第{complete['t_selected']}步的例子，现在建议第 **{selected['t_selected']}** 步停止，实际多付出的误差为初始误差的 **{selected['regret_over_initial']:.2%}**。{nuance}

全部3240个条件中，主方案有 **{primary['recipes_passing_both']}/12** 种任务与特征条件同时过关；此前把交叉项全删的基准是 **{old['recipes_passing_both']}/12**。本轮达到的是预注册的12/12门槛，并不意味着每一条都预测正确。这里补的是公式中的相互作用，没有重新训练网络。

**按幅度选也得到几乎一样的结果，本轮未证明“变化评分”更好。** 幅度法在重点例子中停在第{magnitude_focus['t_selected']}步，全体时间通过{mag['time_pass']}/3137、低额外风险通过{mag['risk_pass']}/3240，与主方案的差距很小。

![已知失败的曲线](figs/focus_cross_terms.png)

## 问题怎么定义

各学习方向的贡献相加时，还会相互抵消或增强，这些影响就是交叉项。本轮主方案先给每对方向打“随训练时间变化多明显”的分数，只保留得分最高的最多5%；对照按幅度选同样多的项，另做10次随机选择。这个例子共有{focus['pair_count']}对，保留{focus['retained_pairs']}对。5%向下取整，不给活动方向很少的条件额外名额。

“修好”有两个要求：建议时间在原最低点的一半到两倍之间，而且按建议停止时，实际额外风险不超过初始风险的1%。下表的实际风险始终由完整模型算，不能拿稀疏模型自己的低估当作进步。风险是对标签噪声取平均的验证误差；最低点仍是固定窗口内格点搜索与局部精化的估计。

{chr(10).join(focus_table)}

## 成立到什么程度

主方案的时间通过 **{rate(pooled['time_pass'], pooled['interior'])}**，低额外风险通过 **{rate(pooled['risk_pass'], pooled['count'])}**。但正式判定逐个任务与特征条件进行：时间通过率至少80%，低额外风险通过率至少90%，两项都要满足。

{chr(10).join(global_table)}

随机10次的时间通过率范围为{min(random_time):.1%}–{max(random_time):.1%}，低额外风险范围为{min(random_risk):.1%}–{max(random_risk):.1%}。它们复用同一批数据，衡量的是选项运气，不能当成10份新数据，也没有挑最好的一次作为结论。完整参照的好成绩来自它本身就是评价基准，不是新发现。

![全部条件的通过率](figs/overall_quality.png)

## 怎么做的：把第1步和第255步拆开看

每个估计由五项相加：学习方向各自的贡献、保留下来的相互作用、对准目标带来的变化、未训练时的目标平方均值、训练噪声代价。同一步的实际风险对所有方法都相同，变化的是各方法对它的估计。负号表示抵消作用，不能把这些项分别当成独立的误差。

{arithmetic_table(focus, 1)}

{arithmetic_table(focus, 255)}

这个例子的关键在第255步：完整交叉项约为 **{full_at_255['retained_cross']:.3f}**，带来很大的抵消作用。全删后，第255步被估成 **{diagonal_at_255['estimated_risk']:.3f}**，比第1步的 **{diagonal_at_1['estimated_risk']:.3f}** 还差，于是它建议提前停。补回少量有针对性的项后，第255步估计重新接近实际的 **{full_at_255['full_risk']:.3f}**，继续训练的好处才显现出来。

短数经过舍入，相加可能有末位差异；完整精度和遗漏交叉项的正负值保存在[逐项拆解](focus_case.json)。同一任务和数据组合的选中名单在九个噪声水平间共享，评分不读取原最低点，也不根据结果调整名单。

## 失败与代价

主方案仍有 **{pooled['invalid_bias']}/3240** 条出现负的无噪声误差，**{pooled['invalid_total']}/3240** 条出现负的总风险。负误差不是好效果，而是近似失效；这些条件没有被删掉。另有 **{pooled['harm_above_10pct']}/3240** 条因选错时间，额外风险超过初值的10%。最差一条是 **{condition_name(worst)}**、n={worst['n']}、噪声方差={worst['sigma2']:g}、seed={worst['seed']}：原最佳位置是第{worst['t_reference']}步，仍被提前选成第{worst['t_selected']}步；实际风险由最佳的{worst['risk_reference']:.3f}变成{worst['full_risk_at_selected']:.3f}，多出的风险为初值的 **{worst['regret_over_initial']:.2%}**。

{failed_sentence}全部103个边界最低点保留在风险评价中；时间倍数只对3137个内部最低点计算。{summary['zero_budget_task_draws']}/360个任务与数据组合因严格取整而保留0对，不能把它们从分母中拿掉。

## 还没解决什么

这是已经看过的数据和失败例子上的压缩检查，不是新任务上的盲测。评分仍用了干净目标及验证分布，而且先检查了全部交叉项；即使通过，也只说明保留下来的表示可以变小，不等于计算更快或得到只看训练数据的早停法。

主方案逐种子结果为：{seed_text}。只有三个独立主种子，样本量、任务和噪声水平间共享数据；3240条并非3240次独立重复。还有{summary['near_degenerate_draws']}/120个数据与特征组合含很接近的特征值，方向基的选择可能影响排名。本轮没有为此事后旋转方向或换规则。

## 证据

预注册与计算合同提交为 `{summary['contract']['commit']}`，早于计算。输入再次核对{inputs['d2_manifest_files']}个D2清单文件、{inputs['receipt_payloads']}个数组回执，以及{inputs['previous_artifacts']}个上一轮产物。全部3240条完整曲线重建的最大归一化差为{verification['maximum_normalized_reconstruction_error']:.2g}；{verification['mask_checks']}个掩码核对的最大差为{verification['maximum_normalized_mask_error']:.2g}；{verification['independent_checks']}次原特征坐标核对的最大绝对差为{verification['maximum_independent_absolute_error']:.2g}。计算约{summary['compute_seconds']:.0f}秒，无模型API或GPU调用。

{audit_note}

[预注册](preregistration.json) · [完整结果](summary.json) · [逐条件结果](rows.jsonl) · [重点例子](focus_case.json) · [选择记录](selection_manifest.json) · [数值核对](numerical_verification.json) · [输入校验](input_verification.json) · [执行记录](execution.json)
'''
    (out / 'report.md').write_text(report)
    print(json.dumps({'report': str(out / 'report.md'),
                      'figures': [str(out / 'figs/focus_cross_terms.png'), str(out / 'figs/overall_quality.png')],
                      'primary_status': summary['hypothesis_status'],
                      'focus_repaired': repaired}, ensure_ascii=False))


if __name__ == '__main__':
    main()
