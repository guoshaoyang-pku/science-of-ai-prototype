from pathlib import Path
import copy
import hashlib
import json
import re
import subprocess

REPO = Path(__file__).resolve().parents[5]
DIRECTION = REPO/'directions/D7_train_gain_test_harm'
STUDY = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fmt(value):
    return f'{value:.6f}'


def ci(metric):
    return '['+', '.join(fmt(v) for v in metric['seed_95pct_t_interval'])+']'


def main():
    summary = json.loads((STUDY/'summary.json').read_text())
    verification = json.loads((STUDY/'executed/independent_verification.json').read_text())
    assert summary['prediction']['status'] == 'refuted' and summary['prediction']['passing_units'] == 1
    assert verification['status'] == 'passed'
    table = ['| 单元 | mean D(.5) [95% 初始化区间] | D>0 | 通过 | 新−旧 mean D [95% 配对初始化区间] |',
             '|---|---:|---:|---|---:|']
    for unit in summary['units']:
        metrics = unit['metrics']
        table.append(f"| {unit['function']}×{unit['width']} | {fmt(metrics['noise_amplification']['mean'])} {ci(metrics['noise_amplification'])} | {unit['positive_amplification_seeds']}/3 | {'是' if unit['pass'] else '否'} | {fmt(metrics['new_minus_previous_half_D']['mean'])} {ci(metrics['new_minus_previous_half_D'])} |")
    table = '\n'.join(table)
    seed_table = ['| 单元 | init | 新 D(.5) | 旧 D(.5) | 新−旧 D(.5) |', '|---|---:|---:|---:|---:|']
    for pair in summary['paired_seed_results']:
        seed_table.append(f"| {pair['function']}×{pair['width']} | {pair['seed']} | {fmt(pair['noise_amplification'])} | {fmt(pair['previous_half_noise_amplification'])} | {fmt(pair['new_minus_previous_half_D'])} |")
    seed_table = '\n'.join(seed_table)
    conclusion = '固定 data7339、σ=.5、含 bias 两层 hidden SiLU、width8/32、init11/29/47、CPU float64 全批量 SGD η=.05/mom0/nodecay/T512，仅把 noise733123 换为733131，原“至少3/4单元 mean D≥.01 且各至少2/3初始化 D>0”预测被推翻：通过率从3/4降为1/4。新四单元 mean D 依次为 .086687、−.008957、−.013877、.003331，正初始化数为3/3、2/3、2/3、2/3。噪声强度和该 recipe 固定时，已测判据仍依赖具体噪声向量。'
    boundary = '全部为 development；新噪声向量在预注册前未生成或查看，但条件与判据依据已见旧结果选择，不称盲 OOD。三初始化只描述初始化差异；两个半强度 noise draw 不能估计跨 draw 概率。结果不拆噪声拟合、核方向变化和隐式偏置的比例，不给 train-only 早停、连续噪声阈值、线性/二次缩放或其他 optimizer/LN/深度/样本量外推。'
    gains = '/'.join(fmt(u['metrics']['half_train_gain']['mean']) for u in summary['units'])
    late_new = [row for row in summary['late_units'] if row['draw'] == 'new']
    real_c = '/'.join(fmt(row['metrics']['real']['late_test_change']['mean']) for row in late_new)
    tangent_c = '/'.join(fmt(row['metrics']['tangent']['late_test_change']['mean']) for row in late_new)
    findings = f'''# r091：新噪声向量推翻半强度放大判据

{conclusion}

## 条件与唯一预测

程序第91轮、D7第6轮只检验一个小问题：固定 data7339 和 σ=.5，仅更换预指定 noise733131 后，半强度放大判据是否保持。d5 Gaussian train32/test256、mixed_sine/radial、train clean目标按均值与ddof0标准差归一化后加独立PCG64 Gaussian噪声，不center、不rescale、不再次归一化；test始终clean。默认float32 Linear初始化后double；两模型共享初始输出与全部weight/bias Jacobian，full-batch halfMSE、SGD eta=.05/mom0/nodecay/T512，检查点0/1/8/32/128/256/512。

Hσ为512步真实网络减初始切线的clean test halfMSE；同init D(.5)=H(.5)−H(0)。P1要求四个function×width单元至少三个 mean D≥.01且各至少两个init D>0。结果齐全后才判定；不调整阈值，不排除seed。三init不算三种独立recipe。

先按inbox六节要求重排累计报告，保留旧数字、表格、结论和反例，单独提交1ac6dbd；inbox已追加指定处理标记。d8e5c64同一提交冻结合同、run.py、analysis.py及verify_saved.py；唯一提交逐字匹配，早于首cell {verification['preregistration_before_first_cell_seconds']:.6f} 秒。旧r079科学收尾7b35a66、预注册652ea06与45科学文件核验通过，239旧文件hash/mtime冻结，72旧成功cell/144文件完整性通过。

12/12新cell、24条新轨迹完成，累计训练 {summary['training_seconds']:.6f} 秒；读取12个已保存sigma0和12个旧sigma=.5 cell，不重跑或改写旧cell。旧clean保留原noise_base733123；新旧的输入、clean目标、test目标、归一化、参数与完整Jacobian精确相同，只有新half噪声向量不同。首cell核验后才续跑11cell。

## 结果与配对区间

{table}

区间为95% Student-t初始化区间，df=2，只描述固定data/noise下三init差异。新四D区间都跨零；只有radial×32的新−旧配对区间排除零。mean D小于.01使mixed_sine×32、radial×8和radial×32不通过，后三个单元仍各有2/3初始化D>0。

{seed_table}

## train增益与晚期test变化

所有12新cell的train gain为正，四单元均值为 {gains}。真实网络128→512 test变化均值为 {real_c}，切线为 {tangent_c}；真实晚期上升初始化数依次为1/3、3/3、2/3、0/3。以上均为描述诊断，无新成功判据。

mixed_sine×32 init47的新 D=−.102542，真实加噪test风险变化=−.072304，切线变化=+.030239，同时noisy train gain=.394740。相对差距减少不要求终点真实优于切线：该cell H(.5)=.124723仍正。radial×32的三init H(.5)与真实晚期C均负；均值H=−.122981、C=−.148465。终点H、相对噪声敏感度D与晚期C分别记录，不能相互替代。

## 核验与边界

保存NPZ复算全部loss、D、跨draw差、区间与主判据通过。重建初始参数、train/test Jacobian、初始及终点真实输出误差0；新旧36cell合计切线train/test检查点最大误差 {max(summary['verification']['maximum_absolute_errors']['tangent_train_checkpoints'],summary['verification']['maximum_absolute_errors']['tangent_test_checkpoints']):.6e}。独立math.fsum端点风险与NumPy差最大 {verification['maximum_scalar_endpoint_vs_numpy_error']:.6e}；解析df2区间与SciPy差最大 {verification['maximum_analytic_df2_vs_scipy_interval_error']:.6e}，按事前1e−12均值/D、1e−10区间核验。239旧文件与24新成功文件hash/mtime不变，首cell未覆盖；恢复new0/reused12，所有新cell均绑定同一预注册commit。训练与分析stderr为空；旧matmul警告与根因未定记录保留。

{boundary}

下一小问题可只换预指定noise733139，在相同条件下检验本轮“不足3/4单元通过”是否再现；先注册相反方向的数值判据、来源hash和pinned源码单一commit后训练，不先生成噪声或看结果。新draw不承担机制解释。

## 证据

- [预注册](../studies/r091_noise_draw_half/preregistration.json)
- [逐cell与配对汇总](../studies/r091_noise_draw_half/summary.json)
- [完整保存证据核验](../studies/r091_noise_draw_half/executed/saved_evidence_verification.json)
- [独立主判据与恢复核验](../studies/r091_noise_draw_half/executed/independent_verification.json)
- [历史收尾审查](../studies/r091_noise_draw_half/executed/prior_closeout_audit.json)
'''
    (DIRECTION/'findings/r091_noise_draw_half.md').write_text(findings)
    report_path = DIRECTION/'report.md'
    report = report_path.read_text()
    report = report.replace('所有12个新noisy终点H1为正，', 'data7339/noise733123、sigma=1的12个终点H1为正，')
    report = report.replace('## Formulation', conclusion+'\n\n## Formulation', 1)
    report = report.replace('σ∈{0,1}', 'σ∈{0,.5,1}')
    report = report.replace('D_s^(ν,ρ) = H_1,s^(ν,ρ)−H_0,s^(ν,ρ)。', 'D_s^(ν,ρ)(σ) = H_σ,s^(ν,ρ)−H_0,s^(ν,ρ)；旧σ=1记作D，本轮σ=.5记作D(.5)。')
    report = report.replace('### data7339：主判据', '### data7339、sigma=.5：更换噪声向量\n\n'+table+'\n\n新四D区间均跨零；新−旧noise的radial×32配对区间排除零，其余三个跨零。该失败只约束已测noise733131，不能把1/4作为跨noise概率。\n\n### data7339：主判据', 1)
    methods = f'新noise733131实验新增12个sigma=.5 cell、24条轨迹，引用12旧clean与12旧half cell；训练 {summary["training_seconds"]:.6f} 秒。新train gain均值 {gains}，12/12为正；真实128→512 test变化均值 {real_c}，晚期上升init数1/3、3/3、2/3、0/3。旧239文件与新24成功文件hash/mtime不变，恢复0新训练；初始参数/J及真实输出重建误差0，全36cell切线检查点最大误差4.996004e−15，独立df2区间差6.822654e−12（事前容差1e−10）。\n\n'
    report = report.replace('### 失败与反例', methods+'### 失败与反例\n\n更换noise733131后，半强度P1由3/4降为1/4，预测被推翻。mixed_sine×32 init47 D=−.102542，真实加噪test风险变化−.072304而切线变化+.030239，noisy train gain=.394740；H(.5)=.124723仍正。radial×32 mean H(.5)=−.122981、真实C=−.148465且三init均负。较低带噪train loss、相对噪声敏感度和晚期test变化不能互相替代。\n\n', 1)
    start = report.index('## 未决问题')
    end = report.index('## 证据')
    report = report[:start]+'''## 未决问题

下一小问题可在development中只换预指定noise733139，固定同data7339/sigma=.5/四recipe/init/SGD/T512，检验本次“不足3/4单元通过”是否再现。先注册新的数值判据、来源hash及pinned源码单一commit，再生成噪声并训练；不把两draw推成跨draw概率。机制比例和train-only早停仍未建立，D7管损害机制边界、D2管U型曲线与早停。

'''+report[end:]
    report += '''
- [更换半强度noise draw的finding](findings/r091_noise_draw_half.md)
- [新draw预注册](studies/r091_noise_draw_half/preregistration.json)
- [新draw保存结果与配对区间](studies/r091_noise_draw_half/summary.json)
- [新draw独立核验与无覆盖恢复](studies/r091_noise_draw_half/executed/independent_verification.json)
- [旧收尾与报告重排审查](studies/r091_noise_draw_half/executed/prior_closeout_audit.json)；报告重排提交1ac6dbd，新draw科学预注册d8e5c64。
'''
    report_path.write_text(report)
    assert re.findall(r'^## (.*)$', report, re.M) == ['结论','Formulation','成立程度','方法与条件','未决问题','证据']
    assert '](studies/' not in report[:report.index('## 证据')]
    kb_path = REPO/'central/kb.json'
    kb = json.loads(kb_path.read_text())
    assert not any(c['id'] in ['D7-008','D7-009'] for c in kb['claims'])
    recipe = 'd5 Gaussian train32/test256、data7339、sigma=.5、mixed_sine/radial×width8/32、含bias两hidden SiLU、默认float32初始化后double、init11/29/47、CPU float64 full-batch SGD eta=.05/mom0/nodecay/T512'
    evidence = [str((STUDY/'summary.json').relative_to(REPO)),str((STUDY/'executed/independent_verification.json').relative_to(REPO))]
    kb['claims'].append({'id':'D7-008','direction':'D7','round':91,'text':recipe+'，仅noise733123→733131后，至少3/4单元mean D(.5)>=.01且各>=2/3 init D>0的预测被推翻，仅1/4通过；四mean D=.086687/−.008957/−.013877/.003331，正init数3/3、2/3、2/3、2/3。','evidence':evidence,'domain':'development','status':'refuted','boundary':boundary})
    kb['claims'].append({'id':'D7-009','direction':'D7','round':91,'text':recipe+'的radial×32，固定sigma=.5只换noise733123→733131，三init配对D变化均负，mean新−旧=−.057162、95% df2初始化区间[−.072850,−.041473]。','evidence':evidence,'domain':'development','status':'measured','boundary':'仅该固定data和两noise draw的配对测量；区间只含三init差异，无跨noise概率、幅度稳定性、机制占比、连续强度或其他recipe外推。'})
    save(kb_path, kb)
    state_path = REPO/'central/state.json'
    state = json.loads(state_path.read_text())
    before = copy.deepcopy(state)
    save(STUDY/'executed/state_before_handoff.json', before)
    direction = state['directions']['D7_train_gain_test_harm']
    direction['last_round'] = {'round':91,'direction':'D7_train_gain_test_harm','result':'ok','summary':'12/12新noise733131 sigma=.5 cell完成，半强度放大P1仅1/4通过而refuted；239旧/24新文件与独立、恢复核验通过。'}
    direction['next_question'] = 'r091完成12/12新cell/24轨迹，先核验科学收尾与executed/final_commit_verification及独立审查；不得重跑或覆盖全部72旧成功cell与本轮12新cell。报告按inbox六节重排1ac6dbd，唯一预注册d8e5c64早于首cell14.285938秒。固定data7339/sigma.5仅noise733123→733131，P1 refuted1/4；mixed8/32 radial8/32 meanD=.086687438792/−.008956801978/−.013877214669/.003330746377，正init3/3、2/3、2/3、2/3，四D CI均跨零；radial32新−旧meanD−.057161515 CI[−.072849746,−.041473284]排除零，其余跨零。新train gain12/12正、均值.420113665/.372169430/.337023065/.222037485；真实C均值−.004234709/.133417873/.031635355/−.148464868。独立math.fsum误差1.110223e−16、df2区间差6.822654e−12按事前1e−10核验，全36cell切线检查点误差4.996004e−15；239旧/24新hashmtime与恢复new0/reused12通过。下一小问题development保持全部同条件，仅换预指定noise733139，检验不足3/4单元meanD>=.01且各2/3init>0是否再现；先注册新的方向判据、pinned源码和输入hash单一commit，再生成噪声/训练，严禁先看结果。旧半强度radial8反例、sigma1三条件4/4与所有终点/晚期分歧及matmul根因未定均保留；不外推跨draw概率/机制比例/连续sigma阈值/缩放/train-only早停。'
    after_without = copy.deepcopy(state)
    for field in ['last_round','next_question']:
        after_without['directions']['D7_train_gain_test_harm'][field] = before['directions']['D7_train_gain_test_harm'][field]
    assert after_without == before
    save(state_path, state)
    save(STUDY/'executed/central_update_audit.json', {'status':'passed','only_D7_last_round_and_next_question_changed':True,'round_before':before['round'],'round_after':state['round'],'rounds_done_before':before['directions']['D7_train_gain_test_harm']['rounds_done'],'rounds_done_after':direction['rounds_done'],'KB_claim_ids_added':['D7-008','D7-009']})
    progress = REPO/'reports/PROGRESS.md'
    with progress.open('a') as f:
        f.write('\n2026-10-07 r091 D7（方向第6轮）：仅data7339/sigma=.5的noise733123→733131，半强度放大判据是否保持？预注册d8e5c64后完成12/12新cell与24轨迹，P1被推翻、仅1/4通过；四meanD=.086687/−.008957/−.013877/.003331，四CI跨零，radial32跨draw配对差CI排除零。239旧与24新成功文件hash/mtime、独立复算及恢复new0/reused12通过；限development两draw。证据：[finding](../directions/D7_train_gain_test_harm/findings/r091_noise_draw_half.md)、[summary](../directions/D7_train_gain_test_harm/studies/r091_noise_draw_half/summary.json)。\n')
    save(STUDY/'executed/preregistration_commit_attempt.json', {'status':'succeeded','preregistration_commit':summary['preregistration_commit'],'gate_before_first_cell_seconds':verification['preregistration_before_first_cell_seconds'],'report_reorder_commit':'1ac6dbd','note':'git add -A依流程纳入当时supervisor已存在的日志/repair产物；研究代码未修改supervisor或repair。'})
    print('findings/report/KB/state/progress written; only D7 last_round and next_question changed')


if __name__ == '__main__':
    main()
