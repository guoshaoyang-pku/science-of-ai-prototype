"""用保存summary更新发现、方向报告与中心交接，并验证状态边界。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
DIRECTION = STUDY.parent.parent


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    verification = json.loads((STUDY / 'executed/independent_verification.json').read_text())
    config = json.loads((STUDY / 'preregistration.json').read_text())
    assert summary['predictions'][0]['status'] == 'refuted'
    assert summary['new']['interior'] and not summary['new']['operational_u_shape']
    assert verification['status'] == 'verified'
    commit = summary['preregistration_commit']
    findings = '''# r095：seed413 的回升幅度低于事前区间

本轮仅检验预指定 n32、seed413 的 population 条件：复用已保存 signal_bias 与 variance_unit，把 σ²=.125 降至 .0625，终点风险减最低风险是否仍至少 .05？完成 **1/1 新评价 cell、0 新训练**。差值为 .014188114079254938，低于事前区间 [.015,.045] 的下界 .000811885920745061。P1 refuted，不能因 .05 判据失败而把整个区间预测记为支持。最低点仍在内部，风险回升仍为正。全部 development。

## 先读、报告重排与旧证据

按序读取 GOAL、AGENTS、state、task、全部 findings/report，最后读取 inbox。优先执行 inbox 六节顺序，保留全文、全部数字与结论，把失败与反例归入方法与条件；重排独立提交 fcce095。已追加指定处理时间标记。

先核验上一轮科学收尾 c8a9e3a：12 个 study/findings 对象与当前文件匹配。旧 final_commit_verification.json 缺失，如实登记，不补造旧回执；真实科学提交存在。独立只读审查确认旧 4 份冻结文件、13 项来源 pin、281 项历史 hash/mtime 和 4 项成功结果/summary hash/mtime。旧风险误差、Decimal 与 peer review 一致。未运行失效 r039 草稿，未重跑或覆盖 r003 的 60、r019 的 8、r051 的 8、r083 的 1 个成功 cell。本轮另冻结 300 个历史 study 文件 hash/mtime。

## 预注册与预测来源

新合同与 4 份源码一次提交为 FREEZE，绑定该唯一 commit 后才组合新风险。18 项来源 hash 同时匹配文件与该提交。提交早于新评价 LEAD 秒；条件、区间、源码均未改。区间 [.015,.045] 来自旧 .125 差值 .058105292352594984、粗略减半约 .029 和已见 seed412 的 .0625 结果 .02215220201337556。seed413 由上一轮预指定。注册前未读取该 seed 的曲线数值以计算 .0625 整条风险、端点或 argmin。属于 development、非盲发现，不是封存 OOD。

固定 d4 对称 Gaussian、目标 (x₀+.5x₀x₁)/√1.25、冻结 ReLU width128/scale .03、原训练特征居中/RMS、含 bias 零 head、float64、全批量 GD η=.3/mom0/nodecay、audit2048、整数步 0..16384。只改变噪声方差；九项原数据数组固定。一 recipe 配对、一 seed；逐步网格不是独立重复。

## 保存配对结果

| σ² | 首个全局最低点（步） | 最低风险 | 第 16384 步风险 | 终点减最低风险 | .05 判据 |
|---:|---:|---:|---:|---:|---|
| .125（旧保存对照） | 230 | .15135295688426875 | .20945824923686374 | .058105292352594984 | 满足 |
| .0625（新评价） | 433 | .11506551092432435 | .1292536250035793 | .014188114079254938 | 不满足 |

最低点延后 1.882608695652174 倍。差值减少 .043917178273340046，最低风险减少 .0362874459599444，终点风险减少 .08020462423328445。新差值对 .05 的余量为 −.035811885920745065；对事前下界的偏差不是浮点误差。噪声项线性缩放，最低点重新选取；不假定差值精确减半。

## 数值核验与执行记录

保存曲线组合耗时 .025099708 秒，stderr 为空。分析仅从新 NPZ 复算。独立从原 data 重构 K/eigh，以 log1p/expm1 响应复算全步信号/单位方差/风险，最大误差分别为 9.769963e−15/8.726353e−14/1.437739e−14；差值误差 6.467049e−15，argmin 同为 433。60 位 Decimal 的差值为 .014188114079254930677276291817179298959672451019287109375，最大组合舍入误差 4.553649e−17，P1 同样 refuted。数值核验不算新科学 claim。

恢复为 0 新 cell、1 复用 cell、0 训练；三个成功文件和 summary 共四个文件的 hash/mtime 不变。300 个旧文件 hash/mtime 通过。准备阶段出现 inline 换行 SyntaxError、apply_patch JS 引号 SyntaxError，均未执行风险计算。首次 git show 暂时返回路径不存在；git tree 与 cwd 复核确认文件存在，原准备脚本重试通过，根因未定。原日志与记录保留，未改科学合同。

## 结论边界与下一问题

两条预指定 seed412/413 保存曲线均在 .0625 下未达到 .05 回升幅度，且均保留内部最低点和正回升。本轮 P1 区间失败与上一轮区间支持分别保留。两 seed 不能给跨 seed 概率或连续噪声阈值；只作各自旧新噪声配对，不把两 seed 作为独立 recipe。不外推 train-only 早停、谱/输入因果、其他 n/width、learned hidden、Adam、小批量、CE、无限训练或 sealed OOD。旧 40/44 严格折叠失败、57/60 粗略幂律支持、201/62=3.241935、324/87=3.724138、NumPy bool 及失效轮均保留。

下一小问题可继续零训练：只改为预指定 n32/seed411，旧 .125 差值 .1773813164971647；评价 .0625 是否仍达到 .05。先核验本轮科学收尾与最终回执，冻结新数值区间、来源 hash 与源码，一次提交后才组合。该未测新条件未先算端点或最低点。

## 证据

- [预注册](../studies/r095_low_noise_seed413/preregistration.json)、[提交绑定](../studies/r095_low_noise_seed413/executed/execution_binding.json)、[历史审计](../studies/r095_low_noise_seed413/executed/prior_closeout_audit.json)与[独立旧证据审查](../studies/r095_low_noise_seed413/executed/independent_prior_review.json)。
- [summary](../studies/r095_low_noise_seed413/summary.json)、[新结果](../studies/r095_low_noise_seed413/results/n32_seed413_var0.0625.npz)、[metadata](../studies/r095_low_noise_seed413/results/n32_seed413_var0.0625.json)与[receipt](../studies/r095_low_noise_seed413/results/receipt.json)。
- [独立数值复算](../studies/r095_low_noise_seed413/executed/independent_verification.json)、[执行记录](../studies/r095_low_noise_seed413/executed/evaluation_run.json)、[分析记录](../studies/r095_low_noise_seed413/executed/analysis_run.json)与[恢复核验](../studies/r095_low_noise_seed413/executed/resume_verification.json)。
'''.replace('FREEZE', commit).replace('LEAD', str(verification['commit_precedes_evaluation_seconds']))
    with (DIRECTION / 'findings/r095_low_noise_seed413.md').open('x') as stream:
        stream.write(findings)
    report_path = DIRECTION / 'report.md'
    before_report = report_path.read_text()
    report = before_report
    conclusion = 'n32、seed413 的同一 population 条件中，σ² 从 0.125 降为 0.0625，终点减最低风险从 0.058105292352594984 降为 0.014188114079254938。新差值低于事前区间 [0.015,0.045]，推翻了该区间预测；也低于 0.05。最低点从 230 延后至 433 步，仍在内部，风险回升仍为正。'
    report = report.replace('## Formulation', conclusion + '\n\n## Formulation', 1)
    report = report.replace('更低噪声条件只验证了一个配对、一个 seed。', 'n32、seed412 的更低噪声评价为一个配对、一个 seed。', 1)
    extent = '预指定 seed413 的新配对同样未达到 0.05，但区间预测失败。差值 0.014188114079254938 低于事前下界 0.015，偏差为 −0.000811885920745061；独立差值误差仅 6.467049e−15。最低点延后 1.882608695652174 倍。两个指定 seed412/413 的阈值失败只描述这两条曲线，不给跨 seed 概率；两者都有内部最优点和正回升。'
    report = report.replace('全部条件为 development', extent + '\n\n全部条件为 development', 1)
    method = '''seed413 评价复用其原九项数据数组与保存分解，不新增训练。只将 σ²=.125 改为 .0625，新增一个评价 cell。它由上一轮交接预指定；事前区间 [0.015,0.045] 来自旧 .058105292 差值、粗略减半判断和已见 seed412 结果。注册前未组合该 seed 的新风险、端点或最低点。

| seed413 的 σ² | 最低点（步） | 最低风险 | 第 16384 步风险 | 终点减最低风险 | 0.05 判据 |
|---:|---:|---:|---:|---:|---|
| 0.125 | 230 | 0.15135295688426875 | 0.20945824923686374 | 0.058105292352594984 | 满足 |
| 0.0625 | 433 | 0.11506551092432435 | 0.1292536250035793 | 0.014188114079254938 | 不满足 |

'''
    report = report.replace('### 失败与反例', method + '### 失败与反例', 1)
    failure = 'seed413 的差值区间 [0.015,0.045] 被推翻。实测 0.014188114079254938 低于下界；不能因同一预测中“低于 0.05”的方向符合结果，就把整个 P1 写成支持。原 .05 操作性 U 型判据同样失败，保留内部最低点与正回升。'
    report = report.replace('执行失败保留为执行失败', failure + '\n\n执行失败保留为执行失败', 1)
    old_question = '跨 seed 的低噪声幅度判据仍未验证。n32、seed412 的 0.0625 已不满足 0.05 判据；下一小问题可只换为预指定 n32、seed413，复用其保存 0.125 分解曲线评价 0.0625，检验是否同样未达阈值。先注册新数值区间、来源 hash（内容指纹）与源码，再作零新训练的曲线算术。未评价条件不先算新端点后称盲预测，不据两个噪声点推断连续临界值。'
    new_question = '两个指定 seed 的低噪声评价未给出跨 seed 概率。下一小问题可只换为预指定 n32、seed411；其旧 0.125 差值为 0.1773813164971647。复用保存分解评价 0.0625，检验是否仍达到 0.05。先注册新数值区间、来源 hash（内容指纹）与源码，再作零新训练的曲线算术。未评价条件不先算新端点后称盲预测，不据两个噪声点推断连续临界值。'
    assert old_question in report
    report = report.replace(old_question, new_question)
    evidence = f'''seed413 新评价耗时 0.025099708 秒。唯一预注册 {commit[:7]} 早于新评价 {verification['commit_precedes_evaluation_seconds']} 秒；18 项来源 hash、4 份源码及 300 个旧 study 文件 hash/mtime 固定。独立风险误差≤1.437739e−14，60 位 Decimal 与 argmin 一致。恢复为 0 新 cell，四个成功结果/summary 文件 hash/mtime 不变。旧低噪声科学收尾存在，但缺少最终提交回执，本次按真实提交重核并记录缺口。

- [seed413 预测失败 findings](findings/r095_low_noise_seed413.md)、[预注册](studies/r095_low_noise_seed413/preregistration.json)、[summary](studies/r095_low_noise_seed413/summary.json)、[独立复算](studies/r095_low_noise_seed413/executed/independent_verification.json)、[历史收尾审计](studies/r095_low_noise_seed413/executed/prior_closeout_audit.json)与[恢复核验](studies/r095_low_noise_seed413/executed/resume_verification.json)。

'''
    report = report.replace('## 证据\n\n', '## 证据\n\n' + evidence, 1)
    expected_headers = ['结论', 'Formulation', '成立程度', '方法与条件', '未决问题', '证据']
    assert re.findall(r'^## (.+)$', report, re.M) == expected_headers
    assert not re.search(r'\]\(', report.split('## 证据')[0])
    for heading, end in [('结论', 'Formulation'), ('成立程度', '方法与条件')]:
        content = report.split('## ' + heading + '\n')[1].split('## ' + end)[0]
        assert 'studies/' not in content and 'commit' not in content and 'r095' not in content
    report_path.write_text(report)
    state_path = ROOT / 'central/state.json'
    before_state = json.loads(state_path.read_text())
    initial_state = json.loads((STUDY / 'executed/state_before.json').read_text())
    assert before_state['round'] == initial_state['round'] == 95
    assert before_state['directions']['D2_overfitting_u_curve']['rounds_done'] == 6
    state = json.loads(json.dumps(before_state))
    d = state['directions']['D2_overfitting_u_curve']
    d['last_round'] = {'round': 95, 'direction': 'D2_overfitting_u_curve', 'result': 'ok',
                       'summary': '1/1保存曲线评价、0新训练；seed413 σ²=.0625的终点减最低风险=.014188114低于注册[.015,.045]，P1 refuted；.05判据失败，内部最优433步与正回升保留。'}
    d['next_question'] = ('r095完成1/1保存曲线评价、0新训练；先核验科学收尾与executed/final_commit_verification、independent_verification及独立审查。'
        '不重跑/覆盖r00360、r0198、r0518、r0831与r0951成功cell，不运行r039失效草稿。报告按inbox六节重排fcce095。'
        f'唯一冻结{commit[:7]}早于新评价{verification["commit_precedes_evaluation_seconds"]}秒，18pins与4源码固定；300旧文件及4新成功/summary hashmtime与恢复0新cell通过。'
        '同n32seed413/population/原九数组/eta.3mom0nodecay/T16384，sigma².125→.0625，t*230→433，比1.882608696；'
        '终点减最低风险.058105292352594984→.014188114079254938；P1[.015,.045]refuted，低于下界.000811885921，.05判据失败但内部最低点与正回升仍在。'
        '注册前未组合新risk/端点/argmin；预测区间来自旧差值粗略减半和已见seed412，development非盲发现。独立风险误差1.437739e-14/差值6.467049e-15与60位Decimal一致。'
        'r083真实科学收尾c8a9e3a匹配12对象，但final_commit_verification缺失，缺口记于本轮而不补造旧回执。'
        '旧40/44失败、57/60支持、201/62=3.241935、324/87=3.724138、NumPy bool/失效r039/matmul未定保留。'
        '下一小问题优先0新训练：仅换预指定n32seed411，复用其保存signal_bias/variance_unit、sigma².125→.0625，检验终点减最低风险是否仍>=.05，旧.125差值.1773813164971647。'
        '先注册新具体数值区间、来源hash与pinned源码单一commit，再组合新risk，不预先算新端点后称盲预测。'
        '无连续噪声阈值/跨seed概率/train-only早停/谱或输入因果/其他算法或sealed OOD外推。')
    check = json.loads(json.dumps(state))
    for key in ['last_round', 'next_question']:
        check['directions']['D2_overfitting_u_curve'][key] = before_state['directions']['D2_overfitting_u_curve'][key]
    assert check == before_state
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + chr(10))
    kb_path = ROOT / 'central/kb.json'
    kb = json.loads(kb_path.read_text())
    old_claims = json.loads(json.dumps(kb['claims']))
    assert not any(row['id'] in ['D2-006', 'D2-007'] for row in old_claims)
    boundary = ('仅development：d4 antithetic Gaussian、目标(x0+.5x0x1)/sqrt(1.25)、frozen ReLU width128/scale.03、原train feature mean/RMS、含bias零head、float64全批量GD eta.3/mom0/nodecay、n32/seed413、audit2048、整数0..16384。'
        '一recipe配对一seed、复用保存分解、0新训练；条件预指定，区间来自旧差值粗略减半和已见seed412，非盲发现/OOD。'
        '原分解是已知理论；幅度判据失败不等于无内部最低点/回升。不报连续噪声阈值、跨seed概率、train-only早停、谱/输入因果；不外推其他n/width、learned hidden、Adam、小批量、CE或无限训练。')
    shared = {'direction': 'D2', 'round': 95, 'evidence': [str((STUDY / 'summary.json').relative_to(ROOT)),
                    str((STUDY / 'executed/independent_verification.json').relative_to(ROOT))], 'domain': 'development', 'boundary': boundary}
    kb['claims'] += [
        {**shared, 'id': 'D2-006', 'status': 'refuted', 'text': 'd4 Gaussian混合目标、冻结ReLU width128/scale.03、population标签、含bias零head、float64全批量GD eta.3/mom0/nodecay、n32/seed413/0..16384下，sigma²=.0625终点减最低风险的预注册区间[.015,.045]被推翻：实测.014188114079254938，低于下界.000811885920745061。'},
        {**shared, 'id': 'D2-007', 'status': 'measured', 'text': 'd4 Gaussian混合目标、冻结ReLU width128/scale.03、population标签、含bias零head、float64全批量GD eta.3/mom0/nodecay、n32/seed413/0..16384的保存曲线配对中，仅sigma².125→.0625使终点减最低风险.058105292352594984→.014188114079254938<.05；原操作性U判据失败，内部最低点230→433步且正回升保留。'},
    ]
    assert kb['claims'][:-2] == old_claims
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + chr(10))
    with (ROOT / 'reports/PROGRESS.md').open('a') as stream:
        stream.write('\n2026-10-07 · r095 · D2：预指定 n32/seed413 的保存曲线在 σ²=.125→.0625 后是否仍达到 .05 回升幅度？1/1评价、0训练；差值 .014188114079254938，P1 [.015,.045] 被推翻，.05 判据失败但内部最优433步与正回升保留。证据：[findings](../directions/D2_overfitting_u_curve/findings/r095_low_noise_seed413.md)、[summary](../directions/D2_overfitting_u_curve/studies/r095_low_noise_seed413/summary.json)。\n')
    audit = json.loads((STUDY / 'executed/prior_closeout_audit.json').read_text())
    for relative, record in audit['historical_study_files'].items():
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        assert path.stat().st_mtime_ns == record['mtime_ns']
    links = re.findall(r'\]\(([^)]+)\)', report)
    assert all((DIRECTION / relative).exists() for relative in links)
    save(STUDY / 'executed/round095_completion.json', {
        'status': 'verified', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'round': 95, 'direction_round': 7, 'preregistration_commit': commit,
        'saved_evaluation_cells': 1, 'new_training_cells': 0, 'prediction_status': 'refuted',
        'historical_files_hash_mtime_unchanged': len(audit['historical_study_files']),
        'new_success_and_summary_hash_mtime_unchanged': 4, 'resume_new_cells': 0,
        'round_unchanged': state['round'], 'rounds_done_unchanged': d['rounds_done'],
        'only_D2_last_round_next_question_changed': True, 'old_claims_unchanged': True,
        'added_claims': ['D2-006', 'D2-007'], 'report_sections': expected_headers,
        'report_style_checklist': '通过；inbox六节优先；失败反例保留在方法与条件；数字/原结论保留，新增有保存证据',
        'all_evidence_links_resolve': True, 'evidence_links_count': len(links),
        'prior_report_sha256': hashlib.sha256(before_report.encode()).hexdigest(),
        'report_sha256': hashlib.sha256(report.encode()).hexdigest(),
        'peer_review': '独立审查另保存，不以源码核验冒充统计证据',
    })
    print(json.dumps({'closeout_written': True, 'prediction': 'refuted', 'report_links': len(links),
                      'only_state_fields': ['D2.last_round', 'D2.next_question']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
