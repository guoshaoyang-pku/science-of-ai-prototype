from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]
BASE = STUDY.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


checks = []
old_names = ['r028_gap_cascade', 'r032_modal_formula', 'r036_large_ratio_formula', 'r044_low_ratio_formula']
for name in old_names:
    receipt_path = BASE / name / 'executed/final_commit_verification.json'
    receipt = json.loads(receipt_path.read_text())
    commit = receipt['scientific_closeout_commit']
    for pin in receipt['verified_files']:
        blob = subprocess.check_output(['git', 'show', f'{commit}:{pin["path"]}'], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == pin['sha256'], pin['path']
    subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['preregistration_commit'], commit], cwd=ROOT, check=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    checks.append({'study': name, 'scientific_closeout_commit': commit,
                   'preregistration_commit': receipt['preregistration_commit'],
                   'verified_commit_file_hashes': len(receipt['verified_files']),
                   'receipt_sha256': sha(receipt_path), 'status': 'passed'})
old_audit = json.loads((BASE / 'r044_low_ratio_formula/executed/previous_closeout_audit.json').read_text())
for pin in old_audit['previous_study_file_manifest']:
    path = ROOT / pin['path']
    assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns'], pin['path']
old_roots = [BASE / name for name in old_names] + [
    ROOT / 'directions/D4_sigmoid_training_curve/studies' / name
    for name in ['r002_two_mode_logistic', 'r018_slow_energy_logistic']]
manifest = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
            for directory in old_roots for path in sorted(directory.rglob('*')) if path.is_file()]
contracts = []
for directory in old_roots:
    for pattern in ['metadata.json', 'evaluation.json']:
        for path in sorted((directory / 'results').rglob(pattern)):
            data = json.loads(path.read_text())
            assert data['status'] == 'success'
            assert sha(path.parent / 'arrays.npz') == data['arrays_sha256']
            contracts.append({'path': str(path.relative_to(ROOT)), 'status': 'passed'})
write(STUDY / 'executed/previous_closeout_audit.json', {
    'status': 'passed', 'created_at': datetime.now().astimezone().isoformat(),
    'closeout_checks': checks, 'previous_study_file_manifest': manifest,
    'saved_cell_contract_checks': contracts, 'old_229_hash_mtime_verified': True,
    'audit_scope': '仅hash/mtime/合同/commit blob核验；没有训练、拟合或计算旧公式误差。旧中心文件只与对应commit比对。',
    'pre_existing_worktree_changes': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)})
write(STUDY / 'executed/state_before.json', json.loads((ROOT / 'central/state.json').read_text()))
for relative in ['executed/run.py', 'analysis.py', 'executed/verify_saved.py']:
    (STUDY / relative).write_bytes((BASE / 'r036_large_ratio_formula' / relative).read_bytes())
analysis_path = STUDY / 'analysis.py'
source = analysis_path.read_text().replace(
    "    summary = {'study':",
    "    intervals = prereg['predictions'][1]['intervals']\n"
    "    interval_failures = [{'cell_id': r['cell_id'], 'metric': m, 'value': r[m],\n"
    "                          'interval': intervals[str(r['ratio'])][m]}\n"
    "                         for r in rows for m in ['rmse', 'maxabs']\n"
    "                         if not intervals[str(r['ratio'])][m][0] <= r[m] <= intervals[str(r['ratio'])][m][1]]\n"
    "    summary = {'study':")
source = source.replace(
    "'counterexamples': failures}},",
    "'counterexamples': failures},\n"
    "                               'P2': {'status': 'refuted' if interval_failures else 'supported',\n"
    "                                      'counterexamples': interval_failures}},")
analysis_path.write_text(source)
prereg = json.loads((BASE / 'r036_large_ratio_formula/preregistration.json').read_text())
prereg.update(study='r076_lower_ratio_formula', round=76, direction_round=5, created_date='2026-10-07',
              created_at=datetime.now().astimezone().isoformat(), ratios=[3, 10],
              question='固定逐模态h/b、原150点等权网格、平台1/0、幅度.5/.5及.03/.05阈值，D4保存r3/10能否通过，误差是否落入注册区间？')
prereg['cells'] = [{'cell_id': f'ratio_{ratio:03d}_seed_{seed}', 'ratio': ratio, 'seed': seed,
                   'arrays_path': f'directions/D4_sigmoid_training_curve/studies/r002_two_mode_logistic/results/ratio_{ratio:03d}_seed_{seed}/arrays.npz',
                   'metadata_path': f'directions/D4_sigmoid_training_curve/studies/r002_two_mode_logistic/results/ratio_{ratio:03d}_seed_{seed}/metadata.json'}
                  for ratio in [3, 10] for seed in [0, 1, 2]]
prereg['data_contract']['source'] = '仅读D4 r002的r3/10保存成功曲线；0新训练/拟合；QR坐标seed0/1/2只检查数值稳健性。'
prereg['predictions'] = [
    {'id': 'P1', 'kind': 'controlled_numerical_prediction',
     'prediction': 'r3/10各3/3坐标seed均RMSE<=.03且maxabs<=.05。',
     'rmse_upper_bound': .03, 'maxabs_upper_bound': .05, 'required_success_cells': 6,
     'criterion': '任一有效cell不同时满足两阈值即refuted；保存反例，不调h/b/幅度/网格，不训练或拟合。',
     'basis': '已有r12/15/20/25/30/100通过；旧r1局部单模态公式RMSE=.03334976043153364、maxabs=.06636634806294173失败，低谱比风险不能忽略。不同模态的近似误差可能部分错开，作为development风险预测；未预计算r3/10新公式误差，不称盲发现。'},
    {'id': 'P2', 'kind': 'controlled_numerical_interval_prediction',
     'prediction': '每个坐标seed，r3的RMSE在[.025,.030]、maxabs在[.045,.050]；r10的RMSE在[.025,.030]、maxabs在[.040,.050]。',
     'intervals': {'3': {'rmse': [.025, .030], 'maxabs': [.045, .050]},
                   '10': {'rmse': [.025, .030], 'maxabs': [.040, .050]}},
     'criterion': '闭区间；任一有效cell任一指标在对应区间外即refuted；保留原区间与反例。',
     'basis': '沿用已测r12/15的误差量级，将低谱比视作误差重叠风险；是未校准的宽区间数值预测，不是连续谱比公式。'}]
prereg['boundaries'] = [line.replace('r30/100', 'r3/10') for line in prereg['boundaries']]
prereg['boundaries'].append('旧r1不计算、不重判；旧纯快/alpha.01原最小二乘失败保留；六旧格点不重评，新格点不定位连续临界或独立归因于误差重叠。')
prereg['pre_evaluation_note'] = '无inbox；报告符合骨架。首批读操作并行，随后完整按规定顺序重读；一次错误相对目录查找只读失败，一次functions.exec JavaScript SyntaxError未启动shell或写文件。仅旧摘要/源码/合同/hash/mtime和收尾审计，未运行新公式误差、不计算或改判r1。'
paths = [ROOT / 'directions/D4_sigmoid_training_curve/studies/r002_two_mode_logistic' / relative
         for relative in ['preregistration.json', 'summary.json', 'analysis.py', 'executed/run.py']]
paths += [BASE / name / relative for name in old_names[1:]
          for relative in ['preregistration.json', 'summary.json', 'executed/run.py', 'executed/final_commit_verification.json']]
paths += [BASE / 'r028_gap_cascade/executed/final_commit_verification.json', STUDY / 'executed/previous_closeout_audit.json']
paths += [ROOT / cell[key] for cell in prereg['cells'] for key in ['arrays_path', 'metadata_path']]
prereg['input_manifest'] = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in paths]
prereg['source_hashes'] = {str((STUDY / relative).relative_to(ROOT)): sha(STUDY / relative)
                         for relative in ['executed/run.py', 'analysis.py', 'executed/verify_saved.py', 'executed/prepare.py']}
write(STUDY / 'preregistration.json', prereg)
print(json.dumps({'historical_files': len(manifest), 'saved_contracts': len(contracts),
                  'closeout_pins': [item['verified_commit_file_hashes'] for item in checks],
                  'input_pins': len(paths), 'source_pins': len(prereg['source_hashes']),
                  'scientific_evaluation_performed': False}, indent=2))
