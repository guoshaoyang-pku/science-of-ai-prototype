import ast
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
        handle.write(chr(10))


def main():
    names = ['r028_gap_cascade', 'r032_modal_formula', 'r036_large_ratio_formula',
             'r044_low_ratio_formula', 'r076_lower_ratio_formula', 'r088_weighted_formula']
    checks = []
    for name in names:
        path = BASE / name / 'executed/final_commit_verification.json'
        receipt = json.loads(path.read_text())
        commit = receipt['scientific_closeout_commit']
        for pin in receipt['verified_files']:
            blob = subprocess.check_output(['git', 'show', f'{commit}:{pin["path"]}'], cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == pin['sha256'], pin['path']
        subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['preregistration_commit'], commit], cwd=ROOT, check=True)
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
        checks.append({'study': name, 'scientific_closeout_commit': commit,
                       'preregistration_commit': receipt['preregistration_commit'],
                       'verified_commit_file_hashes': len(receipt['verified_files']),
                       'receipt_sha256': sha(path), 'status': 'passed'})
    prior = BASE / 'r088_weighted_formula/executed'
    previous = json.loads((prior / 'previous_closeout_audit.json').read_text())['previous_study_file_manifest']
    recent = json.loads((prior / 'resume_verification.json').read_text())['result_hash_and_mtime_unchanged']
    for pin in previous + recent:
        path = ROOT / pin['path']
        assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns'], pin['path']
    old_roots = [BASE / name for name in names] + [
        ROOT / 'directions/D4_sigmoid_training_curve/studies' / name
        for name in ['r002_two_mode_logistic', 'r018_slow_energy_logistic']]
    manifest = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
                for directory in old_roots for path in sorted(directory.rglob('*')) if path.is_file()]
    contracts = []
    for directory in old_roots:
        for path in sorted((directory / 'results').rglob('*.json')):
            if path.name not in ['metadata.json', 'evaluation.json']:
                continue
            data = json.loads(path.read_text())
            assert data['status'] == 'success' and sha(path.parent / 'arrays.npz') == data['arrays_sha256']
            contracts.append({'path': str(path.relative_to(ROOT)), 'status': 'passed'})
    write(STUDY / 'executed/previous_closeout_audit.json', {
        'status': 'passed', 'created_at': datetime.now().astimezone().isoformat(),
        'closeout_checks': checks, 'previous_study_file_manifest': manifest,
        'saved_cell_contract_checks': contracts, 'old_283_hash_mtime_verified': len(previous) == 283,
        'r088_6_result_hash_mtime_verified': len(recent) == 6,
        'audit_scope': '仅hash/mtime/合同/提交blob核验；无训练、拟合、旧公式重评。旧中央文件只与对应commit比对。',
        'pre_existing_worktree_changes': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)})
    write(STUDY / 'executed/state_before.json', json.loads((ROOT / 'central/state.json').read_text()))
    old = ROOT / 'directions/D4_sigmoid_training_curve/studies/r018_slow_energy_logistic'
    old_summary = json.loads((old / 'summary.json').read_text())
    grid = old_summary['grid']
    assert grid == json.loads((BASE / 'r076_lower_ratio_formula/summary.json').read_text())['grid']
    assert len(grid) == 150 and grid[0] == 0 and grid[-1] == 4997
    cells = [{'cell_id': f'alpha_0750_seed_{seed}', 'ratio': 100, 'alpha': .75, 'seed': seed,
              'arrays_path': str((old / f'results/alpha_0750_seed_{seed}/arrays.npz').relative_to(ROOT)),
              'metadata_path': str((old / f'results/alpha_0750_seed_{seed}/metadata.json').relative_to(ROOT))}
             for seed in [0, 1, 2]]
    paths = [old / name for name in ['preregistration.json', 'summary.json', 'analysis.py', 'executed/run.py']]
    paths += [ROOT / cell[key] for cell in cells for key in ['arrays_path', 'metadata_path']]
    paths += [BASE / name / 'executed/final_commit_verification.json' for name in names]
    paths += [BASE / 'r088_weighted_formula' / name for name in ['summary.json', 'preregistration.json', 'executed/run.py']]
    paths += [STUDY / 'executed/previous_closeout_audit.json']
    sources = ['executed/run.py', 'analysis.py', 'executed/verify_saved.py', 'executed/prepare.py', 'executed/resume.py']
    for source in sources:
        ast.parse((STUDY / source).read_text())
    prereg = {
        'study': 'r106_weighted_slow_formula', 'round': 106, 'direction_round': 7, 'direction': 'D12_cascade_sigmoid',
        'created_at': datetime.now().astimezone().isoformat(), 'domain': 'development',
        'question': '仅改变保存的初始慢损失权重为alpha=.75，固定h/b的加权双段公式在r100是否同时通过原.03/.05阈值，误差是否落入新注册区间？',
        'data_contract': {'sample_count': 2, 'dimension': 2, 'dtype': 'float64',
            'source': '仅读D4 r018的r100 alpha=.75三条成功保存曲线；QR坐标seed0/1/2仅数值稳健性，非独立数据重复。',
            'loss': 'L=||Xtheta-y||²/(2n)，L0=1，theta0=0，模态初始慢/快损失.75/.25；只从NPZ mode_weights取幅度，不拟合权重。',
            'optimizer': '固定特征，全批量SGD eta=.1/momentum0/nodecay，lambda=[.01,1]，10000步。',
            'paired_control': '同seed保存D4 r018原最小二乘单logistic best_errors仅描述性对照；段数与参数取得方式同时改变，不独立归因。'},
        'learning_rate': .1, 'cells': cells,
        'formula': {'curve': 'C(t)=alpha H_s(t)+(1-alpha)H_f(t)，H_i=1/[1+(t/h_i)^b]，H_i(0)=1。',
                    'h': 'h_i=log(2)/[-2log(1-eta lambda_i)]，单位步。',
                    'b': '两模态b=2log(2)，无量纲，禁止校准。',
                    'alpha': 'NPZ mode_weights[0]，无量纲初始慢损失占比；不是参数、特征或梯度能量占比。',
                    'derivation': '谱递推与局部半值/对log(t)斜率匹配是已知解析工具，不登记为新发现。'},
        'evaluation': {'grid': grid, 'origin': '直接沿用D4 r018原150点等权网格0..4997；逐点与D12 r076相同。',
                       'platforms': [1, 0], 'amplitudes_slow_fast': [.75, .25],
                       'adequacy': {'rmse_maximum': .03, 'maxabs_maximum': .05, 'logic': '两项同时<=才通过'},
                       'aggregation': '1个recipe×3个坐标seed；保存每cell及指标min/max、最大残差步、同seed原LS对照差，不计算统计CI。'},
        'predictions': [
            {'id': 'P1', 'kind': 'controlled_numerical_prediction',
             'prediction': '3/3坐标seed均RMSE<=.03但maxabs>.05，因此加权候选不同时通过原描述阈值。',
             'criterion': '任一cell RMSE>.03或maxabs<=.05即refuted；保持原候选和失败记录。',
             'basis': '既有等权r100通过；单模态局部近似最大误差存在失败背景。已有alpha=.25反例RMSE=.022740946/maxabs=.053989623；改由慢模态占多数仍可能保留局部近似尾部误差，未计算本轮候选误差，非OOD或盲发现。'},
            {'id': 'P2', 'kind': 'controlled_numerical_interval_prediction',
             'prediction': '3/3坐标seed RMSE在[.025,.035]，maxabs在[.050,.060]。',
             'intervals': {'rmse': [.025, .035], 'maxabs': [.050, .060]},
             'criterion': '闭区间，任一cell任一指标在区间外即refuted，保留原区间与反例。',
             'basis': '按已有等权r100与alpha=.25、旧单模态局部公式的development误差量级给出更宽的新数值范围；未校准区间，不是置信区间，也不保证P1。'}],
        'previous_summary': str((old / 'summary.json').relative_to(ROOT)),
        'input_manifest': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in paths],
        'source_hashes': {str((STUDY / source).relative_to(ROOT)): sha(STUDY / source) for source in sources},
        'resume_protocol': '唯一完整预注册commit必须是HEAD祖先；当前JSON和全部pinned源码等于该commit blob，全部输入与历史hash/mtime通过才评价。全局先扫描所有receipt/结果：混合commit、冲突pin、额外cell/文件、孤立或不完整结果均拒绝。成功cell仅读取；新cell独占创建，绝不覆盖。',
        'compute_budget_seconds': 1200,
        'execution_environment': 'python3，本机CPU单线程；0训练/拟合，无网络/GPU/solver/模型API。',
        'boundaries': ['仅development，n=d=2固定特征half-MSE，float64，全批量SGD eta=.1/mom0/nodecay，lambda_s=.01，r100，L0=1，alpha=.75。',
                       '仅原150点等权网格0..4997，平台1/0；除保存幅度外h/b及判据全部固定；不校准幅度。',
                       '8个等权通过格点不代表连续临界；不独立检验误差重叠机制、参数可辨认、物理merging或grokking。',
                       'alpha=.25新反例与八个等权离散通过格点保留；旧r1局部公式、D4纯快与alpha=.01原LS失败保留且不重算/改判；不重评任何旧成功cell。',
                       '原单段LS为描述性对照，不把差异归因于段数；不外推MLP、learned features、momentum、mini-batch、test或OOD。'],
        'pre_evaluation_note': '已按序读必要文件并执行inbox；六节全文重排且全部数字逐项保全，独立commit 2605e7b。冻结前工具JS SyntaxError及重排Python字符串SyntaxError均未写文件或科学评价；修正后只做重排与静态审计。准备只核验hash/mtime/合同、复制原grid和注册源码，未运行alpha=.75候选误差；非盲或封存OOD。'}
    write(STUDY / 'preregistration.json', prereg)
    print(json.dumps({'historical_files': len(manifest), 'saved_contracts': len(contracts),
                      'closeout_pins': [r['verified_commit_file_hashes'] for r in checks],
                      'input_pins': len(paths), 'source_pins': len(sources), 'scientific_evaluation_performed': False}))


if __name__ == '__main__':
    main()
