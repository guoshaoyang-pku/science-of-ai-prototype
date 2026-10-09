"""只读取旧合同、元数据与文件字节，准备预注册；不读取曲线数值。"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r083_low_noise_u_threshold'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))


def main():
    historical = {}
    for path in sorted(STUDY.parent.rglob('*')):
        if not path.is_file() or STUDY in path.parents or '__pycache__' in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        assert subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT) == path.read_bytes(), relative
        historical[relative] = {'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
    closeout = subprocess.check_output(['git', 'rev-parse', 'c8a9e3a'], cwd=ROOT, text=True).strip()
    matched = {}
    files = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', closeout], cwd=ROOT, text=True).splitlines()
    for relative in files:
        if '/studies/r083_' not in relative and '/findings/r083_' not in relative:
            continue
        path = ROOT / relative
        assert subprocess.check_output(['git', 'show', closeout + ':' + relative], cwd=ROOT) == path.read_bytes()
        matched[relative] = sha(path)
    prior_review = {
        'status': 'verified', 'method': '独立agent只读审查经消息返回；root落盘',
        'r083_scientific_closeout': closeout, 'r083_frozen_commit': '86f82ac7bfeca40fee8005609798de736d4fdbf3',
        'frozen_files_verified': 4, 'input_pins_verified': 13,
        'historical_hash_mtime_verified': 281, 'r083_success_and_summary_hash_mtime_verified': 4,
        'scientific_closeout_objects_verified': len(matched), 'r051_closeout_source_objects_verified': 10,
        'r083_final_commit_verification_receipt_exists': False,
        'gap': '旧final_commit_verification.json缺失；以真实c8a9e3a提交重核，不补造旧回执。',
        'commit_precedes_evaluation_seconds': 24.078096,
        'r083_t_star': 461, 'r083_delta': 0.02215220201337556,
        'new_seed413_condition_evaluations': 0, 'new_training_cells': 0,
        'r083_independent_verification_and_peer_review_consistent': True,
    }
    save(STUDY / 'executed/independent_prior_review.json', prior_review)
    save(STUDY / 'executed/prior_closeout_audit.json', {
        'status': 'verified', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'historical_study_files': historical, 'historical_files_count': len(historical),
        'r083_scientific_closeout_commit': closeout, 'r083_scientific_closeout_objects': matched,
        'missing_old_receipt': prior_review['gap'], 'new_condition_evaluations': 0, 'new_training_cells': 0,
        'report_reorder_commit': 'fcce095',
    })
    save(STUDY / 'executed/state_before.json', json.loads((ROOT / 'central/state.json').read_text()))
    for relative in ['executed/run.py', 'analysis.py', 'executed/independent_verify.py']:
        code = (OLD / relative).read_text().replace('seed412', 'seed413')
        code = code.replace("'round': 83", "'round': 95").replace("'direction_round': 6", "'direction_round': 7")
        with (STUDY / relative).open('x') as stream:
            stream.write(code)
        ast.parse(code)
    config = json.loads((OLD / 'preregistration.json').read_text())
    config['study'], config['round'], config['direction_round'] = STUDY.name, 95, 7
    config['question'] = '仅复用预指定n32/seed413保存分解曲线，将噪声方差.125减半至.0625，终点减最低风险是否仍至少.05？'
    config['cell']['seed'] = 413
    for field in ['source_curve', 'source_data']:
        config[field] = config[field].replace('seed412', 'seed413')
    config['evaluation_contract']['baseline'] = '同n32/seed413的旧.125保存risk，配对报告t*、最低/终点风险与delta差'
    config['predictions'] = [{
        'id': 'P1', 'kind': 'development_threshold_prediction', 'interval': [0.015, 0.045],
        'claim': 'sigma²=.0625的终点减最低风险在[.015,.045]内，低于.05操作性U型阈值。',
        'criterion': 'supported iff .015<=delta<=.045 and operational_u_shape=false；否则refuted。最低点只作描述。',
        'rationale': '旧seed413的.125差值为.058105292352594984，粗略减半约.029；已知seed412新差值为.02215220201337556。给宽区间，不假定delta线性减半；未组合seed413新曲线或端点。',
    }]
    config['prediction_provenance'] = {
        'new_sigma_evaluated_before_registration': False,
        'new_endpoint_or_minimum_computed_before_registration': False,
        'selection': '上一轮预指定seed413。已知development条件；此前seed412依旧近阈值余量选择。本轮非盲发现、非封存OOD。',
        'old_known': {'noise_variance': 0.125, 't_star': 230, 'minimum_risk': 0.15135295688426875,
                      'final_risk': 0.20945824923686374, 'delta': 0.058105292352594984},
        'known_seed412_delta_at_0625': 0.02215220201337556,
        'numeric_interval_origin': '旧seed413已保存summary差值、粗略减半判断及已见seed412结果；未读取未注册seed413曲线数值。',
    }
    config['boundaries'] = [item.replace('seed412', 'seed413') for item in config['boundaries']]
    paths = [config['source_curve'], config['source_curve'].replace('.npz', '.json'),
             config['source_data'], config['source_data'].replace('.npz', '.json')]
    paths += ['directions/D2_overfitting_u_curve/studies/r051_noise_halving/' + relative
              for relative in ['results/receipt.json', 'summary.json', 'preregistration.json',
                               'executed/run.py', 'analysis.py', 'executed/final_commit_verification.json',
                               'executed/independent_verification.json']]
    paths += [str((OLD / relative).relative_to(ROOT)) for relative in
              ['preregistration.json', 'summary.json', 'executed/execution_binding.json',
               'executed/independent_verification.json', 'executed/round083_completion.json']]
    paths += [str((STUDY / 'executed' / name).relative_to(ROOT))
              for name in ['prior_closeout_audit.json', 'independent_prior_review.json']]
    config['input_sha256'] = {relative: sha(ROOT / relative) for relative in paths}
    sources = ['executed/run.py', 'analysis.py', 'executed/independent_verify.py', 'executed/prepare_study.py']
    config['source_sha256'] = {relative: sha(STUDY / relative) for relative in sources}
    config['source_commits']['previous_evaluation_scientific_closeout'] = closeout
    config['source_commits']['new_execution'] = '唯一新冻结提交由execution_binding.json绑定；合同和全部四源码逐字节匹配才评价'
    config['created_at'] = datetime.now(timezone.utc).isoformat()
    save(STUDY / 'preregistration.json', config)
    save(STUDY / 'executed/static_validation.json', {
        'status': 'verified', 'ast_sources': sources, 'input_pins': len(paths), 'historical_files': len(historical),
        'new_condition_evaluations': 0, 'new_training_cells': 0, 'old_scripts_not_executed': True,
        'new_curve_numbers_not_read': True,
        'preparation_failures': ['inline重排命令换行SyntaxError，未执行Python', 'apply_patch调用JS引号SyntaxError，未执行工具'],
    })
    print(json.dumps({'prepared': config['study'], 'historical_files': len(historical), 'input_pins': len(paths), 'new_evaluations': 0}))


if __name__ == '__main__':
    main()
