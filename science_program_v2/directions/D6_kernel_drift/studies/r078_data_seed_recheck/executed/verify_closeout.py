import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit')
    args = parser.parse_args()
    config = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    assert summary['saved_cells'] == config['planned_cells'] == 12
    assert summary['predictions'] == {'P1_failure_reproduced': 'refuted'}
    before = json.loads((STUDY / 'executed/state_before.json').read_text())
    after = json.loads((REPO / 'central/state.json').read_text())
    for field in ['last_round', 'next_question']:
        after['directions']['D6_kernel_drift'][field] = before['directions']['D6_kernel_drift'][field]
    assert before == after, 'Only D6 last_round and next_question may change'
    for name, expected in config['source_sha256'].items():
        assert sha(STUDY / name) == expected
    for name, expected in config['input_sha256'].items():
        assert sha(REPO / name) == expected
    prior = REPO / config['previous_draw_study']
    historical_files = 0
    for name in ['historical_evidence_verification.json', 'saved_evidence_verification.json']:
        receipt = json.loads((prior / 'executed' / name).read_text())
        for relative, expected in receipt['files'].items():
            path = REPO / relative
            assert sha(path) == expected['sha256']
            assert path.stat().st_mtime_ns == expected['mtime_ns']
            historical_files += 1
    successful = json.loads((STUDY / 'executed/saved_evidence_verification.json').read_text())
    assert successful['status'] == 'passed' and successful['resume_hash_and_mtime_unchanged']
    actual_files = {str(p.relative_to(REPO)) for p in (STUDY / 'results').iterdir() if p.is_file()}
    assert actual_files == set(successful['files'])
    for name, expected in successful['files'].items():
        path = REPO / name
        assert sha(path) == expected['sha256'] and path.stat().st_mtime_ns == expected['mtime_ns']
    independent = json.loads((STUDY / 'executed/independent_verification.json').read_text())
    assert independent['status'] == 'passed' and independent['checkpoints_rebuilt'] == 36
    assert independent['summary_sha256'] == successful['summary_sha256'] == sha(STUDY / 'summary.json')
    previous = json.loads((STUDY / 'executed/previous_draw_verification.json').read_text())
    assert previous['status'] == 'passed' and previous['cells_paired'] == 12
    report = (REPO / 'directions/D6_kernel_drift/report.md').read_text()
    assert re.findall(r'^## (.+)$', report, re.M) == ['规律与公式', '现象与解释', '失败与反例', '方法与条件', '未决问题', '证据']
    body = report.split('## 证据')[0]
    assert all(text not in body for text in ['](', 'studies/', 'findings/'])
    assert not re.search(r'[0-9a-f]{40}', body)
    for name in ['first_cell.log', 'remaining_cells.log', 'resume_verification.log',
                 'analysis.log', 'independent_verification.log', 'previous_draw_verification.log']:
        assert 'RuntimeWarning' not in (STUDY / 'executed' / name).read_text()
    paths = [p for p in STUDY.rglob('*') if p.is_file() and p.name not in
             ['precloseout_verification.json', 'final_commit_verification.json']]
    paths += [REPO / p for p in ['directions/D6_kernel_drift/findings/r078_data_seed_recheck.md',
              'directions/D6_kernel_drift/report.md', 'directions/D6_kernel_drift/inbox.md',
              'central/kb.json', 'central/state.json', 'reports/PROGRESS.md']]
    scientific = {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}
    prereg_commit = independent['preregistration_commits'][0]
    if args.commit:
        subprocess.run(['git', 'merge-base', '--is-ancestor', prereg_commit, args.commit], cwd=REPO, check=True)
        for name, expected in scientific.items():
            blob = subprocess.run(['git', 'show', f'{args.commit}:{name}'], cwd=REPO, capture_output=True, check=True).stdout
            assert hashlib.sha256(blob).hexdigest() == expected, name
    result = {'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
              'scientific_closeout_commit': args.commit, 'preregistration_commit': prereg_commit,
              'preregistration_is_ancestor': bool(args.commit), 'saved_cells': 12,
              'checked_files': len(scientific), 'all_owned_scientific_blobs_match_worktree': bool(args.commit),
              'scientific_sha256': scientific, 'supervisor_round_and_rounds_done_unchanged': True,
              'only_D6_last_round_and_next_question_changed': True,
              'historical_input_hashes_unchanged': len(config['input_sha256']),
              'historical_files_hash_and_mtime_unchanged': historical_files,
              'successful_files_hash_and_mtime_unchanged': len(successful['files']),
              'mae': summary['mae'], 'paired_mae_reduction': summary['paired_mae_reduction'],
              'predictions': summary['predictions'], 'registered_components': summary['registered_components'],
              'independent_checkpoints_rebuilt': 36, 'no_new_runtime_warning': True,
              'read_only_evidence_review': '第二代理只读复算12NPZ/J、E/Q/L、三MAE与四recipe区间最大差0；pins与时序通过；0训练/0写入。',
              'note': '科学收尾commit后生成最终回执；中心supervisor输出日志允许继续变化，不纳入科学blob匹配。'}
    name = 'final_commit_verification.json' if args.commit else 'precloseout_verification.json'
    (STUDY / 'executed' / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'scientific_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
