from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit')
    args = parser.parse_args()
    config = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    independent = json.loads((STUDY / 'executed/independent_verification.json').read_text())
    saved = json.loads((STUDY / 'executed/saved_evidence_verification.json').read_text())
    assert summary['saved_cells'] == summary['planned_cells'] == 12
    assert summary['predictions'] == {'P1_negative_reduction': 'supported'}
    assert independent['status'] == 'passed' and independent['cells_checked'] == 12
    assert independent['checkpoints_rebuilt'] == 36 and independent['new_training_cells'] == 0
    assert saved['status'] == 'passed' and saved['resume_hash_and_mtime_unchanged']

    prereg = str(STUDY.joinpath('preregistration.json').relative_to(REPO))
    history = subprocess.run(['git', 'log', '--format=%H', '--', prereg], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    prereg_commit = 'f6e9aa1193cd4e7107fe8b80bd892c61e618f5e0'
    assert history == [prereg_commit]
    subprocess.run(['git', 'merge-base', '--is-ancestor', prereg_commit, 'HEAD'], cwd=REPO, check=True)
    assert independent['preregistration_commits'] == [prereg_commit]

    actual = {p.name for p in (STUDY / 'results').iterdir() if p.is_file()}
    expected = {f'{fn}_w{w}_s{s}{suffix}' for fn in config['functions']
                for w in config['widths'] for s in config['seeds']
                for suffix in ('.json', '.npz', '.early.json')}
    assert actual == expected
    for relative, record in saved['files'].items():
        path = REPO / relative
        assert sha(path) == record['sha256']
        assert path.stat().st_mtime_ns == record['mtime_ns']
    report = (REPO / 'directions/D6_kernel_drift/report.md').read_text()
    assert re.findall(r'^## (.+)$', report, re.M) == ['结论', 'Formulation', '成立程度', '方法与条件', '失败与反例', '未决问题', '证据']
    body = report.split('## 证据')[0]
    assert not any(token in body for token in ('](', 'studies/', 'findings/'))
    assert not re.search(r'[0-9a-f]{40}', body)
    kb = json.loads((REPO / 'central/kb.json').read_text())
    claim = next(c for c in kb['claims'] if c['id'] == 'D6-009')
    assert claim['status'] == 'measured' and claim['round'] == 108
    state = json.loads((REPO / 'central/state.json').read_text())
    d6 = state['directions']['D6_kernel_drift']
    assert d6['rounds_done'] == 6 and d6['last_round']['round'] == 108

    scientific_paths = [p for p in STUDY.rglob('*') if p.is_file()
                        and p.name not in {'final_commit_verification.json'}
                        and '__pycache__' not in p.parts]
    scientific_paths += [REPO / p for p in [
        'directions/D6_kernel_drift/findings/r108_data_seed_transfer.md',
        'directions/D6_kernel_drift/report.md', 'central/kb.json',
        'reports/PROGRESS.md']]
    scientific = {str(p.relative_to(REPO)): sha(p) for p in sorted(scientific_paths)}
    result = {
        'status': 'passed',
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'scientific_closeout_commit': args.commit,
        'preregistration_commit': prereg_commit,
        'preregistration_is_ancestor': bool(args.commit),
        'saved_cells': 12,
        'new_training_cells': 12,
        'mae': summary['mae'],
        'paired_mae_reduction': summary['paired_mae_reduction'],
        'predictions': summary['predictions'],
        'independent_checkpoints_rebuilt': 36,
        'resume_hash_and_mtime_unchanged': True,
        'scientific_sha256': scientific,
        'supervisor_rounds_done_unchanged': True,
        'note': '仅保留r108_data_seed_transfer；重复目录r108_data_seed260610未纳入科学产物。'
    }
    out = STUDY / 'executed' / ('final_commit_verification.json' if args.commit else 'precloseout_verification.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'scientific_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
