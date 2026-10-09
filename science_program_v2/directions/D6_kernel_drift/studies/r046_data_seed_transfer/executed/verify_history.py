import hashlib
import json
from pathlib import Path
import subprocess

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
OLD = REPO / 'directions/D6_kernel_drift/studies'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archived(commit, path):
    return subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO,
                          check=True, capture_output=True).stdout


def main():
    receipt = json.loads((OLD / 'r022_early_amplitude/executed/final_commit_verification.json').read_text())
    commit = receipt['scientific_closeout_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['preregistration_commit'], commit],
                   cwd=REPO, check=True)
    for name, expected in receipt['scientific_sha256'].items():
        assert sha(archived(commit, name)) == expected, name
        if '/studies/' in name or '/findings/' in name:
            assert sha((REPO / name).read_bytes()) == expected, name
    baseline = STUDY / 'executed/historical_evidence_verification.json'
    if baseline.exists():
        previous = json.loads(baseline.read_text())
        for name, expected in previous['files'].items():
            path = REPO / name
            assert sha(path.read_bytes()) == expected['sha256'], name
            assert path.stat().st_mtime_ns == expected['mtime_ns'], name
        print(json.dumps({'status': 'passed', 'unchanged_historical_files': len(previous['files'])}))
        return
    files = {}
    for folder in [OLD / 'r006_early_direction', OLD / 'r022_early_amplitude',
                   REPO / 'directions/D6_kernel_drift/findings']:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(REPO))] = {'sha256': sha(path.read_bytes()),
                                                     'mtime_ns': path.stat().st_mtime_ns}
    for path in sorted((OLD / 'r006_early_direction/results').glob('*')):
        name = str(path.relative_to(REPO))
        assert archived('3dd7adf', name) == path.read_bytes(), name
    for folder in ['r006_early_direction', 'r022_early_amplitude']:
        paths = sorted((OLD / folder / 'results').glob('*.npz'))
        assert len(paths) == 12
        for path in paths:
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['status'] == 'success' and sha(path.read_bytes()) == meta['arrays_sha256']
    result = {'status': 'passed', 'receipt_archived_blobs_checked': len(receipt['scientific_sha256']),
              'r022_scientific_closeout_commit': commit,
              'r022_preregistration_commit': receipt['preregistration_commit'],
              'r006_scientific_closeout_commit': '3dd7adf',
              'r006_log_only_commit': '03a3741', 'saved_historical_cells': 24,
              'files': files,
              'note': '历史receipt的50个归档blob核验；当前中心共享文件/方向活报告允许更新，旧study与findings字节和mtime冻结。'}
    baseline.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
