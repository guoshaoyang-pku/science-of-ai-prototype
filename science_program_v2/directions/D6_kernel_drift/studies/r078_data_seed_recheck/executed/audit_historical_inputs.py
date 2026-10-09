import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
PRIOR = REPO / 'directions/D6_kernel_drift/studies'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived(commit, name):
    return subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=REPO,
                          check=True, capture_output=True).stdout


def main():
    transfer = PRIOR / 'r046_data_seed_transfer'
    receipt = json.loads((transfer / 'executed/final_commit_verification.json').read_text())
    assert receipt['status'] == 'passed' and receipt['saved_cells'] == 12
    commit = receipt['scientific_closeout_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['preregistration_commit'], commit],
                   cwd=REPO, check=True)
    for name, expected in receipt['scientific_sha256'].items():
        assert hashlib.sha256(archived(commit, name)).hexdigest() == expected, name
        if '/studies/' in name or '/findings/' in name:
            assert sha(REPO / name) == expected, name
    historical = json.loads((transfer / 'executed/historical_evidence_verification.json').read_text())
    successful = json.loads((transfer / 'executed/saved_evidence_verification.json').read_text())
    assert historical['status'] == successful['status'] == 'passed'
    for snapshot in [historical, successful]:
        for name, expected in snapshot['files'].items():
            path = REPO / name
            assert sha(path) == expected['sha256'], name
            assert path.stat().st_mtime_ns == expected['mtime_ns'], name
    scans = []
    for slug in ['r006_early_direction', 'r022_early_amplitude', 'r046_data_seed_transfer']:
        root = PRIOR / slug
        config = json.loads((root / 'preregistration.json').read_text())
        expected = set()
        cells = []
        for function in config['functions']:
            for width in config['widths']:
                for seed in config['seeds']:
                    cell = f'{function}_w{width}_s{seed}'
                    meta_path = root / 'results' / f'{cell}.json'
                    arrays_path = root / 'results' / f'{cell}.npz'
                    required = [meta_path, arrays_path]
                    if 'early_step' in config:
                        required.append(root / 'results' / f'{cell}.early.json')
                    expected.update(path.name for path in required)
                    meta = json.loads(meta_path.read_text())
                    assert meta['status'] == 'success', cell
                    assert sha(arrays_path) == meta['arrays_sha256'], cell
                    if 'early_step' in config:
                        assert sha(required[2]) == meta['early_sha256'], cell
                    cells.append(cell)
        actual = {path.name for path in (root / 'results').iterdir() if path.is_file()}
        assert actual == expected, {'study': slug, 'extra': sorted(actual - expected),
                                    'missing': sorted(expected - actual)}
        assert len(cells) == 12
        scans.append({'study': slug, 'successful_cells': len(cells),
                      'successful_files': len(expected), 'extra_files': [], 'missing_files': []})
    independent = json.loads((transfer / 'executed/independent_verification.json').read_text())
    assert independent['status'] == 'passed' and independent['checkpoints_rebuilt'] == 36
    assert sha(transfer / 'executed/independent_verification.py') == independent['verification_source_sha256']
    result = {
        'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'new_training_cells': 0, 'r046_preregistration_commit': receipt['preregistration_commit'],
        'r046_scientific_closeout_commit': commit, 'preregistration_is_ancestor': True,
        'receipt_archived_blob_checks': len(receipt['scientific_sha256']),
        'current_prior_studies_and_findings_match_receipt': True,
        'historical_hash_mtime_files': len(historical['files']),
        'r046_success_hash_mtime_files': len(successful['files']),
        'all_hashes_and_mtimes_unchanged': True, 'artifact_scans': scans,
        'archived_r046_independent_verification': {
            'checkpoints_rebuilt': independent['checkpoints_rebuilt'],
            'max_initial_parameter_error': independent['max_initial_parameter_error'],
            'max_output_error': independent['max_output_error'],
            'max_jacobian_error': independent['max_jacobian_error'],
            'min_preregistration_commit_lead_seconds': independent['min_preregistration_commit_lead_seconds'],
            'note': '这里只核验旧独立重建证据的归档与hash，不重新运行或覆盖旧重建脚本。'},
        'source_sha256': sha(Path(__file__)),
        'note': '只读审计旧证据；无新数据生成、训练、拟合。r046 archived scientific blobs全部核验；共享中心文件和活报告允许本轮更新。'
    }
    destination = STUDY / 'executed/historical_input_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['artifact_scans', 'archived_r046_independent_verification']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
