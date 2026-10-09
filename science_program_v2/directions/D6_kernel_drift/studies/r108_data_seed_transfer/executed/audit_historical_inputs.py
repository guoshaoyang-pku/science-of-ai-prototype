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


def audit():
    previous = PRIOR / 'r090_negative_transfer'
    receipt = json.loads((previous / 'executed/final_commit_verification.json').read_text())
    assert receipt['status'] == 'passed' and receipt['saved_cells'] == 12
    commit = receipt['scientific_closeout_commit']
    for target in [commit, 'HEAD']:
        ancestor = receipt['preregistration_commit'] if target == commit else commit
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, target],
                       cwd=REPO, check=True)
    immutable = 0
    for name, expected in receipt['scientific_sha256'].items():
        blob = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=REPO,
                              check=True, capture_output=True).stdout
        assert hashlib.sha256(blob).hexdigest() == expected, name
        if '/studies/' in name or '/findings/' in name:
            assert sha(REPO / name) == expected, name
            immutable += 1
    old_config = json.loads((previous / 'preregistration.json').read_text())
    for name, expected in old_config['input_sha256'].items():
        assert sha(REPO / name) == expected, name
    snapshots = [
        PRIOR / 'r046_data_seed_transfer/executed/historical_evidence_verification.json',
        PRIOR / 'r046_data_seed_transfer/executed/saved_evidence_verification.json',
        previous / 'executed/saved_evidence_verification.json',
    ]
    checked = {}
    for snapshot in snapshots:
        saved = json.loads(snapshot.read_text())
        assert saved['status'] == 'passed'
        for name, expected in saved['files'].items():
            path = REPO / name
            assert sha(path) == expected['sha256'], name
            assert path.stat().st_mtime_ns == expected['mtime_ns'], name
            checked[name] = expected
    scans = []
    for slug in ['r006_early_direction', 'r022_early_amplitude',
                 'r046_data_seed_transfer', 'r078_data_seed_recheck',
                 'r090_negative_transfer']:
        root = PRIOR / slug
        config = json.loads((root / 'preregistration.json').read_text())
        expected = set()
        cells = 0
        for function in config['functions']:
            for width in config['widths']:
                for seed in config['seeds']:
                    cell = f'{function}_w{width}_s{seed}'
                    path = root / 'results' / f'{cell}.json'
                    required = [path, path.with_suffix('.npz')]
                    if 'early_step' in config:
                        required.append(path.with_name(f'{cell}.early.json'))
                    expected.update(p.name for p in required)
                    meta = json.loads(path.read_text())
                    assert meta['status'] == 'success', cell
                    assert sha(required[1]) == meta['arrays_sha256'], cell
                    if len(required) == 3:
                        assert sha(required[2]) == meta['early_sha256'], cell
                    cells += 1
        actual = {p.name for p in (root / 'results').iterdir()}
        assert actual == expected, (slug, sorted(actual - expected), sorted(expected - actual))
        assert cells == 12
        scans.append({'study': slug, 'successful_cells': cells, 'files': len(expected)})
    independent = json.loads((previous / 'executed/independent_verification.json').read_text())
    assert independent['status'] == 'passed' and independent['checkpoints_rebuilt'] == 36
    assert sha(previous / 'executed/independent_verification.py') == independent['verification_source_sha256']
    result = {
        'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'new_training_cells': 0, 'r078_scientific_closeout_commit': commit,
        'r078_preregistration_commit': receipt['preregistration_commit'],
        'archived_scientific_blob_checks': len(receipt['scientific_sha256']),
        'immutable_scientific_files': immutable,
        'prior_input_hashes_checked': len(old_config['input_sha256']),
        'hash_mtime_files_checked': len(checked), 'files': checked,
        'artifact_scans': scans, 'source_sha256': sha(Path(__file__)),
        'note': '只读核验旧证据；0新数据生成/训练/拟合。共享报告与中心文件仅核验归档blob。'
    }
    destination = STUDY / 'executed/historical_input_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    result = audit()
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, ensure_ascii=False))
