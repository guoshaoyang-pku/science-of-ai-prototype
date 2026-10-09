import json
import subprocess
import sys

from run import ROOT, STUDY, check_frozen, scan_results, sha, write_json


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit, = summary['preregistration_commits']
    prereg = check_frozen(commit)
    pins = scan_results(prereg, commit)
    assert len(pins) == 3
    manifest = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
                for path in sorted((STUDY / 'results').rglob('*')) if path.is_file()]
    assert len(manifest) == 6
    run = subprocess.run([sys.executable, '-B', str(STUDY / 'executed/run.py'),
                          '--preregistration-commit', commit], cwd=ROOT, capture_output=True, text=True, check=True)
    receipt = json.loads(sorted((STUDY / 'executed').glob('run_receipt_*.json'))[-1].read_text())
    assert receipt['new_successes'] == [] and len(receipt['resumed_successes']) == 3
    assert receipt['new_training_cells'] == receipt['new_fits'] == 0
    for pin in manifest:
        path = ROOT / pin['path']
        assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns']
    check_frozen(commit)
    assert scan_results(prereg, commit) == pins
    audit = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    verification = {'status': 'passed', 'preregistration_commit': commit,
                    'new_evaluation_cells': 0, 'new_training_cells': 0, 'new_fits': 0, 'resumed_cells': 3,
                    'result_hash_and_mtime_unchanged': manifest,
                    'historical_hash_and_mtime_unchanged': len(audit['previous_study_file_manifest']),
                    'run_stdout': run.stdout, 'run_stderr': run.stderr}
    write_json(STUDY / 'executed/resume_verification.json', verification)
    print(json.dumps({k: v for k, v in verification.items() if k != 'result_hash_and_mtime_unchanged'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
