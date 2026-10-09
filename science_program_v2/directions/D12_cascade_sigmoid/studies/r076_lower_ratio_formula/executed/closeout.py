import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scientific-commit')
    args = parser.parse_args()
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    independent = json.loads((STUDY / 'executed/independent_verification.json').read_text())
    resume = json.loads((STUDY / 'executed/resume_verification.json').read_text())
    audit = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    assert summary['counts'] == {'planned_cells': 6, 'saved_cells': 6,
                                'condition_recipe_units': 2, 'coordinate_seeds_per_unit': 3,
                                'new_training_cells': 0, 'new_fits': 0}
    assert all(summary['predictions'][key]['status'] == 'supported' for key in ['P1', 'P2'])
    assert independent['status'] == resume['status'] == audit['status'] == 'passed'
    assert resume['new_evaluation_cells'] == 0 and resume['resumed_cells'] == 6
    for pin in audit['previous_study_file_manifest'] + resume['result_hash_and_mtime_unchanged']:
        path = ROOT / pin['path']
        assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns'], pin['path']
    prereg_commit, = summary['preregistration_commits']
    for relative, expected in {str((STUDY / 'preregistration.json').relative_to(ROOT)):
                              sha(STUDY / 'preregistration.json'), **prereg['source_hashes']}.items():
        blob = subprocess.check_output(['git', 'show', f'{prereg_commit}:{relative}'], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == expected == sha(ROOT / relative), relative
    for pin in prereg['input_manifest']:
        assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
    state = json.loads((ROOT / 'central/state.json').read_text())
    before = json.loads((STUDY / 'executed/state_before.json').read_text())
    current_summary = state['directions']['D12_cascade_sigmoid']['last_round']['summary']
    current_question = state['directions']['D12_cascade_sigmoid']['next_question']
    before['directions']['D12_cascade_sigmoid']['last_round']['summary'] = current_summary
    before['directions']['D12_cascade_sigmoid']['next_question'] = current_question
    assert state == before, 'Only D12 last_round.summary/next_question may change'
    assert state['round'] == 76 and state['directions']['D12_cascade_sigmoid']['rounds_done'] == 4
    kb = json.loads((ROOT / 'central/kb.json').read_text())
    old_kb = json.loads(subprocess.check_output(['git', 'show', f'{prereg_commit}:central/kb.json'], cwd=ROOT))
    assert kb['claims'][:-1] == old_kb['claims']
    assert kb['claims'][-1]['id'] == 'D12-006'
    for evidence in kb['claims'][-1]['evidence']:
        assert (ROOT / evidence).is_file(), evidence
    report_path = ROOT / 'directions/D12_cascade_sigmoid/report.md'
    report = report_path.read_text()
    assert re.findall(r'^## (.+)$', report, flags=re.M) == [
        '规律与公式', '现象与解释', '失败与反例', '方法与条件', '未决问题', '证据']
    body, evidence_section = report.split('## 证据', 1)
    assert '](studies/' not in body and 'findings/' not in body and 'commit' not in body
    assert not re.search(r'\br\d{3}_', body)
    assert all(value in body for value in ['.066366348', '.06476095', '.07462910', '.000638477'])
    deferred_receipt = STUDY / 'executed/final_commit_verification.json'
    for target in re.findall(r'\]\(([^)]+)\)', report):
        resolved = (report_path.parent / target).resolve()
        assert resolved.is_file() or resolved == deferred_receipt, target
    progress = (ROOT / 'reports/PROGRESS.md').read_text()
    old_progress = subprocess.check_output(['git', 'show', f'{prereg_commit}:reports/PROGRESS.md'], cwd=ROOT, text=True)
    assert progress.startswith(old_progress) and 'r076' in progress[len(old_progress):]
    validation = {'status': 'passed', 'checked_at': datetime.now().astimezone().isoformat(),
                  'preregistration_commit': subprocess.check_output(
                      ['git', 'rev-parse', prereg_commit], cwd=ROOT, text=True).strip(),
                  'historical_files_unchanged': len(audit['previous_study_file_manifest']),
                  'new_result_files_unchanged': len(resume['result_hash_and_mtime_unchanged']),
                  'only_D12_summary_and_next_question_changed': True,
                  'supervisor_round_and_rounds_done_unchanged': True,
                  'report_six_sections_and_evidence_links_valid': True,
                  'old_claims_preserved': True, 'new_claim_count': 1}
    if not args.scientific_commit:
        write(STUDY / 'executed/delivery_validation.json', validation)
    else:
        commit = args.scientific_commit
        subprocess.run(['git', 'merge-base', '--is-ancestor', prereg_commit, commit], cwd=ROOT, check=True)
        required = [ROOT / 'central/kb.json', ROOT / 'central/state.json', ROOT / 'reports/PROGRESS.md',
                    report_path, ROOT / 'directions/D12_cascade_sigmoid/findings/r076_lower_ratio_formula.md']
        required += [path for path in sorted(STUDY.rglob('*')) if path.is_file()
                     and path != deferred_receipt]
        pins = []
        for path in required:
            relative = str(path.relative_to(ROOT))
            blob = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == sha(path), relative
            pins.append({'path': relative, 'sha256': sha(path)})
        validation.update(scientific_closeout_commit=commit, preregistration_is_ancestor=True,
                          all_required_artifacts_committed=True, verified_files=pins,
                          verified_file_count=len(pins))
        write(deferred_receipt, validation)
    print(json.dumps({key: value for key, value in validation.items() if key != 'verified_files'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
