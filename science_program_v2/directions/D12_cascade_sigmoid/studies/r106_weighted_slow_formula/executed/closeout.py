import argparse
from datetime import datetime
import hashlib
import json
import re
import subprocess

from run import ROOT, STUDY, check_frozen, scan_results, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scientific-commit')
    args = parser.parse_args()
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit, = summary['preregistration_commits']
    prereg = check_frozen(commit)
    assert len(scan_results(prereg, commit)) == 3
    assert summary['counts'] == {'planned_cells': 3, 'saved_cells': 3, 'condition_recipe_units': 1,
                                'coordinate_seeds_per_unit': 3, 'new_training_cells': 0, 'new_fits': 0}
    assert all(summary['predictions'][key]['status'] == 'refuted' for key in ['P1', 'P2'])
    assert summary['aggregate']['adequate_coordinate_seeds'] == 3
    assert all(r['rmse'] <= .03 and r['maxabs'] <= .05 and r['adequate'] for r in summary['cells'])
    independent = json.loads((STUDY / 'executed/independent_verification.json').read_text())
    resume = json.loads((STUDY / 'executed/resume_verification.json').read_text())
    audit = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    assert independent['status'] == resume['status'] == audit['status'] == 'passed'
    assert independent['commit_before_every_evaluation'] and independent['initial_saved_loss_weights_verified']
    assert resume['new_evaluation_cells'] == resume['new_training_cells'] == resume['new_fits'] == 0
    assert resume['resumed_cells'] == 3
    for pin in audit['previous_study_file_manifest'] + resume['result_hash_and_mtime_unchanged']:
        path = ROOT / pin['path']
        assert sha(path) == pin['sha256'] and path.stat().st_mtime_ns == pin['mtime_ns'], pin['path']
    receipts = [json.loads(p.read_text()) for p in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    assert len(receipts[0]['new_successes']) == 3 and not receipts[0]['resumed_successes']
    assert sum(len(r['new_successes']) for r in receipts) == 3
    assert all(r['new_training_cells'] == r['new_fits'] == 0 for r in receipts)
    assert summary['evaluation_receipt_seconds'] == receipts[0]['elapsed_seconds']
    state = json.loads((ROOT / 'central/state.json').read_text())
    before = json.loads((STUDY / 'executed/state_before.json').read_text())
    original_round = before['round']
    original_rounds_done = before['directions']['D12_cascade_sigmoid']['rounds_done']
    for key in ['summary']:
        before['directions']['D12_cascade_sigmoid']['last_round'][key] = state['directions']['D12_cascade_sigmoid']['last_round'][key]
    before['directions']['D12_cascade_sigmoid']['next_question'] = state['directions']['D12_cascade_sigmoid']['next_question']
    assert state == before, 'Only D12 last_round.summary/next_question may change'
    assert state['round'] == original_round == 106
    assert state['directions']['D12_cascade_sigmoid']['rounds_done'] == original_rounds_done == 6
    kb = json.loads((ROOT / 'central/kb.json').read_text())
    old_kb = json.loads(subprocess.check_output(['git', 'show', f'{commit}:central/kb.json'], cwd=ROOT))
    assert kb['claims'][:-2] == old_kb['claims'] and [r['id'] for r in kb['claims'][-2:]] == ['D12-008', 'D12-009']
    assert [r['status'] for r in kb['claims'][-2:]] == ['measured', 'refuted'] and all(r['round'] == 106 for r in kb['claims'][-2:])
    for evidence in kb['claims'][-1]['evidence']:
        assert (ROOT / evidence).is_file(), evidence
    report_path = ROOT / 'directions/D12_cascade_sigmoid/report.md'
    report = report_path.read_text()
    assert re.findall(r'^## (.+)$', report, flags=re.M) == [
        '结论', 'Formulation', '成立程度', '方法与条件', '未决问题', '证据']
    body, _ = report.split('## 证据', 1)
    assert not re.search(r'\]\([^)]+\)', body)
    assert 'findings/' not in body and 'studies/' not in body and 'commit' not in body
    assert not re.search(r'\br\d{3}_', body)
    old_report = subprocess.check_output(['git', 'show', f'{commit}:directions/D12_cascade_sigmoid/report.md'], cwd=ROOT, text=True)
    old_numbers = re.findall(r'\d+(?:\.\d+)?', old_report)
    current_numbers = re.findall(r'\d+(?:\.\d+)?', report)
    from collections import Counter
    assert not (Counter(old_numbers) - Counter(current_numbers)), 'An earlier measured number was lost'
    assert all(value in body for value in ['.066366348', '.06476095', '.07462910', '.000638477',
                                          '.029596700', '.048893158', '.053989623'])
    deferred_receipt = STUDY / 'executed/final_commit_verification.json'
    delivery = STUDY / 'executed/delivery_validation.json'
    finding = ROOT / 'directions/D12_cascade_sigmoid/findings/r088_weighted_formula.md'
    for path in [report_path, finding]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            resolved = (path.parent / target).resolve()
            assert resolved.is_file() or resolved in [deferred_receipt, delivery], target
    inbox = (ROOT / 'directions/D12_cascade_sigmoid/inbox.md').read_text()
    assert '[已处理 2026-10-07T23:24:59+08:00]' in inbox
    progress = (ROOT / 'reports/PROGRESS.md').read_text()
    old_progress = subprocess.check_output(['git', 'show', f'{commit}:reports/PROGRESS.md'], cwd=ROOT, text=True)
    assert progress.startswith(old_progress) and 'r106' in progress[len(old_progress):]
    validation = {'status': 'passed', 'checked_at': datetime.now().astimezone().isoformat(),
                  'preregistration_commit': commit, 'historical_files_unchanged': len(audit['previous_study_file_manifest']),
                  'new_result_files_unchanged': len(resume['result_hash_and_mtime_unchanged']),
                  'only_D12_summary_and_next_question_changed': True,
                  'supervisor_round_and_rounds_done_unchanged': True,
                  'inbox_six_sections_with_failures_in_support_and_evidence_links_valid': True,
                  'prior_report_numbers_preserved': True,
                  'old_claims_preserved': True, 'new_claim_count': 2,
                  'saved_cells': 3, 'adequate_cells': 3, 'prediction_statuses': summary['predictions'],
                  'new_training_cells': 0, 'new_fits': 0, 'resume_new_evaluation_cells': 0}
    if not args.scientific_commit:
        write_json(delivery, validation)
    else:
        scientific = args.scientific_commit
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit, scientific], cwd=ROOT, check=True)
        subprocess.run(['git', 'merge-base', '--is-ancestor', scientific, 'HEAD'], cwd=ROOT, check=True)
        required = [ROOT / 'central/kb.json', ROOT / 'central/state.json', ROOT / 'reports/PROGRESS.md',
                    report_path, finding, ROOT / 'directions/D12_cascade_sigmoid/inbox.md']
        required += [p for p in sorted(STUDY.rglob('*')) if p.is_file() and p != deferred_receipt]
        pins = []
        for path in required:
            relative = str(path.relative_to(ROOT))
            blob = subprocess.check_output(['git', 'show', f'{scientific}:{relative}'], cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == sha(path), relative
            pins.append({'path': relative, 'sha256': sha(path)})
        validation.update(scientific_closeout_commit=scientific, preregistration_is_ancestor=True,
                          all_required_artifacts_committed=True, verified_files=pins, verified_file_count=len(pins))
        write_json(deferred_receipt, validation)
    print(json.dumps({key: value for key, value in validation.items() if key != 'verified_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
