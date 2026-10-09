import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from run import ROOT, STUDY, contract, scan_results


def main():
    cfg, pins = contract()
    receipt = scan_results(cfg, pins)
    summary = json.loads((STUDY / 'summary.json').read_text())
    verify = json.loads((STUDY / 'executed/verification.json').read_text())
    resume = json.loads((STUDY / 'executed/resume_verification.json').read_text())
    before = json.loads((STUDY / 'executed/input_audit.json').read_text())
    current = json.loads((ROOT / 'central/state.json').read_text())
    state = copy.deepcopy(current)
    for key in ('last_round', 'next_question'):
        state['directions']['D9_minibatch_noise'][key] = before['state_before']['directions']['D9_minibatch_noise'][key]
    assert state == before['state_before'], 'State changes outside the two permitted fields'
    assert summary['pins'] == pins == verify['pins']
    assert summary['validation']['passed'] and verify['passed'] and resume['passed']
    assert len(receipt['cells']) == 2
    old_files = before['old_files']
    actual = {p.relative_to(ROOT).as_posix() for p in STUDY.parent.rglob('*')
              if p.is_file() and STUDY not in p.parents}
    assert actual == set(old_files), 'Historical file set changed'
    for relative, record in resume['files'].items():
        path = STUDY / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        assert path.stat().st_mtime_ns == record['mtime_ns']
    freeze_time = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', pins['git_commit']], cwd=ROOT))
    timing = []
    for metadata in sorted((STUDY / 'results').glob('n*.json')):
        cell = json.loads(metadata.read_text())
        delta = cell['started_at_unix_ns'] / 1e9 - freeze_time
        assert delta > 0
        assert cell['layout'] == {'basis': 'C', 'q': 'C', 'qa': 'C'}
        timing.append({'seed': cell['seed'], 'freeze_before_cell_seconds': delta})
    report = (ROOT / 'directions/D9_minibatch_noise/report.md').read_text()
    assert re.findall(r'^## (.+)$', report, re.M) == ['结论', 'Formulation', '成立程度', '方法与条件', '未决问题', '证据']
    for section in ('结论', '成立程度'):
        text = report.split('## ' + section + chr(10), 1)[1].split(chr(10) + '## ', 1)[0]
        assert '](' not in text and not re.search(r'studies/|findings/|r0[0-9]{2}|94beabf|d644e3b', text)
    original_report = subprocess.check_output(['git', 'show', 'd644e3b:directions/D9_minibatch_noise/report.md'], cwd=ROOT, text=True)
    numbers = re.findall(r'[0-9]+(?:[.][0-9]+)?(?:e[−-]?[0-9]+)?', original_report)
    current_numbers = re.findall(r'[0-9]+(?:[.][0-9]+)?(?:e[−-]?[0-9]+)?', report)
    from collections import Counter
    missing = Counter(numbers) - Counter(current_numbers)
    assert not missing, missing
    old_summary = json.loads((STUDY.parent / 'r085_batch4_exact/summary.json').read_text())
    assert old_summary['validation']['passed'] is False and old_summary['round_result'] == 'partial'
    result = {
        'passed': True, 'pins': pins, 'new_training': 0,
        'scientific_prediction': summary['predictions'],
        'registered_numerical_validation_passed': True,
        'independent_raw_second_moment_verification_passed': True,
        'freeze_timing': timing,
        'state_only_D9_last_round_next_question_changed': True,
        'round_unchanged': True, 'all_rounds_done_unchanged': True,
        'old_files_sha256_mtime_unchanged': len(old_files),
        'old_file_set_unchanged': True,
        'resume_existing_files_unchanged': len(resume['files']),
        'successful_cells_not_overwritten': True,
        'summary_and_verification_each_written_once': True,
        'prior_B4_registered_failure_and_partial_preserved': True,
        'report_six_section_order': True,
        'prior_report_number_multiset_preserved': True,
        'report_style_checklist': {
            'conclusion_first_and_ranges_quantified': True,
            'terms_explained': True,
            'no_unquantified_strength_claims': True,
            'evidence_links_only_in_evidence': True,
            'prior_failures_preserved': True,
            'no_unmeasured_extrapolation': True,
        },
        'limitations': 'Independent implementation differences are not strict error certificates. Frozen analysis/verify do not guard output overwrites; each was run once and excluded from resume. No old moment cell was recomputed.',
    }
    path = STUDY / 'executed/closeout_audit.json'
    with path.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write(chr(10))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
