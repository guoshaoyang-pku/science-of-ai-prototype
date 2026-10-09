import json
import subprocess
from run import ROOT, STUDY, cell_label, contract, save, scan_results, sha
import numpy as np


def main():
    cfg, pins = contract()
    scan_results(cfg, pins)
    summary = json.loads((STUDY / 'summary.json').read_text())
    verification = json.loads((STUDY / 'executed/verification.json').read_text())
    resume = json.loads((STUDY / 'executed/resume_verification.json').read_text())
    assert verification['passed'] and resume['passed']
    assert summary['round_result'] == 'partial' and not summary['validation']['passed']
    freeze_seconds = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', pins['git_commit']], cwd=ROOT, text=True))
    cells = []
    for condition, row in zip(cfg['cells'], summary['cells']):
        path = STUDY / 'results' / (cell_label(condition['seed']) + '.json')
        metadata = json.loads(path.read_text())
        lead = metadata['started_at_unix_ns']/1e9 - freeze_seconds
        assert lead > 0
        with np.load(path.with_suffix('.npz')) as current, np.load(ROOT / condition['baseline']) as previous:
            same_basis = np.array_equal(current['basis'], previous['basis'])
            assert same_basis
            cells.append({'seed': condition['seed'], 'freeze_before_cell_seconds': lead,
                          'same_basis_values': same_basis,
                          'B8_basis_F_contiguous': bool(previous['basis'].flags.f_contiguous),
                          'B8_basis_C_contiguous': bool(previous['basis'].flags.c_contiguous),
                          'B4_basis_F_contiguous': bool(current['basis'].flags.f_contiguous),
                          'B4_basis_C_contiguous': bool(current['basis'].flags.c_contiguous),
                          'registered_exact_mean_equality_pass': row['mean_coordinates_B8_maxabs'] == 0,
                          'mean_coordinates_maxabs': row['mean_coordinates_B8_maxabs'],
                          'mean_risk_maxabs': row['mean_risk_B8_maxabs']})
    state = json.loads((ROOT / 'central/state.json').read_text())
    before = json.loads((STUDY / 'executed/input_audit.json').read_text())['state_before']
    direction = cfg['direction']
    left, right = json.loads(json.dumps(before)), json.loads(json.dumps(state))
    for value in (left, right):
        for name in ('last_round', 'next_question'):
            value['directions'][direction].pop(name, None)
    assert left == right
    result = {'scientific_prediction': summary['predictions'], 'round_result': 'partial',
              'registered_validation_failed': 'B4 mean_coordinates和mean_risk与保存B8必须逐位相等：两个seed均失败。',
              'other_numerical_validation': '新B4独立raw二阶矩与argmin通过；有限/窗内/PSD/full原样继承/保存full容差通过。',
              'no_posthoc_tolerance_change': True, 'original_analysis_and_log_preserved': True,
              'successful_cells_not_overwritten': resume['all_protected_sha256_mtime_unchanged'],
              'state_only_D9_last_round_next_question_changed': True, 'round_unchanged': state['round']==before['round'],
              'rounds_done_unchanged': state['directions'][direction]['rounds_done']==before['directions'][direction]['rounds_done'],
              'cells': cells, 'pins': pins,
              'layout_interpretation': 'basis值相等而保存内存顺序不同；这是einsum舍入路径候选，未做layout干预或重算，根因未确证。',
              'not_a_strict_error_certificate': True}
    save(STUDY / 'executed/closeout_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
