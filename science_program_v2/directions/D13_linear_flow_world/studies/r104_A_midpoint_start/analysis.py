import argparse
import json
from pathlib import Path

from executed.run_experiment import frozen_contract, saved_rows, sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    rows = saved_rows(contract, contract_sha, args.freeze)
    assert len(rows) == 2
    baseline = json.loads((REPO / contract['baseline_fit']).read_text())
    base_tau = baseline['reference']['tau_A']
    assert base_tau == contract['baseline_tau_A']
    comparisons = []
    for cell in contract['cells']:
        row = rows[cell['id']]
        delta = row['tau_A'] - base_tau
        relative = abs(delta) / abs(base_tau)
        comparisons.append({'cell_id': cell['id'], 'h0_log_argument': cell['h0_log_argument'], 'tau_A': row['tau_A'],
                            'success': row['A']['success'], 'signed_difference': delta,
                            'relative_difference': relative,
                            'point_error': row['tau_A'] - contract['predictions']['P1']['per_start'][cell['id']]['point_tau_A'],
                            'supports_P1': bool(row['A']['success'] and relative <= .01),
                            'normalized_rmse': row['A']['normalized_rmse'], 'r2': row['A']['r2']})
    supported = sum(c['supports_P1'] for c in comparisons)
    times = [c['tau_A'] for c in comparisons]
    summary = {'study_id': contract['study_id'], 'freeze_commit': args.freeze,
               'preregistration_sha256': contract_sha, 'new_training_cells': 0,
               'new_fit_cells': 2, 'baseline_fit_recomputed': False,
               'baseline_tau_A': base_tau, 'comparisons': comparisons,
               'supported_cells': supported, 'total_cells': 2,
               'prediction_P1': 'supported' if supported == 2 else 'refuted',
               'tau_A_range': [min(times), max(times)],
               'max_relative_difference': max(c['relative_difference'] for c in comparisons),
               'result_sha256': {'results/' + cell['id'] + '.json': sha(ROOT / 'results' / (cell['id'] + '.json'))
                                 for cell in contract['cells']},
               'boundary': contract['boundary']}
    out = ROOT / 'summary.json'
    if out.exists():
        assert json.loads(out.read_text()) == summary, 'saved summary differs'
    else:
        with out.open('x') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write(chr(10))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
