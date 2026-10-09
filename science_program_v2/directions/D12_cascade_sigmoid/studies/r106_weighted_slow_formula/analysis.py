import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / 'executed'))
from run import check_frozen, scan_results, sha, write_json


def main():
    receipts = [json.loads(p.read_text()) for p in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    commit, = {r['preregistration_commit'] for r in receipts}
    prereg = check_frozen(commit)
    pins = scan_results(prereg, commit)
    if set(pins) != {r['cell_id'] for r in prereg['cells']}:
        raise RuntimeError('Incomplete evaluation')
    rows = []
    for request in prereg['cells']:
        result_dir = STUDY / 'results' / request['cell_id']
        row = json.loads((result_dir / 'evaluation.json').read_text())
        with np.load(result_dir / 'arrays.npz', allow_pickle=False) as arrays:
            if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                raise RuntimeError('Nonfinite result')
            residual = arrays['prediction'] - arrays['observed']
            metrics = {'rmse': float(np.sqrt(np.mean(residual ** 2))), 'maxabs': float(np.max(np.abs(residual)))}
            if (not np.array_equal(residual, arrays['residual'])
                    or any(abs(metrics[m] - row[m]) > 1e-14 for m in metrics)
                    or row['maxabs_step'] != int(arrays['grid'][np.argmax(np.abs(residual))])):
                raise RuntimeError('Saved metric mismatch')
        row['result_hashes'] = pins[request['cell_id']]
        rows.append(row)
    p1_failures = [r['cell_id'] for r in rows if not (r['rmse'] <= .03 and r['maxabs'] > .05)]
    intervals = prereg['predictions'][1]['intervals']
    p2_failures = [{'cell_id': r['cell_id'], 'metric': m, 'value': r[m], 'interval': intervals[m]}
                   for r in rows for m in ['rmse', 'maxabs'] if not intervals[m][0] <= r[m] <= intervals[m][1]]
    aggregate = {'ratio': 100, 'alpha': .75, 'adequate_coordinate_seeds': sum(r['adequate'] for r in rows),
                 'formula_error_ranges': {m: [min(r[m] for r in rows), max(r[m] for r in rows)] for m in ['rmse', 'maxabs']},
                 'paired_formula_minus_fitted_ranges': {m: [min(r['paired_formula_minus_fitted'][m] for r in rows),
                                                            max(r['paired_formula_minus_fitted'][m] for r in rows)]
                                                       for m in ['rmse', 'maxabs']},
                 'rmse_threshold_margins': [.03 - max(r['rmse'] for r in rows), .03 - min(r['rmse'] for r in rows)],
                 'maxabs_threshold_excess': [min(r['maxabs'] for r in rows) - .05, max(r['maxabs'] for r in rows) - .05],
                 'maxabs_steps': sorted({r['maxabs_step'] for r in rows})}
    summary = {'study': prereg['study'], 'round': 106, 'direction_round': 7, 'direction': prereg['direction'],
               'domain': 'development', 'status': 'analyzed',
               'counts': {'planned_cells': 3, 'saved_cells': len(rows), 'condition_recipe_units': 1,
                          'coordinate_seeds_per_unit': 3, 'new_training_cells': 0, 'new_fits': 0},
               'preregistration_sha256': sha(STUDY / 'preregistration.json'), 'preregistration_commits': [commit],
               'source_hashes': prereg['source_hashes'], 'grid': prereg['evaluation']['grid'],
               'predictions': {'P1': {'status': 'refuted' if p1_failures else 'supported', 'counterexamples': p1_failures},
                               'P2': {'status': 'refuted' if p2_failures else 'supported', 'counterexamples': p2_failures}},
               'aggregate': aggregate, 'cells': rows, 'boundaries': prereg['boundaries'],
               'evaluation_receipt_seconds': sum(r['elapsed_seconds'] for r in receipts)}
    write_json(STUDY / 'summary.json', summary)
    print(json.dumps({'predictions': summary['predictions'], 'aggregate': aggregate}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
