import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    if sha(Path(__file__)) != prereg['source_hashes'][str(Path(__file__).resolve().relative_to(ROOT))]:
        raise RuntimeError('Analysis source changed')
    receipts = [json.loads(p.read_text()) for p in sorted(
        (STUDY / 'executed').glob('run_receipt_*.json'))]
    pins = {}
    for receipt in receipts:
        for item in receipt['new_successes'] + receipt['resumed_successes']:
            if item['cell_id'] in pins and pins[item['cell_id']] != item:
                raise RuntimeError('Conflicting cell hash')
            pins[item['cell_id']] = item
    if set(pins) != {r['cell_id'] for r in prereg['cells']}:
        raise RuntimeError('Incomplete evaluation')
    rows = []
    for request in prereg['cells']:
        cell_id = request['cell_id']
        result_dir = STUDY / 'results' / cell_id
        if (sha(result_dir / 'arrays.npz') != pins[cell_id]['arrays_sha256']
                or sha(result_dir / 'evaluation.json') != pins[cell_id]['evaluation_sha256']):
            raise RuntimeError('Saved result changed')
        row = json.loads((result_dir / 'evaluation.json').read_text())
        with np.load(result_dir / 'arrays.npz', allow_pickle=False) as arrays:
            if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                raise RuntimeError('Nonfinite result')
            residual = arrays['prediction'] - arrays['observed']
            metrics = {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                       'maxabs': float(np.max(np.abs(residual)))}
            if any(abs(metrics[m] - row[m]) > 1e-14 for m in metrics):
                raise RuntimeError('Saved metric mismatch')
        row['result_hashes'] = pins[cell_id]
        rows.append(row)
    aggregate = []
    for ratio in prereg['ratios']:
        group = [r for r in rows if r['ratio'] == ratio]
        aggregate.append({'ratio': ratio, 'adequate_coordinate_seeds': sum(r['adequate'] for r in group),
                          'formula_error_ranges': {m: [min(r[m] for r in group), max(r[m] for r in group)]
                                                   for m in ['rmse', 'maxabs']},
                          'paired_formula_minus_fitted_ranges': {
                              m: [min(r['paired_formula_minus_fitted'][m] for r in group),
                                  max(r['paired_formula_minus_fitted'][m] for r in group)]
                              for m in ['rmse', 'maxabs']}})
    failures = [r['cell_id'] for r in rows if not r['adequate']]
    summary = {'study': prereg['study'], 'round': 32, 'direction_round': 2,
               'direction': prereg['direction'], 'domain': 'development', 'status': 'analyzed',
               'counts': {'planned_cells': 6, 'saved_cells': len(rows), 'condition_recipe_units': 2,
                          'coordinate_seeds_per_unit': 3, 'new_training_cells': 0, 'new_fits': 0},
               'preregistration_sha256': sha(STUDY / 'preregistration.json'),
               'preregistration_commits': sorted({r['preregistration_commit'] for r in receipts}),
               'source_hashes': prereg['source_hashes'], 'grid': prereg['evaluation']['grid'],
               'predictions': {'P1': {'status': 'refuted' if failures else 'supported',
                                      'counterexamples': failures}},
               'aggregate': aggregate, 'cells': rows, 'boundaries': prereg['boundaries'],
               'evaluation_receipt_seconds': sum(r['elapsed_seconds'] for r in receipts)}
    (STUDY / 'summary.json').write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': summary['predictions'], 'aggregate': aggregate},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
