import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    if sha(Path(__file__)) != prereg['analysis_source_sha256']:
        raise RuntimeError('Analysis source differs from preregistration')
    for relative, expected in prereg['input_hashes'].items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f'Input changed: {relative}')
    receipts = [json.loads(path.read_text()) for path in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    metadata_pins = {}
    for receipt in receipts:
        for cell in receipt['new_cells'] + receipt['resumed_cells']:
            if metadata_pins.get(cell['cell_id'], cell['metadata_sha256']) != cell['metadata_sha256']:
                raise RuntimeError('Conflicting metadata pins')
            metadata_pins[cell['cell_id']] = cell['metadata_sha256']
    rows = []
    for item in prereg['inputs']:
        folder = STUDY / 'results' / item['cell_id']
        metadata = json.loads((folder / 'metadata.json').read_text())
        if (sha(folder / 'metadata.json') != metadata_pins.get(item['cell_id'])
                or sha(folder / 'arrays.npz') != metadata['arrays_sha256']
                or metadata['request']['input'] != item
                or metadata['request']['preregistration_sha256'] != sha(STUDY / 'preregistration.json')
                or metadata['request']['source_sha256'] != prereg['execution_source_sha256']):
            raise RuntimeError('Result contract or hash mismatch')
        with np.load(folder / 'arrays.npz') as arrays:
            grid, observed = arrays['grid'].copy(), arrays['observed'].copy()
            ls, mm = arrays['least_squares_prediction'].copy(), arrays['minimax_prediction'].copy()
            parameters = {name: arrays[name + '_parameters'].tolist() for name in ['least_squares', 'minimax']}
        if grid.tolist() != prereg['grid']:
            raise RuntimeError('Grid changed')
        errors = {}
        for name, prediction in [('least_squares', ls), ('minimax', mm)]:
            residual = prediction - observed
            errors[name] = {'rmse': float(np.sqrt(np.mean(residual**2))),
                            'maxabs': float(np.max(np.abs(residual))),
                            'maximum_residual_step': int(grid[np.argmax(np.abs(residual))])}
        if any(abs(errors['least_squares'][m] - metadata['old_errors'][m]) > 1e-12 for m in ['rmse', 'maxabs']):
            raise RuntimeError('Saved LS comparator differs from original summary')
        delta = {m: errors['minimax'][m] - errors['least_squares'][m] for m in ['rmse', 'maxabs']}
        rows.append({'cell_id': item['cell_id'], 'alpha': item['alpha'], 'seed': item['seed'],
                     'errors': errors, 'parameters_log_half_beta': parameters,
                     'paired_minimax_minus_ls': delta, 'certificate': {k: v for k, v in metadata['certificate'].items() if k != 'trace'},
                     'adequate': errors['minimax']['rmse'] <= .03 and errors['minimax']['maxabs'] <= .05})
    aggregates = []
    for alpha in prereg['alphas']:
        group = [r for r in rows if r['alpha'] == alpha]
        aggregates.append({'alpha': alpha, 'adequate_cells': sum(r['adequate'] for r in group),
                           'errors_seed_ranges': {name: {m: [min(r['errors'][name][m] for r in group), max(r['errors'][name][m] for r in group)] for m in ['rmse', 'maxabs']} for name in ['least_squares', 'minimax']},
                           'paired_seed_ranges': {m: [min(r['paired_minimax_minus_ls'][m] for r in group), max(r['paired_minimax_minus_ls'][m] for r in group)] for m in ['rmse', 'maxabs']},
                           'parameter_seed_ranges': {name: {'t50': [min(float(np.exp(r['parameters_log_half_beta'][name][0])) for r in group), max(float(np.exp(r['parameters_log_half_beta'][name][0])) for r in group)],
                                                                    'beta': [min(r['parameters_log_half_beta'][name][1] for r in group), max(r['parameters_log_half_beta'][name][1] for r in group)]} for name in ['least_squares', 'minimax']}})
    predicates = {
        'P1': lambda r: r['adequate'],
        'P2': lambda r: r['paired_minimax_minus_ls']['maxabs'] <= -.015 and r['paired_minimax_minus_ls']['rmse'] >= .003,
        'P3': lambda r: (.025 if r['alpha'] == 0 else .020) <= r['errors']['minimax']['maxabs'] <= .045,
    }
    predictions = {name: {'status': 'refuted' if any(not predicate(r) for r in rows) else 'supported',
                          'counterexamples': [r['cell_id'] for r in rows if not predicate(r)]} for name, predicate in predicates.items()}
    summary = {'study': prereg['study'], 'round': 38, 'direction_round': 4,
               'direction': prereg['direction'], 'domain': 'development', 'status': 'analyzed',
               'training_started': False, 'new_training_cells': 0, 'new_fitting_cells': len(rows),
               'counts': {'recipe_units': 2, 'coordinate_seeds_per_unit': 3, 'saved_cells': len(rows)},
               'preregistration_sha256': sha(STUDY / 'preregistration.json'),
               'execution_source_sha256': prereg['execution_source_sha256'],
               'analysis_source_sha256': prereg['analysis_source_sha256'],
               'execution_heads': sorted({r['execution_head'] for r in receipts}),
               'successful_call_seconds': sum(r['elapsed_seconds'] for r in receipts),
               'grid': prereg['grid'], 'predictions': predictions, 'aggregate': aggregates, 'cells': rows,
               'boundaries': prereg['boundaries']}
    (STUDY / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': predictions, 'aggregate': aggregates}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
