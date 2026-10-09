from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r013_gram_condition_review'
spec = importlib.util.spec_from_file_location('measure_runner', STUDY / 'executed/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return {'mean': mean, 'min': float(values.min()), 'max': float(values.max()),
            'paired_seed_t95': [mean-half, mean+half]}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x-x.mean(), y-y.mean()
    denominator = float(np.sqrt(np.sum(x*x)*np.sum(y*y)))
    return float(np.sum(x*y)/denominator) if denominator > 0 else None


def correlations(rows, centered=False):
    columns = {name: np.asarray([r[name] for r in rows])
               for name in ('X_R', 'X_G', 'B')}
    if centered:
        for function in sorted({r['function'] for r in rows}):
            mask = np.asarray([r['function'] == function for r in rows])
            for values in columns.values():
                values[mask] -= values[mask].mean()
    r_r = correlation(columns['X_R'], columns['B'])
    r_g = correlation(columns['X_G'], columns['B'])
    return {'rayleigh_pearson_r': r_r, 'gram_pearson_r': r_g,
            'advantage': r_r-r_g if r_r is not None and r_g is not None else None}


def main():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'executed/receipt.json').read_text())
    runner.gate(receipt['preregistration_commit'], cfg)
    expected = {(fn, label, seed) for fn in cfg['functions'] for label in cfg['recipes']
                for seed in cfg['seeds']}
    measured, q_error = {}, 0.0
    for path in sorted((STUDY / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        key = tuple(row['contract'][n] for n in ('function', 'label', 'seed'))
        assert key in expected and key not in measured
        req, _ = runner.request(*key, cfg, receipt)
        runner.validate(path, req)
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        with np.load(path.with_suffix('.npz')) as data:
            assert all(np.isfinite(data[n]).all() for n in data.files)
            yc = data['centered_target']
            y = data['train_y'].astype(np.float64).ravel()
            assert np.array_equal(yc, y-y.mean())
            norm = 0.0
            for name in row['parameter_names']:
                gradient = data['vjp__' + name]
                assert runner.pin(gradient) == row['gradient_pins'][name]
                norm += float(np.sum(gradient.astype(np.float64)**2))
            denominator = len(y)*float(np.sum(yc**2))
            q = norm/denominator
            q_error = max(q_error, abs(q-row['target_rayleigh']))
            assert q > 0 and abs(q-row['target_rayleigh']) < 1e-14
            measured[key] = {'q': q, 'parameter_count': row['parameter_count'],
                             'seconds': row['seconds']}
    complete = set(measured) == expected
    old, floor_hits, gram_error = {}, 0, 0.0
    for fn in cfg['functions']:
        for label in cfg['recipes']:
            for offset in cfg['offsets']:
                for seed in cfg['seeds']:
                    path = OLD / 'results' / f'{fn}_{label}_{offset}_{seed}.json'
                    row = json.loads(path.read_text())
                    assert row['status'] == 'completed'
                    assert runner.sha(path.with_suffix('.npz')) == row['arrays_sha256']
                    with np.load(path.with_suffix('.npz')) as data:
                        h = data['hidden_256']
                        hc = h-h.mean(0, keepdims=True)
                        gram = np.einsum('ni,nj->ij', hc, hc, optimize=False)/len(h)
                        error = float(np.max(np.abs(gram-data['gram_256'])))
                        gram_error = max(gram_error, error)
                        assert error < 1e-12
                        eigenvalues = np.linalg.eigvalsh(gram)
                        floor_hits += int(eigenvalues[0] < 1e-12)
                        logk = float(np.log10(max(eigenvalues[-1], 1e-12)/max(eigenvalues[0], 1e-12)))
                        assert abs(logk-row['log10_condition']['256']) < 1e-8
                        residual = data['predictions_256'].astype(float)-data['test_y'].astype(float)
                        residual -= residual.mean()
                        e = float(np.mean(residual**2))
                    old[(fn, label, offset, seed)] = {'E': e, 'logk': logk}
    pairs, functions = [], []
    if complete:
        for fn in cfg['functions']:
            local = []
            for seed in cfg['seeds']:
                recipe_values = {}
                for label in cfg['recipes']:
                    cells = {m: old[(fn, label, m, seed)] for m in cfg['offsets']}
                    recipe_values[label] = {'chord': (cells[-3]['E']+cells[3]['E'])/2-cells[0]['E'],
                        'mean_logk': float(np.mean([cells[m]['logk'] for m in cfg['offsets']])),
                        'q': measured[(fn, label, seed)]['q']}
                ln, no = [recipe_values[label] for label in ('LN010_w64', 'noLN_w64')]
                row = {'function': fn, 'seed': seed, 'recipes': recipe_values,
                       'delta_chord': ln['chord']-no['chord'],
                       'B': no['chord']-ln['chord'],
                       'X_R': float(np.log10(ln['q']/no['q'])),
                       'X_G': no['mean_logk']-ln['mean_logk']}
                pairs.append(row)
                local.append(row)
            functions.append({'function': fn, **correlations(local),
                **{name: interval([row[name] for row in local]) for name in ('X_R', 'X_G', 'B')}})
    overall = correlations(pairs) if complete else None
    centered = correlations(pairs, True) if complete else None
    predictions = {}
    for name, values, minimum, advantage in [('P1', overall, .5, .15), ('P2', centered, .3, .1)]:
        passed = (values is not None and values['rayleigh_pearson_r'] is not None
                  and values['advantage'] is not None and values['rayleigh_pearson_r'] >= minimum
                  and values['advantage'] >= advantage)
        predictions[name] = {'status': 'supported' if passed else 'refuted' if complete else 'not_evaluated',
            'observed': values, 'required_rayleigh_r': minimum, 'required_advantage': advantage}
    loo = []
    if complete:
        for seed in cfg['seeds']:
            subset = [r for r in pairs if r['seed'] != seed]
            loo.append({'omitted_seed': seed, 'overall': correlations(subset),
                        'function_centered': correlations(subset, True)})
    result = {'study': cfg['study'], 'round': 21, 'direction_round': 3,
        'domain': 'development', 'status': 'complete' if complete else 'partial',
        'saved_measurement_cells': len(measured), 'planned_measurement_cells': 40,
        'new_training_cells': 0, 'reused_training_cells': 120,
        'condition_units': 8, 'functions': 4, 'seed_repeats': 5,
        'measurement_seconds': sum(r['seconds'] for r in measured.values()),
        'receipt': receipt, 'verification': {'maximum_saved_q_error': q_error,
            'maximum_hidden_gram_error': gram_error, 'terminal_floor_hits': floor_hits,
            'old_training_cells_verified': len(old), 'complete_measurement_grid': complete},
        'predictions': predictions, 'overall': overall, 'function_centered': centered,
        'function_summary': functions, 'same_seed_pairs': pairs,
        'descriptive_leave_one_seed_out': loo, 'boundaries': cfg['boundaries']}
    runner.save(STUDY / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'saved_measurement_cells', 'verification',
                    'predictions', 'function_summary')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
