from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', STUDY / 'executed/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def interval(values):
    a = np.asarray(values, dtype=np.float64)
    half = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return {'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max()),
            'paired_seed_t95': [float(a.mean()-half), float(a.mean()+half)]}


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x-x.mean(), y-y.mean()
    d = float(np.sqrt(np.sum(x*x)*np.sum(y*y)))
    return float(np.sum(x*y)/d) if d > 0 else None


def correlations(rows, centered=False):
    c = {n: np.asarray([r[n] for r in rows]) for n in ('X_L', 'X_R', 'B')}
    if centered:
        for fn in sorted({r['function'] for r in rows}):
            mask = np.asarray([r['function'] == fn for r in rows])
            for a in c.values():
                a[mask] -= a[mask].mean()
    rl, rr = corr(c['X_L'], c['B']), corr(c['X_R'], c['B'])
    return {'coupling_pearson_r': rl, 'rayleigh_pearson_r': rr,
            'advantage': rl-rr if rl is not None and rr is not None else None,
            'raw_ln_over_noln_log_ratio_pearson_r': -rl if rl is not None else None}


def main():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'executed/receipt.json').read_text())
    runner.gate(receipt['preregistration_commit'], cfg)
    expected = {(fn, label, seed) for fn in cfg['functions']
                for label in cfg['recipes'] for seed in cfg['seeds']}
    measured, maximum_error = {}, 0.0
    for path in sorted((STUDY / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        key = tuple(row['contract'][n] for n in ('function', 'label', 'seed'))
        assert key in expected and key not in measured
        req, _ = runner.request(*key, cfg, receipt)
        runner.validate(path, req)
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        with np.load(path.with_suffix('.npz')) as a:
            assert all(np.isfinite(a[n]).all() for n in a.files)
            v = a['kernel_ones']
            u = v-v.mean()
            assert np.array_equal(u, a['centered_kernel_ones'])
            assert runner.pin(v) == row['kernel_ones_pin']
            assert runner.pin(u) == row['centered_kernel_ones_pin']
            for name in row['parameter_names']:
                assert runner.pin(a['vjp_ones__'+name]) == row['gradient_pins'][name]
            ell = float(np.mean(u*u))
            maximum_error = max(maximum_error, abs(ell-row['mean_to_centered_coupling']))
            assert ell > 0 and ell == row['mean_to_centered_coupling']
        measured[key] = {'ell': ell, 'seconds': row['seconds']}
    complete = set(measured) == expected
    prior = json.loads((runner.RAYLEIGH / 'summary.json').read_text())
    pairs, functions = [], []
    chord_error = rayleigh_error = 0.0
    if complete:
        for fn in cfg['functions']:
            local = []
            for seed in cfg['seeds']:
                values = {}
                for label in cfg['recipes']:
                    losses = []
                    for m in cfg['offsets']:
                        path = runner.OLD / 'results' / f'{fn}_{label}_{m}_{seed}.npz'
                        with np.load(path) as a:
                            residual = a['predictions_256'].astype(float)-a['test_y'].astype(float)
                            residual -= residual.mean()
                            losses.append(float(np.mean(residual**2)))
                    qpath = runner.RAYLEIGH / 'results' / f'{fn}_{label}_{seed}.json'
                    qrow = json.loads(qpath.read_text())
                    assert runner.sha(qpath.with_suffix('.npz')) == qrow['arrays_sha256']
                    with np.load(qpath.with_suffix('.npz')) as a:
                        norm = sum(float(np.sum(a['vjp__'+n].astype(float)**2))
                                   for n in qrow['parameter_names'])
                        yc = a['centered_target']
                        q = norm/(len(yc)*float(np.sum(yc**2)))
                    assert abs(q-qrow['target_rayleigh']) < 1e-14
                    values[label] = {'C': .5*(losses[0]+losses[2])-losses[1],
                                     'q': q, 'ell': measured[(fn, label, seed)]['ell']}
                ln, no = values['LN010_w64'], values['noLN_w64']
                raw = float(np.log10(ln['ell']/no['ell']))
                row = {'function': fn, 'seed': seed, 'recipes': values,
                       'B': no['C']-ln['C'], 'delta_chord': ln['C']-no['C'],
                       'log10_ell_ln_over_noln': raw, 'ell_ln_over_noln': ln['ell']/no['ell'],
                       'X_L': -raw, 'X_R': float(np.log10(ln['q']/no['q']))}
                old = next(r for r in prior['same_seed_pairs'] if r['function']==fn and r['seed']==seed)
                chord_error = max(chord_error, abs(row['B']-old['B']))
                rayleigh_error = max(rayleigh_error, abs(row['X_R']-old['X_R']))
                assert chord_error < 1e-10 and rayleigh_error < 1e-10
                pairs.append(row)
                local.append(row)
            functions.append({'function': fn, **correlations(local),
                **{n: interval([r[n] for r in local]) for n in ('X_L', 'X_R', 'B', 'ell_ln_over_noln')}})
    overall = correlations(pairs) if complete else None
    centered = correlations(pairs, True) if complete else None
    predictions = {}
    for name, observed, threshold, gap in [('P1', overall, .5, .15), ('P2', centered, .3, .1)]:
        passed = (observed is not None and observed['coupling_pearson_r'] is not None
                  and observed['advantage'] is not None and observed['coupling_pearson_r'] >= threshold
                  and observed['advantage'] >= gap)
        predictions[name] = {'status': 'supported' if passed else 'refuted' if complete else 'not_evaluated',
                             'observed': observed, 'required_r': threshold, 'required_advantage': gap}
    result = {'study': cfg['study'], 'round': 45, 'direction_round': 4, 'domain': 'development',
        'status': 'complete' if complete else 'partial', 'saved_measurement_cells': len(measured),
        'planned_measurement_cells': 40, 'new_training_cells': 0, 'reused_training_cells': 120,
        'reused_rayleigh_cells': 40, 'condition_units': 8, 'functions': 4, 'seed_repeats': 5,
        'measurement_seconds': sum(r['seconds'] for r in measured.values()), 'receipt': receipt,
        'verification': {'maximum_saved_ell_error': maximum_error,
                         'maximum_old_B_error': chord_error, 'maximum_old_X_R_error': rayleigh_error},
        'predictions': predictions, 'overall': overall, 'function_centered': centered,
        'function_summary': functions, 'same_seed_pairs': pairs,
        'descriptive_leave_one_seed_out': [{'omitted_seed': seed,
            'overall': correlations([r for r in pairs if r['seed'] != seed]),
            'function_centered': correlations([r for r in pairs if r['seed'] != seed], True)}
            for seed in cfg['seeds']] if complete else [], 'boundaries': cfg['boundaries']}
    runner.save(STUDY / 'summary.json', result)
    print(json.dumps({n: result[n] for n in ('status', 'saved_measurement_cells', 'predictions',
                                            'function_summary', 'measurement_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
