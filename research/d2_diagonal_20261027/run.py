#!/usr/bin/env python3
"""One preregistered ablation of immutable, previously saved D2 data."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import importlib.util
import io
import contextlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from collections import defaultdict

import numpy as np
import scipy
from numerics import base_grid, diagonal_bias, select_integer, time_factor, quantiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / 'preregistration.json').read_text())
DATA_ROOT = Path(os.environ.get('AIQ_KB_DATA_ROOT', SPEC['data_root'])).resolve()
OUT = DATA_ROOT / SPEC['run_relative']
INPUT = Path(SPEC['input_root'])
RELATIVE = Path(SPEC['source_study'])
OLD = ROOT / RELATIVE
NEW_INPUT = INPUT / RELATIVE
module_spec = importlib.util.spec_from_file_location('legacy_d2', ROOT / 'research/d2_scaling_20261009/run.py')
legacy = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(legacy)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def contract():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = [HERE / name for name in ['preregistration.json', 'run.py', 'numerics.py', 'test_numerics.py', 'test_feature_space.py']]
    paths += [ROOT / 'research/d2_scaling_20261009/run.py',
              ROOT / 'research/d2_scaling_20261009/preregistration.json',
              ROOT / 'evidence/manifests/d2.json', ROOT / 'evidence/handoff_manifest.json']
    hashes = {}
    for p in paths:
        stored = subprocess.check_output(['git', 'show', f'{head}:{p.relative_to(ROOT)}'], cwd=ROOT)
        if stored != p.read_bytes():
            raise RuntimeError(f'Uncommitted contract: {p.relative_to(ROOT)}')
        hashes[str(p.relative_to(ROOT))] = digest(p)
    return {'commit': head, 'files': hashes}


def verify_inputs():
    manifest = json.loads((ROOT / 'evidence/manifests/d2.json').read_text())
    records = []
    for row in manifest['files']:
        p = (INPUT if row['delivery'] == 'release' else ROOT) / row['path']
        if not p.is_file() or p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
            raise RuntimeError(f'Changed input: {row["path"]}')
        records.append({'path': row['path'], 'sha256': row['sha256'], 'delivery': row['delivery']})
    data = json.loads((OLD / 'summary.json').read_text())
    rows = data['rows']
    actual = dict(curves=len(rows), draws=len(set(r['data'] for r in rows)),
                  recipes=len(set((r['protocol'], r['task']) for r in rows)),
                  interior=sum(r['interior'] for r in rows), boundary=sum(not r['interior'] for r in rows))
    if actual != SPEC['expected_counts']:
        raise RuntimeError(f'Denominator mismatch: {actual}')
    receipt_files = {}
    for p in sorted((NEW_INPUT / 'results').glob('*.json')):
        receipt = json.loads(p.read_text())
        for rel, pin in receipt['files'].items():
            if digest(NEW_INPUT / rel) != pin:
                raise RuntimeError(f'Receipt mismatch: {rel}')
            receipt_files[rel] = pin
    if len(receipt_files) != 3360:
        raise RuntimeError('Receipt denominator mismatch')
    result = {'status': 'verified', 'manifest_sha256': digest(ROOT / 'evidence/manifests/d2.json'),
              'files_checked': len(records), 'receipt_payloads_checked': len(receipt_files),
              'counts': actual, 'files': records}
    write_json(OUT / 'input_verification.json', result)
    return rows


def metric_summary(rows):
    interior = [r for r in rows if r['reference_interior']]
    temporal = sum(r['time_pass'] for r in interior)
    low = sum(r['risk_pass'] for r in rows)
    return {'count': len(rows), 'interior': len(interior), 'boundary': len(rows)-len(interior),
            'time_pass': temporal, 'time_rate': temporal/len(interior) if interior else None,
            'risk_pass': low, 'risk_rate': low/len(rows),
            'stopping_pass': bool(interior and temporal/len(interior) >= .8 and low/len(rows) >= .9),
            'invalid_bias': sum(r['invalid_bias'] for r in rows),
            'invalid_total': sum(r['invalid_total'] for r in rows),
            'valid_and_risk_pass': sum(r['risk_pass'] and not r['invalid_bias'] and not r['invalid_total'] for r in rows),
            'predicted_boundary': sum(not r['predicted_interior'] for r in rows),
            'boundary_status_agreement': sum(r['reference_interior'] == r['predicted_interior'] for r in rows),
            'boundary_confusion': {f'ref_{ref}_pred_{pred}': sum((not r['reference_interior']) == ref and (not r['predicted_interior']) == pred for r in rows) for ref in (False, True) for pred in (False, True)},
            'time_factor_positive_predictions': quantiles([r['time_factor'] for r in interior if r['time_factor'] is not None]),
            'time_zero_predictions_on_interior': sum(r['t_diagonal'] == 0 for r in interior),
            'regret_over_initial': quantiles([r['regret_over_initial'] for r in rows]),
            'omitted_cross_max_abs_over_initial': quantiles([r['cross_max_abs_over_initial'] for r in rows]),
            'baseline_time_pass': sum(r['baseline_time_pass'] for r in interior),
            'baseline_risk_pass': sum(r['baseline_risk_pass'] for r in rows),
            'baseline_clipped': sum(r['baseline_clipped'] for r in rows),
            'baseline_regret_over_initial': quantiles([r['baseline_regret_over_initial'] for r in rows])}


def independent_feature_check(data, model, terms, t, sigma2, expected):
    f, _ = legacy.response(model, [t])
    # Reconstruct predictions in original feature coordinates, not h/linear space.
    w = model['vectors'] @ (f[0, :, None] * terms['g'])
    bias = float(np.mean((data['val_phi'] @ w - terms['vy'])**2))
    response = (model['vectors'] * f[0]) @ model['vectors'].T @ data['phi'].T / len(data['phi'])
    noise = float(np.sum((data['val_phi'] @ response)**2)/len(data['val_phi']))
    direct = bias + sigma2*noise
    return {'t': int(t), 'raw_feature_risk': direct, 'spectral_risk': expected,
            'absolute_error': abs(direct-expected)}


def main():
    frozen = contract()
    if OUT.exists():
        raise RuntimeError(f'Run directory already exists; refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    (OUT / 'preregistration.json').write_bytes((HERE / 'preregistration.json').read_bytes())
    started = time.time()
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    execution = {'status': 'verifying_inputs', 'contract': frozen, 'started_unix': started,
                 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                 'platform': platform.platform(), 'blas': blas.getvalue(),
                 'source_summary_sha256': digest(OLD / 'summary.json'), 'model_calls': 0}
    write_json(OUT / 'execution.json', execution)
    try:
        original = verify_inputs()
        execution['status'] = 'running'
        execution['compute_started_unix'] = time.time()
        write_json(OUT / 'execution.json', execution)
        compute_start = time.monotonic()
        grouped = defaultdict(list)
        for r in original:
            grouped[r['data']].append(r)
        grid = base_grid()
        output_rows, checks, spacings = [], [], []
        max_curve_error = 0.
        with (OUT / 'rows.jsonl').open('x') as sink:
            for draw_index, (data_name, original_rows) in enumerate(sorted(grouped.items())):
                if time.monotonic()-compute_start > SPEC['budget']['compute_wall_seconds']:
                    raise RuntimeError('Compute budget reached')
                with np.load(NEW_INPUT / data_name, allow_pickle=False) as z:
                    data = {k: z[k] for k in z.files}
                model = legacy.spectral_model(data)
                active_lam = model['lam'][model['active']]
                spacing = float(np.min(np.diff(active_lam))/np.max(active_lam)) if len(active_lam)>1 else None
                spacings.append({'data': data_name, 'rank': model['rank'], 'minimum_relative_spacing': spacing,
                                 'near_degenerate': spacing is not None and spacing <= 1e-10})
                f, _ = legacy.response(model, grid)
                task_groups = defaultdict(list)
                for r in original_rows:
                    task_groups[r['task']].append(r)
                for task, task_rows in sorted(task_groups.items()):
                    terms = legacy.target_terms(model, data, task)
                    b, noise, _, _ = legacy.expected_curves(model, terms, grid)
                    bd = diagonal_bias(f, terms['h'], terms['linear'], terms['val_y2'])
                    if not np.all(np.isfinite(bd)):
                        raise RuntimeError('Nonfinite diagonal curve')
                    for old in sorted(task_rows, key=lambda r: r['sigma2']):
                        sigma2 = old['sigma2']
                        scale = max(1., old['risk_initial'])
                        with np.load(NEW_INPUT / old['curve'], allow_pickle=False) as z:
                            idx = np.searchsorted(z['t'], grid)
                            if not np.array_equal(z['t'][idx], grid):
                                raise RuntimeError('Saved common grid missing')
                            error = max(float(np.max(np.abs(b-z['signal_bias'][idx]))),
                                        float(np.max(np.abs(noise-z['variance_unit'][idx]))),
                                        float(np.max(np.abs(b+sigma2*noise-z['expected_clean_val'][idx]))))/scale
                        max_curve_error = max(max_curve_error, error)
                        if error > 1e-8:
                            raise RuntimeError(f'Full reconstruction differs: {old["curve"]}: {error}')
                        def diag_objective(t):
                            ft, _ = legacy.response(model, t)
                            bn = diagonal_bias(ft, terms['h'], terms['linear'], terms['val_y2'])
                            nn = (ft*ft) @ (np.diag(model['hval'])*model['lam']/model['n'])
                            return bn + sigma2*nn
                        t = select_integer(grid, bd+sigma2*noise, diag_objective)
                        baseline_unclipped = int(round(old['linear_prediction_U']/legacy.SPEC['eta']))
                        bt = int(np.clip(baseline_unclipped, 0, grid[-1]))
                        positions = [t, old['t_star'], bt]
                        bx, nx, _, _ = legacy.expected_curves(model, terms, positions)
                        rx = bx+sigma2*nx
                        if abs(rx[1]-old['risk_minimum']) > 1e-8*scale:
                            raise RuntimeError('Saved reference minimum risk mismatch')
                        raw_regret = float(rx[0]-old['risk_minimum'])
                        base_regret = float(rx[2]-old['risk_minimum'])
                        if min(raw_regret, base_regret) < -1e-8*scale:
                            raise RuntimeError(f'Search-discrepancy: {old["curve"]}, regrets={raw_regret},{base_regret}')
                        ft, _ = legacy.response(model, [t])
                        bd_at = float(diagonal_bias(ft, terms['h'], terms['linear'], terms['val_y2'])[0])
                        min_bias = min(float(bd.min()), bd_at)
                        min_total = min(float((bd+sigma2*noise).min()), bd_at+sigma2*float(nx[0]))
                        factor = time_factor(t, old['t_star']) if old['interior'] else None
                        baseline_factor = time_factor(old['linear_prediction_U'], old['U_rise']) if old['interior'] else None
                        regret_normalized = max(0., raw_regret)/old['risk_initial']
                        base_normalized = max(0., base_regret)/old['risk_initial']
                        row = {k: old[k] for k in ('protocol','task','n','seed','sigma2','curve','data')}
                        row.update({'t_reference': old['t_star'], 't_diagonal': t, 't_baseline': bt,
                                    'reference_interior': old['interior'], 'predicted_interior': bool(0<t<grid[-1]),
                                    'risk_initial': old['risk_initial'], 'risk_reference': old['risk_minimum'],
                                    'full_risk_at_diagonal': float(rx[0]), 'diagonal_risk_at_diagonal': bd_at+sigma2*float(nx[0]),
                                    'regret_raw': raw_regret, 'regret_over_initial': regret_normalized,
                                    'time_factor': factor, 'time_pass': bool(old['interior'] and factor is not None and factor<=2),
                                    'risk_pass': bool(regret_normalized<=.01),
                                    'min_diagonal_bias': min_bias, 'min_diagonal_risk': min_total,
                                    'invalid_bias': bool(min_bias < -1e-10*scale), 'invalid_total': bool(min_total < -1e-10*scale),
                                    'cross_max_abs_over_initial': float(np.max(np.abs(b-bd))/old['risk_initial']),
                                    'cross_at_diagonal_over_initial': float((bx[0]-bd_at)/old['risk_initial']),
                                    'baseline_original_U': old['linear_prediction_U'], 'baseline_time_factor': baseline_factor,
                                    'baseline_time_pass': bool(old['interior'] and baseline_factor is not None and baseline_factor<=2),
                                    'baseline_clipped': baseline_unclipped != bt, 'baseline_risk_pass': bool(base_normalized<=.01),
                                    'baseline_regret_raw': base_regret, 'baseline_regret_over_initial': base_normalized})
                        sink.write(json.dumps(row, allow_nan=False)+'\n')
                        output_rows.append(row)
                        if old['n']==64 and old['seed']==6101 and sigma2==1 and task in ('interaction','vector','quadratic','linear'):
                            record = {k: old[k] for k in ('protocol','task','n','seed','sigma2')}
                            record['checks'] = [independent_feature_check(data,model,terms,int(p),sigma2,float(v)) for p,v in zip(positions[:2],rx[:2])]
                            for c in record['checks']:
                                if c['absolute_error']/scale > 1e-8:
                                    raise RuntimeError('Independent feature-space gate failed')
                            checks.append(record)
                sink.flush()
                print(json.dumps({'draws': draw_index+1, 'of':len(grouped), 'curves':len(output_rows),
                                  'compute_seconds':round(time.monotonic()-compute_start,1)}), flush=True)
        if len(output_rows)!=3240 or len(checks)!=8:
            raise RuntimeError('Output or independent-check denominator mismatch')
        recipes = sorted(set((r['protocol'],r['task']) for r in output_rows))
        by_recipe = [{'protocol':p, 'task':t, **metric_summary([r for r in output_rows if (r['protocol'],r['task'])==(p,t)])} for p,t in recipes]
        by_seed = [{'seed':s, **metric_summary([r for r in output_rows if r['seed']==s])} for s in (6101,6102,6103)]
        by_recipe_seed = [{'protocol':p,'task':t,'seed':s,**metric_summary([r for r in output_rows if (r['protocol'],r['task'],r['seed'])==(p,t,s)])} for p,t in recipes for s in (6101,6102,6103)]
        overall = metric_summary(output_rows)
        overall['pooled_stopping_pass'] = overall.pop('stopping_pass')
        summary = {'study':SPEC['study'], 'contract':frozen, 'domain':'development',
                   'overall':overall, 'recipes':by_recipe, 'seeds':by_seed,
                   'recipe_seeds':by_recipe_seed, 'recipes_passing_both':sum(r['stopping_pass'] for r in by_recipe),
                   'hypothesis_status':'supported_on_development' if all(r['stopping_pass'] for r in by_recipe) else 'refuted_on_development',
                   'near_degenerate_draws':sum(s['near_degenerate'] for s in spacings), 'spectral_spacing':spacings,
                   'rows_sha256':digest(OUT / 'rows.jsonl'), 'compute_seconds':time.monotonic()-compute_start}
        write_json(OUT / 'summary.json', summary)
        write_json(OUT / 'numerical_verification.json', {'status':'verified', 'common_grid_curves_checked':3240,
                   'maximum_normalized_saved_curve_error':max_curve_error,'independent_cells':len(checks),
                   'maximum_independent_absolute_error':max(c['absolute_error'] for r in checks for c in r['checks']), 'checks':checks})
        if contract() != frozen:
            raise RuntimeError('Source contract changed during execution')
        execution.update(status='complete', finished_unix=time.time(), compute_seconds=summary['compute_seconds'])
        write_json(OUT / 'execution.json', execution)
        print(json.dumps({'output':str(OUT),'status':summary['hypothesis_status'], 'overall':summary['overall']},ensure_ascii=False))
    except Exception as exc:
        execution.update(status='failed', finished_unix=time.time(), error=str(exc))
        write_json(OUT / 'execution.json', execution)
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
