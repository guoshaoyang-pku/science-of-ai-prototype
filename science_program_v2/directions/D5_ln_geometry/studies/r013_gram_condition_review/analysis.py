from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
from scipy.stats import pearsonr, t

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return {'n': len(values), 'mean': mean, 'min': float(values.min()),
            'max': float(values.max()), 'paired_seed_t95': [mean-half, mean+half]}


def correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def main():
    cfg = json.loads((STUDY/'preregistration.json').read_text())
    receipt = json.loads((STUDY/'executed/receipt.json').read_text())
    commit = receipt['preregistration_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for relative in ('preregistration.json', 'executed/run.py', 'analysis.py'):
        path = STUDY/relative
        saved = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if saved != path.read_bytes():
            raise RuntimeError('committed source mismatch: '+relative)
        if relative in cfg['source_sha256'] and sha(path) != cfg['source_sha256'][relative]:
            raise RuntimeError('pinned source hash mismatch')
    if receipt['preregistration_sha256'] != sha(STUDY/'preregistration.json'):
        raise RuntimeError('receipt preregistration mismatch')
    expected = {(fn, label, mean, seed) for fn in cfg['functions'] for label in cfg['recipes']
                for mean in cfg['means'] for seed in cfg['seeds']}
    rows, raw, seen = [], {}, set()
    gram_error = eigen_error = head_prediction_error = 0.0
    all_input_pins = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        req = row['contract']
        key = tuple(req[name] for name in ('function', 'label', 'mean', 'seed'))
        if key not in expected or key in seen:
            raise RuntimeError('unexpected or duplicate cell')
        seen.add(key)
        if (req['recipe'] != cfg['recipes'][req['label']] or
                req['intervention'] != cfg['intervention'] or
                req['preregistration_commit'] != commit or
                req['preregistration_sha256'] != sha(STUDY/'preregistration.json') or
                req['source_sha256'] != cfg['source_sha256'] or
                row['contract_sha256'] != digest(req) or row['status'] != 'completed'):
            raise RuntimeError('cell contract/status mismatch')
        if row['started_at_epoch'] < receipt['gate_passed_at_epoch']:
            raise RuntimeError('training precedes execution gate')
        if sha(path.with_suffix('.npz')) != row['arrays_sha256']:
            raise RuntimeError('NPZ hash mismatch')
        conditions, spectra, errors = {}, {}, {}
        with np.load(path.with_suffix('.npz')) as arrays:
            for name, pin in req['inputs'].items():
                if array_pin(arrays[name]) != pin:
                    raise RuntimeError('input tensor pin mismatch')
            all_input_pins[key] = req['inputs']
            if array_pin(arrays['batch_indices'])['sha256'] != row['batch_sha256']:
                raise RuntimeError('minibatch hash mismatch')
            for role in ('weight', 'bias'):
                if not np.array_equal(arrays[f'head_{role}_initial'], arrays[f'head_{role}_final']):
                    raise RuntimeError('frozen head moved')
            for step in cfg['checkpoints']:
                gram = arrays[f'gram_{step}']
                if gram.dtype != np.float64 or not np.isfinite(gram).all():
                    raise RuntimeError('Gram dtype or finite audit failed')
                eig = np.linalg.eigvalsh(gram)[::-1]
                error = float(np.max(np.abs(eig-np.asarray(row['spectrum'][str(step)]))))
                eigen_error = max(eigen_error, error)
                if error > 1e-12 or eig[-1] < -1e-12:
                    raise RuntimeError('spectrum audit failed')
                floor = cfg['mediator']['eigenvalue_floor']
                value = float(np.log10(max(eig[0], floor)/max(eig[-1], floor)))
                if abs(value-row['log10_condition'][str(step)]) > 1e-9:
                    raise RuntimeError('condition number audit failed')
                conditions[str(step)] = value
                spectra[str(step)] = {'lambda_max': float(eig[0]), 'lambda_min': float(eig[-1]),
                                      'floor_hits': int(np.sum(eig < floor))}
                residual = arrays[f'predictions_{step}'].astype(float)-arrays['test_y'].astype(float)
                if not np.isfinite(residual).all():
                    raise RuntimeError('nonfinite residual')
                errors[str(step)] = float(np.mean((residual-residual.mean())**2))
                if step in (0, cfg['recipes'][req['label']]['steps']):
                    hidden = arrays[f'hidden_{step}']
                    centered = hidden-hidden.mean(0, keepdims=True)
                    recomputed = centered.T@centered/len(hidden)
                    mismatch = float(np.max(np.abs(recomputed-gram)))
                    gram_error = max(gram_error, mismatch)
                    if mismatch > 1e-12:
                        raise RuntimeError('saved hidden/Gram mismatch')
                    prediction = hidden@arrays['head_weight_final'].astype(float).T+arrays['head_bias_final']
                    prediction_error = float(np.max(np.abs(prediction-arrays[f'train_predictions_{step}'])))
                    head_prediction_error = max(head_prediction_error, prediction_error)
                    if prediction_error > 1e-5:
                        raise RuntimeError('hidden/fixed-head prediction mismatch')
            for name in ('train_x', 'test_x'):
                pin_key = (req['function'], name)
                if pin_key in raw and raw[pin_key] != req['inputs'][name]:
                    raise RuntimeError('unpaired input dataset')
                raw[pin_key] = req['inputs'][name]
        rows.append({'function': req['function'], 'label': req['label'], 'mean': req['mean'],
                     'seed': req['seed'], 'log10_condition': conditions, 'spectrum_audit': spectra,
                     'centered_error': errors, 'seconds': row['seconds'],
                     'started_at_epoch': row['started_at_epoch']})
        raw[key] = row
    pairing_checks = 0
    for fn in cfg['functions']:
        for seed in cfg['seeds']:
            available = [key for key in seen if key[0] == fn and key[3] == seed]
            if len(available) > 1:
                first = raw[available[0]]
                for key in available[1:]:
                    if (raw[key]['initial_linear'] != first['initial_linear'] or
                            raw[key]['batch_sha256'] != first['batch_sha256']):
                        raise RuntimeError('unpaired linear initialization/minibatch stream')
                    pairing_checks += 1
        for mean in cfg['means']:
            available = [key for key in seen if key[0] == fn and key[2] == mean]
            if len(available) > 1 and any(all_input_pins[key] != all_input_pins[available[0]]
                                          for key in available[1:]):
                raise RuntimeError('unpaired labels at same offset')
    by_key = {(row['function'], row['label'], row['mean'], row['seed']): row for row in rows}
    complete = seen == expected
    spectrum_contrasts, chord_contrasts, function_summary = [], [], []
    if complete:
        for fn in cfg['functions']:
            for mean in cfg['means']:
                deltas = [by_key[(fn, 'LN010_w64', mean, seed)]['log10_condition']['256']-
                          by_key[(fn, 'noLN_w64', mean, seed)]['log10_condition']['256']
                          for seed in cfg['seeds']]
                negative = int(np.sum(np.asarray(deltas) < 0))
                spectrum_contrasts.append({'function': fn, 'mean': mean,
                    'LN_minus_noLN_log10_condition': deltas, 'negative_seeds': negative,
                    'passes': negative >= 3, **interval(deltas)})
            local = []
            for seed in cfg['seeds']:
                checkpoints = {}
                for step in cfg['checkpoints']:
                    values = {}
                    for label in cfg['recipes']:
                        errors = {mean: by_key[(fn, label, mean, seed)]['centered_error'][str(step)]
                                  for mean in cfg['means']}
                        values[label] = (errors[-3]+errors[3])/2-errors[0]
                    k_ln = by_key[(fn, 'LN010_w64', 0, seed)]['log10_condition'][str(step)]
                    k_no = by_key[(fn, 'noLN_w64', 0, seed)]['log10_condition'][str(step)]
                    checkpoints[str(step)] = {'LN_chord': values['LN010_w64'],
                        'noLN_chord': values['noLN_w64'],
                        'delta_chord': values['LN010_w64']-values['noLN_w64'],
                        'delta_log10_condition_mean0': k_ln-k_no}
                contrast = {'function': fn, 'seed': seed, 'checkpoints': checkpoints,
                            **checkpoints['256']}
                local.append(contrast)
                chord_contrasts.append(contrast)
            function_summary.append({'function': fn,
                'delta_chord': interval([row['delta_chord'] for row in local]),
                'delta_log10_condition_mean0': interval([row['delta_log10_condition_mean0'] for row in local]),
                'LN_chord': interval([row['LN_chord'] for row in local]),
                'noLN_chord': interval([row['noLN_chord'] for row in local]),
                'within_function_pearson_r': correlation(
                    [row['delta_log10_condition_mean0'] for row in local],
                    [row['delta_chord'] for row in local])})
    corr = correlation([row['delta_log10_condition_mean0'] for row in chord_contrasts],
                       [row['delta_chord'] for row in chord_contrasts])
    centered_x, centered_y = [], []
    for fn in cfg['functions']:
        local = [row for row in chord_contrasts if row['function'] == fn]
        if local:
            x = np.asarray([row['delta_log10_condition_mean0'] for row in local])
            y = np.asarray([row['delta_chord'] for row in local])
            centered_x.extend(x-x.mean())
            centered_y.extend(y-y.mean())
    passes = sum(row['passes'] for row in spectrum_contrasts)
    predictions = {
        'P1': {'status': 'supported' if complete and passes >= 8 else
                         'refuted' if complete else 'not_evaluated',
               'passing_function_offset_units': passes, 'total_units': 12,
               'rule': '至少8/12单元，每单元至少3/5 seed的终点LN−noLN log10κ<0'},
        'P2': {'status': 'supported' if complete and corr is not None and corr >= .5 else
                         'refuted' if complete else 'not_evaluated',
               'pearson_r': corr, 'paired_repeats': len(chord_contrasts),
               'rule': '20函数×seed配对，offset0终点Δlog10κ与Δchord的Pearson r≥.5；无方差记失败'}
    }
    result = {'study': cfg['study'], 'round': 13, 'direction_round': 2,
        'direction': cfg['direction'], 'domain': cfg['domain'],
        'status': 'measurements_complete' if complete else 'partial',
        'saved_cells': len(rows), 'planned_cells': len(expected),
        'condition_units': 24, 'same_seed_LN_pairs': sum(
            all((fn, label, mean, seed) in seen for label in cfg['recipes'])
            for fn in cfg['functions'] for mean in cfg['means'] for seed in cfg['seeds']),
        'experiment_seconds': sum(row['seconds'] for row in rows),
        'receipt': receipt, 'verification': {'status': 'passed', 'verified_cells': len(rows),
            'initial_linear_and_batch_comparisons': pairing_checks,
            'maximum_hidden_gram_abs_error': gram_error, 'maximum_eigenvalue_abs_error': eigen_error,
            'maximum_hidden_head_prediction_abs_error': head_prediction_error,
            'floor_hit_checkpoint_count': sum(s['floor_hits'] > 0 for row in rows
                                             for s in row['spectrum_audit'].values()),
            'complete_grid': complete},
        'predictions': predictions, 'spectrum_contrasts': spectrum_contrasts,
        'descriptive_within_function_centered_pearson_r': correlation(centered_x, centered_y),
        'chord_contrasts': chord_contrasts, 'function_summary': function_summary,
        'rows': rows, 'boundaries': cfg['boundaries']}
    path = STUDY/'summary.json'
    temp = path.with_name('.summary.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)
    print(json.dumps({'saved_cells': len(rows), 'verification': result['verification'],
                      'predictions': predictions, 'function_summary': function_summary},
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
