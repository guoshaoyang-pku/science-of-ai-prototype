#!/usr/bin/env python3
"""从保存结果复算噪声减半对照、配对最低点和冻结判据。"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('execution', STUDY / 'executed/run.py')
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)
OLD = execution.OLD


def analyze():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    receipt_path = STUDY / 'results/receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'cells': {}, 'compute_seconds': 0}
    old_receipt = json.loads((OLD / 'results/receipt.json').read_text())
    rows = []
    for recipe in config['recipes']:
        n, variance = recipe['n'], recipe['noise_variance']
        for seed in config['seeds']:
            label = f'n{n}_seed{seed}_var{variance:g}'
            path = STUDY / 'results' / f'{label}.json'
            if not path.exists():
                if path.with_suffix('.npz').exists() or label in receipt['cells']:
                    raise RuntimeError('Partial saved result')
                continue
            metadata = json.loads(path.read_text())
            pins = {k: execution.sha(STUDY / k) for k in metadata['pins']['sha256']}
            if pins != metadata['pins']['sha256'] or execution.sha(path) != receipt['cells'][label] or execution.sha(path.with_suffix('.npz')) != metadata['arrays_sha256']:
                raise RuntimeError('Result/source hash mismatch')
            data_path = STUDY / metadata['data_file']
            contract = hashlib.sha256(json.dumps({'cell': metadata['cell'], 'pins': pins, 'data_sha256': execution.sha(data_path)}, sort_keys=True).encode()).hexdigest()
            if execution.sha(data_path) != metadata['data_sha256'] or contract != metadata['contract_sha256']:
                raise RuntimeError('Data/cell hash mismatch')
            old_var = 0.25 if n == 32 else 0.5
            old_label = f'n{n}_seed{seed}_var{old_var:g}'
            old_path = OLD / 'results' / f'{old_label}.json'
            old_metadata = json.loads(old_path.read_text())
            if execution.sha(old_path) != old_receipt['cells'][old_label] or execution.sha(old_path.with_suffix('.npz')) != old_metadata['arrays_sha256']:
                raise RuntimeError('Old result hash mismatch')
            with np.load(path.with_suffix('.npz')) as z, np.load(old_path.with_suffix('.npz')) as oz, np.load(data_path) as d, np.load(OLD / old_metadata['data_file']) as od:
                arrays = {k: z[k] for k in z.files}
                old_arrays = {k: oz[k] for k in oz.files}
                data = {k: d[k] for k in d.files}
                old_data = {k: od[k] for k in od.files}
            unchanged = {k: bool(np.array_equal(data[k], old_data[k])) for k in ['train_x', 'audit_x', 'epsilon', 'hidden_weight', 'train_features', 'audit_features', 'train_y', 'audit_y', 'target_mean_rms']}
            if not all(unchanged.values()):
                raise RuntimeError('Controlled invariant changed')
            bias, noise = arrays['signal_bias'], arrays['variance_unit']
            risk = bias + variance * noise
            if len(risk) != config['optimizer']['steps'] + 1:
                raise RuntimeError('Step grid mismatch')
            risk_error = float(np.abs(risk - arrays['expected_risk']).max())
            q = 1 - 2 * config['optimizer']['lr'] * arrays['eigenvalues']
            factors = np.zeros(len(q))
            max_bias_error, max_noise_error = 0., 0.
            for step in range(len(risk)):
                coefficients = factors * arrays['signal_projection']
                spectral_bias = np.einsum('i,ij,j->', coefficients, arrays['basis_gram'], coefficients) - 2 * np.dot(arrays['basis_target'], coefficients) + float(arrays['audit_target_square'])
                spectral_noise = np.dot(np.diag(arrays['basis_gram']), factors ** 2)
                max_bias_error = max(max_bias_error, abs(spectral_bias - bias[step]))
                max_noise_error = max(max_noise_error, abs(spectral_noise - noise[step]))
                factors = q * factors + 2 * config['optimizer']['lr']
            tx, ty = data['audit_features'], data['audit_y']
            direct_final = float(np.mean((np.einsum('mi,i->m', tx, arrays['final_heads'][:, 1]) - ty) ** 2))
            t, old_t = int(np.argmin(risk)), int(np.argmin(old_arrays['expected_risk']))
            audit = {'max_train_increase': float(np.diff(arrays['train_mse']).max()),
                     'max_saved_risk_error': risk_error, 'max_spectral_signal_error': max_bias_error,
                     'max_spectral_variance_error': max_noise_error,
                     'max_old_new_signal_error': float(np.abs(bias - old_arrays['signal_bias']).max()),
                     'max_old_new_variance_error': float(np.abs(noise - old_arrays['variance_unit']).max()),
                     'final_direct_bias_error': abs(direct_final - bias[-1])}
            p1 = all(np.isfinite(v).all() for v in arrays.values()) and all(unchanged.values()) and audit['max_train_increase'] <= 1e-10 and all(v <= 1e-9 for k, v in audit.items() if k != 'max_train_increase')
            rows.append({**metadata['cell'], 'label': label, 't_star': t, 'old_t_star': old_t,
                         't_halved_over_baseline': t / old_t if old_t else None,
                         'interior': 0 < t < config['optimizer']['steps'], 'minimum_risk': float(risk[t]),
                         'final_risk': float(risk[-1]), 'u_shape': bool(0 < t < len(risk) - 1 and risk[-1] - risk[t] >= .05),
                         'invariants': unchanged, 'audit': audit, 'P1_pass': bool(p1)})
    pairs = []
    for seed in config['seeds']:
        pair = [r for r in rows if r['seed'] == seed]
        if len(pair) != 2:
            continue
        left, right = sorted(pair, key=lambda r: r['n'])
        ratio = right['t_star'] / left['t_star'] if left['t_star'] else None
        old_ratio = right['old_t_star'] / left['old_t_star']
        pairs.append({'seed': seed, 'left': left['label'], 'right': right['label'],
                      't_star_left': left['t_star'], 't_star_right': right['t_star'],
                      'old_ratio': old_ratio, 'halved_ratio': ratio,
                      'log2_ratio': float(np.log2(ratio)) if ratio and ratio > 0 else None,
                      'log2_halved_over_old_ratio': float(np.log2(ratio / old_ratio)) if ratio and ratio > 0 else None,
                      'factor2_pass': bool(left['interior'] and right['interior'] and ratio is not None and .5 <= ratio <= 2)})
    complete = len(rows) == 8
    p2 = next((p for p in pairs if p['seed'] == 414), None)
    predictions = {'P1': {'status': 'not_evaluated'}, 'P2': {'status': 'not_evaluated'}}
    if complete:
        predictions['P1'] = {'status': 'supported' if all(r['P1_pass'] for r in rows) else 'refuted', 'passed': sum(r['P1_pass'] for r in rows), 'total': 8, 'register_as_discovery': False}
        value = p2['halved_ratio']
        predictions['P2'] = {'status': 'supported' if value is not None and 2 < value <= 4 and all(r['interior'] for r in rows if r['seed'] == 414) else 'refuted',
                             'seed': 414, 'predicted_interval': '(2,4]', 'measured_ratio': value,
                             'counterexample_retained': bool(value is not None and value > 2 and all(r['interior'] for r in rows if r['seed'] == 414))}
    summary = {'study': config['study'], 'round': 39, 'direction': config['direction'], 'domain': 'development',
               'status': 'complete' if complete else ('partial' if rows else 'not_started'),
               'preregistration_sha256': execution.sha(STUDY / 'preregistration.json'),
               'execution_source_sha256': execution.sha(STUDY / 'executed/run.py'), 'analysis_source_sha256': execution.sha(__file__),
               'counts': {'recipe_units': 2, 'seeds_per_recipe': 4, 'planned_cells': 8, 'saved_cells': len(rows), 'paired_recipe_groups': 1, 'seed_pairs': len(pairs)},
               'compute_seconds': receipt['compute_seconds'], 'predictions': predictions, 'cells': rows, 'pairs': pairs,
               'boundaries': config['boundaries']}
    execution.save(STUDY / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'saved_cells': len(rows), 'predictions': predictions}, ensure_ascii=False))


if __name__ == '__main__':
    analyze()
