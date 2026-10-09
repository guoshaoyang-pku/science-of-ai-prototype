from pathlib import Path
import hashlib
import json
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def stats(values):
    x = np.array(values, dtype=np.float64)
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {'mean': mean, 'min': float(x.min()), 'max': float(x.max()),
            'paired_seed_t95': [mean-half, mean+half]}


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def sign(value):
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def calculate():
    cfg = json.loads((STUDY/'preregistration.json').read_text())
    cells = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        assert row['status'] == 'completed'
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == row['arrays_sha256']
        with np.load(path.with_suffix('.npz')) as data:
            energies = {}
            for m in cfg['offsets']:
                residual = data[f'train_predictions_256__{m}'].astype(np.float64).ravel()-data[f'train_y__{m}'].astype(np.float64).ravel()
                centered = residual-residual.mean()
                energies[str(m)] = float(np.mean(centered**2))
            true = (energies['-3']+energies['3'])/2-energies['0']
            fixed = {str(step): float(9*np.mean(data[f'centered_propagated_ones_{step}']**2)) for step in (1, 256)}
            assert all(abs(fixed[k]-row['fixed_kernel_train_chord'][k]) < 1e-14 for k in fixed)
        req = row['contract']
        cells[(req['function'], req['label'], req['seed'])] = {
            'real_train_chord': true, 'real_train_centered_residual_variance': energies,
            'fixed_kernel_train_chord': fixed, 'seconds': row['seconds'],
            'lambda_max': row['kernel_eigenvalue_max']}
    pairs = []
    for function in cfg['functions']:
        for seed in cfg['seeds']:
            if not all((function, label, seed) in cells for label in cfg['recipes']):
                continue
            ln = cells[(function, 'LN010_w64', seed)]
            no = cells[(function, 'noLN_w64', seed)]
            actual = ln['real_train_chord']-no['real_train_chord']
            fixed = {step: ln['fixed_kernel_train_chord'][step]-no['fixed_kernel_train_chord'][step] for step in ('1', '256')}
            pairs.append({'function': function, 'seed': seed, 'real_train_delta_chord': actual,
                          'real_train_chord_LN': ln['real_train_chord'], 'real_train_chord_noLN': no['real_train_chord'],
                          'fixed_delta_chord': fixed,
                          'fixed_chord_LN': ln['fixed_kernel_train_chord'], 'fixed_chord_noLN': no['fixed_kernel_train_chord'],
                          'direction_match': {step: sign(fixed[step]) == sign(actual) and sign(actual) != 0 for step in fixed}})
    counts = {step: sum(p['direction_match'][step] for p in pairs) for step in ('1', '256')}
    complete = len(cells) == 40 and len(pairs) == 20
    verdicts = {'P1': 'supported' if counts['256'] >= 16 else 'refuted',
                'P2': 'supported' if counts['256']-counts['1'] >= 10 else 'refuted'} if complete else {'P1': 'not_evaluated', 'P2': 'not_evaluated'}
    functions = []
    for function in cfg['functions']:
        rows = [p for p in pairs if p['function'] == function]
        if rows:
            functions.append({'function': function, 'real_train_delta_chord': stats([p['real_train_delta_chord'] for p in rows]),
                              'fixed_delta_chord': {step: stats([p['fixed_delta_chord'][step] for p in rows]) for step in ('1', '256')},
                              'direction_matches': {step: sum(p['direction_match'][step] for p in rows) for step in ('1', '256')}})
    correlations = {step: correlation([p['fixed_delta_chord'][step] for p in pairs], [p['real_train_delta_chord'] for p in pairs]) for step in ('1', '256')} if pairs else {}
    return {'study': cfg['study'], 'round': 77, 'domain': 'development',
            'status': 'complete' if complete else 'partial', 'saved_measurement_cells': len(cells),
            'new_training_cells': 0, 'reused_training_cells': 120, 'pair_count': len(pairs),
            'measurement_seconds': sum(v['seconds'] for v in cells.values()),
            'predictions': verdicts, 'direction_matches': counts,
            'direction_match_improvement': counts['256']-counts['1'],
            'real_train_negative_delta_count': sum(sign(p['real_train_delta_chord']) == -1 for p in pairs),
            'fixed_negative_delta_count': {step: sum(sign(p['fixed_delta_chord'][step]) == -1 for p in pairs) for step in ('1', '256')},
            'descriptive_correlation': correlations, 'functions': functions, 'pairs': pairs,
            'fixed_chord_range': {step: [min(v['fixed_kernel_train_chord'][step] for v in cells.values()), max(v['fixed_kernel_train_chord'][step] for v in cells.values())] for step in ('1', '256')} if cells else {},
            'max_eta_lambda': .001*max(v['lambda_max'] for v in cells.values()) if cells else None,
            'boundary': cfg['boundaries']}


if __name__ == '__main__':
    summary = calculate()
    (STUDY/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: summary[k] for k in ('status', 'saved_measurement_cells', 'measurement_seconds', 'predictions', 'direction_matches', 'direction_match_improvement')}, ensure_ascii=False))
