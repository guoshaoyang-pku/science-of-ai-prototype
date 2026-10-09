from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
FUNCTIONS = ('trigonometric8', 'quadratic8', 'interaction12', 'radial12')
SEEDS = (100, 101, 102, 103, 104)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return {'mean': mean, 'min': float(x.min()), 'max': float(x.max()), 'paired_seed_t95': [mean-half, mean+half]}


def sign(value):
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def compute():
    receipt = json.loads((STUDY / 'executed/receipt.json').read_text())
    manifest = json.loads((STUDY / 'executed/input_manifest.json').read_text())
    expected = {STUDY/'results'/f'{function}_{label}_{seed}{suffix}'
                for function in FUNCTIONS for label in ('LN010_w64', 'noLN_w64')
                for seed in SEEDS for suffix in ('.json', '.npz')}
    assert set((STUDY/'results').iterdir()) == expected, 'incomplete or extra cells'
    cells = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        assert row['status'] == 'completed'
        assert row['contract']['preregistration_commit'] == receipt['preregistration_commit']
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == row['arrays_sha256']
        request = row['contract']
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
            energies = {}
            for offset in (-3, 0, 3):
                residual = (data[f'train_predictions_256__{offset}'].astype(np.float64).ravel()
                            - data[f'train_y__{offset}'].astype(np.float64).ravel())
                residual -= residual.mean()
                energies[str(offset)] = float(np.mean(residual**2))
            real = (energies['-3'] + energies['3']) / 2 - energies['0']
            batch = float(9 * np.mean(data['centered_propagated_ones_256'] ** 2))
            full_vector = data['full_propagated_ones_256']
            full = float(9 * np.mean((full_vector - full_vector.mean()) ** 2))
            assert abs(batch-row['batch64_train_chord_256']) < 1e-14
        cells[(request['function'], request['label'], request['seed'])] = {
            'real': real, 'batch': batch, 'full': full, 'seconds': row['seconds']}
    pairs = []
    for function in FUNCTIONS:
        for seed in SEEDS:
            ln = cells[(function, 'LN010_w64', seed)]
            no = cells[(function, 'noLN_w64', seed)]
            real, batch, full = (ln[key] - no[key] for key in ('real', 'batch', 'full'))
            pairs.append({'function': function, 'seed': seed, 'real_train_delta_chord': real,
                          'batch64_delta_chord_256': batch, 'full_delta_chord_256': full,
                          'batch_minus_full_delta': batch-full,
                          'batch64_chord_LN': ln['batch'], 'batch64_chord_noLN': no['batch'],
                          'direction_match_batch64': sign(real) != 0 and sign(batch) == sign(real),
                          'direction_match_full': sign(real) != 0 and sign(full) == sign(real)})
    batch_matches = sum(p['direction_match_batch64'] for p in pairs)
    full_matches = sum(p['direction_match_full'] for p in pairs)
    maximum_shift = max(abs(p['batch_minus_full_delta']) for p in pairs)
    per = []
    for function in FUNCTIONS:
        rows = [p for p in pairs if p['function'] == function]
        per.append({'function': function,
                    **{key: stats([p[key] for p in rows]) for key in
                       ('real_train_delta_chord', 'batch64_delta_chord_256', 'full_delta_chord_256', 'batch_minus_full_delta')},
                    'direction_matches': sum(p['direction_match_batch64'] for p in rows)})
    prediction = {'P1': 'supported' if batch_matches == 0 else 'refuted',
                  'P2': 'supported' if maximum_shift <= .005 else 'refuted'}
    return {'study': STUDY.name, 'round': 107, 'direction_round': 7, 'domain': 'development',
            'status': 'complete', 'saved_evaluation_cells': len(cells), 'pair_count': len(pairs),
            'new_training_cells': 0, 'new_kernel_cells': 0, 'reused_training_cells': 120,
            'reused_kernel_cells': 40, 'evaluation_seconds': sum(row['seconds'] for row in cells.values()),
            'preregistration_commit': receipt['preregistration_commit'],
            'preregistration_sha256': receipt['preregistration_sha256'],
            'manifest_sha256': receipt['manifest_sha256'], 'prediction': prediction,
            'direction_match_count_batch64': batch_matches, 'direction_match_count_full': full_matches,
            'direction_match_improvement': batch_matches-full_matches,
            'batch64_delta_range': [min(p['batch64_delta_chord_256'] for p in pairs), max(p['batch64_delta_chord_256'] for p in pairs)],
            'real_train_delta_range': [min(p['real_train_delta_chord'] for p in pairs), max(p['real_train_delta_chord'] for p in pairs)],
            'max_abs_batch_minus_full_delta': maximum_shift, 'functions': per, 'pairs': pairs,
            'batch_order': manifest['batch_order'],
            'boundary': '固定初始化核、同已保存batch64顺序、无decay的train线性化；不识别核漂移、decay、非线性或test动力学，不作因果或sealed OOD外推。'}


if __name__ == '__main__':
    result = compute()
    (STUDY/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'saved_evaluation_cells', 'direction_match_count_batch64', 'prediction')}, ensure_ascii=False))
