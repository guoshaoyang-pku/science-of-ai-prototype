from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(values):
    if not values:
        return None
    a = np.array(values, dtype=float)
    mean = float(a.mean())
    half = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.
    return {'mean': mean, 'min': float(a.min()), 'max': float(a.max()),
            'seed_95pct_t_interval': [mean - half, mean + half]}


def derivatives(b):
    s = expit(b)
    q, h = s * (1 - s), 1 - 2 * s
    return (s + b * q, q * (2 + b * h), q * (3 * h + b * (h * h - 2 * q)),
            q * (4 * (h * h - 2 * q) + b * (h ** 3 - 8 * q * h)))


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    for name, expected in config['source_sha256'].items():
        if sha(STUDY / name) != expected:
            raise RuntimeError(f'Pinned source mismatch: {name}')
    baseline = json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())
    for name, expected in baseline['files'].items():
        if sha(REPO / name) != expected['sha256'] or (REPO / name).stat().st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError(f'Old successful evidence changed: {name}')
    manifest_path = STUDY / 'executed/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    commit = manifest['preregistration_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO, check=True)
    for name in ['preregistration.json', *config['source_sha256']]:
        path = STUDY / name
        data = subprocess.run(['git', 'show', f'{commit}:{path.relative_to(REPO)}'],
                              cwd=REPO, capture_output=True, check=True).stdout
        if data != path.read_bytes():
            raise RuntimeError('Execution commit did not match preregistration and pinned sources')
    if manifest['preregistration_sha256'] != sha(STUDY / 'preregistration.json'):
        raise RuntimeError('Manifest preregistration mismatch')
    snapshot = STUDY / 'executed/input_snapshot.npz'
    if manifest['input_sha256'] != sha(snapshot):
        raise RuntimeError('Input snapshot hash mismatch')
    with np.load(snapshot, allow_pickle=False) as z:
        inputs = {name: z[name].copy() for name in z.files}
    for name, a in inputs.items():
        if hashlib.sha256(a.tobytes()).hexdigest() != manifest['input_arrays'][name]['sha256']:
            raise RuntimeError('Input array hash mismatch')
    coefficients = []
    for seed in config['seeds']:
        u = inputs[f'u_{seed}']
        u2, u4 = u ** 2, u ** 4
        v2, v4 = u2 - u2.mean(0), u4 - u4.mean(0)
        m2, m22, m24, m44 = [float(np.mean(a)) for a in (u2, v2*v2, v2*v4, v4*v4)]
        f1, f2, f3, f4 = derivatives(config['bias_root'])
        coefficients.append({'seed': seed, 'M2': m2, 'M22': m22, 'M24': m24, 'M44': m44,
                             'kappa': float(abs(f4 / (12*f3)) * np.sqrt(m44/m22)),
                             'rho': float(m24 / np.sqrt(m22*m44))})
    rows = []
    max_reconstruction = 0.
    measurement_seconds = 0.
    for path in sorted((STUDY / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        request = row['request']
        cell = request['cell']
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if row['status'] != 'success' or row['contract_sha256'] != contract:
            raise RuntimeError('Cell contract mismatch')
        if request['manifest_sha256'] != sha(manifest_path) or row['arrays_sha256'] != sha(path.with_suffix('.npz')):
            raise RuntimeError('Cell manifest/arrays hash mismatch')
        if (cell['delta'] != config['bias_offsets'][cell['offset_index']]
                or request['bias'] != config['bias_root'] + cell['delta']):
            raise RuntimeError('Bias differs from frozen grid')
        if cell['seed'] not in config['seeds'] or cell['scale'] not in config['scales']:
            raise RuntimeError('Cell not in frozen design')
        expected_label = f"s{cell['seed']}_d{cell['offset_index']:02d}_a{cell['scale']:g}"
        if row['cell_id'] != expected_label or path.stem != expected_label:
            raise RuntimeError('Cell label does not match frozen grid')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            odd, even, center = [z[name].copy() for name in ('odd_raw', 'even_centered_raw', 'feature_center')]
            if not all(np.isfinite(z[name]).all() for name in z.files):
                raise RuntimeError('Nonfinite saved array')
        u, a, b = inputs[f"u_{cell['seed']}"], cell['scale'], request['bias']
        raw = (b+a*u)*expit(b+a*u)
        reflected = (b-a*u)*expit(b-a*u)
        reconstructed = ((raw-reflected)/2, (raw+reflected)/2 - raw.mean(0), raw.mean(0))
        error = max(float(np.max(np.abs(x-y))) for x, y in zip((odd, even, center), reconstructed))
        max_reconstruction = max(max_reconstruction, error)
        if error > config['criteria']['array_reconstruction_tolerance']:
            raise RuntimeError('Independent NumPy activation reconstruction failed')
        f1, f2, f3, f4 = derivatives(b)
        u2, u4 = u*u, u**4
        e2 = (f2/2)*a*a*(u2-u2.mean(0))
        e4 = (f4/24)*a**4*(u4-u4.mean(0))
        o1 = f1*a*u
        r = float(np.mean(even*even)/np.mean(odd*odd))
        mixed = float(np.mean((e2+e4)**2)/np.mean(o1*o1))
        uncoupled = float((np.mean(e2*e2)+np.mean(e4*e4))/np.mean(o1*o1))
        cross = float(2*np.mean(e2*e4)/np.mean(o1*o1))
        m = next(x for x in coefficients if x['seed']==cell['seed'])
        coeff2, coeff4 = f2/2, f4/24
        a_equal = float(np.sqrt(abs(coeff2/coeff4)*np.sqrt(m['M22']/m['M44']))) if coeff2 else 0.
        rows.append({**cell, 'cell_id': row['cell_id'], 'bias': b, 'R': r, 'R24': mixed,
                     'relative_error': abs(r/mixed-1), 'R_no_cross': uncoupled,
                     'no_cross_relative_error': abs(r/uncoupled-1), 'cross_term': cross,
                     'a_equal_component_energy': a_equal, 'array_reconstruction_maxabs': error})
        measurement_seconds += row['seconds']
    if len({r['cell_id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate cell id')
    units = []
    paired = []
    small, large = config['primary_scales']
    for delta in config['bias_offsets']:
        pairs = []
        for seed in config['seeds']:
            low = next((r for r in rows if r['seed']==seed and r['delta']==delta and r['scale']==small), None)
            high = next((r for r in rows if r['seed']==seed and r['delta']==delta and r['scale']==large), None)
            if low is None or high is None:
                continue
            growth = high['R']/low['R']
            prediction = high['R24']/low['R24']
            pairs.append({'seed': seed, 'delta': delta, 'growth': growth,
                          'exponent': float(np.log2(growth)), 'predicted_growth': prediction,
                          'growth_prediction_relative_error': abs(growth/prediction-1)})
        paired.extend(pairs)
        if pairs:
            units.append({'delta': delta, 'paired_seeds': len(pairs),
                          'growth': stats([r['growth'] for r in pairs]),
                          'exponent': stats([r['exponent'] for r in pairs]),
                          'growth_prediction_relative_error': stats([r['growth_prediction_relative_error'] for r in pairs])})
    observed = {(r['seed'], r['offset_index'], r['scale']) for r in rows}
    expected = {(seed, index, scale) for seed in config['seeds']
                for index in range(len(config['bias_offsets'])) for scale in config['scales']}
    if len(observed) != len(rows) or not observed.issubset(expected):
        raise RuntimeError('Duplicated or unexpected design cell')
    complete = observed == expected
    primary = [r for r in rows if r['scale'] in config['primary_scales']]
    near = [r for r in paired if abs(r['delta'])<=1e-5]
    negative = [r for r in paired if r['delta']==-.01]
    positive = [r for r in paired if r['delta']==.01]
    differences = []
    for seed in config['seeds']:
        plus = next((r for r in paired if r['seed']==seed and r['delta']==.001), None)
        minus = next((r for r in paired if r['seed']==seed and r['delta']==-.001), None)
        if plus and minus:
            differences.append({'seed': seed, 'positive_growth': plus['growth'], 'negative_growth': minus['growth'],
                                'difference': plus['growth']-minus['growth']})
    p1 = complete and all(r['relative_error']<=.02 for r in primary)
    p2 = complete and all(60<=r['growth']<=68 for r in near) and all(5<=r['growth']<=8 for r in negative) and all(1.5<=r['growth']<=3.5 for r in positive)
    p3 = complete and all(r['difference']>=20 for r in differences) and sum(r['positive_growth']>100 for r in differences)>=2
    summary = {'study': STUDY.name, 'round': 24, 'direction_round': 3, 'domain': 'development',
               'question': config['question'], 'status': 'complete' if complete else 'partial',
               'saved_cells': len(rows), 'planned_cells': config['planned_cells'], 'bias_recipes': len(config['bias_offsets']),
               'bias_scale_units': len(config['bias_offsets'])*len(config['scales']), 'training_steps': 0, 'measurement_seconds': measurement_seconds,
               'preregistration_commit': commit,
               'predictions': [{'id': name, 'status': ('supported' if passed else 'refuted') if complete else 'not_evaluated'}
                               for name, passed in [('P1',p1),('P2',p2),('P3',p3)]],
               'primary_mixed_relative_error': stats([r['relative_error'] for r in primary]),
               'all_scale_mixed_relative_error': stats([r['relative_error'] for r in rows]),
               'primary_no_cross_relative_error': stats([r['no_cross_relative_error'] for r in primary]),
               'units': units, 'paired_seed_results': paired, 'asymmetry_pairs': differences,
               'asymmetry_difference': stats([r['difference'] for r in differences]) if differences else None,
               'coefficients': coefficients, 'rows': rows,
               'verification': {'old_files_unchanged': len(baseline['files']),
                                'max_array_reconstruction_error': max_reconstruction,
                                'source_commit_hash_contract_checks': 'pass'},
               'boundary': config['boundary']}
    save(STUDY / 'summary.json', summary)
    print(json.dumps({'saved_cells':len(rows),'predictions':summary['predictions'],
                      'max_primary_relative_error':summary['primary_mixed_relative_error']['max'],
                      'max_reconstruction_error':max_reconstruction}, ensure_ascii=False))


if __name__ == '__main__':
    main()
