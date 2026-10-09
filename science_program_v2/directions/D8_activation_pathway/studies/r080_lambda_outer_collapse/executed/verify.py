import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths):
    return {str(p.relative_to(STUDY)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in paths}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    summary = json.loads((STUDY / 'summary.json').read_text())
    config = json.loads((STUDY / 'preregistration.json').read_text())
    protected = json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())['files']
    for name, expected in protected.items():
        p = REPO / name
        assert sha(p) == expected['sha256'] and p.stat().st_mtime_ns == expected['mtime_ns'], name
    paths = sorted((STUDY / 'results').glob('*'))
    before = snapshot(paths)
    assert len(paths) == (2 if args.first else 72)
    forecasts = json.loads((STUDY / 'executed/analytic_forecasts.json').read_text())
    root = np.longdouble(str(config['bias_root']))
    sig = 1/(1+np.exp(-root))
    coefficients = np.array([0, 1], dtype=np.longdouble)
    derivatives = [sig]
    for order in range(1, 5):
        derivative = np.arange(1, len(coefficients))*coefficients[1:]
        coefficients = np.polynomial.polynomial.polymul(derivative, [0, 1, -1])
        derivatives.append(np.polynomial.polynomial.polyval(sig, coefficients))
    f1 = derivatives[0]+root*derivatives[1]
    f3 = 3*derivatives[2]+root*derivatives[3]
    f4 = 4*derivatives[3]+root*derivatives[4]
    independent_F = {}
    with np.load(STUDY / 'executed/input_snapshot.npz', allow_pickle=False) as z:
        for seed in config['seeds']:
            u = z[f'u_{seed}'].astype(np.longdouble)
            v2, v4 = u**2, u**4
            v2, v4 = v2-v2.mean(0), v4-v4.mean(0)
            for lam in config['lambdas']:
                a, b = f3*lam/2, f4/24
                numerator = a*a*np.mean(v2*v2)+2*a*b*np.mean(v2*v4)+b*b*np.mean(v4*v4)
                independent_F[(seed, lam)] = float(numerator/(f1*f1*np.mean(u*u)))
    F_error = max(abs(independent_F[(r['seed'], r['lambda'])]/r['F']-1) for r in forecasts['rows'])
    assert F_error < 1e-10
    rq_errors, measured = [], {}
    for row in summary['rows']:
        path = STUDY / 'results' / (row['cell_id'] + '.npz')
        with np.load(path, allow_pickle=False) as z:
            numerator = math.fsum(float(x)*float(x) for x in z['even_centered_raw'].ravel())
            denominator = math.fsum(float(x)*float(x) for x in z['odd_raw'].ravel())
        r = numerator/denominator
        q = r/row['scale']**6
        rq_errors.extend([abs(r/row['R']-1), abs(q/row['Q']-1)])
        assert abs(q/independent_F[(row['seed'], row['lambda'])]-1) <= config['criteria']['relative_F_error_max'] or summary['predictions'][1]['status']=='refuted'
        measured[(row['seed'], row['lambda'], row['scale'])] = q
    assert max(rq_errors, default=0) < 1e-12
    commit = summary['preregistration_commit']
    commit_epoch = int(subprocess.run(['git', 'show', '-s', '--format=%ct', commit], cwd=REPO, capture_output=True, text=True, check=True).stdout)
    assert all(p.stat().st_mtime_ns > commit_epoch*10**9 for p in paths)
    result = {'status': 'pass', 'saved_cells': summary['saved_cells'], 'training_steps': 0, 'preregistration_commit': commit, 'commit_before_all_result_files': True, 'independent_polynomial_derivative_F_relative_error_max': F_error, 'independent_fsum_R_Q_relative_difference_max': max(rq_errors, default=0), 'old_files_hash_mtime_unchanged': len(protected), 'result_files': before, 'verification_source_sha256': sha(Path(__file__))}
    if args.first:
        result['independent_array_reconstruction_maxabs'] = summary['verification']['max_array_reconstruction_error']
        save(STUDY / 'executed/first_cell_audit.json', result)
    else:
        pairs = []
        for seed in config['seeds']:
            for lam in config['lambdas']:
                qs = [measured[(seed, lam, a)] for a in config['scales']]
                pairs.append(max(qs)/min(qs)-1)
        assert abs(max(pairs)-summary['fold_spread']['max']) < 1e-12
        first = json.loads((STUDY / 'executed/first_cell_audit.json').read_text())
        assert all(before[name] == row for name, row in first['result_files'].items())
        recovery = subprocess.run([sys.executable, str(STUDY / 'executed/run.py')], cwd=REPO, capture_output=True, text=True, check=True)
        recovered = json.loads(recovery.stdout)
        assert recovered['new']==0 and recovered['reused']==36
        assert snapshot(paths) == before
        result.update({'independent_spread_range': [min(pairs), max(pairs)], 'first_cell_unchanged': True, 'new_result_files_hash_mtime_unchanged': len(paths), 'recovery': recovered, 'historical_overlap_cells': 0})
        save(STUDY / 'executed/independent_verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='result_files'}))


if __name__ == '__main__':
    main()
