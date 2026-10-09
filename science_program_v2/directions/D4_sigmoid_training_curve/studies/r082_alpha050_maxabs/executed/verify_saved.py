import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon(epsilon, grid, values):
    log_low, log_high = np.log(.01), np.log(100000.)
    vertices = [np.array([.1, .1 * log_low]), np.array([.1, .1 * log_high]),
                np.array([10., 10. * log_high]), np.array([10., 10. * log_low])]
    for step, y in zip(grid, values):
        if step == 0:
            if abs(y - 1) > epsilon:
                return []
            continue
        x = np.log(step)
        constraints = []
        if y + epsilon < 1:
            z = y + epsilon
            constraints.append((np.array([-x, 1.]), np.log(z) - np.log1p(-z)))
        if y - epsilon > 0:
            z = y - epsilon
            constraints.append((np.array([x, -1.]), -np.log(z) + np.log1p(-z)))
        for normal, bound in constraints:
            clipped = []
            if not vertices:
                return []
            for first, second in zip(vertices, vertices[1:] + vertices[:1]):
                first_slack = bound - normal @ first
                second_slack = bound - normal @ second
                if first_slack >= 0:
                    clipped.append(first)
                if (first_slack >= 0) != (second_slack >= 0):
                    fraction = first_slack / (first_slack - second_slack)
                    clipped.append(first + fraction * (second - first))
            vertices = clipped
    return vertices


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    assert sha(Path(__file__)) == prereg['verification_source_sha256']
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='certificate_verification.json')
    args = parser.parse_args()
    summary_path = STUDY / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    previous = json.loads((STUDY / 'executed/previous_closeout_audit.json').read_text())
    for relative, expected in prereg['input_hashes'].items():
        assert sha(REPO / relative) == expected
    assert sha(STUDY / 'executed/previous_closeout_audit.json') == prereg['previous_audit_sha256']
    for item in previous['files']:
        path = REPO / item['file']
        assert sha(path) == item['sha256'] and path.stat().st_mtime_ns == item['mtime_ns']
    pins = {}
    for receipt_path in sorted((STUDY / 'executed').glob('run_receipt_*.json')):
        receipt = json.loads(receipt_path.read_text())
        for row in receipt['new_cells'] + receipt['resumed_cells']:
            assert pins.get(row['cell_id'], row['metadata_sha256']) == row['metadata_sha256']
            pins[row['cell_id']] = row['metadata_sha256']
    checked = []
    for item in prereg['inputs']:
        folder = STUDY / 'results' / item['cell_id']
        if not folder.exists():
            continue
        metadata = json.loads((folder / 'metadata.json').read_text())
        assert sha(folder / 'metadata.json') == pins[item['cell_id']]
        assert sha(folder / 'arrays.npz') == metadata['arrays_sha256']
        assert metadata['status'] == 'success'
        assert metadata['request']['input'] == item
        assert metadata['request']['preregistration_sha256'] == sha(STUDY / 'preregistration.json')
        assert metadata['request']['source_sha256'] == prereg['execution_source_sha256']
        row = next(r for r in summary['cells'] if r['cell_id'] == item['cell_id']) if summary else None
        with np.load(folder / 'arrays.npz') as arrays, np.load(REPO / item['arrays']) as source:
            assert all(np.all(np.isfinite(arrays[name])) for name in arrays.files)
            grid, values = arrays['grid'].copy(), arrays['observed'].copy()
            assert grid.tolist() == prereg['grid']
            assert np.array_equal(values, source['normalized_loss'][grid])
            for name in ['minimax', 'least_squares']:
                log_half, beta = arrays[name + '_parameters']
                prediction = np.ones(len(grid))
                positive = grid > 0
                prediction[positive] = expit(beta * (log_half - np.log(grid[positive])))
                assert np.max(np.abs(prediction - arrays[name + '_prediction'])) <= 1e-14
                residual = prediction - values
                error = {'rmse': float(np.sqrt(np.mean(residual**2))),
                         'maxabs': float(np.max(np.abs(residual)))}
                if row:
                    assert all(abs(error[m] - row['errors'][name][m]) <= 1e-14 for m in error)
            log_half, beta = arrays['minimax_parameters']
            assert .1 <= beta <= 10 and np.log(.01) <= log_half <= np.log(100000)
        cert = metadata['certificate']
        lower_poly = polygon(cert['lower_bound'], grid, values)
        upper_poly = polygon(cert['feasible_upper_bound'], grid, values)
        assert not lower_poly and upper_poly
        assert cert['gap'] <= prereg['bisection_tolerance']
        checked.append({'cell_id': item['cell_id'], 'lower_polygon_empty': True,
                        'upper_polygon_vertices': len(upper_poly), 'certificate_gap': cert['gap']})
    result = {'status': 'verified', 'checked_at': datetime.now().astimezone().isoformat(),
              'summary_sha256': sha(summary_path) if summary else None, 'verified_cells': len(checked),
              'old_files_hash_mtime_unchanged': len(previous['files']), 'cells': checked,
              'independent_algorithm': '在(beta,c)参数盒多边形上逐点半平面裁剪，独立核对保存误差下界不可行/上界可行；不优化、不拟合、不训练。',
              'verification_source_sha256': sha(Path(__file__))}
    output = STUDY / 'executed' / args.output
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
