import hashlib
import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    rows = []
    prediction_difference = 0.0
    for item in prereg['inputs']:
        folder = STUDY / 'results' / item['cell_id']
        with np.load(folder / 'arrays.npz') as saved, np.load(REPO / item['arrays']) as source:
            grid, observed = saved['grid'], saved['observed']
            assert np.array_equal(observed, source['normalized_loss'][grid])
            row = next(r for r in summary['cells'] if r['cell_id'] == item['cell_id'])
            for name in ['minimax', 'least_squares']:
                log_half, beta = saved[name + '_parameters']
                predicted = np.ones(len(grid))
                positive = grid > 0
                predicted[positive] = expit(beta * (log_half - np.log(grid[positive])))
                difference = float(np.max(np.abs(predicted - saved[name + '_prediction'])))
                prediction_difference = max(prediction_difference, difference)
                residual = predicted - observed
                errors = {'rmse': float(np.sqrt(np.mean(residual**2))),
                          'maxabs': float(np.max(np.abs(residual)))}
                assert all(abs(errors[k] - row['errors'][name][k]) <= 1e-14 for k in errors)
            points = {t: float(observed[np.flatnonzero(grid == t)[0]]) for t in [1, 10, 15, 152]}
        with localcontext() as context:
            context.prec = 60
            epsilon = Decimal('.05')
            def logit(value):
                return value.ln() - (1 - value).ln()
            beta_lower = (logit(Decimal.from_float(points[1]) - epsilon)
                          - logit(Decimal.from_float(points[10]) + epsilon)) / Decimal(10).ln()
            beta_upper = (logit(Decimal.from_float(points[15]) + epsilon)
                          - logit(Decimal.from_float(points[152]) - epsilon)) / (Decimal(152).ln() - Decimal(15).ln())
            assert beta_lower > beta_upper
            rows.append({'cell_id': item['cell_id'], 'saved_points': points,
                         'required_beta_lower': str(beta_lower),
                         'required_beta_upper': str(beta_upper),
                         'lower_exceeds_upper': True})
    result = {'status': 'verified', 'threshold': '.05', 'decimal_precision': 60,
              'independent_prediction_maxabs_difference': prediction_difference,
              'summary_sha256': sha(STUDY / 'summary.json'),
              'verification_source_sha256': sha(Path(__file__)), 'cells': rows,
              'interpretation': '固定平台logit不等式的直接矛盾；只读保存网格数据及已拟合参数，不拟合/二分/训练。Decimal高精度核验保存float64值，不是严格区间算术。'}
    with (STUDY / 'executed/independent_threshold_verification.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
