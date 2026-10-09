import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import expit

from run_experiment import frozen_contract, saved_rows, sha

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    rows = saved_rows(contract, contract_sha, args.freeze)
    summary = json.loads((ROOT / 'summary.json').read_text())
    with np.load(REPO / contract['source_npz']) as archive:
        loss = archive['loss'].copy()
    clock = .0003125 * np.arange(len(loss), dtype=float) / .02
    centered = loss - math.fsum(loss) / len(loss)
    denominator = math.fsum(centered ** 2)
    base_tau = json.loads((REPO / contract['baseline_fit']).read_text())['reference']['tau_A']
    max_rmse_error = max_r2_error = max_tau_error = 0.
    supported = 0
    freeze_time = subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', args.freeze], cwd=REPO)
    details = []
    for cell, comparison in zip(contract['cells'], summary['comparisons']):
        row = rows[cell['id']]
        p = row['A']
        predicted = p['c'] + (p['a'] - p['c']) * expit(-p['k'] * (np.log1p(clock) - p['logtc']))
        squared = (predicted - loss) ** 2
        rmse = math.sqrt(math.fsum(squared) / len(loss))
        r2 = 1 - math.fsum(squared) / denominator
        tau = .02 * math.expm1(p['logtc'])
        relative = abs(tau - base_tau) / abs(base_tau)
        matches = bool(p['success'] and relative <= .01)
        assert comparison['cell_id'] == cell['id'] and comparison['supports_P1'] == matches
        assert abs(relative - comparison['relative_difference']) <= 1e-12
        assert abs(rmse / (loss[0] - loss[-1]) - p['normalized_rmse']) <= 1e-12
        assert 0 <= p['a'] <= max(1., float(loss.max()) * 3)
        assert 0 <= p['c'] <= max(1., float(loss.max()) * 3)
        assert 1e-5 <= p['k'] <= 30 and 0 <= p['logtc'] <= math.log1p(1e7)
        assert row['started_unix'] > int(freeze_time)
        max_rmse_error = max(max_rmse_error, abs(rmse - p['rmse']))
        max_r2_error = max(max_r2_error, abs(r2 - p['r2']))
        max_tau_error = max(max_tau_error, abs(tau - row['tau_A']))
        supported += matches
        details.append({'cell_id': cell['id'], 'independent_tau_A': tau,
                        'independent_rmse': rmse, 'independent_r2': r2,
                        'freeze_to_fit_seconds': row['started_unix'] - int(freeze_time)})
    assert max(max_rmse_error, max_r2_error, max_tau_error) <= 1e-12
    assert summary['supported_cells'] == supported
    assert summary['prediction_P1'] == ('supported' if supported == 4 else 'refuted')
    check = {'status': 'passed', 'freeze_commit': args.freeze, 'preregistration_sha256': contract_sha,
             'old_hash_mtime_verified': len(contract['old_input_pins']),
             'independent_fit_metrics_checked': 4, 'new_training_cells': 0, 'verification_new_fits': 0,
             'max_rmse_error': max_rmse_error, 'max_r2_error': max_r2_error, 'max_tau_error': max_tau_error,
             'prediction_recomputed': True, 'details': details,
             'source_sha256': sha(ROOT / 'executed/verify.py'),
             'scope': 'expit与math.fsum重算保存参数指标及判据；不调用least_squares，不证明全局最优。'}
    out = ROOT / 'executed/independent_verification.json'
    if out.exists():
        assert json.loads(out.read_text()) == check
    else:
        with out.open('x') as stream:
            json.dump(check, stream, ensure_ascii=False, indent=2)
            stream.write(chr(10))
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
