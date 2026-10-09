import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    old_summary = json.loads((REPO / 'directions/D4_sigmoid_training_curve/studies/'
                              'r002_two_mode_logistic/summary.json').read_text())
    assert summary['grid'] == old_summary['grid']
    grid = np.array(summary['grid'])
    assert len(grid) == 150 and grid[0] == 0 and grid[-1] == 4997
    error_disagreements, sse_disagreements, paired = [], [], []
    successful_starts = {'single': 0, 'double': 0}
    saved_manifest = {}
    for row in summary['cells']:
        cell_dir = STUDY / 'results' / row['cell_id']
        fits = json.loads((cell_dir / 'fits.json').read_text())
        recorded = fits.pop('content_sha256')
        assert recorded == hashlib.sha256(json.dumps(
            fits, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert sha(cell_dir / 'fits.json') == row['fit_file_sha256']
        with np.load(cell_dir / 'arrays.npz', allow_pickle=False) as saved:
            values = saved['normalized_loss'][grid]
        independently_computed = {}
        for name in ['single', 'double']:
            valid = [fit for fit in fits[name] if fit['success']]
            successful_starts[name] += len(valid)
            best = min(valid, key=lambda f: f['sse'])
            params = np.exp(best['parameters']).reshape(-1, 2)
            prediction = np.zeros(len(grid))
            for half_time, beta in params:
                prediction += 1 / (1 + (grid / half_time) ** beta) / len(params)
            residual = prediction - values
            errors = {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                      'maxabs': float(np.max(np.abs(residual)))}
            for metric in errors:
                error_disagreements.append(abs(errors[metric] - row[name][metric]))
            sse_disagreements.append(abs(float(residual @ residual) - best['sse']))
            assert (errors['rmse'] <= .03 and errors['maxabs'] <= .05) == row[name]['adequate']
            independently_computed[name] = errors
        pair = {'cell_id': row['cell_id'], 'ratio': row['ratio'], 'seed': row['seed']}
        for metric in ['rmse', 'maxabs']:
            difference = independently_computed['double'][metric] - independently_computed['single'][metric]
            assert abs(difference - row['paired_difference_double_minus_single'][metric]) < 1e-12
            pair[metric + '_double_minus_single'] = difference
        paired.append(pair)
        for file in sorted(cell_dir.iterdir()):
            saved_manifest[str(file.relative_to(STUDY))] = {
                'sha256': sha(file), 'mtime_ns': file.stat().st_mtime_ns}
    assert max(error_disagreements) < 1e-12 and max(sse_disagreements) < 1e-12
    assert summary['single_pass_ratios'] == [12, 15]
    assert summary['single_fail_ratios'] == [20, 25]
    assert all(row['double']['adequate'] for row in summary['cells'])
    assert all(summary['predictions'][name]['status'] == 'supported'
               for name in ['P1', 'P2', 'P3', 'P4'])
    output = {'status': 'passed', 'original_D4_grid_identical': True,
              'grid_points': len(grid), 'counts': summary['counts'],
              'max_error_recomputation_disagreement': max(error_disagreements),
              'max_sse_recomputation_disagreement': max(sse_disagreements),
              'successful_starts': successful_starts, 'paired_error_differences': paired,
              'saved_manifest': saved_manifest}
    with (STUDY / 'executed/independent_analysis_verification.json').open('x') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ['paired_error_differences', 'saved_manifest']}, indent=2))


if __name__ == '__main__':
    main()
