from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import subprocess
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((v - mean)**2 for v in values) / 2)
    quantile = math.sqrt(2 * .95**2 / (1 - .95**2))
    radius = quantile * sd / math.sqrt(3)
    return mean, [mean - radius, mean + radius]


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    assert summary['saved_new_cells'] == summary['complete_noise_pairs'] == 12
    assert summary['analysis_cells'] == 36
    commit = summary['preregistration_commit']
    ct = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', commit], cwd=REPO, text=True).strip())
    for name in ['preregistration.json', *cfg['source_sha256']]:
        assert subprocess.check_output(['git', 'show', f'{commit}:{(STUDY/name).relative_to(REPO)}'], cwd=REPO) == (STUDY/name).read_bytes()
    old = json.loads((STUDY / 'executed/old_files_before.json').read_text())
    for name, item in old.items():
        assert sha(REPO/name) == item['sha256'] and (REPO/name).stat().st_mtime_ns == item['mtime_ns']
    snapshot = {str(p.relative_to(REPO)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                for p in (STUDY/'results').iterdir()}
    pairs = []
    max_loss_error = 0.0
    first_margin = math.inf
    for pair in summary['paired_seed_results']:
        fn, width, seed = pair['function'], pair['width'], pair['seed']
        gaps = []
        for source, sigma in [(REPO/cfg['previous_study'], 0),
                              (STUDY, .5), (REPO/cfg['previous_half_study'], .5)]:
            cell = f'{fn}_w{width}_s{seed}_sigma{sigma:g}'
            path = source/'results'/f'{cell}.json'
            meta = json.loads(path.read_text())
            assert sha(path.with_suffix('.npz')) == meta['arrays_sha256']
            if source == STUDY:
                assert meta['preregistration_commit'] == commit
                margin = (datetime.fromisoformat(meta['started_at']) - ct).total_seconds()
                assert margin > 0
                first_margin = min(first_margin, margin)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
                risks = [math.fsum((float(f)-float(y))**2 for f, y in zip(pred, a['test_y'])) / (2*len(a['test_y']))
                         for pred in a['test_outputs'][-1]]
                gaps.append(risks[0]-risks[1])
        expected = [pair['clean_test_gap'], pair['half_test_gap'], pair['previous_half_test_gap']]
        max_loss_error = max(max_loss_error, *(abs(x-y) for x,y in zip(gaps, expected)))
        assert max_loss_error < 1e-12
        new_d, old_d = gaps[1]-gaps[0], gaps[2]-gaps[0]
        assert abs(new_d-pair['noise_amplification']) < 1e-12
        assert abs(new_d-old_d-pair['new_minus_previous_half_D']) < 1e-12
        pairs.append({'function': fn, 'width': width, 'seed': seed,
                      'D_half': new_d, 'old_D_half': old_d, 'new_minus_old_D_half': new_d-old_d})
    hits, max_ci_error = 0, 0.0
    for unit in summary['units']:
        selected = [p for p in pairs if (p['function'],p['width']) == (unit['function'],unit['width'])]
        for key, metric in [('D_half','noise_amplification'),('new_minus_old_D_half','new_minus_previous_half_D')]:
            values = [p[key] for p in selected]
            mean, bounds = interval(values)
            saved = unit['metrics'][metric]
            assert abs(mean-saved['mean']) < 1e-12
            error = max(abs(x-y) for x,y in zip(bounds, saved['seed_95pct_t_interval']))
            assert error < 1e-10
            max_ci_error = max(max_ci_error, error)
        values = [p['D_half'] for p in selected]
        passed = math.fsum(values)/3 >= .01 and sum(v>0 for v in values) >= 2
        assert passed == unit['pass']
        hits += passed
    assert hits == summary['prediction']['passing_units']
    assert summary['prediction']['status'] == ('supported' if hits < 3 else 'refuted')
    resume = subprocess.run([sys.executable, '-B', str(STUDY/'executed/run.py')], capture_output=True, text=True)
    assert resume.returncode == 0, resume.stderr
    assert '"success": true' not in resume.stdout and resume.stdout.count('"resumed": true') == 12
    for name, item in snapshot.items():
        assert sha(REPO/name) == item['sha256'] and (REPO/name).stat().st_mtime_ns == item['mtime_ns']
    for name, item in old.items():
        assert sha(REPO/name) == item['sha256'] and (REPO/name).stat().st_mtime_ns == item['mtime_ns']
    receipt = {'status': 'passed', 'primary_pairs': pairs, 'passing_units': hits,
               'old_files_hash_mtime_unchanged': len(old), 'new_files_hash_mtime_unchanged': len(snapshot),
               'maximum_scalar_endpoint_vs_numpy_error': max_loss_error,
               'maximum_analytic_df2_vs_scipy_interval_error': max_ci_error,
               'means_and_pair_atol': 1e-12, 'interval_atol': 1e-10,
               'preregistration_before_first_cell_seconds': first_margin,
               'resume_new_cells': 0, 'resume_reused_cells': 12,
               'resume_stdout': resume.stdout, 'resume_stderr': resume.stderr,
               'new_success_snapshot': snapshot}
    (STUDY/'executed/independent_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['primary_pairs','resume_stdout','new_success_snapshot']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
