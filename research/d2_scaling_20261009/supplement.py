#!/usr/bin/env python3
"""Post hoc descriptors; preserve the frozen scan and all successful outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

SOURCE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d2_run', SOURCE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
STUDY = runner.STUDY


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    old = json.loads((runner.LEGACY / 'studies/r003_noise_sample_scaling/summary.json').read_text())['cells']
    factors = [max(.28 * c['n'] / c['noise_variance'] / (.3 * c['t_star']), (.3 * c['t_star']) / (.28 * c['n'] / c['noise_variance'])) for c in old]
    tasks = []
    for task in summary['tasks']:
        rows = [r for r in summary['rows'] if r['protocol'] == task['protocol'] and r['task'] == task['task']]
        interior = [r for r in rows if r['interior']]
        by_n = []
        for n in sorted({r['n'] for r in rows}):
            selected = [r for r in rows if r['n'] == n]
            by_n.append({'n': n, 'rank': selected[0]['rank'], 'linear_within2': sum(r['linear_factor_error'] is not None and r['linear_factor_error'] <= 2 for r in selected), 'total': len(selected), 'boundary': sum(not r['interior'] for r in selected)})
        tasks.append({'protocol': task['protocol'], 'task': task['task'], 'train_at_rise_median': float(np.median([r['expected_train_fraction_at_rise'] for r in interior])), 'noise_at_half_median': float(np.median([r['noise_at_half_over_at_rise'] for r in interior])), 'interior': len(interior), 'by_n': by_n})
    old_run = runner.DATA_ROOT / 'runs/d2_micromanage_20261009'
    target = runner.RUN / 'studies/legacy_curve_analysis'
    if not target.exists():
        shutil.copytree(old_run / 'studies/curve_analysis_20261009', target)
    figs = runner.RUN / 'figs'
    figs.mkdir(exist_ok=True)
    for path in (old_run / 'figs').glob('legacy_*'):
        shutil.copy2(path, figs / path.name)
    post_rise = [r['curvature_inflection_over_rise'] for r in summary['rows'] if r['interior'] and r['curvature_inflection_over_rise'] is not None]
    result = {'domain': 'post_hoc_development', 'source_summary_sha256': hashlib.sha256((STUDY / 'summary.json').read_bytes()).hexdigest(), 'display_legacy_c_0_28': {'within2': sum(f <= 2 for f in factors), 'total': len(factors), 'max_factor': max(factors), 'note': 'The fitted unrounded 0.28334 coefficient passes 57/60; displayed 0.28 passes 58/60. Leave-n-out models pass 57/60.'}, 'expanded_models_within2': {name: {'passed': sum(t['models'][name]['expanded']['within2'] for t in summary['tasks']), 'total': 2700} for name in ('linear', 'power', 'log')}, 'curvature_inflections': {'found': len(post_rise), 'interior_total': summary['counts']['interior_minima'], 'ratio_quantiles': np.quantile(post_rise, [0, .1, .5, .9, 1]).tolist(), 'resolution': 'Analytic second derivative sign crossing interpolated on the registered time grid'}, 'tasks': tasks}
    (STUDY / 'supplement_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({'copied_legacy_figures': len(list(figs.glob('legacy_*'))), 'supplement': str(STUDY / 'supplement_summary.json')}))


if __name__ == '__main__':
    main()
