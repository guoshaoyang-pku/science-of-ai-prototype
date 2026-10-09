import json
from pathlib import Path
import numpy as np
from scipy.stats import linregress
ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'analysis/raw_summary.json').read_text())
rows = data['rows']
for r in rows:
    r['A_pass'] = bool(r['A'].get('success') and r['A'].get('normalized_rmse',1) <= 0.05 and r['A'].get('r2',0) >= 0.99)
    r['AB_ratio'] = r['A']['tc']/r['B']['tc'] if r['B']['tc'] > 0 else None
    r['AB_agree'] = 0.5 <= r['AB_ratio'] <= 2.0
by = {}
for L in sorted(set(r['depth'] for r in rows)):
    z = [r for r in rows if r['depth'] == L]
    vals = np.array([r['A']['tc'] for r in z])
    by[str(L)] = {'n':len(z), 'tc_A_mean':float(vals.mean()), 'tc_A_sd':float(vals.std(ddof=1)), 'tc_A_cv':float(vals.std(ddof=1)/vals.mean()), 'tc_B_mean':float(np.mean([r['B']['tc'] for r in z])), 'AB_ratio_min':float(min(r['AB_ratio'] for r in z)), 'AB_ratio_max':float(max(r['AB_ratio'] for r in z))}
Lm = np.array([int(k) for k in by]); ym = np.array([by[k]['tc_A_mean'] for k in by])
fit = linregress(Lm, ym); pred = fit.intercept + fit.slope*Lm; rel = np.abs(pred-ym)/ym
summary = {'study_id':'r001_chain_tc', 'n_cells':len(rows), 'A_pass_count':sum(r['A_pass'] for r in rows), 'A_pass_rate':sum(r['A_pass'] for r in rows)/len(rows), 'AB_agree_count':sum(r['AB_agree'] for r in rows), 'AB_agree_rate':sum(r['AB_agree'] for r in rows)/len(rows), 'AB_ratio_range':[float(min(r['AB_ratio'] for r in rows)), float(max(r['AB_ratio'] for r in rows))], 'seed_cv_by_depth':{k:v['tc_A_cv'] for k,v in by.items()}, 'by_depth':by, 'depth_linear_fit':{'intercept':float(fit.intercept),'slope':float(fit.slope),'r2':float(fit.rvalue**2),'predicted_relative_errors':rel.tolist(),'max_relative_error':float(rel.max())}, 'predictions':{'P1_single_sigmoid':'supported','P2_protocol_agreement':'refuted','P3_seed_robustness':'refuted','P4_depth_scaling':'refuted'}, 'rows':rows}
(ROOT/'analysis/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
