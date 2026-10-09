from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
config = json.loads((STUDY / 'preregistration.json').read_text())
summary = json.loads((STUDY / 'summary.json').read_text())
old = REPO / config['reference_study']
with np.load(STUDY / 'executed/input_snapshot.npz', allow_pickle=False) as z:
    inputs = {name: z[name].copy() for name in z.files}
rows = {}
max_scalar, max_summary, overlap = 0., 0., 0
for path in sorted((STUDY / 'results').glob('*.json')):
    saved = json.loads(path.read_text())
    cell = saved['request']['cell']
    assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == saved['arrays_sha256']
    assert saved['contract_sha256'] == hashlib.sha256(json.dumps(saved['request'], sort_keys=True).encode()).hexdigest()
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
        arrays = {name:z[name].copy() for name in z.files}
    assert all(np.isfinite(a).all() for a in arrays.values())
    measured = float(np.sum(arrays['even_centered_raw']**2)/np.sum(arrays['odd_raw']**2))
    u = inputs[f"u_{cell['seed']}"]
    v2, v4 = u**2 - (u**2).mean(0), u**4 - (u**4).mean(0)
    b, a = config['bias_root'] + cell['delta'], cell['scale']
    s = expit(b)
    # Differentiate z*sigmoid(z) via derivatives of sigmoid.
    s1 = s*(1-s)
    s2 = s1*(1-2*s)
    s3 = s1*((1-2*s)**2-2*s1)
    s4 = s1*((1-2*s)**3-8*s1*(1-2*s))
    f1, c2, c4 = s+b*s1, (2*s1+b*s2)/2, (4*s3+b*s4)/24
    predicted = float((c2*c2*a*a*np.mean(v2*v2)+2*c2*c4*a**4*np.mean(v2*v4)+c4*c4*a**6*np.mean(v4*v4))/(f1*f1*np.mean(u*u)))
    matched = next(r for r in summary['rows'] if r['cell_id']==saved['cell_id'])
    max_summary = max(max_summary, abs(measured/matched['R']-1))
    max_scalar = max(max_scalar, abs(predicted/matched['R24']-1))
    key = (cell['seed'], cell['offset_index'], a)
    assert key not in rows
    rows[key] = {'R':measured, 'predicted':predicted, 'error':abs(measured/predicted-1)}
    if cell['delta']==0 and a in [.025,.05,.1]:
        with np.load(old / f"results/s{cell['seed']}_inflection_a{a:g}.npz", allow_pickle=False) as z:
            assert np.array_equal(arrays['odd_raw'],z['odd_raw'])
            assert np.array_equal(arrays['even_centered_raw'],z['even_centered_raw'])
        overlap += 1
expected = {(seed,index,a) for seed in config['seeds'] for index in range(len(config['bias_offsets'])) for a in config['scales']}
assert set(rows)==expected
pairs = {(seed,delta):rows[(seed,index,.05)]['R']/rows[(seed,index,.025)]['R']
         for seed in config['seeds'] for index,delta in enumerate(config['bias_offsets'])}
p1 = all(r['error']<=.02 for key,r in rows.items() if key[2] in [.025,.05])
p2 = (all(60<=v<=68 for (seed,delta),v in pairs.items() if abs(delta)<=1e-5)
      and all(5<=pairs[(seed,-.01)]<=8 for seed in config['seeds'])
      and all(1.5<=pairs[(seed,.01)]<=3.5 for seed in config['seeds']))
p3 = (all(pairs[(seed,.001)]-pairs[(seed,-.001)]>=20 for seed in config['seeds'])
      and sum(pairs[(seed,.001)]>100 for seed in config['seeds'])>=2)
assert [p1,p2,p3]==[r['status']=='supported' for r in summary['predictions']]
for name,value in json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())['files'].items():
    p=REPO/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256'] and p.stat().st_mtime_ns==value['mtime_ns']
first=json.loads((STUDY/'executed/first_cell_audit.json').read_text())
for name,value in first['successful_cell_fingerprints'].items():
    p=REPO/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256'] and p.stat().st_mtime_ns==value['mtime_ns']
record = {'saved_cells':len(rows),'complete_cartesian_grid':True,'hash_contract_finite_checks':'pass',
          'independent_scalar_R24_max_relative_discrepancy':max_scalar,
          'independent_R_max_relative_discrepancy':max_summary,'predictions':[p1,p2,p3],
          'old_delta_zero_arrays_bitwise_identical_cells':overlap,'old_evidence_files_unchanged':150,
          'first_cell_hash_and_mtime_unchanged':True,'positive_001_above100':sum(pairs[(seed,.001)]>100 for seed in config['seeds']),
          'seed104_positive_001_growth':pairs[(104,.001)]}
(STUDY/'executed/independent_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record))
