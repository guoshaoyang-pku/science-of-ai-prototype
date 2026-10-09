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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    if not values:
        return None
    return {'min': min(values), 'max': max(values), 'mean': float(np.mean(values))}


def seed_stats(values):
    a = np.asarray(values)
    half = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return {**describe(values), 'seed_95pct_t_interval': [float(a.mean()-half), float(a.mean()+half)]}


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    manifest_path = STUDY / 'executed/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    commit = manifest['preregistration_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO, check=True)
    for name in ['preregistration.json', *config['source_sha256']]:
        p = STUDY / name
        if name in config['source_sha256'] and sha(p) != config['source_sha256'][name]:
            raise RuntimeError(f'Pinned file changed: {name}')
        blob = subprocess.run(['git', 'show', f'{commit}:{p.relative_to(REPO)}'], cwd=REPO, capture_output=True, check=True).stdout
        if blob != p.read_bytes():
            raise RuntimeError('Execution commit mismatch')
    if manifest['preregistration_sha256'] != sha(STUDY / 'preregistration.json') or manifest['source_sha256'] != config['source_sha256']:
        raise RuntimeError('Manifest differs')
    baseline = json.loads((STUDY / 'executed/old_evidence_manifest.json').read_text())['files']
    for name, expected in baseline.items():
        p = REPO / name
        if sha(p) != expected['sha256'] or p.stat().st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError(f'Old evidence changed: {name}')
    forecasts = json.loads((STUDY / 'executed/analytic_forecasts.json').read_text())
    with np.load(STUDY / 'executed/input_snapshot.npz', allow_pickle=False) as z:
        inputs = {name: z[name].copy() for name in z.files}
    for name, a in inputs.items():
        if hashlib.sha256(a.tobytes()).hexdigest() != forecasts['input_arrays'][name]['sha256']:
            raise RuntimeError('Input array differs')
    expected = {}
    for seed in config['seeds']:
        for index, lam in enumerate(config['lambdas']):
            for scale in config['scales']:
                cell = {'seed': seed, 'lambda_index': index, 'lambda': lam, 'scale': scale, 'delta': lam*scale**2}
                expected[f's{seed}_l{index}_a{scale:g}'] = cell
    rows, reconstruction = [], []
    for path in sorted((STUDY / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        cell = expected[path.stem]
        request = {'cell': cell, 'bias': config['bias_root']+cell['delta'], 'manifest_sha256': sha(manifest_path)}
        contract = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if row['status']!='success' or row['cell_id']!=path.stem or row['request']!=request or row['contract_sha256']!=contract or row['arrays_sha256']!=sha(path.with_suffix('.npz')) or row['training_steps']!=0:
            raise RuntimeError('Result contract/hash mismatch')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            odd, even, center = [z[k].copy() for k in ('odd_raw','even_centered_raw','feature_center')]
            if not all(np.isfinite(z[k]).all() for k in z.files):
                raise RuntimeError('Nonfinite saved arrays')
        u, a, b = inputs[f"u_{cell['seed']}"], cell['scale'], request['bias']
        raw, ref = (b+a*u)*expit(b+a*u), (b-a*u)*expit(b-a*u)
        checks = ((raw-ref)/2, (raw+ref)/2-raw.mean(0), raw.mean(0))
        err = max(float(np.max(np.abs(x-y))) for x,y in zip((odd,even,center), checks))
        if err > config['criteria']['array_reconstruction_tolerance']:
            raise RuntimeError('Independent NumPy reconstruction failed')
        reconstruction.append(err)
        f = next(r['F'] for r in forecasts['rows'] if r['seed']==cell['seed'] and r['lambda']==cell['lambda'])
        r = float(np.mean(even*even)/np.mean(odd*odd))
        q = r/a**6
        rows.append({**cell, 'cell_id': path.stem, 'bias': b, 'R': r, 'Q': q, 'F': f,
                     'relative_F_error': abs(q/f-1), 'array_reconstruction_maxabs': err, 'seconds': row['seconds']})
    observed = {r['cell_id'] for r in rows}
    complete = observed==set(expected)
    if len(observed)!=len(rows) or {p.stem for p in (STUDY/'results').glob('*.npz')}!=observed:
        raise RuntimeError('Duplicate or orphan results')
    pairs = []
    for seed in config['seeds']:
        for lam in config['lambdas']:
            cells = sorted([r for r in rows if r['seed']==seed and r['lambda']==lam], key=lambda r:r['scale'])
            if len(cells)==len(config['scales']):
                qs = [r['Q'] for r in cells]
                pairs.append({'seed':seed,'lambda':lam,'scales':config['scales'],'Q':qs,'F':cells[0]['F'],
                              'spread':max(qs)/min(qs)-1,'large_small_ratio':qs[-1]/qs[0]})
    units=[]
    for lam in config['lambdas']:
        selected=[p for p in pairs if p['lambda']==lam]
        if selected:
            units.append({'lambda':lam,'paired_seeds':len(selected),'spread':seed_stats([p['spread'] for p in selected]),
                          'large_small_ratio':seed_stats([p['large_small_ratio'] for p in selected]),
                          'F':describe([p['F'] for p in selected])})
    p1=complete and all(p['spread']<=config['criteria']['fold_spread_max'] for p in pairs)
    p2=complete and all(r['relative_F_error']<=config['criteria']['relative_F_error_max'] for r in rows)
    summary={'study':STUDY.name,'round':80,'direction_round':5,'domain':'development','question':config['question'],
             'status':'complete' if complete else 'partial','saved_cells':len(rows),'planned_cells':config['planned_cells'],
             'lambda_scale_conditions':6,'paired_seed_lambda_units':len(pairs),'training_steps':0,
             'measurement_seconds':sum(r['seconds'] for r in rows),'preregistration_commit':commit,
             'predictions':[{'id':k,'status':('supported' if v else 'refuted') if complete else 'not_evaluated'} for k,v in [('P1',p1),('P2',p2)]],
             'fold_spread':describe([p['spread'] for p in pairs]),'relative_F_error':describe([r['relative_F_error'] for r in rows]),
             'F_range':forecasts['F_range'],'units':units,'paired_seed_results':pairs,'rows':rows,
             'verification':{'old_files_hash_mtime_unchanged':len(baseline),'max_array_reconstruction_error':max(reconstruction,default=0),
                             'historical_overlap_cells':0,'source_commit_input_contract_checks':'pass'},
             'boundary':config['boundary']}
    (STUDY/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('saved_cells','predictions','fold_spread','relative_F_error','verification')},ensure_ascii=False))


if __name__=='__main__':
    main()
