"""只从保存结果复算注册的跨 n 系数极差；诊断不另增科学问题。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from executed.run import STUDY, ROOT, contract, saved, save, sha


def coefficient(rows, norm, n):
    group = [r for r in rows if r['normalization'] == norm and r['n'] == n]
    return .3 * math.fsum(r['x']*r['t_star'] for r in group) / math.fsum(r['x']**2 for r in group)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--commit', required=True); args = ap.parse_args()
    cfg = contract(args.commit); receipt = saved(cfg, args.commit)
    assert len(receipt['cells']) == 32
    target = STUDY/'summary.json'
    if target.exists():
        assert json.loads(target.read_text())['commit'] == args.commit
        print('saved summary retained'); return
    rows = []
    for c in cfg['cells']:
        with np.load(STUDY/'results'/(c['label']+'.npz'), allow_pickle=False) as z:
            r = z['expected_risk']; t = int(np.argmin(r))
            rows.append({**c, 't_star': t, 'U_rise': .3*t, 'k_point': .3*t/c['x'], 'interior': 0<t<16384,
                         'minimum_risk': float(r[t]), 'endpoint_risk': float(r[-1])})
    coeff = {norm: {str(n): coefficient(rows, norm, n) for n in [32,64]} for norm in ['sample','population']}
    spread = {norm: abs(v['64']-v['32']) for norm,v in coeff.items()}
    ratio = spread['population']/spread['sample'] if spread['sample'] > 0 else None
    passed = ratio is not None and 0<=ratio<=.5 and all(r['interior'] for r in rows)
    per_seed = []
    for seed in cfg['seeds']:
        subset = [r for r in rows if r['seed']==seed]
        item = {'seed': seed}
        for norm in ['sample','population']:
            vals = [coefficient(subset,norm,n) for n in [32,64]]
            item[norm] = {'c32': vals[0], 'c64': vals[1], 'signed_difference': vals[1]-vals[0], 'range': abs(vals[1]-vals[0])}
        per_seed.append(item)
    diagnostic = []
    for seed in cfg['seeds']:
        curves = {}
        for n in [32,64]:
            cell = next(c for c in cfg['cells'] if c['seed']==seed and c['n']==n and c['normalization']=='population')
            with np.load(STUDY/'results'/(cell['label']+'.npz'), allow_pickle=False) as z:
                curves[n] = (z['signal_bias'].copy(), n*z['variance_unit'].copy())
        b32,m32=curves[32]; b64,m64=curves[64]
        sl=slice(64,4097)
        values={'seed':seed}
        for name,left,right in [('B',b32,b64),('M',m32,m64),('delta_B',np.diff(b32),np.diff(b64)),('delta_M',np.diff(m32),np.diff(m64))]:
            values[name+'_relative_maxabs'] = float(np.max(np.abs(right[sl]-left[sl]))/np.max(np.abs(left[sl])))
        diagnostic.append(values)
    summary={'study':cfg['study'],'round':113,'direction':'D2','domain':'development','status':'complete','commit':args.commit,
             'preregistration_sha256':sha(STUDY/'preregistration.json'),
             'counts':{'evaluation_cells':32,'underlying_data_feature_draws':8,'noise_grid_per_draw':2,'normalizations':2,
                       'new_noise_grid_cells':4,'saved_noise_grid_recalculations':28,'new_training_cells':0,'seed_pairs':4},
             'coefficients':coeff,'absolute_spreads':spread,'spread_ratio':ratio,
             'P1':{'status':'supported' if passed else 'refuted','registered_interval':[0,.5],'all_minima_interior':all(r['interior'] for r in rows)},
             'per_seed':per_seed,'diagnostics':diagnostic,'cells':rows,'boundaries':cfg['boundaries']}
    save(target,summary)
    print(json.dumps({k:summary[k] for k in ['coefficients','absolute_spreads','spread_ratio','P1']},ensure_ascii=False))


if __name__=='__main__': main()
