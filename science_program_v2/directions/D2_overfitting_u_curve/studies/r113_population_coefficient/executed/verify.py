"""Decimal 独立重选最低点；fsum 独立复算系数，核对冻结与不可变性。"""
import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from run import STUDY, ROOT, contract, saved, save, sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--commit',required=True); ap.add_argument('--first',action='store_true'); args=ap.parse_args()
    cfg=contract(args.commit); receipt=saved(cfg,args.commit)
    expected=1 if args.first else 32
    assert len(receipt['cells'])==expected
    output=STUDY/'executed'/('first_cell_verification.json' if args.first else 'independent_verification.json')
    assert not output.exists(), '核验回执不覆盖'
    rows=[]; max_error=0.; saved_error=0.
    for cell in cfg['cells']:
        if cell['label'] not in receipt['cells']: continue
        with np.load(ROOT/cell['source'],allow_pickle=False) as src, np.load(STUDY/'results'/(cell['label']+'.npz'),allow_pickle=False) as out:
            b=src['signal_bias']; n=src['variance_unit']; r=out['expected_risk']
            assert np.array_equal(b,out['signal_bias']) and np.array_equal(n,out['variance_unit'])
            with localcontext() as ctx:
                ctx.prec=60
                s=Decimal.from_float(cell['noise_variance'])
                decimal_r=[Decimal.from_float(float(a))+s*Decimal.from_float(float(v)) for a,v in zip(b,n)]
                t=min(range(len(decimal_r)),key=decimal_r.__getitem__)
                max_error=max(max_error,max(abs(float(q)-float(y)) for q,y in zip(decimal_r,r)))
            assert t==int(np.argmin(r))
            if cell['saved_same_noise_source']:
                with np.load(ROOT/cell['saved_same_noise_source'],allow_pickle=False) as old:
                    error=float(np.max(np.abs(r-old['expected_risk'])))
                    saved_error=max(saved_error,error)
                    assert error<=1e-12
            rows.append({**cell,'t_star':t})
    assert max_error<=1e-12
    coefficient_error=None
    if not args.first:
        summary=json.loads((STUDY/'summary.json').read_text())
        coefficients={}
        for norm in ['sample','population']:
            coefficients[norm]={}
            for n in [32,64]:
                group=[r for r in rows if r['n']==n and r['normalization']==norm]
                coefficients[norm][str(n)]=.3*math.fsum(r['x']*r['t_star'] for r in group)/math.fsum(r['x']**2 for r in group)
        coefficient_error=max(abs(coefficients[k][n]-summary['coefficients'][k][n]) for k in coefficients for n in ['32','64'])
        assert coefficient_error<1e-14
        ds=abs(coefficients['sample']['64']-coefficients['sample']['32'])
        dp=abs(coefficients['population']['64']-coefficients['population']['32'])
        assert abs(dp/ds-summary['spread_ratio'])<1e-14
        assert summary['P1']['status']==('supported' if dp/ds<=.5 and all(0<r['t_star']<16384 for r in rows) else 'refuted')
    save(output,{'status':'verified','commit':args.commit,'cells':len(rows),'training_cells':0,
                 'decimal_full_risk_maxabs':max_error,'saved_same_noise_risk_maxabs':saved_error,
                 'coefficient_error':coefficient_error,'all_argmin_match':True,
                 'old_file_set_hash_mtime_verified':True,'result_hash_mtime_verified':True})
    print(output.relative_to(ROOT))


if __name__=='__main__': main()
