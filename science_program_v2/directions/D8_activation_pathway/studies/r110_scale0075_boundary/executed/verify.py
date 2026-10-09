import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import expit

from run import STUDY, REPO, gate, save, sha


def snapshot():
    return {str(p.relative_to(STUDY)):{'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns} for p in sorted((STUDY/'results').iterdir())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first',action='store_true')
    args = parser.parse_args()
    config,commit,_,inputs = gate()
    summary = json.loads((STUDY/('executed/first_cell_summary.json' if args.first else 'summary.json')).read_text())
    before = snapshot()
    assert len(before)==(2 if args.first else 24)
    forecasts = json.loads((STUDY/'executed/numeric_forecasts.json').read_text())
    root = np.longdouble(str(config['bias_root']))
    sig = 1/(1+np.exp(-root))
    coeff = np.array([0,1],dtype=np.longdouble)
    derivatives = [sig]
    for _ in range(4):
        d = np.arange(1,len(coeff))*coeff[1:]
        coeff = np.polynomial.polynomial.polymul(d,[0,1,-1])
        derivatives.append(np.polynomial.polynomial.polyval(sig,coeff))
    f1 = derivatives[0]+root*derivatives[1]
    f3 = 3*derivatives[2]+root*derivatives[3]
    f4 = 4*derivatives[3]+root*derivatives[4]
    independent_F = {}
    for seed in config['seeds']:
        u = inputs[f'u_{seed}'].astype(np.longdouble)
        u2,u4 = u**2,u**4
        u2,u4 = u2-u2.mean(0),u4-u4.mean(0)
        for lam in config['lambdas']:
            c2,c4 = f3*lam/2,f4/24
            numerator = c2*c2*np.mean(u2*u2)+2*c2*c4*np.mean(u2*u4)+c4*c4*np.mean(u4*u4)
            independent_F[(seed,lam)] = float(numerator/(f1*f1*np.mean(u*u)))
    f_error = max(abs(independent_F[(r['seed'],r['lambda'])]/r['F']-1) for r in forecasts['rows'])
    assert f_error<=1e-10
    reconstructed,energy_errors,qs = [],[],{}
    controls = json.loads((STUDY/'executed/readonly_controls.json').read_text())['rows']
    for row in controls+summary['rows']:
        path = (REPO/row['metadata_path']).with_suffix('.npz') if 'metadata_path' in row else STUDY/'results'/(row['cell_id']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            odd,even,center = [z[k].copy() for k in ('odd_raw','even_centered_raw','feature_center')]
        ratio = math.fsum(float(x)*float(x) for x in even.ravel())/math.fsum(float(x)*float(x) for x in odd.ravel())
        q = ratio/row['scale']**6
        energy_errors.extend([abs(ratio/row['R']-1),abs(q/row['Q']-1)])
        qs[(row['seed'],row['lambda'],row['scale'])] = q
        # Reconstruction is only allowed for the 12 newly measured conditions.
        if 'metadata_path' not in row:
            u,a,b = inputs[f'u_{row["seed"]}'],row['scale'],row['bias']
            raw,ref = (b+a*u)*expit(b+a*u),(b-a*u)*expit(b-a*u)
            err = max(float(np.max(np.abs(x-y))) for x,y in zip((odd,even,center),((raw-ref)/2,(raw+ref)/2-raw.mean(0),raw.mean(0))))
            assert err<=config['criteria']['array_reconstruction_tolerance']
            reconstructed.append(err)
    assert max(energy_errors)<=1e-12
    stamp = int(subprocess.run(['git','show','-s','--format=%ct',commit],cwd=REPO,capture_output=True,text=True,check=True).stdout)
    assert all(r['mtime_ns']>stamp*10**9 for r in before.values())
    result = {'status':'pass','saved_cells':summary['saved_cells'],'preregistration_commit':commit,'training_steps':0,'commit_before_all_result_files':True,'independent_derivative_moment_F_relative_error_max':f_error,'independent_fsum_R_Q_relative_difference_max':max(energy_errors),'independent_new_array_reconstruction_maxabs':max(reconstructed),'old_control_activation_calls':0,'old_files_hash_mtime_unchanged':len(json.loads((STUDY/'executed/old_evidence_manifest.json').read_text())['files']),'result_files':before,'verification_source_sha256':sha(Path(__file__))}
    if args.first:
        assert not (STUDY/'executed/first_cell_audit.json').exists()
        save(STUDY/'executed/first_cell_audit.json',result)
    else:
        p1,p2 = 0,0
        errors,spreads = [],[]
        for row in summary['rows']:
            signed = qs[(row['seed'],row['lambda'],config['scales'][0])]/independent_F[(row['seed'],row['lambda'])]-1
            errors.append(abs(signed))
            forecast = next(f for f in forecasts['rows'] if f['seed']==row['seed'] and f['lambda']==row['lambda'])
            lo,hi = forecast['absolute_relative_error_interval']
            p1 += int(lo<=abs(signed)<=hi and signed*row['lambda']<0 and abs(signed)>.02)
            vals = [qs[(row['seed'],row['lambda'],a)] for a in config['control_scales']+config['scales']]
            spread = max(vals)/min(vals)-1
            spreads.append(spread)
            lo,hi = forecast['four_scale_spread_interval']
            p2 += int(lo<=spread<=hi and spread>.02)
        assert p1==summary['predictions'][0]['passed_cells'] and p2==summary['predictions'][1]['passed_pairs']
        assert abs(max(spreads)-summary['fold_spread']['max'])<=1e-12
        assert abs(max(errors)-summary['relative_F_error']['max'])<=1e-10
        first = json.loads((STUDY/'executed/first_cell_audit.json').read_text())
        assert all(before[name]==r for name,r in first['result_files'].items())
        recovery = subprocess.run([sys.executable,str(STUDY/'executed/run.py')],cwd=REPO,capture_output=True,text=True,check=True)
        recovered = json.loads(recovery.stdout)
        assert recovered['new']==0 and recovered['reused']==12
        assert snapshot()==before
        result.update({'independent_F_error_range':[min(errors),max(errors)],'independent_spread_range':[min(spreads),max(spreads)],'independent_prediction_counts':[p1,p2],'first_cell_unchanged':True,'new_result_files_hash_mtime_unchanged':len(before),'recovery':recovered})
        assert not (STUDY/'executed/independent_verification.json').exists()
        save(STUDY/'executed/independent_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='result_files'}))


if __name__=='__main__':
    main()
