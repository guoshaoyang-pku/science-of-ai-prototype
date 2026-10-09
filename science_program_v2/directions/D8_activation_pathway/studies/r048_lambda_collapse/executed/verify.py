import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamps(paths):
    return {str(p.relative_to(STUDY)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in paths}


def main():
    summary = json.loads((STUDY/'summary.json').read_text())
    config = json.loads((STUDY/'preregistration.json').read_text())
    manifest_path = STUDY/'executed/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    commit = manifest['preregistration_commit']
    timestamp = int(subprocess.run(['git','show','-s','--format=%ct',commit],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip())
    paths = sorted((STUDY/'results').glob('*'))
    before = stamps(paths)
    forecasts = json.loads((STUDY/'executed/analytic_forecasts.json').read_text())
    b=config['bias_root']
    sig=1/(1+np.exp(-b))
    sp1=sig*(1-sig)
    sp2=sp1*(1-2*sig)
    sp3=sp1*(1-6*sig+6*sig**2)
    sp4=sp1*(1-14*sig+36*sig**2-24*sig**3)
    f1=sig+b*sp1
    f3=3*sp2+b*sp3
    f4=4*sp3+b*sp4
    fs={}
    with np.load(STUDY/'executed/input_snapshot.npz',allow_pickle=False) as z:
        for seed in config['seeds']:
            u=z[f'u_{seed}']
            v2=u**2-np.mean(u**2,axis=0)
            v4=u**4-np.mean(u**4,axis=0)
            m2=float(np.einsum('ij,ij->',u,u)/u.size)
            m22=float(np.einsum('ij,ij->',v2,v2)/u.size)
            m24=float(np.einsum('ij,ij->',v2,v4)/u.size)
            m44=float(np.einsum('ij,ij->',v4,v4)/u.size)
            for lam in config['lambdas']:
                c=lam*f3/2
                d=f4/24
                fs[seed,lam]=(c*c*m22+2*c*d*m24+d*d*m44)/(f1*f1*m2)
    max_f=max(abs(fs[r['seed'],r['lambda']]/r['F']-1) for r in forecasts['rows'])
    if max_f>1e-11:
        raise RuntimeError('Independent derivative/scalar F differs')
    rows={}
    max_r=0.
    for row in summary['rows']:
        path=STUDY/'results'/(row['cell_id']+'.json')
        saved=json.loads(path.read_text())
        if saved['arrays_sha256']!=sha(path.with_suffix('.npz')) or saved['request']['manifest_sha256']!=sha(manifest_path):
            raise RuntimeError('Saved hash differs')
        if timestamp>path.stat().st_mtime or timestamp>path.with_suffix('.npz').stat().st_mtime:
            raise RuntimeError('Measurement predates execution commit')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            odd,even=z['odd_raw'],z['even_centered_raw']
            r=float(np.einsum('ij,ij->',even,even)/np.einsum('ij,ij->',odd,odd))
        q=r/row['scale']**6
        max_r=max(max_r,abs(r/row['R']-1),abs(q/row['Q']-1))
        rows[row['seed'],row['lambda'],row['scale']]=q
    spread=[]
    for pair in summary['paired_seed_results']:
        qs=[rows[pair['seed'],pair['lambda'],a] for a in config['scales']]
        value=max(qs)/min(qs)-1
        if abs(value-pair['spread'])>1e-13:
            raise RuntimeError('Pair spread differs')
        spread.append(value)
    first=json.loads((STUDY/'executed/first_cell_audit.json').read_text())
    for name,expected in first['files'].items():
        if stamps([STUDY/name])[name]!=expected:
            raise RuntimeError('First successful cell changed')
    old=json.loads((STUDY/'executed/old_evidence_manifest.json').read_text())['files']
    for name,expected in old.items():
        p=REPO/name
        if sha(p)!=expected['sha256'] or p.stat().st_mtime_ns!=expected['mtime_ns']:
            raise RuntimeError('Old evidence changed')
    result=subprocess.run([config['execution']['python'],'-B',str(STUDY/'executed/run.py')],cwd=REPO,capture_output=True,text=True,check=True)
    recovery=json.loads(result.stdout)
    if recovery['new']!=0 or recovery['reused']!=90 or before!=stamps(paths):
        raise RuntimeError('Recovery overwrote successful results')
    audit={'status':'pass','saved_cells':len(rows),'training_steps':0,'preregistration_commit':commit,
           'commit_before_all_result_files':True,'scalar_F_relative_error_max':max_f,
           'independent_saved_R_Q_relative_difference_max':max_r,'independent_spread_range':[min(spread),max(spread)],
           'first_cell_unchanged':True,'old_files_hash_mtime_unchanged':len(old),
           'new_result_files_hash_mtime_unchanged':len(paths),'recovery':recovery,'result_files':before,
           'verification_source_sha256':sha(Path(__file__))}
    (STUDY/'executed/independent_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='result_files'}))


if __name__=='__main__':
    main()
