from pathlib import Path
import json,hashlib,subprocess,math,sys
from datetime import datetime
import numpy as np
B=Path(__file__).resolve().parents[1]; R=B.parents[3]; O=R/'directions/D7_train_gain_test_harm/studies/r047_data_seed_transfer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot():return {str(p.relative_to(R)):{'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns} for p in (B/'results').glob('*')}
def ci(v):
    a=np.asarray(v);m=float(a.mean());q=math.sqrt(2*.95**2/(1-.95**2));rad=q*float(a.std(ddof=1))/math.sqrt(3)
    return [m-rad,m+rad]
def main():
    summary=json.loads((B/'summary.json').read_text());cfg=json.loads((B/'preregistration.json').read_text());commit=summary['preregistration_commit']
    for name in ['preregistration.json',*cfg['source_sha256']]:
        assert subprocess.check_output(['git','show',f'{commit}:{(B/name).relative_to(R)}'],cwd=R)==(B/name).read_bytes()
    old=json.loads((B/'executed/old_files_before.json').read_text())
    for name,item in old.items():assert sha(R/name)==item['sha256'] and (R/name).stat().st_mtime_ns==item['mtime_ns'],name
    before=snapshot();pairs=[];max_new=0.;max_ci=0.
    for p in summary['paired_seed_results']:
        fn,w,s=p['function'],p['width'],p['seed'];vals=[]
        for sigma,source in [(0,O),(.5,B),(1,O)]:
            name=f'{fn}_w{w}_s{s}_sigma{sigma:g}';mp=source/'results'/f'{name}.json';meta=json.loads(mp.read_text());assert sha(mp.with_suffix('.npz'))==meta['arrays_sha256']
            if sigma==.5:
                ct=datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',commit],cwd=R,text=True).strip());assert ct<datetime.fromisoformat(meta['started_at'])
            with np.load(mp.with_suffix('.npz'),allow_pickle=False) as a:
                loss=np.square(a['test_outputs'][-1]-a['test_y']).mean(axis=1)/2;h=float(loss[0]-loss[1]);vals.append(h)
                if sigma==.5:
                    j,jt=a['initial_jacobian_train'],a['initial_jacobian_test'];k=np.einsum('ip,jp->ij',j,j,optimize=False)/32;cross=np.einsum('ip,jp->ij',jt,j,optimize=False)/32
                    residual=a['train_outputs'][0,1]-a['train_y'];test=a['test_outputs'][0,1].copy();train_pred=[];test_pred=[]
                    for step in range(513):
                        if step in cfg['checkpoints']:train_pred.append(residual+a['train_y']);test_pred.append(test.copy())
                        if step<512:test-=.05*np.einsum('ij,j->i',cross,residual,optimize=False);residual-=.05*np.einsum('ij,j->i',k,residual,optimize=False)
                    e=max(float(np.max(np.abs(np.asarray(train_pred)-a['train_outputs'][:,1]))),float(np.max(np.abs(np.asarray(test_pred)-a['test_outputs'][:,1]))));assert e<1e-11;max_new=max(max_new,e)
        d=vals[1]-vals[0];assert abs(d-p['noise_amplification'])<1e-12;assert abs(vals[1]-p['half_test_gap'])<1e-12;assert abs(vals[1]-vals[2]-p['half_minus_full_D'])<1e-12;pairs.append({'function':fn,'width':w,'seed':s,'D_half':d,'H_half':vals[1]})
    hits=0
    for u in summary['units']:
        ds=[p['D_half'] for p in pairs if p['function']==u['function'] and p['width']==u['width']];m=float(np.mean(ds));bounds=ci(ds);metric=u['metrics']['noise_amplification'];assert abs(m-metric['mean'])<1e-12
        e=max(abs(x-y) for x,y in zip(bounds,metric['seed_95pct_t_interval']));assert e<1e-10;max_ci=max(max_ci,e)
        passed=m>=.01 and sum(d>0 for d in ds)>=2;assert passed==u['pass'];hits+=passed
    assert hits==summary['prediction']['passing_units']==3 and summary['prediction']['status']=='supported'
    r=subprocess.run([sys.executable,str(B/'executed/run.py')],capture_output=True,text=True);assert r.returncode==0 and '"success": true' not in r.stdout and r.stdout.count('"resumed": true')==12
    assert before==snapshot()
    for name,item in old.items():assert sha(R/name)==item['sha256'] and (R/name).stat().st_mtime_ns==item['mtime_ns'],name
    gate=json.loads((B/'executed/pre_execution_audit.json').read_text());assert datetime.fromisoformat(gate['commit_timestamp'])<datetime.fromisoformat(gate['verified_at'])
    first=min(datetime.fromisoformat(json.loads(p.read_text())['started_at']) for p in (B/'results').glob('*.json'));assert datetime.fromisoformat(gate['verified_at'])<first
    receipt={'status':'passed','primary_pairs':pairs,'passing_units':hits,'old_files_hash_mtime_unchanged':len(old),'new_files_hash_mtime_unchanged':len(before),'maximum_new_tangent_checkpoint_error':max_new,'maximum_analytic_df2_vs_scipy_interval_error':max_ci,'interval_verification_atol':1e-10,'means_and_pair_verification_atol':1e-12,'resume_new_cells':0,'resume_stdout':r.stdout,'resume_stderr':r.stderr,'commit_before_gate_before_first_training':True,'new_success_snapshot':before}
    (B/'executed/independent_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['primary_pairs','resume_stdout','new_success_snapshot']},ensure_ascii=False))
if __name__=='__main__':main()
