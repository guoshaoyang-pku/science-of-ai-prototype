"""One preregistered CPU round; outputs external, historical data read-only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import json, hashlib, importlib.util, subprocess, time, signal, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SPEC=json.loads((HERE/'preregistration.json').read_text())
BASE=Path(SPEC['output_root'])
OUT=BASE/SPEC['round_id']
RAW=Path(SPEC['input_root'])

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
legacy=load('d2_original',ROOT/'research/d2_scaling_20261009/run.py')
num=load('d2_numerics',ROOT/'research/d2_diagonal_20261027/numerics.py')

def terms_for(model,data,alpha):
    x,vx=data['train_x'],data['val_x']
    y=((x[:,0]+alpha*x[:,0]*x[:,1])/np.sqrt(1+alpha**2))[:,None]
    vy=((vx[:,0]+alpha*vx[:,0]*vx[:,1])/np.sqrt(1+alpha**2))[:,None]
    g=model['vectors'].T@data['phi'].T@y/len(y);g[~model['active']]=0
    vg=model['vectors'].T@data['val_phi'].T@vy/len(vy)
    return dict(y=y,vy=vy,g=g,h=model['hval']*(g@g.T),linear=(g*vg)[:,0],val_y2=float(np.mean(vy**2)),train_y2=float(np.mean(y**2)))

def direct(model,data,terms,t):
    f=legacy.response(model,[t])[0][0]
    w=model['vectors']@(f[:,None]*terms['g'])
    b=float(np.mean((data['val_phi']@w-terms['vy'])**2))
    a=(model['vectors']*f)@model['vectors'].T@data['phi'].T/model['n']
    noise=float(np.sum((data['val_phi']@a)**2)/len(data['val_phi']))
    return b,noise

def main():
    BASE.mkdir(exist_ok=True,parents=True)
    lock=BASE/'active.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'round':SPEC['round_id']}).encode());os.close(fd)
    def timeout_handler(signum,frame): raise TimeoutError('Registered computation budget exceeded')
    signal.signal(signal.SIGALRM,timeout_handler)
    start=time.monotonic();signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists(): raise RuntimeError('Round already exists; never overwrite or rerun completed output')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deps=[HERE/'run.py',HERE/'preregistration.json',ROOT/'research/d2_scaling_20261009/run.py',ROOT/'research/d2_scaling_20261009/preregistration.json',ROOT/'research/d2_diagonal_20261027/numerics.py',ROOT/'evidence/manifests/d2.json',ROOT/'evidence/d2_scaling_20261009/studies/n_sigma_tasks/summary.json']
        pins={}
        for p in deps:
            rel=str(p.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)!=p.read_bytes(): raise RuntimeError('Uncommitted dependency '+rel)
            pins[rel]=sha(p)
        old=json.loads((ROOT/'evidence/d2_scaling_20261009/studies/n_sigma_tasks/summary.json').read_text())
        manifests=json.loads((ROOT/'evidence/manifests/d2.json').read_text())['files']
        mp={r['path']:r for r in manifests}
        input_pins={}
        for n in SPEC['n']:
            for seed in SPEC['seeds']:
                rel=f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{n}_seed{seed}_data.npz'
                p=RAW/rel; expected=mp[rel]
                if sha(p)!=expected['sha256'] or p.stat().st_size!=expected['bytes']: raise RuntimeError('Input changed '+rel)
                input_pins[rel]=sha(p)
        OUT.mkdir();(OUT/'curves').mkdir()
        save(OUT/'execution.json',{'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'commit':commit,'source_hashes':pins,'input_hashes':input_pins,'spec':SPEC})
        grid=num.base_grid(SPEC['max_steps']);rows=[];checks=[];endpoints=[]
        for n in SPEC['n']:
            for seed in SPEC['seeds']:
                p=RAW/f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{n}_seed{seed}_data.npz'
                with np.load(p) as z: data={k:z[k] for k in z.files}
                model=legacy.spectral_model(data);noise_reference=None
                for alpha in SPEC['alpha']:
                    terms=terms_for(model,data,alpha)
                    b,noise,tb,tn=legacy.expected_curves(model,terms,grid)
                    if noise_reference is None: noise_reference=noise.copy()
                    if not np.array_equal(noise,noise_reference): raise RuntimeError('Noise response changed with target')
                    np.savez_compressed(OUT/'curves'/f'n{n}_s{seed}_a{alpha:g}.npz',t=grid,B=b,N=noise,train_B=tb,train_N=tn)
                    for sigma2 in SPEC['sigma2']:
                        objective=lambda t: (lambda parts:parts[0]+sigma2*parts[1])(legacy.expected_curves(model,terms,t))
                        t=num.select_integer(grid,b+sigma2*noise,objective)
                        late=2*t
                        vals=legacy.expected_curves(model,terms,[t,late])
                        bs,ns=vals[0],vals[1];r0=terms['val_y2'];tol=1e-10*max(1,r0)
                        db=float(bs[1]-bs[0]);dn=float(sigma2*(ns[1]-ns[0]));dr=db+dn
                        train_delta=float(vals[2][1]-vals[2][0]+sigma2*(vals[3][1]-vals[3][0]))
                        if not np.all(np.isfinite([db,dn,dr,train_delta])): raise RuntimeError('Nonfinite response')
                        if dn < -tol or train_delta > tol: raise RuntimeError('Noise/train monotonicity verification failed')
                        if late<=SPEC['max_steps'] and dr < -tol: raise RuntimeError('Search or numerical inconsistency')
                        interior=0<t<SPEC['max_steps']
                        if sigma2==0: category='no_noise_control'
                        elif not interior: category='boundary_minimum'
                        elif late>SPEC['max_steps']: category='late_outside_window'
                        elif dr/r0<=SPEC['material_rise_fraction']: category='flat_or_negligible'
                        elif db<=tol and dn>tol: category='noise_overtakes_signal'
                        elif db>tol: category='signal_also_worsens'
                        else: category='other'
                        row=dict(n=n,seed=seed,alpha=alpha,sigma2=sigma2,t_star=t,interior=interior,t_late=late,R0=r0,R_star=float(bs[0]+sigma2*ns[0]),B_star=float(bs[0]),noise_star=float(sigma2*ns[0]),delta_B=db,delta_noise=dn,delta_R=dr,delta_R_over_R0=dr/r0,category=category,expected_noisy_train_delta=train_delta,late_within_window=late<=SPEC['max_steps'])
                        rows.append(row)
                        if sigma2==1:
                            for at in [t,late]:
                                direct_b,direct_n=direct(model,data,terms,at)
                                cb,cn,*_=legacy.expected_curves(model,terms,[at])
                                err=max(abs(direct_b-cb[0]),abs(direct_n-cn[0]))
                                checks.append(float(err))
                                if err>1e-8*max(1,r0): raise RuntimeError('Feature-space verification failed')
                        if alpha in [0,.5] and sigma2>0:
                            task='linear' if alpha==0 else 'interaction'
                            oldrow=next(r for r in old['rows'] if r['protocol']=='d4_homogeneous' and r['task']==task and r['n']==n and r['seed']==seed and r['sigma2']==sigma2)
                            risk_old=float(objective([oldrow['t_star']])[0]);err=abs(risk_old-oldrow['risk_minimum'])
                            endpoints.append({'n':n,'seed':seed,'alpha':alpha,'sigma2':sigma2,'old_t':oldrow['t_star'],'new_t':t,'risk_error_at_old_t':err})
                            if err>1e-8*max(1,r0): raise RuntimeError('Saved endpoint reconstruction failed')
                print(f'completed n={n} seed={seed}',flush=True)
        assert len(rows)==SPEC['conditions']
        categories={k:sum(r['category']==k for r in rows) for k in sorted(set(r['category'] for r in rows))}
        eligible=[r for r in rows if r['category'] in ['noise_overtakes_signal','signal_also_worsens','other']]
        def fraction(rs): return sum(r['category']=='noise_overtakes_signal' for r in rs)/len(rs) if rs else None
        proportion=fraction(eligible)
        byalpha=[]
        for a in SPEC['alpha']:
            rr=[r for r in rows if r['alpha']==a];ee=[r for r in eligible if r['alpha']==a]
            byalpha.append({'alpha':a,'count':len(rr),'eligible':len(ee),'noise_overtakes_signal':sum(r['category']=='noise_overtakes_signal' for r in ee),'fraction':fraction(ee),'positive_noise_median_t':float(np.median([r['t_star'] for r in rr if r['sigma2']>0]))})
        byseed=[{'seed':s,'eligible':sum(r['seed']==s for r in eligible),'fraction':fraction([r for r in eligible if r['seed']==s])} for s in SPEC['seeds']]
        save(OUT/'rows.json',rows)
        save(OUT/'summary.json',{'question':SPEC['question'],'domain':'development','count':len(rows),'positive_noise_count':sum(r['sigma2']>0 for r in rows),'categories':categories,'eligible':len(eligible),'coverage_of_positive_noise':len(eligible)/270,'noise_overtakes_signal_of_all_positive':sum(r['category']=='noise_overtakes_signal' for r in rows)/270,'fraction':proportion,'hypothesis_status':'inconclusive' if proportion is None else 'supported_on_development' if proportion>=SPEC['threshold'] else 'refuted_on_development','by_alpha':byalpha,'by_seed':byseed})
        save(OUT/'verification.json',{'feature_space_checks':len(checks),'max_abs_error':max(checks),'endpoint_checks':endpoints,'noise_response_identical_across_alpha':True})
        execution=json.loads((OUT/'execution.json').read_text());execution.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);save(OUT/'execution.json',execution)
        artifacts=[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file()]
        save(OUT/'manifest.json',{'artifacts':artifacts})
        progress=json.loads((BASE/'progress.json').read_text());progress.update(status='awaiting_interpretation',active_round=None);progress['rounds'].append({'id':SPEC['round_id'],'status':'computed','commit':commit});save(BASE/'progress.json',progress)
        print(json.dumps({'status':'complete','output':str(OUT),'count':len(rows),'fraction':proportion,'categories':categories},ensure_ascii=False))
    except Exception:
        if OUT.exists(): save(OUT/'failure.json',{'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start})
        raise
    finally:
        signal.alarm(0);lock.unlink(missing_ok=True)
if __name__=='__main__': main()
