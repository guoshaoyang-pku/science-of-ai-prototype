"""Population integration of previously frozen D2 models; no new training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib, json, signal, subprocess, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
import mpmath as mp
from scipy.integrate import quad
from scipy.special import ndtr
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SPEC=json.loads((HERE/'preregistration.json').read_text())
BASE=Path(SPEC['output_root']);OUT=BASE/SPEC['round_id'];PREV=BASE/SPEC['previous_round']
RAW=Path(SPEC['input_root'])

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def density(z): return np.exp(-z*z/2)/np.sqrt(2*np.pi)
def integrate(f):
    value,error=quad(f,0,np.inf,epsabs=1e-12,epsrel=1e-12,limit=200)
    if error>1e-9*max(1,abs(value)): raise RuntimeError('Quadrature accuracy not sufficient')
    return value

def raw_moments(weight):
    radii=np.linalg.norm(weight,axis=1)
    if np.any(radii<=0): raise RuntimeError('Zero feature norm')
    rho=np.clip(weight@weight.T/np.outer(radii,radii),-1,1)
    np.fill_diagonal(rho,1.)
    theta=np.arccos(rho)
    K=np.outer(radii,radii)*(np.sin(theta)+(np.pi-theta)*rho)/(2*np.pi)
    mu=radii/np.sqrt(2*np.pi)
    linear=weight[:,0]/2
    interaction=weight[:,0]*weight[:,1]/(radii*np.sqrt(2*np.pi))
    return K,mu,linear,interaction,rho

def pair_integral(w,v):
    r=np.linalg.norm(w);s=np.linalg.norm(v);rho=np.clip(w@v/(r*s),-1,1)
    if 1-abs(rho)<1e-14: return r*s/2 if rho>0 else 0.
    tau=np.sqrt(1-rho*rho)
    return r*s*integrate(lambda z:z*density(z)*(rho*z*ndtr(rho*z/tau)+tau*density(rho*z/tau)))

def verify_moments(weight,K,linear,interaction,rho):
    pairs=list(map(tuple,SPEC['fixed_pairs']))
    mask=np.eye(len(weight),dtype=bool)
    pairs.extend([tuple(map(int,np.unravel_index(np.argmin(np.where(mask,np.inf,rho)),rho.shape))),
                  tuple(map(int,np.unravel_index(np.argmax(np.where(mask,-np.inf,rho)),rho.shape)))])
    checks=[]
    for i,j in pairs:
        direct=pair_integral(weight[i],weight[j]);err=abs(direct-K[i,j])
        checks.append(dict(i=i,j=j,quadrature=float(direct),analytic=float(K[i,j]),error=float(err)))
        if err>1e-9*max(1,abs(K[i,j])): raise RuntimeError('Pair moment failed')
    target=[]
    for i,w in enumerate(weight):
        radius=np.linalg.norm(w)
        lin=integrate(lambda z:w[0]*z*z*density(z))
        cross=integrate(lambda z:w[0]*w[1]/radius*(z**3-z)*density(z))
        err=max(abs(lin-linear[i]),abs(cross-interaction[i]))
        if err>1e-9*max(1,abs(linear[i]),abs(interaction[i])): raise RuntimeError('Target moment failed')
        target.append(dict(feature=i,linear_quad=float(lin),interaction_quad=float(cross),error=float(err)))
    return dict(pairs=checks,target_moments=target)

def high_precision_moments(weight):
    mp.mp.dps=60
    w=[[mp.mpf(float(v)) for v in row] for row in weight]
    radii=[mp.sqrt(mp.fsum(v*v for v in row)) for row in w]
    K=mp.matrix(len(w));mu=mp.matrix([r/mp.sqrt(2*mp.pi) for r in radii])
    for i in range(len(w)):
        for j in range(i,len(w)):
            rho=mp.mpf(1) if i==j else max(mp.mpf(-1),min(mp.mpf(1),mp.fsum(a*b for a,b in zip(w[i],w[j]))/(radii[i]*radii[j])))
            theta=mp.acos(rho)
            K[i,j]=K[j,i]=radii[i]*radii[j]*(mp.sin(theta)+(mp.pi-theta)*rho)/(2*mp.pi)
    return K,mu,mp.matrix([r[0]/2 for r in w]),mp.matrix([r[0]*r[1]/(norm*mp.sqrt(2*mp.pi)) for r,norm in zip(w,radii)])

def high_precision_risks(moments,beta,center,scale,alpha):
    K,mu,lin,cross=moments
    s=mp.mpf(float(scale));c=mp.matrix([mp.mpf(float(x)) for x in center]);a=mp.mpf(float(alpha))
    h=(lin+a*cross)/mp.sqrt(1+a*a)
    raw=[mp.matrix([mp.mpf(float(x))/s for x in b[:-1]]) for b in beta]
    intercept=[mp.mpf(float(b[-1]))-mp.fdot(v,c) for b,v in zip(beta,raw)]
    risks=[mp.fdot(v,K*v)+2*b*mp.fdot(v,mu)+b*b-2*mp.fdot(v,h)+1 for v,b in zip(raw,intercept)]
    da,sa=raw[1]-raw[0],raw[1]+raw[0];db,sb=intercept[1]-intercept[0],intercept[1]+intercept[0]
    delta=mp.fdot(da,K*sa)+db*mp.fdot(sa,mu)+sb*mp.fdot(da,mu)+db*sb-2*mp.fdot(da,h)
    return risks,delta

def main():
    fd=os.open(BASE/'active.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'round':SPEC['round_id']}).encode());os.close(fd)
    created=False;start=time.monotonic()
    def timeout(signum,frame): raise TimeoutError('Registered 30 minute limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists(): raise RuntimeError('Output exists; never overwrite')
        OUT.mkdir();created=True;(OUT/'moments').mkdir()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deps=[HERE/'run.py',HERE/'preregistration.json',HERE/'formulation.md',ROOT/'research/d2_continuous_20261103/r002_validation_replication/result_manifest.json']
        pins={}
        for p in deps:
            rel=str(p.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)!=p.read_bytes(): raise RuntimeError('Uncommitted dependency '+rel)
            pins[rel]=sha(p)
        execution=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),commit=commit,source_hashes=pins,spec=SPEC,numpy_version=np.__version__,scipy_version=scipy.__version__,mpmath_version=mp.__version__)
        save(OUT/'execution.json',execution)
        manifest=json.loads(deps[-1].read_text())
        for item in manifest['artifacts']:
            p=PREV/item['path']
            if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise RuntimeError('Prior artifact mismatch '+str(p))
        old_execution=json.loads((PREV/'execution.json').read_text())
        for rel,digest in old_execution['input_hashes'].items():
            if sha(RAW/rel)!=digest: raise RuntimeError('Input mismatch '+rel)
        execution['input_hashes']=old_execution['input_hashes'];save(OUT/'execution.json',execution)
        progress=json.loads((BASE/'progress.json').read_text());progress.update(status='running',active_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        previous=json.loads((PREV/'rows.json').read_text())
        weights=np.load(PREV/'frozen_weights.npz')['weights']
        assert len(previous)==SPEC['conditions'] and weights.shape==(SPEC['evaluated'],2,129)
        # Toy checks are mathematical gates, not data-driven research conditions.
        toy=np.array([[1.,0,0,0],[-1.,0,0,0],[0.,1,0,0],[1.,1,0,0]])
        tk,tm,tl,ti,_=raw_moments(toy)
        toy_err=max(abs(tk[0,0]-.5),abs(tk[0,1]),abs(tk[0,2]-1/(2*np.pi)),abs(tl[3]-.5),abs(ti[3]-1/(2*np.sqrt(np.pi))))
        if toy_err>1e-12: raise RuntimeError('Toy identity failed')
        cache={};checks={};rows=[];precise_cache={};precise_weights={}
        for old in previous:
            row={k:old[k] for k in ['n','seed','alpha','t_star','t_late','old_R0','train_delta','status']}
            row['sigma2']=0
            if old['status']!='evaluated':
                row['reason']=old['reason'];rows.append(row);continue
            key=(old['n'],old['seed'])
            if key not in cache:
                rel=f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{key[0]}_seed{key[1]}_data.npz'
                with np.load(RAW/rel) as z:data={k:z[k] for k in z.files}
                if np.any(data['hidden_bias']!=0): raise RuntimeError('Nonzero hidden bias unsupported')
                if old['seed'] not in precise_cache:
                    precise_cache[old['seed']]=high_precision_moments(data['weight']);precise_weights[old['seed']]=data['weight'].copy()
                elif not np.array_equal(precise_weights[old['seed']],data['weight']): raise RuntimeError('Feature weights differ within seed')
                K,mu,lin,cross,rho=raw_moments(data['weight']);c=data['feature_center'];s=float(data['feature_scale'])
                Q=np.empty((129,129));Q[:-1,:-1]=(K-np.outer(mu,c)-np.outer(c,mu)+np.outer(c,c))/(s*s)
                Q[:-1,-1]=(mu-c)/s;Q[-1,:-1]=Q[:-1,-1];Q[-1,-1]=1
                eig=np.linalg.eigvalsh(Q)
                if eig.min() < -1e-10*max(1,eig.max()) or np.max(abs(Q-Q.T))>1e-12: raise RuntimeError('Invalid population Gram')
                check=verify_moments(data['weight'],K,lin,cross,rho)
                check.update(min_eigenvalue=float(eig.min()),max_eigenvalue=float(eig.max()))
                checks[f'n{key[0]}_s{key[1]}']=check
                np.savez_compressed(OUT/'moments'/f'n{key[0]}_s{key[1]}.npz',Q=Q,K=K,mu=mu,linear=lin,interaction=cross,center=c,scale=s,weight=data['weight'])
                cache[key]=(Q,K,mu,lin,cross,c,s)
            Q,K,mu,lin,cross,c,s=cache[key];a=old['alpha'];hraw=(lin+a*cross)/np.sqrt(1+a*a);h=np.r_[hraw/s,0.]
            beta=weights[old['weight_index']];early,late=beta
            risks=np.einsum('ij,jk,ik->i',beta,Q,beta)-2*beta@h+1
            delta=float((late-early)@(Q@(late+early)-2*h))
            raw_a=beta[:,:-1]/s;raw_b=beta[:,-1]-raw_a@c
            raw_risk=np.einsum('ij,jk,ik->i',raw_a,K,raw_a)+2*raw_b*(raw_a@mu)+raw_b**2-2*raw_a@hraw+1
            da=raw_a[1]-raw_a[0];sa=raw_a[1]+raw_a[0];db=raw_b[1]-raw_b[0];sb=raw_b[1]+raw_b[0]
            raw_delta=float(da@K@sa+db*(sa@mu)+sb*(da@mu)+db*sb-2*da@hraw)
            error=float(max(np.max(abs(risks-raw_risk)),abs(delta-raw_delta),abs(delta-(risks[1]-risks[0]))))
            tol=SPEC['numerical_tolerance']*max(1,old['old_R0']);threshold=SPEC['effect_fraction']*old['old_R0']
            if not np.isfinite([*risks,delta,raw_delta,error]).all() or min(risks)<-tol or error>tol: raise RuntimeError('Risk numerical verification failed')
            mp_risks,mp_delta=high_precision_risks(precise_cache[old['seed']],beta,c,s,a)
            precise_error=float(max(abs(mp.mpf(float(r))-m) for r,m in zip([*risks,delta],[*mp_risks,mp_delta])))
            primary_state='numerical_uncertain' if precise_error>tol or abs(delta-threshold)<=tol else 'confirmed' if delta>threshold else 'counterexample'
            row.update(weight_index=old['weight_index'],early_mse=float(risks[0]),late_mse=float(risks[1]),delta=delta,
                high_precision_early=mp.nstr(mp_risks[0],60),high_precision_late=mp.nstr(mp_risks[1],60),high_precision_delta=mp.nstr(mp_delta,60),high_precision_error=precise_error,primary_state=primary_state,
                numerical_tolerance=tol,effect_threshold=threshold,category='numerical_uncertain' if precise_error>tol else 'positive' if delta>tol else 'negative' if delta< -tol else 'numerically_negligible',
                material_positive=primary_state=='confirmed',previous_primary_pass=old['primary_pass'],previous_mc_delta=old['delta'],
                previous_ci_lower=old['ci_lower'],previous_ci_upper=old['ci_upper'],
                within_previous_ci=old['ci_lower']-tol<=delta<=old['ci_upper']+tol,raw_expansion_error=error)
            rows.append(row)
        evaluated=[r for r in rows if r['status']=='evaluated'];primary=[r for r in evaluated if r['previous_primary_pass']];uncertain=[r for r in evaluated if not r['previous_primary_pass']]
        assert len(primary)==SPEC['primary_required'] and len(evaluated)==SPEC['evaluated']
        def aggregate(rs):return dict(count=len(rs),material_positive=sum(r['material_positive'] for r in rs),categories={k:sum(r['category']==k for r in rs) for k in ['positive','negative','numerically_negligible','numerical_uncertain']},within_previous_ci=sum(r['within_previous_ci'] for r in rs))
        primary_passes=sum(r['material_positive'] for r in primary)
        primary_states={k:sum(r['primary_state']==k for r in primary) for k in ['confirmed','counterexample','numerical_uncertain']}
        summary=dict(question=SPEC['question'],conditions=len(rows),excluded=len(rows)-len(evaluated),all_evaluated=aggregate(evaluated),previously_uncertain=aggregate(uncertain),primary=aggregate(primary),primary_states=primary_states,hypothesis_status='supported_on_fixed_development_models' if primary_passes==SPEC['primary_required'] else 'refuted' if primary_states['counterexample'] else 'inconclusive',by_alpha=[dict(alpha=a,**aggregate([r for r in evaluated if r['alpha']==a])) for a in sorted({r['alpha'] for r in rows})],by_seed=[dict(seed=s,**aggregate([r for r in evaluated if r['seed']==s])) for s in sorted({r['seed'] for r in rows})])
        save(OUT/'rows.json',rows);save(OUT/'summary.json',summary)
        save(OUT/'verification.json',dict(previous_artifacts_verified=len(manifest['artifacts']),input_files_verified=len(old_execution['input_hashes']),toy_max_error=toy_err,draws=checks,risk_checks=len(evaluated),max_raw_expansion_error=max(r['raw_expansion_error'] for r in evaluated),high_precision_checks=len(evaluated),max_high_precision_error=max(r['high_precision_error'] for r in evaluated)))
        execution.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);save(OUT/'execution.json',execution)
        save(OUT/'manifest.json',{'artifacts':[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]})
        progress.update(status='awaiting_interpretation',active_round=None);progress['rounds'].append(dict(id=SPEC['round_id'],status='computed',commit=commit));save(BASE/'progress.json',progress)
        print(json.dumps(summary,ensure_ascii=False))
    except Exception:
        if created:
            save(OUT/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start))
            progress=json.loads((BASE/'progress.json').read_text());progress.update(status='blocked_failure',active_round=None,failure_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        raise
    finally:
        signal.alarm(0);(BASE/'active.lock').unlink(missing_ok=True)
if __name__=='__main__':main()
