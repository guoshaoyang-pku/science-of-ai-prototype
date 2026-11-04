"""One preregistered target intervention. External outputs, frozen old times."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib, importlib.util, json, signal, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import mpmath as mp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SPEC=json.loads((HERE/'preregistration.json').read_text())
BASE=Path(SPEC['output_root']);OUT=BASE/SPEC['round_id'];PREV=BASE/SPEC['previous_round'];RAW=Path(SPEC['input_root'])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def response(model,teacher,steps):
    lam,V,active,q=model
    factors=np.zeros((len(steps),len(lam)))
    factors[:,active]=-np.expm1(np.outer(steps,np.log(q[active])))
    return (factors*(V.T@teacher))@V.T

def risk_pair(beta,teacher,Q):
    errors=beta-teacher
    risks=np.einsum('ij,jk,ik->i',errors,Q,errors)
    delta=float((beta[1]-beta[0])@Q@(beta[1]+beta[0]-2*teacher))
    return risks,delta

def mp_risk_pair(beta,teacher,center,scale,rawQ):
    c=mp.matrix([mp.mpf(float(v)) for v in center]);s=mp.mpf(float(scale));es=[]
    for b in beta:
        # Subtract in mp before any float64 cancellation.
        e=mp.matrix([mp.mpf(float(v))-mp.mpf(float(t)) for v,t in zip(b,teacher)])
        a=e[:-1,0]/s;intercept=e[e.rows-1]-mp.fdot(a,c)
        es.append(mp.matrix(list(a)+[intercept]))
    risks=[mp.fdot(e,rawQ*e) for e in es]
    delta=mp.fdot(es[1]-es[0],rawQ*(es[1]+es[0]))
    return risks,delta

def main():
    fd=os.open(BASE/'active.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'round':SPEC['round_id']}).encode());os.close(fd)
    created=False;start=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('Registered30minute limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists():raise RuntimeError('Existing output; no overwrite')
        OUT.mkdir();created=True;(OUT/'models').mkdir()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deps=[HERE/'run.py',HERE/'preregistration.json',HERE/'formulation.md',ROOT/'research/d2_continuous_20261103/r003_population_risk/run.py',ROOT/'research/d2_continuous_20261103/r003_population_risk/preregistration.json',ROOT/'research/d2_continuous_20261103/r003_population_risk/formulation.md',ROOT/'research/d2_continuous_20261103/r003_population_risk/result_manifest.json',ROOT/'research/d2_continuous_20261103/r002_validation_replication/result_manifest.json']
        pins={}
        for p in deps:
            rel=str(p.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)!=p.read_bytes():raise RuntimeError('Uncommitted dependency '+rel)
            pins[rel]=sha(p)
        execution=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),commit=commit,source_hashes=pins,spec=SPEC,numpy_version=np.__version__,mpmath_version=mp.__version__)
        save(OUT/'execution.json',execution)
        prev_manifest=json.loads(deps[6].read_text())
        for item in prev_manifest['artifacts']:
            p=PREV/item['path']
            if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise RuntimeError('Prior artifact mismatch '+str(p))
        input_hashes=json.loads((PREV/'execution.json').read_text())['input_hashes']
        for rel,digest in input_hashes.items():
            if sha(RAW/rel)!=digest:raise RuntimeError('Input mismatch '+rel)
        weight_path=BASE/'r002_validation_replication/frozen_weights.npz'
        weight_pin=next(r for r in json.loads(deps[7].read_text())['artifacts'] if r['path']=='frozen_weights.npz')
        if sha(weight_path)!=weight_pin['sha256']:raise RuntimeError('Frozen old weights mismatch')
        execution.update(input_hashes=input_hashes,old_weight_sha256=sha(weight_path));save(OUT/'execution.json',execution)
        progress=json.loads((BASE/'progress.json').read_text());progress.update(status='running',active_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        previous=json.loads((PREV/'rows.json').read_text());old_weights=np.load(weight_path)['weights']
        assert len(previous)==54 and sum(r.get('category')=='positive' for r in previous)==50
        save(OUT/'frozen_conditions.json',previous)
        r3=load('r003_population_moments',deps[3]);mp.mp.dps=60
        seeds={};draws={};checks=[];direct=[];rows=[];teacher_records={}
        for old in previous:
            row={k:old[k] for k in ['n','seed','alpha','t_star','t_late','old_R0','status']};row['sigma2']=0
            if old['status']!='evaluated':row['reason']=old['reason'];rows.append(row);continue
            key=(old['n'],old['seed']);seed=old['seed'];alpha=old['alpha'];tol=SPEC['numerical_tolerance']*max(1,old['old_R0'])
            if key not in draws:
                rel=f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{key[0]}_seed{key[1]}_data.npz'
                with np.load(RAW/rel) as z:data={k:z[k] for k in z.files}
                with np.load(PREV/'moments'/f'n{key[0]}_s{key[1]}.npz') as z:mom={k:z[k] for k in z.files}
                Q=mom['Q'];phi=data['phi'];H=phi.T@phi/len(phi);lam,V=np.linalg.eigh(H);active=lam>max(lam.max()*1e-12,1e-14)
                q=1-2*SPEC['eta']*np.where(active,lam,0)
                if q.min()<=0 or q.max()>1 or np.linalg.eigvalsh(Q).min()<=0:raise RuntimeError('Invalid stable GD or population Gram')
                model=(lam,V,active,q);A=np.eye(len(H))-2*SPEC['eta']*H
                draws[key]=(data,mom,model,A)
                if seed not in seeds:
                    K,mu,linear,cross=r3.high_precision_moments(data['weight']);m=len(mu);rawQ=mp.matrix(m+1)
                    rawQ[:m,:m]=K;rawQ[:m,m]=mu;rawQ[m,:m]=mu.T;rawQ[m,m]=1
                    lin=mp.matrix(list(linear)+[0]);interaction=mp.matrix(list(cross)+[0])
                    bl=mp.lu_solve(rawQ,lin);bc=mp.lu_solve(rawQ,interaction)
                    seeds[seed]=(rawQ,bl,bc,data['weight'].copy())
                    teacher_records[str(seed)]={'raw_linear_coefficients':[mp.nstr(v,60) for v in bl],'raw_interaction_coefficients':[mp.nstr(v,60) for v in bc],'coordinates':'128 raw ReLU coefficients then constant','shared_across_n':True}
                elif not np.array_equal(seeds[seed][3],data['weight']):raise RuntimeError('Seed feature space changed across n')
            data,mom,model,A=draws[key];Q=mom['Q'];c=data['feature_center'];s=float(data['feature_scale']);phi=data['phi'];rawQ,bl,bc,_=seeds[seed]
            raw_teacher=(bl+mp.mpf(alpha)*bc)/mp.sqrt(1+mp.mpf(alpha)**2)
            teacher=np.array([float(mp.mpf(s)*v) for v in raw_teacher[:-1,0]]+[float(raw_teacher[raw_teacher.rows-1]+mp.fdot(raw_teacher[:-1,0],[mp.mpf(float(v)) for v in c]))])
            h=np.r_[(mom['linear']+alpha*mom['interaction'])/(s*np.sqrt(1+alpha*alpha)),0.]
            ortho=float(np.max(abs(Q@teacher-h))/max(1,np.max(abs(h))))
            float_teacher=np.linalg.solve(Q,h);difference=float_teacher-teacher
            teacher_q_error=float(difference@Q@difference)
            P=float(teacher@Q@teacher);floor=float(1-2*teacher@h+P)
            teacher_valid=ortho<=1e-10 and abs(teacher_q_error)<=1e-12 and P>1e-8 and floor>=-tol
            steps=[old['t_star'],old['t_late']];beta=response(model,teacher,steps);target=phi@teacher
            risks,delta=risk_pair(beta,teacher,Q)
            train=np.mean((phi@beta.T-target[:,None])**2,axis=0);train_delta=float(train[1]-train[0])
            # Full Gram recurrence retains every numerical mode, independently of eigencutoff.
            full_beta=np.array([teacher-np.linalg.matrix_power(A,int(t))@teacher for t in steps])
            full_risks,full_delta=risk_pair(full_beta,teacher,Q)
            dynamics_error=float(max(np.max(abs(full_risks-risks)),abs(full_delta-delta)))
            if dynamics_error>1e-8:raise RuntimeError('Full-Gram GD verification failed')
            high_risks,high_delta=mp_risk_pair(beta,teacher,c,s,rawQ)
            high_error=float(max(abs(mp.mpf(float(x))-y) for x,y in zip([*risks,delta],[*high_risks,high_delta])))
            original_metric=np.einsum('ij,jk,ik->i',beta,Q,beta)-2*beta@h+1
            identity_error=float(np.max(abs(original_metric-risks-floor)))
            # Restore original target only as a baseline integrity gate, never refit times.
            x=data['train_x'];y=(x[:,0]+alpha*x[:,0]*x[:,1])/np.sqrt(1+alpha*alpha)
            lam,V,active,q=model;g=V.T@phi.T@y/len(y);f=np.zeros((2,len(lam)))
            f[:,active]=-np.expm1(np.outer(steps,np.log(q[active])))/lam[active]
            original_beta=(f*g)@V.T
            original_delta=float((original_beta[1]-original_beta[0])@(Q@(original_beta[1]+original_beta[0])-2*h))
            baseline_error=float(max(np.max(abs(original_beta-old_weights[old['weight_index']])),abs(original_delta-old['delta'])))
            if baseline_error>1e-8 or abs(original_delta-old['delta'])>tol:raise RuntimeError('Old baseline reconstruction failed')
            if alpha==.5:
                b=np.zeros(len(teacher));frozen=set(SPEC['direct_steps'])
                expected=response(model,teacher,SPEC['direct_steps'])
                for step in range(max(frozen)+1):
                    if step in frozen:
                        at=SPEC['direct_steps'].index(step);candidate=expected[at]
                        weight_error=float(np.max(abs(candidate-b)))
                        dr=float((b-teacher)@Q@(b-teacher));sr=float((candidate-teacher)@Q@(candidate-teacher))
                        tr=float(np.mean((phi@b-target)**2));st=float(np.mean((phi@candidate-target)**2))
                        err=max(abs(dr-sr),abs(tr-st))
                        direct.append(dict(n=key[0],seed=key[1],step=step,weight_error=weight_error,risk_error=err))
                        if weight_error>1e-8 or err>1e-10:raise RuntimeError('Direct GD gate failed')
                    if step<max(frozen):b-=2*SPEC['eta']*phi.T@(phi@b-target)/len(target)
            numeric_valid=teacher_valid and high_error<=tol and dynamics_error<=tol and identity_error<=tol
            if not np.isfinite([*risks,delta,*train,P,floor,high_error,dynamics_error]).all() or min(risks)<-tol or train_delta>tol:raise RuntimeError('Invalid risk/train result')
            threshold=SPEC['effect_fraction']*old['old_R0']
            qualified=numeric_valid and delta>threshold+tol and train_delta< -tol
            category='undetermined' if not numeric_valid else 'positive' if delta>tol else 'negative' if delta< -tol else 'numerically_negligible'
            row.update(primary_member=old['category']=='positive',original_delta=old['delta'],original_train_delta=old['train_delta'],
                original_category=old['category'],early_mse=float(risks[0]),late_mse=float(risks[1]),delta=delta,delta_over_projected_R0=delta/P,
                projected_R0=P,residual_floor=floor,train_early=float(train[0]),train_late=float(train[1]),train_delta=train_delta,
                threshold=threshold,tolerance=tol,numeric_valid=numeric_valid,category=category,witness=qualified,
                delta_change=delta-old['delta'],comparison='undetermined' if not numeric_valid else 'reduced' if delta<old['delta']-tol else 'increased' if delta>old['delta']+tol else 'unchanged',
                high_precision_early=mp.nstr(high_risks[0],60),high_precision_late=mp.nstr(high_risks[1],60),high_precision_delta=mp.nstr(high_delta,60))
            name=f'n{key[0]}_s{key[1]}_a{alpha:g}'
            np.savez_compressed(OUT/'models'/f'{name}.npz',teacher=teacher,beta=beta,train_target=target,full_gram_beta=full_beta,Q=Q,h=h,steps=steps)
            checks.append(dict(condition=name,teacher_valid=teacher_valid,orthogonality_residual=ortho,teacher_q_error=teacher_q_error,projection_energy_identity_error=abs(P-float(teacher@h)),
                baseline_error=baseline_error,high_precision_error=high_error,full_gram_risk_error=dynamics_error,full_gram_weight_error=float(np.max(abs(beta-full_beta))),risk_identity_error=identity_error,numeric_valid=numeric_valid))
            rows.append(row)
        evaluated=[r for r in rows if r['status']=='evaluated'];primary=[r for r in evaluated if r['primary_member']]
        assert len(evaluated)==SPEC['evaluated'] and len(primary)==SPEC['primary_denominator']
        def aggregate(rs):return dict(count=len(rs),witnesses=sum(r['witness'] for r in rs),categories={k:sum(r['category']==k for r in rs) for k in ['positive','negative','numerically_negligible','undetermined']},comparisons={k:sum(r['comparison']==k for r in rs) for k in ['reduced','increased','unchanged','undetermined']})
        summary=dict(question=SPEC['question'],conditions=len(rows),excluded=3,primary=aggregate(primary),all_evaluated=aggregate(evaluated),original_negative=aggregate([r for r in evaluated if not r['primary_member']]),hypothesis_status='witness_found_on_development' if any(r['witness'] for r in primary) else 'not_found',by_alpha=[dict(alpha=a,**aggregate([r for r in primary if r['alpha']==a])) for a in sorted({r['alpha'] for r in rows})],by_seed=[dict(seed=s,**aggregate([r for r in primary if r['seed']==s])) for s in sorted({r['seed'] for r in rows})])
        save(OUT/'rows.json',rows);save(OUT/'summary.json',summary);save(OUT/'teacher_bases_60decimal.json',teacher_records)
        save(OUT/'verification.json',dict(previous_artifacts_verified=len(prev_manifest['artifacts']),raw_inputs_verified=len(input_hashes),conditions=checks,direct_gd_checks=direct))
        execution.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);save(OUT/'execution.json',execution)
        save(OUT/'manifest.json',{'artifacts':[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]})
        progress.update(status='awaiting_interpretation',active_round=None);progress['rounds'].append(dict(id=SPEC['round_id'],status='computed',commit=commit));save(BASE/'progress.json',progress)
        print(json.dumps(summary,ensure_ascii=False))
    except Exception:
        if created:
            save(OUT/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start))
            progress=json.loads((BASE/'progress.json').read_text());progress.update(status='blocked_failure',active_round=None,failure_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        raise
    finally:signal.alarm(0);(BASE/'active.lock').unlink(missing_ok=True)
if __name__=='__main__':main()
