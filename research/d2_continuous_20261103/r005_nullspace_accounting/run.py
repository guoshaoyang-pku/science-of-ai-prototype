"""Frozen-witness nullspace accounting, with independent structural projection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import hashlib,importlib.util,json,signal,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import mpmath as mp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SPEC=json.loads((HERE/'preregistration.json').read_text());BASE=Path(SPEC['output_root']);OUT=BASE/SPEC['round_id'];PREV=BASE/SPEC['previous_round'];RAW=Path(SPEC['input_root'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def components(beta,brow,bnull,Q):
    e=beta-brow
    rr=np.einsum('ij,jk,ik->i',e,Q,e);nn=float(bnull@Q@bnull);cross=-2*e@Q@bnull
    dr=float((beta[1]-beta[0])@Q@(beta[1]+beta[0]-2*brow));dc=float(-2*(beta[1]-beta[0])@Q@bnull)
    return rr,nn,cross,dr,dc

def dual_gd(phi,y,steps):
    n=len(phi);G=phi@phi.T/n;lam,U=np.linalg.eigh(G);lam=np.asarray(lam,dtype=np.longdouble)
    # Polynomial geometric response, including every numerical eigenvalue.
    response=[]
    for t in steps:
        f=np.empty_like(lam);zero=lam==0
        f[zero]=2*np.longdouble(SPEC['eta'])*t
        f[~zero]=-np.expm1(np.longdouble(t)*np.log1p(-2*np.longdouble(SPEC['eta'])*lam[~zero]))/lam[~zero]
        response.append(np.asarray(phi.T@U@(np.asarray(f,dtype=float)*(U.T@y))/n,dtype=float))
    return np.array(response)

def structural(data,r3):
    mp.mp.dps=60
    w=mp.matrix([[mp.mpf(float(x)) for x in row] for row in data['weight']]);x=mp.matrix([[mp.mpf(float(z)) for z in row] for row in data['train_x']])
    c=mp.matrix([mp.mpf(float(z)) for z in data['feature_center']]);s=mp.mpf(float(data['feature_scale']));n=x.rows;m=w.rows
    raw=x*w.T;phi=mp.matrix(n,m+1)
    for i in range(n):
        for j in range(m):phi[i,j]=(max(mp.mpf(0),raw[i,j])-c[j])/s
        phi[i,m]=1
    B=mp.matrix(n//2+4,m+1)
    for i in range(n//2):
        for j in range(m+1):B[i,j]=(phi[2*i,j]+phi[2*i+1,j])/2
    for k in range(4):
        for j in range(m):B[n//2+k,j]=w[j,k]/s
    K,mu,lin,interaction=r3.high_precision_moments(data['weight']);Q=mp.matrix(m+1)
    for i in range(m):
        for j in range(m):Q[i,j]=(K[i,j]-mu[i]*c[j]-c[i]*mu[j]+c[i]*c[j])/(s*s)
        Q[i,m]=Q[m,i]=(mu[i]-c[i])/s
    Q[m,m]=1
    return phi,B,Q

def high_components(beta,teacher,B,Q):
    b=mp.matrix([mp.mpf(float(z)) for z in teacher]);br=B.T*mp.lu_solve(B*B.T,B*b);bn=b-br
    es=[mp.matrix([mp.mpf(float(z)) for z in row])-br for row in beta]
    rr=[mp.fdot(e,Q*e) for e in es];nn=mp.fdot(bn,Q*bn);cross=[-2*mp.fdot(e,Q*bn) for e in es]
    dr=mp.fdot(es[1]-es[0],Q*(es[1]+es[0]));dc=-2*mp.fdot(es[1]-es[0],Q*bn)
    return br,bn,rr,nn,cross,dr,dc

def main():
    fd=os.open(BASE/'active.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'round':SPEC['round_id']}).encode());os.close(fd)
    created=False;start=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('30minute budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists():raise RuntimeError('No overwrite')
        OUT.mkdir();created=True;(OUT/'decompositions').mkdir()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deps=[HERE/'run.py',HERE/'preregistration.json',HERE/'formulation.md',ROOT/'research/d2_continuous_20261103/r004_representable_target/result_manifest.json',ROOT/'research/d2_continuous_20261103/r003_population_risk/run.py',ROOT/'research/d2_continuous_20261103/r003_population_risk/preregistration.json',ROOT/'research/d2_continuous_20261103/r003_population_risk/formulation.md']
        pins={}
        for p in deps:
            rel=str(p.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)!=p.read_bytes():raise RuntimeError('Uncommitted source '+rel)
            pins[rel]=sha(p)
        execution=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),commit=commit,source_hashes=pins,spec=SPEC,numpy_version=np.__version__,mpmath_version=mp.__version__);save(OUT/'execution.json',execution)
        manifest=json.loads(deps[3].read_text())
        for item in manifest['artifacts']:
            p=PREV/item['path']
            if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise RuntimeError('Prior artifact mismatch '+str(p))
        hashes=json.loads((PREV/'execution.json').read_text())['input_hashes']
        for rel,digest in hashes.items():
            if sha(RAW/rel)!=digest:raise RuntimeError('Raw input changed '+rel)
        execution['input_hashes']=hashes;save(OUT/'execution.json',execution)
        ledger=json.loads((PREV/'rows.json').read_text());save(OUT/'frozen_previous_ledger.json',ledger)
        selected=[r for r in ledger if r.get('witness')];assert len(selected)==SPEC['conditions']
        progress=json.loads((BASE/'progress.json').read_text());progress.update(status='running',active_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        r3=load('r3_for_r005',deps[4]);cache={};rows=[];checks=[];direct=[]
        for old in selected:
            key=(old['n'],old['seed']);name=f'n{key[0]}_s{key[1]}_a{old["alpha"]:g}'
            with np.load(PREV/'models'/f'{name}.npz') as z:teacher=z['teacher'];beta=z['beta'];Q=z['Q']
            if key not in cache:
                rel=f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{key[0]}_seed{key[1]}_data.npz'
                with np.load(RAW/rel) as z:data={k:z[k] for k in z.files}
                phi=data['phi'];n,p=phi.shape
                if np.any(data['hidden_bias']!=0) or not np.array_equal(data['train_x'][::2],-data['train_x'][1::2]) or np.linalg.matrix_rank(data['train_x'][::2])!=4:raise RuntimeError('Structural preconditions failed')
                U,singular,Vh=np.linalg.svd(phi,full_matrices=False);tau=np.finfo(float).eps*max(phi.shape)*singular[0];rank=int(np.sum(singular>tau))
                gap=float(singular[rank-1]/singular[rank]) if rank<len(singular) else None
                if rank!=n//2+4 or gap is None or gap<=1e6:raise RuntimeError('Rank structural check failed')
                basis=Vh[:rank].T;P=basis@basis.T;mp_phi,B,mp_Q=structural(data,r3)
                cache[key]=(data,P,singular,tau,rank,gap,mp_phi,B,mp_Q)
            data,P,singular,tau,rank,gap,mp_phi,B,mp_Q=cache[key];phi=data['phi'];brow=P@teacher;bnull=teacher-brow
            rr,nn,cross,dr,dc=components(beta,brow,bnull,Q);total=rr+nn+cross;delta=dr+dc
            labels=phi@teacher;rowlabels=phi@brow;steps=[old['t_star'],old['t_late']]
            full=dual_gd(phi,labels,steps);rowfull=dual_gd(phi,rowlabels,steps)
            full_parts=components(full,brow,bnull,Q);row_parts=components(rowfull,brow,bnull,Q)
            fullrisks=np.einsum('ij,jk,ik->i',full-teacher,Q,full-teacher)
            coeff_error=float(max(np.max(abs(full-beta)),np.max(abs(full-rowfull))))
            dynamic_error=float(max(np.max(abs(fullrisks-[old['early_mse'],old['late_mse']])),abs(full_parts[3]+full_parts[4]-old['delta']),np.max(abs(full_parts[0]-row_parts[0])),abs(full_parts[3]-row_parts[3])))
            high=high_components(beta,teacher,B,mp_Q);hbr,hbn,hrr,hnn,hcross,hdr,hdc=high
            high_error=float(max(abs(mp.mpf(float(x))-y) for x,y in zip([*rr,nn,*cross,dr,dc],[*hrr,hnn,*hcross,hdr,hdc])))
            projection_error=float(max(abs(mp.mpf(float(x))-y) for x,y in zip(brow,hbr)))
            high_label_residual=mp.norm(mp_phi*hbn,p=mp.inf)
            label_error=float(np.max(abs(labels-rowlabels)));label_limit=1e-10*max(1,float(np.max(abs(labels))))
            membership=float(np.max(abs(beta-beta@P)));projector_error=float(max(np.max(abs(P-P.T)),np.max(abs(P@P-P))))
            orthogonality=float(abs(brow@bnull));tol=SPEC['numerical_tolerance']*max(1,old['old_R0']);threshold=SPEC['effect_fraction']*old['old_R0']
            identity_error=float(max(np.max(abs(total-[old['early_mse'],old['late_mse']])),abs(delta-old['delta'])))
            # Same stored labels imply same direct GD; preserve numerical deviations at fixed steps.
            be=np.zeros_like(teacher);br=np.zeros_like(teacher);direct_error=0.
            for t in range(max(SPEC['direct_steps'])+1):
                if t in SPEC['direct_steps']:
                    err=float(np.max(abs(be-br)));direct_error=max(direct_error,err)
                    direct.append(dict(condition=name,step=t,weight_difference=err,train_loss_difference=float(abs(np.mean((phi@be-labels)**2)-np.mean((phi@br-rowlabels)**2)))))
                if t<max(SPEC['direct_steps']):
                    be-=2*SPEC['eta']*phi.T@(phi@be-labels)/len(phi);br-=2*SPEC['eta']*phi.T@(phi@br-rowlabels)/len(phi)
            valid=label_error<=label_limit and coeff_error<=1e-8 and dynamic_error<=tol and high_error<=tol and projection_error<=1e-8 and float(high_label_residual)<=tol and membership<=1e-8 and projector_error<=tol and orthogonality<=tol and identity_error<=tol and direct_error<=1e-8
            if not np.isfinite([*rr,nn,*cross,dr,dc,delta,high_error,dynamic_error]).all():raise RuntimeError('Nonfinite result')
            margins=[-dr,-float(cross[0]),-float(cross[1]),dc-abs(dr)-threshold]
            state='undetermined' if not valid else 'confirmed' if min(margins)>tol else 'refuted' if min(margins)<-tol else 'undetermined'
            if not valid:category='numerical_undetermined'
            elif dr>tol:category='row_itself_worsens'
            elif cross[0]<-tol and cross[1]<-tol and dc>tol:category='negative_cancellation_weakens'
            elif cross[0]<-tol and cross[1]>tol:category='cancellation_turns_positive'
            elif cross[0]>tol and cross[1]>tol:category='positive_cross_term'
            else:category='near_zero_or_other'
            row=dict(n=key[0],seed=key[1],alpha=old['alpha'],t_star=steps[0],t_late=steps[1],old_R0=old['old_R0'],threshold=threshold,tolerance=tol,
                teacher_population_R0=old['projected_R0'],row_population_R0=float(brow@Q@brow),row_early=float(rr[0]),row_late=float(rr[1]),null_risk=nn,cross_early=float(cross[0]),cross_late=float(cross[1]),
                total_early=float(total[0]),total_late=float(total[1]),delta_row=dr,delta_cross=dc,delta_total=delta,previous_delta=old['delta'],train_delta=old['train_delta'],
                rank=rank,parameters=len(teacher),nullity=len(teacher)-rank,numeric_valid=valid,primary_state=state,category=category,
                high_precision_components={'row_early':mp.nstr(hrr[0],60),'row_late':mp.nstr(hrr[1],60),'null_risk':mp.nstr(hnn,60),'cross_early':mp.nstr(hcross[0],60),'cross_late':mp.nstr(hcross[1],60),'delta_row':mp.nstr(hdr,60),'delta_cross':mp.nstr(hdc,60)})
            rows.append(row)
            checks.append(dict(condition=name,rank=rank,singular_values=singular.tolist(),svd_threshold=float(tau),gap=gap,label_residual=label_error,label_limit=label_limit,high_precision_null_label_residual=mp.nstr(high_label_residual,60),projector_error=projector_error,teacher_row_null_dot=orthogonality,beta_row_residual=membership,projection_high_precision_error=projection_error,high_precision_component_error=high_error,full_dynamic_risk_error=dynamic_error,full_dynamic_weight_error=coeff_error,accounting_identity_error=identity_error,direct_weight_difference=direct_error,numeric_valid=valid))
            np.savez_compressed(OUT/'decompositions'/f'{name}.npz',teacher=teacher,beta=beta,b_row=brow,b_null=bnull,P=P,Q=Q,labels=labels,row_labels=rowlabels,full_beta=full,row_full_beta=rowfull,structural_basis=np.array(B.tolist(),dtype=float))
        states={k:sum(r['primary_state']==k for r in rows) for k in ['confirmed','refuted','undetermined']}
        summary=dict(question=SPEC['question'],selected_previous_witnesses=len(rows),previous_ledger_count=len(ledger),states=states,categories={k:sum(r['category']==k for r in rows) for k in sorted({r['category'] for r in rows})},hypothesis_status='supported_on_five_development_witnesses' if states['confirmed']==5 else 'refuted_all_five_prediction' if states['refuted'] else 'inconclusive')
        save(OUT/'rows.json',rows);save(OUT/'summary.json',summary);save(OUT/'verification.json',dict(prior_artifacts_verified=len(manifest['artifacts']),raw_inputs_verified=len(hashes),conditions=checks,direct_checks=direct))
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
