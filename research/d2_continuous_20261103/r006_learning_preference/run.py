"""Preregistered semantic learning-rate profile using saved fixed features."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import hashlib,json,signal,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SPEC=json.loads((HERE/'preregistration.json').read_text());BASE=Path(SPEC['output_root']);OUT=BASE/SPEC['round_id'];RAW=Path(SPEC['input_root']);R3=BASE/'r003_population_risk';R1=BASE/'r001_alpha_mix'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def response(V,q,coeff,steps):
    gains=-np.expm1(np.outer(np.asarray(steps),np.log(q)))
    return np.einsum('ti,ib,ji->tjb',gains,coeff,V,optimize=True)
def threshold_info(values,level):
    hit=np.flatnonzero(values>=level)
    if not len(hit):return dict(t=None,status='censored',after_hit_recross=None)
    t=int(hit[0]);before=float(values[t-1]) if t else None;at=float(values[t])
    near=abs(at-level)<=1e-9 or (before is not None and abs(before-level)<=1e-9)
    return dict(t=t,status='threshold_near' if near else 'resolved',previous=before,value=at,after_hit_recross=bool(np.any(values[t+1:]<level-1e-9)))
def weighted_half(half,energy):
    order=np.argsort(half);weights=energy[order]/energy.sum();return float(half[order[np.searchsorted(np.cumsum(weights),.5)]])

def main():
    fd=os.open(BASE/'active.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'round':SPEC['round_id']}).encode());os.close(fd)
    created=False;start=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('30minute registered budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists():raise RuntimeError('No overwrite')
        OUT.mkdir();created=True;(OUT/'profiles').mkdir()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deps=[HERE/'run.py',HERE/'preregistration.json',HERE/'formulation.md',ROOT/'research/d2_continuous_20261103/r003_population_risk/result_manifest.json',ROOT/'research/d2_continuous_20261103/r001_alpha_mix/result_manifest.json']
        pins={}
        for p in deps:
            rel=str(p.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)!=p.read_bytes():raise RuntimeError('Uncommitted source '+rel)
            pins[rel]=sha(p)
        execution=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),commit=commit,source_hashes=pins,spec=SPEC,numpy_version=np.__version__);save(OUT/'execution.json',execution)
        counts={}
        for directory,path in [(R3,deps[3]),(R1,deps[4])]:
            manifest=json.loads(path.read_text())
            for item in manifest['artifacts']:
                p=directory/item['path']
                if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise RuntimeError('Prior artifact mismatch '+str(p))
            counts[directory.name]=len(manifest['artifacts'])
        input_hashes=json.loads((R3/'execution.json').read_text())['input_hashes']
        for rel,digest in input_hashes.items():
            if sha(RAW/rel)!=digest:raise RuntimeError('Raw mismatch '+rel)
        execution['input_hashes']=input_hashes;save(OUT/'execution.json',execution)
        oldrows=json.loads((R3/'rows.json').read_text())
        progress=json.loads((BASE/'progress.json').read_text());progress.update(status='running',active_round=SPEC['round_id']);save(BASE/'progress.json',progress)
        rows=[];checks=[];milestones=[]
        for n in SPEC['n']:
            for seed in SPEC['seeds']:
                with np.load(RAW/f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{n}_seed{seed}_data.npz') as z:data={k:z[k] for k in z.files}
                with np.load(R3/'moments'/f'n{n}_s{seed}.npz') as z:mom={k:z[k] for k in z.files}
                phi=data['phi'];x=data['train_x'];Q=mom['Q'];s=float(data['feature_scale']);H=phi.T@phi/n
                labels=np.column_stack([x[:,0],x[:,0]*x[:,1]]);mean_y=float(labels.mean(axis=0).sum()/np.sqrt(2))
                h=np.column_stack([np.r_[mom['linear']/s,0.],np.r_[mom['interaction']/s,0.]])
                lam,V=np.linalg.eigh(H);active=lam>max(lam.max()*1e-12,1e-14);q=1-2*SPEC['eta']*np.where(active,lam,0)
                if q.min()<=0 or q.max()>1:raise RuntimeError('Unstable or nonpositive q')
                center_error=float(max(np.max(abs(phi[:,:-1].mean(0))),abs(np.trace(H[:-1,:-1])-1),abs(H[-1,-1]-1)))
                if center_error>1e-12:raise RuntimeError('Center/scale fact failed')
                g=V.T@phi.T@labels/n;coeff=np.zeros_like(g);coeff[active]=g[active]/lam[active,None]
                readout=V.T@h
                # Ordered columns: LL, IL, LI, II (output component first, trained label second).
                modes=np.column_stack([coeff[:,0]*readout[:,0],coeff[:,0]*readout[:,1],coeff[:,1]*readout[:,0],coeff[:,1]*readout[:,1]])
                curve=np.empty((SPEC['max_steps']+1,4))
                for begin in range(0,len(curve),SPEC['chunk_rows']):
                    end=min(len(curve),begin+SPEC['chunk_rows']);ts=np.arange(begin,end)
                    curve[begin:end]=-np.expm1(np.outer(ts,np.log(q)))@modes
                if not np.isfinite(curve).all():raise RuntimeError('Nonfinite response')
                mixed=np.column_stack([curve[:,0]+curve[:,2],curve[:,1]+curve[:,3]])
                series={'mixed_linear':mixed[:,0],'mixed_interaction':mixed[:,1],'pure_linear':curve[:,0],'pure_interaction':curve[:,3]}
                info={name:{str(level):threshold_info(values,level) for level in SPEC['thresholds']} for name,values in series.items()}
                old=next(r for r in oldrows if r['n']==n and r['seed']==seed and r['alpha']==1)
                times=set(SPEC['check_steps']+[SPEC['max_steps'],old['t_star'],old['t_late']])
                for entries in info.values():
                    for value in entries.values():
                        if value['t'] is not None:times.update([value['t'],max(0,value['t']-1)])
                times=sorted(times);betas=response(V,q,coeff,times);A=np.eye(len(H))-2*SPEC['eta']*H
                # Full-Gram finite geometric sum via a block matrix, valid also at zero modes.
                block=np.zeros((2*len(H),2*len(H)));block[:len(H),:len(H)]=A;block[:len(H),len(H):]=np.eye(len(H));block[len(H):,len(H):]=np.eye(len(H))
                gradients=2*SPEC['eta']*phi.T@labels/n
                full_error=0.;old_error=0.;risk_identity_error=0.;constant_error=0.
                for t,b in zip(times,betas):
                    full=np.linalg.matrix_power(block,int(t))[:len(H),len(H):]@gradients
                    weight_error=float(np.max(abs(full-b)));response_error=float(np.max(abs(full.T@h-b.T@h)))
                    if weight_error>1e-8 or response_error>1e-10:raise RuntimeError('Uncut Gram boundary verification failed')
                    full_error=max(full_error,response_error)
                    student=b.sum(axis=1)/np.sqrt(2);vals=np.sqrt(2)*(student@h)
                    ef2=float(student@Q@student);risk=float(ef2-2*student@h.sum(axis=1)/np.sqrt(2)+1)
                    mismatches=(vals-1)**2/2;extra=float(ef2-np.sum((student@h)**2))
                    residual=float(abs(risk-mismatches.sum()-extra));risk_identity_error=max(risk_identity_error,residual)
                    if extra < -1e-10 or residual>1e-10:raise RuntimeError('Population risk decomposition failed')
                    constant_error=max(constant_error,abs(float(student[-1])-(1-.4**int(t))*mean_y))
                    if constant_error>1e-10:raise RuntimeError('Constant mode formula failed')
                    if t in [old['t_star'],old['t_late']]:
                        expected=old['early_mse'] if t==old['t_star'] else old['late_mse'];old_error=max(old_error,abs(risk-expected))
                    milestones.append(dict(n=n,seed=seed,t=t,A_linear=float(vals[0]),A_interaction=float(vals[1]),response_matrix=(b.T@h).T.tolist(),risk=risk,linear_mismatch=float(mismatches[0]),interaction_mismatch=float(mismatches[1]),out_of_target_energy=extra,constant_coefficient=float(student[-1]),population_mean=float(Q[-1]@student),train_mse=float(np.mean((phi@student-labels.sum(axis=1)/np.sqrt(2))**2))))
                if old_error>1e-10:raise RuntimeError('Saved baseline risks mismatch')
                direct=np.zeros((len(H),2));direct_error=0.;direct_risk_error=0.
                for t in range(max(SPEC['check_steps'])+1):
                    if t in SPEC['check_steps']:
                        b=betas[times.index(t)];err=float(np.max(abs(direct.T@h-b.T@h)));dw=float(np.max(abs(direct-b)))
                        ds=direct.sum(axis=1)/np.sqrt(2);bs=b.sum(axis=1)/np.sqrt(2)
                        dr=float(ds@Q@ds-2*ds@h.sum(axis=1)/np.sqrt(2)+1);br=float(bs@Q@bs-2*bs@h.sum(axis=1)/np.sqrt(2)+1)
                        direct_error=max(direct_error,err);direct_risk_error=max(direct_risk_error,abs(dr-br))
                        if err>1e-10 or abs(dr-br)>1e-10 or dw>1e-8:raise RuntimeError('Direct GD verification failed')
                    if t<max(SPEC['check_steps']):direct-=2*SPEC['eta']*phi.T@(phi@direct-labels)/n
                half=np.full_like(lam,np.inf);half[active]=np.log(.5)/np.log(q[active]);energy=np.zeros_like(g);energy[active]=g[active]**2/lam[active,None]
                label_energy=np.mean(labels**2,axis=0);recoverable=energy.sum(axis=0);unfit=label_energy-recoverable
                l=info['mixed_linear']['0.5'];i=info['mixed_interaction']['0.5']
                status='inconclusive' if l['status']!='resolved' or i['status']!='resolved' else 'confirmed' if l['t']<i['t'] else 'refuted'
                rows.append(dict(n=n,seed=seed,status=status,thresholds=info,sample_target_mean=mean_y,rank=int(active.sum()),mode_half_life_range=[float(half[active].min()),float(half[active].max())],training_energy_weighted_median_half_life={'linear':weighted_half(half[active],energy[active,0]),'interaction':weighted_half(half[active],energy[active,1])},training_label_energy=label_energy.tolist(),recoverable_training_energy=recoverable.tolist(),unfit_training_energy=unfit.tolist(),trajectory_ranges={name:dict(minimum=float(values.min()),maximum=float(values.max()),overshoot_above_one=float(max(0,values.max()-1))) for name,values in series.items()}))
                checks.append(dict(n=n,seed=seed,center_scale_error=center_error,full_gram_response_error=full_error,direct_gd_response_error=direct_error,direct_gd_risk_error=direct_risk_error,old_risk_error=old_error,risk_identity_error=risk_identity_error,constant_mode_error=constant_error,verified_milestones=len(times),direct_steps=SPEC['check_steps']))
                np.savez_compressed(OUT/'profiles'/f'n{n}_s{seed}.npz',t=np.arange(len(curve),dtype=np.int32),pure_responses=curve,mixed_responses=mixed,columns=np.array(['linear_from_linear','interaction_from_linear','linear_from_interaction','interaction_from_interaction']),eigenvalues=lam,q=q,active=active,eigenvectors=V,coefficients=coeff,population_readout=readout,signed_mode_readouts=modes,coefficient_half_life=half,training_energies=energy,h=h)
        assert len(rows)==SPEC['conditions'];counts_status={k:sum(r['status']==k for r in rows) for k in ['confirmed','refuted','inconclusive']}
        summary=dict(question=SPEC['question'],conditions=len(rows),states=counts_status,hypothesis_status='supported_on_nine_development_draws' if counts_status['confirmed']==9 else 'refuted_all_nine_prediction' if counts_status['refuted'] else 'inconclusive',scope=SPEC['domain'])
        save(OUT/'rows.json',rows);save(OUT/'summary.json',summary);save(OUT/'milestones.json',milestones);save(OUT/'verification.json',dict(prior_artifacts_verified=counts,raw_inputs_verified=len(input_hashes),conditions=checks))
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
