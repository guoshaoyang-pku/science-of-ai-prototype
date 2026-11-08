"""Independent saved-state, PyTorch trajectory and SciPy calibration audit."""
from __future__ import annotations
import hashlib,json,subprocess,time
import numpy as np
import torch
from torch import nn
from scipy.optimize import nnls
from run_r002 import SOURCE,RUN,REPO
from neural_engine_checks import make_optimizer


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def np_forward(weights,x,activation):
    h=x
    for j,w in enumerate(weights):
        h=h@w.T
        if j==0 and activation=='relu':h=np.maximum(h,0)
    return h[:,0]


def np_tangent(weights,x,delta,activation):
    h=x;t=np.zeros_like(x)
    for j,(w,dw) in enumerate(zip(weights,delta)):
        z=h@w.T;t=t@w.T+h@dw.T
        if j==0 and activation=='relu':t*=z>0;h=np.maximum(z,0)
        else:h=z
    return t[:,0]


def torch_forward(weights,x,activation):
    for j,w in enumerate(weights):
        x=torch.nn.functional.linear(x,w)
        if j==0 and activation=='relu':x=torch.relu(x)
    return x[:,0]


def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    pre=json.loads((RUN/'r002_preregistration.json').read_text());receipt=json.loads((RUN/'execution_receipt.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text());stored_coeff={tuple(json.loads(k)):v for k,v in json.loads((RUN/'calibrated_coefficients.json').read_text()).items()}
    checks=0;failures=[]
    def check(name,condition,details=None):
        nonlocal checks
        checks+=1
        if not bool(condition):failures.append({'name':name,'details':details})
    def close(name,a,b,rtol=2e-9,atol=2e-11):
        aa,bb=np.asarray(a),np.asarray(b)
        check(name,aa.shape==bb.shape and np.allclose(aa,bb,rtol=rtol,atol=atol,equal_nan=True),{'max_abs_error':float(np.nanmax(np.abs(aa-bb))) if aa.size else None})
    for name,sha in pre['sources'].items():
        check('source_hash:'+name,hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha)
        rel=(SOURCE/name).relative_to(REPO).as_posix()
        check('committed_source:'+name,hashlib.sha256(subprocess.check_output(['git','-C',str(REPO),'show',receipt['preregistration_commit']+':'+rel])).hexdigest()==sha)
    caseby={c['case_id']:c for c in pre['design']['case_manifest']}
    refcaseby={c['case_id']:c for c in pre['design']['reference_cases']}
    keyfields=['teacher','input_condition','activation','width','seed']
    groupfields=keyfields[:-1]
    key=lambda c:tuple(c[k] for k in keyfields)
    group=lambda c:tuple(c[k] for k in groupfields)
    initial={};refs={};seen=[];states=0
    for entry in receipt['artifacts']:
        path=RUN/entry['file'];check('artifact_hash:'+entry['file'],hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'])
        if entry['kind']=='initial':initial[entry['file']]=load(path)
    for entry in receipt['artifacts']:
        if entry['kind']!='reference':continue
        z=load(RUN/entry['file']);init=initial[entry['initial_file']]
        for i,cid in enumerate(z['case_ids']):
            good=np.flatnonzero(~z['valid_actual'][i]);end=int(good[0]) if len(good) else z['loss'].shape[1]
            refs[key(refcaseby[str(cid)])]=(z['loss'][i,:end].copy(),float(z['eta'][i]),init['f0'][i],init['f0val'][i])
    specifications=[('SGD','identity',8,2,1.,'linear',1.,8201),('Momentum','identity',32,2,.1,'relu_teacher',10.,8201),
       ('Adam','relu',32,2,10.,'relu_teacher',10.,8201),('AdamW','relu',8,2,.1,'linear',1.,8201),
       ('RMSprop','relu',32,2,.1,'relu_teacher',10.,8202),('Adagrad','identity',32,4,10.,'relu_teacher',10.,8201),
       ('Adam','identity',8,4,1.,'linear',1.,8202),('RMSprop','relu',8,4,10.,'relu_teacher',1.,8201)]
    selected=[];calibration={}
    for entry in receipt['artifacts']:
        if entry['kind']!='candidate':continue
        z=load(RUN/entry['file']);init=initial[entry['initial_file']];depth=len([k for k in init if k.startswith('w0_')])
        for i,cid in enumerate(z['case_ids']):
            c=caseby[str(cid)];seen.append((str(cid),entry['method']));states+=1
            weights=[init[f'w0_{j}'][i] for j in range(depth)]
            actual=[z[f'final_weights_actual_{j}'][i] for j in range(depth)]
            frozen=[z[f'final_weights_const_{j}'][i] for j in range(depth)]
            close('initial_prediction',np_forward(weights,init['X'][i],c['activation']),init['f0'][i])
            close('function_pair',init['f0'][i],refs[key(c)][2],rtol=0,atol=1e-10)
            close('validation_function_pair',init['f0val'][i],refs[key(c)][3],rtol=0,atol=1e-10)
            check('full_case_length',z['loss'].shape[1]==257)
            check('first_step_cap',z['planned_first_V'][i]<=.02+1e-12)
            check('valid_masks_monotone',np.all(np.diff(z['valid_actual'][i].astype(int))<=0) and np.all(np.diff(z['valid_const'][i].astype(int))<=0))
            if z['valid_actual'][i,-1]:
                close('final_loss',.5*np.mean((np_forward(actual,init['X'][i],c['activation'])-init['y'][i])**2),z['loss'][i,-1])
                close('final_validation',.5*np.mean((np_forward(actual,init['Xval'][i],c['activation'])-init['yval'][i])**2),z['validation_loss'][i,-1])
            if z['valid_const'][i,-1]:
                delta=[v-w for v,w in zip(frozen,weights)]
                u=np.sqrt(sum(np.sum(a*a) for a in delta));r=init['f0'][i]-init['y'][i]
                df=np_tangent(weights,init['X'][i],delta,c['activation'])
                close('frozen_U',u,z['u_const'][i,-1]);close('frozen_V',np.linalg.norm(df)/np.linalg.norm(r),z['function_norm_V'][i,-1])
                close('frozen_P',-r@df/(r@r),z['function_projection_P'][i,-1])
                close('planned_V',z['planned_final_V'][i],z['function_norm_V'][i,-1],rtol=2e-8,atol=1e-10)
                close('planned_d',z['planned_final_d'][i],z['relative_d'][i,-1],rtol=2e-8,atol=1e-10)
            signature=(entry['method'],c['activation'],c['width'],c['depth'],c['balance'],c['teacher'],c['input_condition'],c['seed'])
            if signature in specifications and c['radius']==1 and c['eta_protocol']=='function_radius':selected.append((c,entry,z,i,init))
            if c['split']=='calibration' and entry['method'] in ['Adam','RMSprop','Adagrad']:
                reference,eta_ref,*_=refs[key(c)]
                targets=[]
                for value in z['loss'][i]:
                    hits=np.flatnonzero(reference<=value)
                    if not np.isfinite(value) or value>reference[0] or not len(hits):targets.append(np.nan);continue
                    j=int(hits[0]);left=max(0,j-1)
                    step=left+(reference[left]-value)/(reference[left]-reference[j]) if reference[left]!=reference[j] else float(j)
                    targets.append(step*eta_ref)
                targets=np.array(targets)
                for clock,field in [('d','relative_d'),('V','function_norm_V'),('P','function_projection_P')]:
                    x=z[field][i];mask=np.isfinite(x)&np.isfinite(targets)&(x>0)
                    if np.sum(mask)>1:calibration.setdefault((group(c),clock),[]).append((x[mask],targets[mask],np.full(np.sum(mask),1/np.sum(mask))))
    check('candidate_path_denominator',states==5184 and len(set(seen))==5184)
    check('reference_denominator',len(refs)==64)
    check('independent_trajectory_subset_size',len(selected)==len(specifications))
    coefficient_checks=0
    for (g,clock),tracks in calibration.items():
        x,y,w=[np.concatenate([t[j] for t in tracks]) for j in range(3)]
        coefficients=nnls(np.column_stack([x,x*x])*np.sqrt(w)[:,None],y*np.sqrt(w))[0]
        expected=stored_coeff[g][clock]
        close('independent_SciPy_NNLS',coefficients,[expected['A'],expected['B']],rtol=2e-8,atol=1e-9);coefficient_checks+=1
    trajectory_reports=[]
    params={'mu':.9,'beta1':.9,'beta2':.999,'alpha':.99,'eps':1e-8,'weight_decay':0.,'adagrad_init':0.}
    for c,entry,z,row,init in selected:
        depth=c['depth'];w0=[torch.tensor(init[f'w0_{j}'][row]) for j in range(depth)]
        x=torch.tensor(init['X'][row]);y=torch.tensor(init['y'][row]);xv=torch.tensor(init['Xval'][row]);yv=torch.tensor(init['yval'][row])
        real=[nn.Parameter(w.clone()) for w in w0];frozen=[nn.Parameter(w.clone()) for w in w0]
        eta=float(z['eta'][row]);optimizer=make_optimizer(entry['method'],real,eta,params);constopt=make_optimizer(entry['method'],frozen,eta,params)
        initial_loss=.5*torch.mean((torch_forward(real,x,c['activation'])-y)**2)
        g0=torch.autograd.grad(initial_loss,real)
        jac_parts=torch.autograd.functional.jacobian(lambda *ww:torch_forward(ww,x,c['activation']),tuple(w0))
        J=torch.cat([j.reshape(len(x),-1) for j in jac_parts],dim=1)
        close('explicit_autograd_kernel',J.numpy()@J.numpy().T,init['K0'][row],rtol=2e-9,atol=1e-9)
        residual=(torch_forward(w0,x,c['activation'])-y).detach();L=[];val=[];V=[];P=[]
        for t in range(257):
            loss=.5*torch.mean((torch_forward(real,x,c['activation'])-y)**2)
            L.append(float(loss.detach()));val.append(float((.5*torch.mean((torch_forward(real,xv,c['activation'])-yv)**2)).detach()))
            delta=torch.cat([(w-winit).detach().flatten() for w,winit in zip(frozen,w0)]);df=J@delta
            V.append(float(torch.linalg.norm(df)/torch.linalg.norm(residual)));P.append(float(-residual@df/(residual@residual)))
            if t==256:break
            optimizer.zero_grad();loss.backward();optimizer.step()
            for parameter,g in zip(frozen,g0):parameter.grad=g.clone()
            constopt.step()
        close('independent_full_training_path',L,z['loss'][row],rtol=2e-8,atol=2e-10)
        close('independent_full_validation_path',val,z['validation_loss'][row],rtol=2e-8,atol=2e-10)
        close('independent_full_V',V,z['function_norm_V'][row],rtol=2e-8,atol=2e-10)
        close('independent_full_P',P,z['function_projection_P'][row],rtol=2e-8,atol=2e-10)
        trajectory_reports.append({'case_id':c['case_id'],'method':entry['method'],'steps':256,'max_train_abs_error':float(np.max(np.abs(np.array(L)-z['loss'][row])))})
    # Recompute the primary paired comparison from saved scores using only the
    # preregistered confirmation selector; numerical score formulas audited above.
    rows=[json.loads(line) for line in (RUN/'trajectory_scores.jsonl').open(encoding='utf-8')]
    scoreby={(r['case_id'],r['method']):r for r in rows}
    endpointby={(r['case_id'],r['method'],r['clock'],r['q']):r for r in (json.loads(line) for line in (RUN/'endpoint_scores.jsonl').open(encoding='utf-8'))}
    scoring_paths=0;endpoint_checks=0;independent_pairs=[];depth_errors=[];depth_total=0
    for entry in receipt['artifacts']:
        if entry['kind']!='candidate':continue
        z=load(RUN/entry['file'])
        for i,cid in enumerate(z['case_ids']):
            c=caseby[str(cid)];r=scoreby[(str(cid),entry['method'])];ref,eta_ref,*_=refs[key(c)]
            observed=z['loss'][i];L0=observed[0];pred={};times={};local_scores={}
            for clock,field in [('d','relative_d'),('V','function_norm_V'),('P','function_projection_P')]:
                coeff=stored_coeff[group(c)][clock];x=z[field][i];tau=coeff['A']*x+coeff['B']*x*x
                valid=np.isfinite(tau)&(tau>=0)&(tau<=eta_ref*(len(ref)-1))&np.isfinite(observed)
                p=np.full(len(x),np.nan);p[valid]=np.interp(tau[valid]/eta_ref,np.arange(len(ref)),ref)
                mae=float(np.mean(np.abs(p[valid]-observed[valid]))/L0) if valid.any() else None
                close('independent_curve_MAE',mae,r['scores'][clock]['mae'])
                close('independent_curve_coverage',float(np.mean(valid)),r['scores'][clock]['coverage'])
                check('independent_complete_coverage',bool(valid.all())==r['scores'][clock]['complete'])
                pred[clock]=p;times[clock]=tau;local_scores[clock]=(valid,mae)
                for q in [.9,.5,.1]:
                    obs_hit=np.flatnonzero(observed<=q*L0);obs=int(obs_hit[0]) if len(obs_hit) else None
                    bad=np.flatnonzero(observed>q*L0)
                    sustained=(int(bad[-1]+1) if len(bad) and bad[-1]<256 else (0 if not len(bad) else None)) if np.isfinite(observed).all() else None
                    ref_hit=np.flatnonzero(ref<=q*L0)
                    phit=np.flatnonzero(tau>=eta_ref*ref_hit[0]) if len(ref_hit) else []
                    guess=int(phit[0]) if len(phit) else None
                    stored=endpointby[(str(cid),entry['method'],clock,q)]
                    check('independent_first_sustained_prediction',(obs,sustained,guess)==(stored['observed_first'],stored['observed_sustained'],stored['predicted']))
                    endpoint_checks+=1
            common=np.isfinite(pred['d'])&np.isfinite(pred['V'])&np.isfinite(observed)
            cm={clock:float(np.mean(np.abs(pred[clock][common]-observed[common]))/L0) if common.any() else None for clock in ['d','V']}
            close('paired_common_coverage',float(np.mean(common)),r['common_dV_coverage'])
            for clock in ['d','V']:close('paired_common_MAE',cm[clock],r['common_dV_mae'][clock])
            progress=float((L0-np.nanmin(observed))/L0)
            if c['split']=='confirmation' and c['cohort']=='core' and c['balance']!=1 and entry['method'] in ['Adam','RMSprop','Adagrad'] and progress>=.1 and np.mean(common)>=.8:
                independent_pairs.append(cm['d']-cm['V']>1e-10)
            if c['cohort']=='depth_holdout' and entry['method'] in ['Adam','RMSprop','Adagrad']:
                depth_total+=1
                if local_scores['V'][0].all():depth_errors.append(local_scores['V'][1])
            scoring_paths+=1
    candidates=[r for r in rows if r['split']=='confirmation' and r['cohort']=='core' and r['balance']!=1 and r['method'] in ['Adam','RMSprop','Adagrad']]
    eligible=[r for r in candidates if r['progress']>=.1 and r['common_dV_coverage']>=.8]
    wins=sum(r['common_dV_mae']['d']-r['common_dV_mae']['V']>1e-10 for r in eligible)
    check('P2_denominator',len(candidates)==summary['predictions']['P2']['scale_holdout_adaptive_paths'])
    check('P2_eligible',len(eligible)==summary['predictions']['P2']['eligible'])
    check('P2_wins',wins==summary['predictions']['P2']['V_better'])
    check('P2_from_raw_arrays',len(independent_pairs)==len(eligible) and sum(independent_pairs)==wins)
    close('P3_complete_coverage_raw',len(depth_errors)/depth_total,summary['predictions']['P3']['complete_fraction'])
    close('P3_complete_median_raw',float(np.median(depth_errors)),summary['predictions']['P3']['median_complete_mae'])
    report={'status':'passed' if not failures else 'failed','checks':checks,'failures':failures,'initial_and_final_paths_checked':states,
       'independent_calibration_fits':coefficient_checks,'independent_scoring_paths':scoring_paths,'independent_endpoint_records':endpoint_checks,'independent_training_subset':trajectory_reports,'P2_recomputed':{'paths':len(candidates),'eligible':len(eligible),'wins':wins},
       'scope':'all artifact hashes, initial/final identities, raw-array d/V/P scores and thresholds; independent SciPy fits for all16x3 clock groups; eight explicit full PyTorch paths and Jacobians, not all5184 paths independently rerun',
       'elapsed_seconds':time.perf_counter()-start}
    (SOURCE/'r002_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='independent_training_subset'},indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
