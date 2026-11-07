"""Function-matched neural parameterizations, preregistered before training."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
import neural_design as design
from neural_engine import forward_batch,make_initial_cache,jvp_from_initial,initial_kernel,simulate_batch

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]
DATA=Path(os.environ.get('AIQ_KB_DATA_ROOT',str(REPO.parent/'AIQ_KB_DATA'))).resolve()
RUN=DATA/'runs/d1_scaling_exploration_20261107/r002_parameterization'
CHUNK=24
SOURCES=['run_r002.py','neural_design.py','neural_engine.py']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def task_cached(teacher,condition,seed):
    return design.task({'teacher':teacher,'input_condition':condition,'seed':seed})


def batch_inputs(selected):
    seq=[design.weights(c) for c in selected]
    weights=[torch.as_tensor(np.stack([s[i] for s in seq]),dtype=torch.float64) for i in range(selected[0]['depth'])]
    values=[task_cached(c['teacher'],c['input_condition'],c['seed']) for c in selected]
    x,y,xv,yv=[torch.as_tensor(np.stack([v[j] for v in values]),dtype=torch.float64) for j in range(4)]
    return weights,x,y,xv,yv


def norm_layers(arrays):return torch.sqrt(sum(torch.sum(a*a,dim=(1,2)) for a in arrays))


def initial_information(weights,x,y,pattern):
    params=[w.clone().requires_grad_(True) for w in weights]
    pred=forward_batch(params,x,pattern)
    loss=.5*torch.mean((pred-y)**2,dim=1)
    grads=[g.detach() for g in torch.autograd.grad(loss.sum(),params)]
    cache=make_initial_cache(weights,x,pattern)
    return grads,cache,loss.detach()


def planned_eta(selected,weights,x,y,method):
    pattern=design.activation_pattern(selected[0])
    grads,cache,initial_loss=initial_information(weights,x,y,pattern)
    eps=1e-8;T=design.STEPS
    if method=='SGD':
        unit=[-T*g for g in grads];first=[-g for g in grads]
    elif method=='Momentum':
        mu=.9;factor=T/(1-mu)-mu*(1-mu**T)/(1-mu)**2
        unit=[-factor*g for g in grads];first=[-g for g in grads]
    elif method in ['Adam','AdamW']:
        first=[-g/(g.abs()+eps) for g in grads];unit=[T*d for d in first]
    else:
        unit=[torch.zeros_like(g) for g in grads]
        first=None
        for t in range(1,T+1):
            denom_factor=np.sqrt(1-.99**t) if method=='RMSprop' else np.sqrt(t)
            updates=[-g/(denom_factor*g.abs()+eps) for g in grads]
            unit=[a+b for a,b in zip(unit,updates)]
            if t==1:first=updates
    rnorm=torch.linalg.vector_norm(cache['prediction']-y,dim=1)
    pnorm=norm_layers(weights)
    unorm=norm_layers(unit)
    vnorm=torch.linalg.vector_norm(jvp_from_initial(cache,unit),dim=1)
    first_v=torch.linalg.vector_norm(jvp_from_initial(cache,first),dim=1)/rnorm
    radius=torch.tensor([c['radius'] for c in selected],dtype=torch.float64)
    eta=torch.empty(len(selected),dtype=torch.float64)
    for i,c in enumerate(selected):
        eta[i]=radius[i]*(pnorm[i]/unorm[i] if c['eta_protocol']=='parameter_radius' else rnorm[i]/vnorm[i])
    eta_cap=.02/first_v
    eta_final=torch.minimum(eta,eta_cap)
    assert torch.isfinite(eta_final).all() and (eta_final>0).all()
    metadata={'eta_uncapped':eta.numpy(),'eta_cap':eta_cap.numpy(),'eta_capped':(eta>eta_cap).numpy(),
              'initial_loss':initial_loss.numpy(),'initial_gradient_norm':norm_layers(grads).numpy(),
              'target_radius':radius.numpy(),'planned_final_V':(eta_final*vnorm/rnorm).numpy(),
              'planned_final_d':(eta_final*unorm/pnorm).numpy(),'planned_first_V':(eta_final*first_v).numpy()}
    return eta_final.numpy(),metadata


def reference_cases(all_cases):
    result=[];seen=set()
    for c in all_cases:
        key=design.base_key(c)
        if key in seen:continue
        seen.add(key)
        row=design.canonical(c)
        row['case_id']=f'ref{len(result):03d}'
        result.append(row)
    return result


def seal():
    rows=design.cases();refs=reference_cases(rows)
    micro=json.loads((SOURCE/'neural_engine_checks.json').read_text(encoding='utf-8'))
    assert micro['status']=='passed' and micro['engine_sha256']==digest(SOURCE/'neural_engine.py')
    pre={
      'study':'r002_parameterization','question':'Do initial function-space displacement clocks retain a shared historical calibration across function-preserving parameter scale and linear-factor depth interventions better than relative parameter displacement?',
      'sources':{name:digest(SOURCE/name) for name in SOURCES},
      'design':{'case_manifest':rows,'reference_cases':refs,'candidate_paths':len(rows)*len(design.METHODS),'reference_paths':len(refs),
                'steps':design.STEPS,'reference_steps':design.REFERENCE_STEPS,'methods':design.METHODS,'chunk_size':CHUNK,
                'calibration_seeds':design.CALIBRATION_SEEDS,'confirmation_seeds':design.CONFIRMATION_SEEDS},
      'pairing':'Within teacher/covariance/activation/width/seed, every scale c and inserted linear-factor depth has the same initial input-output function. First weight multiplied by c and final weight divided by c, c>0. Deep4 inserts two trainable Linear(I) factors with Identity activations; only the first hidden layer is ReLU in that family. No bias or stochastic layer.',
      'task':'self-generated linear or one-hidden-layer ReLU teacher; fixed 64 training/256 validation Gaussian inputs in dimension8; input covariance condition parameter 1 or10; targets divided by training RMS without centering; no label noise in this round',
      'initialization':'Deep linear first factor has orthonormal columns and final factor reconstructs one fixed norm-.1 linear map per seed; ReLU first/final Gaussian factors define the fixed initial function per width and seed. Identity insertion is controlled factorization, not a claim about standard random deep-network training.',
      'optimizers':{'eta':'constant for every path, determined only from initialization','mu':.9,'betas':[.9,.999],'rms_alpha':.99,'eps':1e-8,'adagrad_initial_accumulator':0,'weight_decay':0,'bias_correction':'Adam/AdamW only','AdamW':'zero-decay exact consistency control, excluded from primary adaptive aggregation'},
      'learning_rate_protocols':{'parameter_radius':'choose eta so frozen ||delta(T)||/||theta0|| equals .25 or1','function_radius':'choose eta so ||J0 delta(T)||/||r0|| equals .25 or1',
                                'shared_initial_step_cap':'cap eta so the first frozen ||J0 delta(1)||/||r0|| <= .02; retain every capped case and report attained radius; stratify both protocols separately to expose any bias favoring a function-scale coordinate'},
      'clocks':{'d':'||delta_const||/||theta0||','V':'||J0 delta_const||/||r0||','P':'-r0 dot J0 delta_const/||r0||^2',
                'warning':'J0 delta is not invariant when the optimizer displacement is recomputed after reparameterization; it is an initial-function-scale candidate, not natural gradient'},
      'reference':'same initial function canonical balanced depth2 SGD using function-radius1 eta over256 planned steps, then run2048 real steps; retain nonmonotonicity and incomplete references',
      'calibration':'16 groups by teacher,covariance,activation,width. For each clock fit nonnegative tau=A*x+B*x^2 using only depth2,c=1,seeds8101/8102 and Adam/RMSprop/Adagrad; pool both radius and eta protocols with each training path equally weighted. tau=eta_ref*reference first-passage step. The evaluation time grid is every step0..256, not selected by target losses.',
      'prediction_rules':{
        'P1':'Every paired initial training and validation prediction agrees with canonical reference within absolute1e-10; no claim of identical gradients or kernels.',
        'P2':'On confirmation depth2 c=.1/10 adaptive paths, V has smaller paired MAE than d on more than60% of paths with >=80% common reference coverage and actual progress >=.1 of initial training loss; improvement requires >1e-10 to exclude ties; report each eta protocol independently.',
        'P3':'On held-out depth4 adaptive paths, V has >=90% complete-reference coverage and median MAE among fully covered paths <=.01 initial training loss; retain partial errors without counting them as complete passes.',
        'P4':'There is at least one paired function-preserving scale intervention that changes normalized K0 shape by >.01 and real loss history by >.01 initial loss; this is a falsifiable existence diagnostic, not a universal mechanism.'},
      'metrics':'curve MAE and max error /initial training loss and /actual achieved progress; complete/pointwise reference coverage; observed/predicted first and sustained thresholds q=.9/.5/.1; train/validation separately; clock reversal and divergence; both fixed common-support and common-complete-path comparisons',
      'class_floor':'bias-free affine least-squares training floor for identity students; nonlinear students only have nonnegative-loss lower bound. Do not infer failure from class-unattainable thresholds.',
      'diagnostics':'initial exact K0=J0J0^T via layer Gram products, K shape distance and speed scale from canonical reference; initial data/weight/function hashes; exact frozen-model loss versus initial-JVP loss; future quantities are diagnostics only',
      'failures':'nonfinite real train/validation/state or trainloss>1e10*L0 stops that path; retain prior finite entries thenNaN, no budget endpoint imputation; failed rows isolated. No retuning eta after observing training.',
      'scope':'full-batch synthetic no-bias regression; this round does not establish real minibatch/new-data/CE/GRU results; new seeds are replication not new tasks',
      'compute':'local single-thread CPU,float64; maximum1800seconds per invocation checked between completed files; explicit resume may verify and skip already saved files but never retrain them; no supervisor/paid model/solver/benchmark access',
      'kernel_checks':{'status':micro['status'],'checks':micro['checks'],'max_abs_error':micro['max_abs_error']}}
    path=SOURCE/'r002_preregistration.json';path.write_text(json.dumps(pre,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'candidate_paths':pre['design']['candidate_paths'],'reference_paths':len(refs),'sources':pre['sources']}))


def pack_initial(selected,weights,x,y,xv,yv):
    pattern=design.activation_pattern(selected[0])
    gradients,cache,loss=initial_information(weights,x,y,pattern)
    kernel=initial_kernel(weights,x,pattern).numpy()
    init={'case_ids':np.array([c['case_id'] for c in selected]),'X':x.numpy(),'y':y.numpy(),'Xval':xv.numpy(),'yval':yv.numpy(),
          'f0':cache['prediction'].numpy(),'f0val':forward_batch(weights,xv,pattern).numpy(),'initial_loss':loss.numpy(),'K0':kernel,
          'initial_loss_floor':np.array([float(np.mean((v[0]@np.linalg.lstsq(v[0],v[1],rcond=None)[0]-v[1])**2)/2) if c['activation']=='identity' else 0. for c,v in zip(selected,[task_cached(c['teacher'],c['input_condition'],c['seed']) for c in selected])])}
    for i,(w,g) in enumerate(zip(weights,gradients)):init[f'w0_{i}']=w.numpy();init[f'g0_{i}']=g.numpy()
    # Verify the function equivalence before any training for these cases.
    for i,c in enumerate(selected):
        canon=design.canonical(c);cw=[torch.tensor(w[None,:,:]) for w in design.weights(canon)]
        fc=forward_batch(cw,x[i:i+1],design.activation_pattern(canon)).numpy()[0]
        fcv=forward_batch(cw,xv[i:i+1],design.activation_pattern(canon)).numpy()[0]
        assert np.max(np.abs(fc-init['f0'][i]))<=1e-10
        assert np.max(np.abs(fcv-init['f0val'][i]))<=1e-10
    return init


def execute(commit,resume=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_default_dtype(torch.float64)
    pre_path=SOURCE/'r002_preregistration.json';pre=json.loads(pre_path.read_text(encoding='utf-8'))
    full_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse',commit],text=True).strip()
    for name,sha in pre['sources'].items():
        rel=(SOURCE/name).relative_to(REPO).as_posix()
        assert digest(SOURCE/name)==sha
        assert hashlib.sha256(subprocess.check_output(['git','-C',str(REPO),'show',f'{full_commit}:{rel}'])).hexdigest()==sha
    rel=pre_path.relative_to(REPO).as_posix()
    assert subprocess.check_output(['git','-C',str(REPO),'show',f'{full_commit}:{rel}'])==pre_path.read_bytes()
    assert not RUN.is_relative_to(REPO)
    journal_path=RUN/'journal.json'
    if RUN.exists():
        assert resume,'Existing run requires explicit --resume; saved paths will not be retrained'
        journal=json.loads(journal_path.read_text());assert journal['preregistration_commit']==full_commit
        for f in journal['artifacts']:assert digest(RUN/f['file'])==f['sha256']
    else:
        RUN.mkdir(parents=True)
        for name in SOURCES+['r002_preregistration.json']:shutil.copy2(SOURCE/name,RUN/name)
        journal={'status':'running','preregistration_commit':full_commit,'preregistration_sha256':digest(pre_path),'artifacts':[],'elapsed_seconds':0.,'torch_version':torch.__version__,'numpy_version':np.__version__}
    completed={f['file'] for f in journal['artifacts']};start=time.perf_counter();previous=journal['elapsed_seconds']
    def save(name,arrays,**metadata):
        dest=RUN/name
        np.savez_compressed(dest,**arrays)
        journal['artifacts'].append(dict(file=name,sha256=digest(dest),**metadata));completed.add(name)
        journal['elapsed_seconds']=previous+time.perf_counter()-start
        journal_path.write_text(json.dumps(journal,indent=2)+'\n')
        print(f"saved {name} paths={metadata.get('paths',0)} elapsed={journal['elapsed_seconds']:.1f}s",flush=True)
        if time.perf_counter()-start>1800:
            journal['status']='invocation_budget_truncated';journal_path.write_text(json.dumps(journal,indent=2)+'\n');raise SystemExit(3)
    for cohort,all_rows in [('reference',pre['design']['reference_cases']),('candidate',pre['design']['case_manifest'])]:
        for activation in ['identity','relu']:
            for width in [8,32]:
                for depth in ([2] if cohort=='reference' else [2,4]):
                    rows=[c for c in all_rows if (c['activation'],c['width'],c['depth'])==(activation,width,depth)]
                    for offset in range(0,len(rows),CHUNK):
                        selected=rows[offset:offset+CHUNK]
                        stem=f'{cohort}_{activation}_w{width}_d{depth}_b{offset//CHUNK:02d}'
                        init_name=stem+'_initial.npz'
                        weights,x,y,xv,yv=batch_inputs(selected)
                        if init_name not in completed:save(init_name,pack_initial(selected,weights,x,y,xv,yv),kind='initial',paths=len(selected),cohort=cohort)
                        for method in (['SGD'] if cohort=='reference' else design.METHODS):
                            name=stem+'_'+method+'.npz'
                            if name in completed:continue
                            eta,metadata=planned_eta(selected,weights,x,y,method)
                            result=simulate_batch(weights,x,y,xv,yv,eta,method,design.activation_pattern(selected[0]),design.REFERENCE_STEPS if cohort=='reference' else design.STEPS,{'weight_decay':0.})
                            arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                            for key in ['final_weights_actual','final_weights_const']:
                                for i,w in enumerate(result[key]):arrays[f'{key}_{i}']=w.numpy()
                            arrays.update(metadata,eta=eta,case_ids=np.array([c['case_id'] for c in selected]))
                            save(name,arrays,kind=cohort,method=method,paths=len(selected),initial_file=init_name)
    journal['status']='completed';journal['elapsed_seconds']=previous+time.perf_counter()-start
    journal['candidate_paths']=sum(a['paths'] for a in journal['artifacts'] if a['kind']=='candidate')
    journal['reference_paths']=sum(a['paths'] for a in journal['artifacts'] if a['kind']=='reference')
    assert journal['candidate_paths']==pre['design']['candidate_paths']
    assert journal['reference_paths']==pre['design']['reference_paths']
    journal_path.write_text(json.dumps(journal,indent=2)+'\n')
    (RUN/'execution_receipt.json').write_text(json.dumps(journal,indent=2)+'\n')
    print(json.dumps({k:v for k,v in journal.items() if k!='artifacts'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--prereg-commit');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.seal:seal()
    elif args.execute and args.prereg_commit:execute(args.prereg_commit,args.resume)
    else:ap.error('select --seal or --execute --prereg-commit HASH')
