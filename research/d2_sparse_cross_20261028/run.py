#!/usr/bin/env python3
"""Preregistered, fixed-budget cross-term selection; saved-data analysis only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from collections import defaultdict
import numpy as np
import scipy
from sparse_numerics import pair_indices, rank_pairs, select_top, masked_h, budget_size

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPEC=json.loads((HERE/'preregistration.json').read_text())
DATA_ROOT=Path(os.environ.get('AIQ_KB_DATA_ROOT',SPEC['data_root'])).resolve()
OUT=DATA_ROOT/SPEC['run_relative']
PREVIOUS=DATA_ROOT/SPEC['previous_run']
INPUT=Path(SPEC['input_root'])
REL=Path(SPEC['source_study'])
RAW=INPUT/REL
METHODS=['change05','magnitude05']+[f'random05_{i:02d}' for i in range(10)]


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


legacy=load_module('saved_d2_source',ROOT/'research/d2_scaling_20261009/run.py')
num=load_module('saved_d2_numerics',ROOT/'research/d2_diagonal_20261027/numerics.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def contract():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[HERE/n for n in ['preregistration.json','run.py','sparse_numerics.py','test_sparse_numerics.py','test_runner.py']]
    files += [ROOT/p for p in ['research/d2_scaling_20261009/run.py','research/d2_scaling_20261009/preregistration.json',
                              'research/d2_diagonal_20261027/numerics.py','research/d2_diagonal_20261027/result_manifest.json',
                              'evidence/manifests/d2.json','evidence/handoff_manifest.json']]
    pins={}
    for p in files:
        rel=str(p.relative_to(ROOT))
        if subprocess.check_output(['git','show',f'{head}:{rel}'],cwd=ROOT)!=p.read_bytes():
            raise RuntimeError(f'Uncommitted dependency: {rel}')
        pins[rel]=digest(p)
    return {'commit':head,'files':pins}


def verify_inputs():
    manifest=json.loads((ROOT/'evidence/manifests/d2.json').read_text())
    for r in manifest['files']:
        p=(INPUT if r['delivery']=='release' else ROOT)/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise RuntimeError(f'Changed D2 input: {r["path"]}')
    receipts={}
    for p in sorted((RAW/'results').glob('*.json')):
        for rel,pin in json.loads(p.read_text())['files'].items():
            if digest(RAW/rel)!=pin:raise RuntimeError(f'Changed receipt payload: {rel}')
            receipts[rel]=pin
    prior_manifest=json.loads((ROOT/'research/d2_diagonal_20261027/result_manifest.json').read_text())
    for r in prior_manifest['artifacts']:
        p=PREVIOUS/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise RuntimeError(f'Changed previous run artifact: {r["path"]}')
    rows=json.loads((ROOT/REL/'summary.json').read_text())['rows']
    old_diag={r['curve']:r for r in map(json.loads,(PREVIOUS/'rows.jsonl').read_text().splitlines())}
    counts={'conditions':len(rows),'draws':len(set(r['data'] for r in rows)),
            'task_draws':len(set((r['data'],r['task']) for r in rows)),
            'recipes':len(set((r['protocol'],r['task']) for r in rows)),
            'interior':sum(r['interior'] for r in rows),'boundary':sum(not r['interior'] for r in rows)}
    if any(v!=SPEC['counts'][k] for k,v in counts.items()) or len(old_diag)!=3240 or len(receipts)!=3360:
        raise RuntimeError('Input denominator mismatch')
    save(OUT/'input_verification.json',{'status':'verified','d2_manifest_files':len(manifest['files']),
         'receipt_payloads':len(receipts),'previous_artifacts':len(prior_manifest['artifacts']),'counts':counts,
         'd2_manifest_sha256':digest(ROOT/'evidence/manifests/d2.json'),
         'previous_manifest_sha256':digest(ROOT/'research/d2_diagonal_20261027/result_manifest.json'),
         'source_summary_sha256':digest(ROOT/REL/'summary.json'),'previous_rows_sha256':digest(PREVIOUS/'rows.jsonl')})
    return rows,old_diag


def f_and_dt(model,times):
    f,powers=legacy.response(model,times)
    df=np.zeros_like(f);a=model['active']
    df[:,a]=powers[:,a]*(-np.log(model['q'][a]))/model['lam'][a]
    return f,df


def signal_curve(f,h,terms):
    return np.einsum('ti,ij,tj->t',f,h,f,optimize=True)-2*f@terms['linear']+terms['val_y2']


def independent_risk(data,model,terms,t,sigma2):
    f,_=legacy.response(model,[t]);v=f[0]
    w=model['vectors']@(v[:,None]*terms['g'])
    b=np.mean((data['val_phi']@w-terms['vy'])**2)
    mapping=(model['vectors']*v)@model['vectors'].T@data['phi'].T/model['n']
    n=np.sum((data['val_phi']@mapping)**2)/len(data['val_phi'])
    return float(b+sigma2*n)


def make_row(old,method,t,risk,estimated,min_bias,min_total,K,M,mask_file):
    scale=max(1.,old['risk_initial'])
    raw=float(risk-old['risk_minimum'])
    if raw < -1e-8*scale:raise RuntimeError(f'Search-discrepancy: {old["curve"]} {method} {raw}')
    factor=num.time_factor(t,old['t_star']) if old['interior'] else None
    regret=max(0.,raw)/old['risk_initial']
    result={k:old[k] for k in ('protocol','task','n','seed','sigma2','data','curve')}
    result.update(method=method,t_reference=old['t_star'],t_selected=int(t),reference_interior=old['interior'],
                  predicted_interior=bool(0<t<262144),risk_initial=old['risk_initial'],risk_reference=old['risk_minimum'],
                  full_risk_at_selected=float(risk),estimated_risk_at_selected=float(estimated),regret_raw=raw,
                  regret_over_initial=regret,time_factor=factor,time_pass=bool(old['interior'] and factor is not None and factor<=2),
                  risk_pass=bool(regret<=.01),min_bias=float(min_bias),min_total=float(min_total),
                  invalid_bias=bool(min_bias < -1e-10*scale),invalid_total=bool(min_total < -1e-10*scale),
                  pair_count=M,retained_pairs=K,retained_fraction=K/M if M else 0.,mask_file=mask_file)
    return result


def aggregate(rows):
    inside=[r for r in rows if r['reference_interior']]
    passed_time=sum(r['time_pass'] for r in inside);passed_risk=sum(r['risk_pass'] for r in rows)
    return {'count':len(rows),'interior':len(inside),'boundary':len(rows)-len(inside),'time_pass':passed_time,
            'time_rate':passed_time/len(inside) if inside else None,'risk_pass':passed_risk,'risk_rate':passed_risk/len(rows),
            'stopping_pass':bool(inside and passed_time/len(inside)>=.8 and passed_risk/len(rows)>=.9),
            'invalid_bias':sum(r['invalid_bias'] for r in rows),'invalid_total':sum(r['invalid_total'] for r in rows),
            'valid_and_low_regret':sum(r['risk_pass'] and not r['invalid_bias'] and not r['invalid_total'] for r in rows),
            'harm_above_10pct':sum(r['regret_over_initial']>.1 for r in rows),
            'zero_time_on_interior':sum(r['t_selected']==0 for r in inside),
            'boundary_status_agreement':sum(r['predicted_interior']==r['reference_interior'] for r in rows),
            'regret_over_initial':num.quantiles([r['regret_over_initial'] for r in rows]),
            'positive_time_factors':num.quantiles([r['time_factor'] for r in inside if r['time_factor'] is not None])}


def is_focus(r):
    return (r['protocol'],r['task'],r['n'],r['seed'],r['sigma2'])==('d1_biased','sine',32,6103,1)


def main():
    frozen=contract()
    if OUT.exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True);(OUT/'basis').mkdir();(OUT/'selections').mkdir()
    (OUT/'preregistration.json').write_bytes((HERE/'preregistration.json').read_bytes())
    config=io.StringIO()
    with contextlib.redirect_stdout(config):np.show_config()
    execution={'status':'verifying_inputs','started_unix':time.time(),'contract':frozen,
               'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
               'platform':platform.platform(),'blas':config.getvalue(),'model_calls':0}
    save(OUT/'execution.json',execution)
    try:
        originals,previous=verify_inputs()
        execution.update(status='running',compute_started_unix=time.time());save(OUT/'execution.json',execution)
        started=time.monotonic();grid=num.base_grid()
        score_times=np.unique(np.r_[np.arange(17),np.rint(np.geomspace(17,262144,128)).astype(int)])
        check_times=np.array([0,1,16,255,262144])
        grouped=defaultdict(list)
        for old in originals:grouped[old['data']].append(old)
        rows=[];basis_records=[];selections=[];direct_checks=[];focus=None
        max_reconstruction=0.;max_mask_error=0.;mask_check_count=0;new_count=0
        def append(row,sink):
            rows.append(row);sink.write(json.dumps(row,allow_nan=False)+'\n')
        with (OUT/'rows.jsonl').open('x') as sink:
            for draw_index,(data_name,draw_rows) in enumerate(sorted(grouped.items())):
                if time.monotonic()-started>SPEC['budget']['compute_wall_seconds']:raise RuntimeError('Compute budget reached')
                with np.load(RAW/data_name,allow_pickle=False) as z:data={k:z[k] for k in z.files}
                model=legacy.spectral_model(data);rank=model['rank'];pairs=pair_indices(model['active']);M=len(pairs);K=budget_size(rank,5)
                if K*100>5*M or M!=rank*(rank-1)//2:raise RuntimeError('Pair budget error')
                draw_id=Path(data_name).stem.removesuffix('_data')
                bp=OUT/'basis'/f'{draw_id}.npz'
                np.savez_compressed(bp,lam=model['lam'],vectors=model['vectors'],active=model['active'],q=model['q'])
                active=model['lam'][model['active']]
                spacing=float(np.min(np.diff(active))/max(active)) if len(active)>1 else None
                basis_records.append({'data':data_name,'file':str(bp.relative_to(OUT)),'sha256':digest(bp),'rank':rank,
                                      'pair_count':M,'retained_pairs':K,'retained_fraction':K/M if M else 0.,
                                      'minimum_relative_spacing':spacing,'near_degenerate':spacing is not None and spacing<=1e-10})
                f,_=legacy.response(model,grid);fs,dfs=f_and_dt(model,score_times)
                fc,_=legacy.response(model,check_times)
                noise=(f*f)@(np.diag(model['hval'])*model['lam']/model['n'])
                by_task=defaultdict(list)
                for old in draw_rows:by_task[old['task']].append(old)
                for task,task_rows in sorted(by_task.items()):
                    terms=legacy.target_terms(model,data,task)
                    changes,magnitudes=rank_pairs(terms['h'],fs,dfs,score_times,f[-1],pairs)
                    masks={'change05':select_top(changes,K),'magnitude05':select_top(magnitudes,K)}
                    rng_seeds={}
                    for rep in range(10):
                        label=f'random05_{rep:02d}'
                        key=f'{SPEC["study"]}|{data_name}|{task}|{rep}'
                        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big');rng_seeds[label]=str(seed)
                        masks[label]=np.sort(np.random.default_rng(seed).choice(M,size=K,replace=False))
                    for sel in masks.values():
                        if len(sel)!=K or len(np.unique(sel))!=K:raise RuntimeError('Invalid selection mask')
                    mask_path=OUT/'selections'/f'{draw_id}_{task}.npz'
                    np.savez_compressed(mask_path,pairs=pairs,change_scores=changes,magnitude_scores=magnitudes,
                                        score_times=score_times,**{f'selected_{k}':v for k,v in masks.items()})
                    mask_rel=str(mask_path.relative_to(OUT))
                    overlap=len(set(masks['change05'])&set(masks['magnitude05']))
                    selections.append({'data':data_name,'task':task,'file':mask_rel,'sha256':digest(mask_path),
                                       'rank':rank,'pair_count':M,'retained_pairs':K,'retained_fraction':K/M if M else 0.,
                                       'change_magnitude_overlap_count':overlap,'random_rng_seeds':rng_seeds})
                    b=signal_curve(f,terms['h'],terms)
                    diag_h=np.diag(np.diag(terms['h']));bd=num.diagonal_bias(f,terms['h'],terms['linear'],terms['val_y2'])
                    # First verify saved full curves and add immutable references.
                    for old in sorted(task_rows,key=lambda r:r['sigma2']):
                        sigma=old['sigma2'];scale=max(1.,old['risk_initial']);prior=previous[old['curve']]
                        with np.load(RAW/old['curve'],allow_pickle=False) as z:
                            idx=np.searchsorted(z['t'],grid)
                            if not np.array_equal(z['t'][idx],grid):raise RuntimeError('Missing common grid')
                            err=max(float(np.max(np.abs(b-z['signal_bias'][idx]))),float(np.max(np.abs(noise-z['variance_unit'][idx]))),
                                    float(np.max(np.abs(b+sigma*noise-z['expected_clean_val'][idx]))))/scale
                        max_reconstruction=max(max_reconstruction,err)
                        if err>1e-8:raise RuntimeError('Full saved-curve reconstruction failed')
                        bx,nx,_,_=legacy.expected_curves(model,terms,[old['t_star'],prior['t_diagonal']])
                        if max(abs(bx[0]+sigma*nx[0]-old['risk_minimum']),abs(bx[1]+sigma*nx[1]-prior['full_risk_at_diagonal']))>1e-8*scale:
                            raise RuntimeError('Saved reference point mismatch')
                        append(make_row(old,'full',old['t_star'],old['risk_minimum'],old['risk_minimum'],float(b.min()),
                                        old['risk_minimum'],M,M,None),sink)
                        append(make_row(old,'diagonal',prior['t_diagonal'],prior['full_risk_at_diagonal'],prior['diagonal_risk_at_diagonal'],
                                        prior['min_diagonal_bias'],prior['min_diagonal_risk'],0,M,None),sink)
                    for method,selected in masks.items():
                        h=masked_h(terms['h'],pairs,selected)
                        masked_bias=bd.copy() if K==0 else signal_curve(f,h,terms)
                        if not np.all(np.isfinite(masked_bias)):raise RuntimeError('Nonfinite masked curve')
                        if K==0 and not np.array_equal(h,diag_h):raise RuntimeError('Zero budget not diagonal')
                        ii,jj=pairs[selected].T
                        explicit=signal_curve(fc,diag_h,terms)+np.sum(2*terms['h'][ii,jj]*fc[:,ii]*fc[:,jj],axis=1)
                        check=signal_curve(fc,h,terms)
                        error=float(np.max(np.abs(explicit-check)))/max(1.,terms['val_y2'])
                        max_mask_error=max(max_mask_error,error);mask_check_count+=1
                        if error>1e-8:raise RuntimeError('Selected-pair versus masked-matrix mismatch')
                        for old in sorted(task_rows,key=lambda r:r['sigma2']):
                            sigma=old['sigma2'];scale=max(1.,old['risk_initial'])
                            def objective(t):
                                ft,_=legacy.response(model,t)
                                signal=num.diagonal_bias(ft,terms['h'],terms['linear'],terms['val_y2']) if K==0 else signal_curve(ft,h,terms)
                                return signal+sigma*((ft*ft)@(np.diag(model['hval'])*model['lam']/model['n']))
                            t=num.select_integer(grid,masked_bias+sigma*noise,objective)
                            bx,nx,_,_=legacy.expected_curves(model,terms,[t]);true=float(bx[0]+sigma*nx[0])
                            ft,_=legacy.response(model,[t])
                            bias_at=float((num.diagonal_bias(ft,terms['h'],terms['linear'],terms['val_y2']) if K==0 else signal_curve(ft,h,terms))[0])
                            estimate=bias_at+sigma*float(nx[0])
                            row=make_row(old,method,t,true,estimate,min(float(masked_bias.min()),bias_at),
                                         min(float((masked_bias+sigma*noise).min()),estimate),K,M,mask_rel)
                            if K==0 and (t!=previous[old['curve']]['t_diagonal'] or abs(estimate-previous[old['curve']]['diagonal_risk_at_diagonal'])>1e-8*scale):
                                raise RuntimeError('Zero-budget result differs from saved diagonal')
                            append(row,sink);new_count+=1
                            if old['n']==64 and old['seed']==6101 and sigma==1 and task in ('interaction','vector','quadratic','linear'):
                                direct=independent_risk(data,model,terms,t,sigma)
                                error=abs(direct-true)
                                if error>1e-8*scale:raise RuntimeError('Independent feature-space check failed')
                                direct_checks.append({'curve':old['curve'],'method':method,'t':t,'spectral_risk':true,
                                                      'direct_risk':direct,'absolute_error':error})
                    focus_old=next((r for r in task_rows if is_focus(r)),None)
                    if focus_old is not None:
                        focus_rows=[r for r in rows if is_focus(r)]
                        extra=np.array(sorted(set([0,1,255]+[r['t_selected'] for r in focus_rows])))
                        focus_grid=np.unique(np.r_[grid,extra]);ff,_=legacy.response(model,focus_grid)
                        pp,_=legacy.response(model,extra)
                        bb,nn,_,_=legacy.expected_curves(model,terms,focus_grid)
                        np_points=(pp*pp)@(np.diag(model['hval'])*model['lam']/model['n'])
                        arrays={'t':focus_grid,'full':bb+nn,'noise':nn,'full_bias':bb}
                        breakdown={}
                        for method in ['full','diagonal']+METHODS:
                            hm=terms['h'] if method=='full' else diag_h if method=='diagonal' else masked_h(terms['h'],pairs,masks[method])
                            mb=signal_curve(ff,hm,terms);arrays[method]=mb+nn
                            own=np.sum(pp*pp*np.diag(terms['h']),axis=1)
                            cross=np.einsum('ti,ij,tj->t',pp,hm,pp,optimize=True)-own
                            allcross=np.einsum('ti,ij,tj->t',pp,terms['h'],pp,optimize=True)-own
                            target=-2*pp@terms['linear']
                            true_b=signal_curve(pp,terms['h'],terms)
                            breakdown[method]=[{'t':int(t),'self_terms':float(own[j]),'retained_cross':float(cross[j]),
                                                'full_cross':float(allcross[j]),'omitted_cross':float(allcross[j]-cross[j]),
                                                'target_match':float(target[j]),'target_squared_mean':terms['val_y2'],
                                                'noise':float(np_points[j]),'estimated_risk':float(own[j]+cross[j]+target[j]+terms['val_y2']+np_points[j]),
                                                'full_risk':float(true_b[j]+np_points[j])} for j,t in enumerate(extra)]
                        np.savez_compressed(OUT/'focus_curves.npz',**arrays)
                        focus={'condition':{k:focus_old[k] for k in ('protocol','task','n','seed','sigma2')},
                               'status':'previously inspected development example','rank':rank,'pair_count':M,'retained_pairs':K,
                               'mask_file':mask_rel,'arrays':'focus_curves.npz','rows':focus_rows,'breakdown':breakdown,
                               'repaired':{r['method']:r['time_pass'] and r['risk_pass'] for r in focus_rows}}
                        save(OUT/'focus_case.json',focus)
                sink.flush()
                if (draw_index+1)%10==0:print(json.dumps({'draws':draw_index+1,'of':120,'new_evaluations':new_count,'seconds':round(time.monotonic()-started,1)}),flush=True)
        if len(rows)!=45360 or new_count!=38880 or len(selections)!=360 or len(direct_checks)!=96 or mask_check_count!=4320 or focus is None:
            raise RuntimeError('Output denominator mismatch')
        recipe_keys=sorted(set((r['protocol'],r['task']) for r in originals))
        method_results={}
        for method in ['full','diagonal']+METHODS:
            subset=[r for r in rows if r['method']==method]
            recipes=[{'protocol':p,'task':t,**aggregate([r for r in subset if (r['protocol'],r['task'])==(p,t)])} for p,t in recipe_keys]
            overall=aggregate(subset);overall['pooled_stopping_pass']=overall.pop('stopping_pass')
            method_results[method]={'overall':overall,'recipes':recipes,
                'seeds':[{'seed':s,**aggregate([r for r in subset if r['seed']==s])} for s in (6101,6102,6103)],
                'recipe_seeds':[{'protocol':p,'task':t,'seed':s,**aggregate([r for r in subset if (r['protocol'],r['task'],r['seed'])==(p,t,s)])} for p,t in recipe_keys for s in (6101,6102,6103)],
                'recipes_passing_both':sum(r['stopping_pass'] for r in recipes)}
        random_summary={key:num.quantiles([method_results[f'random05_{i:02d}']['overall'][key] for i in range(10)]) for key in ['time_rate','risk_rate','harm_above_10pct','invalid_bias','invalid_total']}
        summary={'study':SPEC['study'],'domain':'development','contract':frozen,'counts':SPEC['counts'],
                 'primary_method':'change05','hypothesis_status':'supported_on_development' if method_results['change05']['recipes_passing_both']==12 else 'refuted_on_development',
                 'methods':method_results,'random_replicate_distribution':random_summary,
                 'focus_repaired':focus['repaired'],'zero_budget_task_draws':sum(s['retained_pairs']==0 for s in selections),
                 'near_degenerate_draws':sum(s['near_degenerate'] for s in basis_records),
                 'retained_fraction':num.quantiles([s['retained_fraction'] for s in selections]),
                 'compute_seconds':time.monotonic()-started,'rows_sha256':digest(OUT/'rows.jsonl')}
        save(OUT/'basis_manifest.json',basis_records);save(OUT/'selection_manifest.json',selections)
        save(OUT/'summary.json',summary)
        save(OUT/'numerical_verification.json',{'status':'verified','saved_curves_checked':3240,
             'maximum_normalized_reconstruction_error':max_reconstruction,'mask_checks':mask_check_count,
             'maximum_normalized_mask_error':max_mask_error,'independent_checks':len(direct_checks),
             'maximum_independent_absolute_error':max(r['absolute_error'] for r in direct_checks),'checks':direct_checks})
        if contract()!=frozen:raise RuntimeError('Source drift during execution')
        execution.update(status='complete',finished_unix=time.time(),compute_seconds=summary['compute_seconds'])
        save(OUT/'execution.json',execution)
        print(json.dumps({'out':str(OUT),'status':summary['hypothesis_status'],'primary':method_results['change05']['overall'],
                          'recipes_passing':method_results['change05']['recipes_passing_both'],'focus_repaired':focus['repaired']},ensure_ascii=False))
    except Exception as exc:
        execution.update(status='failed',finished_unix=time.time(),error=str(exc));save(OUT/'execution.json',execution)
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
