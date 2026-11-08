"""Separate calibration from function-matched parameterization holdouts."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from run_r002 import SOURCE,RUN
import neural_design as design

ADAPTIVE=['Adam','RMSprop','Adagrad']
CLOCK_FIELDS={'d':'relative_d','V':'function_norm_V','P':'function_projection_P'}


def arrays(path):
    with np.load(path) as source:return {key:source[key] for key in source.files}


def first_passages(reference,targets):
    targets=np.asarray(targets)
    envelope=np.minimum.accumulate(reference)
    valid=np.isfinite(targets)&(targets<=reference[0])&(targets>=envelope[-1])
    idx=np.minimum(np.searchsorted(-envelope,-targets,side='left'),len(reference)-1)
    left=np.maximum(idx-1,0)
    frac=np.divide(reference[left]-targets,reference[left]-reference[idx],out=np.zeros_like(targets),where=reference[left]!=reference[idx])
    result=left+frac;result[~valid]=np.nan
    return result


def fit_quadratic(x,y,weights):
    x,y,w=map(np.asarray,(x,y,weights))
    mat=np.column_stack([x,x*x])*np.sqrt(w)[:,None];target=y*np.sqrt(w)
    candidates=[np.zeros(2)]
    unconstrained=np.linalg.lstsq(mat,target,rcond=None)[0]
    if (unconstrained>=0).all():candidates.append(unconstrained)
    for j in [0,1]:
        candidate=np.zeros(2);candidate[j]=max(0.,float(mat[:,j]@target/(mat[:,j]@mat[:,j])));candidates.append(candidate)
    result=min(candidates,key=lambda v:float(np.sum((mat@v-target)**2)))
    return {'A':float(result[0]),'B':float(result[1]),'fit_points':len(x)}


def prediction(clock,coeff,reference,eta_ref):
    tau=coeff['A']*clock+coeff['B']*clock*clock
    valid=np.isfinite(tau)&(tau>=0)&(tau<=eta_ref*(len(reference)-1))
    output=np.full(len(clock),np.nan)
    output[valid]=np.interp(tau[valid]/eta_ref,np.arange(len(reference)),reference)
    return output,tau


def scores(prediction,observed,denominator):
    good=np.isfinite(prediction)&np.isfinite(observed)
    error=np.abs(prediction[good]-observed[good])/denominator
    progress=(observed[0]-np.nanmin(observed))/denominator
    return {'coverage':float(np.mean(good)),'complete':bool(good.all()),
            'mae':float(np.mean(error)) if len(error) else None,
            'max_error':float(np.max(error)) if len(error) else None,
            'progress_normalized_mae':float(np.mean(error)/progress) if len(error) and progress>1e-12 else None}


def aggregate(rows):
    result={'paths':len(rows),'diverged':sum(r['diverged'] for r in rows),'capped_paths':sum(r['eta_capped'] for r in rows),'clocks':{}}
    for name in ['d','V','P','initial_linearization','constant']:
        sel=[r['scores'][name] for r in rows]
        values=[s['mae'] for s in sel if s['mae'] is not None]
        complete=[s['mae'] for s in sel if s['complete']]
        result['clocks'][name]={'paths':len(sel),'complete_paths':len(complete),'complete_fraction':len(complete)/len(sel) if sel else None,
             'median_covered_mae':float(np.median(values)) if values else None,'mean_covered_mae':float(np.mean(values)) if values else None,
             'median_complete_mae':float(np.median(complete)) if complete else None,
             'worst_point':max((s['max_error'] for s in sel if s['max_error'] is not None),default=None)}
    return result


def clean(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {key:clean(v) for key,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main():
    receipt=json.loads((RUN/'execution_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='completed'
    pre=json.loads((RUN/'r002_preregistration.json').read_text(encoding='utf-8'))
    caseby={c['case_id']:c for c in pre['design']['case_manifest']}
    refcases={c['case_id']:c for c in pre['design']['reference_cases']}
    refby={};initial_cache={}
    def init_file(name):
        if name not in initial_cache:initial_cache[name]=arrays(RUN/name)
        return initial_cache[name]
    for artifact in receipt['artifacts']:
        if artifact['kind']!='reference':continue
        saved=arrays(RUN/artifact['file']);init=init_file(artifact['initial_file'])
        for i,cid in enumerate(saved['case_ids']):
            row=refcases[str(cid)]
            valid=np.flatnonzero(~saved['valid_actual'][i]);end=int(valid[0]) if len(valid) else saved['loss'].shape[1]
            refby[design.base_key(row)]={'loss':saved['loss'][i,:end].copy(),'validation':saved['validation_loss'][i,:end].copy(),
                 'eta':float(saved['eta'][i]),'K0':init['K0'][i].copy(),'f0':init['f0'][i].copy(),'f0val':init['f0val'][i].copy(),
                 'nonmonotone':bool(np.any(np.diff(saved['loss'][i,:end])>1e-10*saved['loss'][i,0])),'full':end==design.REFERENCE_STEPS+1}
    calibration=defaultdict(lambda:defaultdict(list))
    candidate_files=[a for a in receipt['artifacts'] if a['kind']=='candidate']
    for artifact in candidate_files:
        if artifact['method'] not in ADAPTIVE:continue
        saved=arrays(RUN/artifact['file'])
        for i,cid in enumerate(saved['case_ids']):
            c=caseby[str(cid)]
            if c['split']!='calibration':continue
            ref=refby[design.base_key(c)]
            target=first_passages(ref['loss'],saved['loss'][i])*ref['eta']
            for clock,field in CLOCK_FIELDS.items():
                x=saved[field][i];valid=np.isfinite(target)&np.isfinite(x)&(x>0)
                if np.sum(valid)<2:continue
                calibration[design.calibration_group(c)][clock].append({'x':x[valid],'y':target[valid],
                    'weights':np.full(np.sum(valid),1/np.sum(valid)),'case_id':str(cid),'method':artifact['method']})
    coefficients={}
    for key,clocks in calibration.items():
        coefficients[key]={}
        for clock,tracks in clocks.items():
            coefficients[key][clock]=fit_quadratic(np.concatenate([t['x'] for t in tracks]),np.concatenate([t['y'] for t in tracks]),np.concatenate([t['weights'] for t in tracks]))
            coefficients[key][clock]['training_paths']=len(tracks)
    (RUN/'calibrated_coefficients.json').write_text(json.dumps({json.dumps(k):v for k,v in coefficients.items()},indent=2)+'\n')
    rows=[];endpoints=[];paired={};examples=[]
    for artifact in candidate_files:
        saved=arrays(RUN/artifact['file']);init=init_file(artifact['initial_file'])
        for i,cid in enumerate(saved['case_ids']):
            c=caseby[str(cid)];ref=refby[design.base_key(c)];observed=saved['loss'][i];L0=float(observed[0])
            pred={};times={};all_scores={}
            for name,field in CLOCK_FIELDS.items():
                pred[name],times[name]=prediction(saved[field][i],coefficients[design.calibration_group(c)][name],ref['loss'],ref['eta'])
                all_scores[name]=scores(pred[name],observed,L0)
            linearized=L0*(1-2*saved['function_projection_P'][i]+saved['function_norm_V'][i]**2)
            all_scores['initial_linearization']=scores(linearized,observed,L0)
            all_scores['constant']=scores(np.full(len(observed),L0),observed,L0)
            common=np.isfinite(pred['d'])&np.isfinite(pred['V'])&np.isfinite(observed)
            common_scores={name:float(np.mean(np.abs(pred[name][common]-observed[common]))/L0) if common.any() else None for name in ['d','V']}
            k=init['K0'][i];kr=ref['K0'];shape=float(np.linalg.norm(k/np.linalg.norm(k)-kr/np.linalg.norm(kr)))
            row={**c,'method':artifact['method'],'eta':float(saved['eta'][i]),'eta_capped':bool(saved['eta_capped'][i]),
                 'planned_final_d':float(saved['planned_final_d'][i]),'planned_final_V':float(saved['planned_final_V'][i]),
                 'first_V':float(saved['planned_first_V'][i]),'diverged':bool(saved['diverged_actual'][i]),'scores':all_scores,
                 'progress':float((L0-np.nanmin(observed))/L0),'final_loss_ratio':float(observed[-1]/L0),
                 'initial_floor_ratio':float(init['initial_loss_floor'][i]/L0),
                 'initial_function_max_difference':float(np.max(np.abs(init['f0'][i]-ref['f0']))),
                 'initial_validation_function_max_difference':float(np.max(np.abs(init['f0val'][i]-ref['f0val']))),
                 'kernel_shape_distance':shape,'kernel_trace_ratio':float(np.trace(k)/np.trace(kr)),
                 'common_dV_coverage':float(np.mean(common)),'common_dV_mae':common_scores,
                 'validation_final_ratio':float(saved['validation_loss'][i,-1]/saved['validation_loss'][i,0]),
                 'clock_reversals':{name:int(np.sum(np.diff(saved[field][i]) < -1e-10)) for name,field in CLOCK_FIELDS.items()}}
            rows.append(row)
            pairkey=(design.base_key(c),c['depth'],c['eta_protocol'],c['radius'],artifact['method'])
            paired.setdefault(pairkey,{})[c['balance']]=(row,observed.copy())
            # Fixed illustrative draw, not chosen for favorable clock ranking.
            if c['teacher']=='relu_teacher' and c['input_condition']==10 and c['activation']=='relu' and c['width']==32 and c['seed']==8201 and c['eta_protocol']=='function_radius' and c['radius']==1 and artifact['method']=='Adam':
                examples.append({'case':c,'loss_ratio':(observed/L0).tolist(),'validation_ratio':(saved['validation_loss'][i]/saved['validation_loss'][i,0]).tolist(),
                    'd':saved['relative_d'][i].tolist(),'V':saved['function_norm_V'][i].tolist(),'P':saved['function_projection_P'][i].tolist(),
                    'predictions':{name:(v/L0).tolist() for name,v in pred.items()}})
            for q in [.9,.5,.1]:
                hits=np.flatnonzero(observed<=q*L0);bad=np.flatnonzero(observed>q*L0)
                obs=int(hits[0]) if len(hits) else None
                sustained=(int(bad[-1]+1) if len(bad) and bad[-1]<design.STEPS else (0 if not len(bad) else None)) if np.isfinite(observed).all() else None
                refhit=np.flatnonzero(ref['loss']<=q*L0);target=ref['eta']*int(refhit[0]) if len(refhit) else None
                for name in CLOCK_FIELDS:
                    hits=np.flatnonzero(times[name]>=target) if target is not None else []
                    estimate=int(hits[0]) if len(hits) else None
                    endpoints.append({'case_id':c['case_id'],'method':artifact['method'],'clock':name,'q':q,'observed_first':obs,'observed_sustained':sustained,'predicted':estimate,
                        'class_unattainable':q<row['initial_floor_ratio'],'error':abs(estimate-obs)/obs if estimate is not None and obs is not None and obs>0 else None})
    witnesses=[]
    for items in paired.values():
        if 1. not in items:continue
        base,bl=items[1.]
        for balance,(row,loss) in items.items():
            if balance==1:continue
            diff=float(np.nanmax(np.abs(loss-bl))/bl[0])
            row['paired_scale_max_loss_difference']=diff
            row['paired_scale_eta_ratio']=row['eta']/base['eta']
            if row['split']=='confirmation' and row['cohort']=='core' and row['kernel_shape_distance']>.01 and diff>.01:
                witnesses.append({k:row[k] for k in ['case_id','method','cohort','activation','width','depth','balance','kernel_shape_distance','paired_scale_max_loss_difference']})
    confirm=[r for r in rows if r['split']=='confirmation' and r['method'] in ADAPTIVE]
    scale=[r for r in confirm if r['cohort']=='core' and r['balance']!=1]
    eligible=[r for r in scale if r['progress']>=.1 and r['common_dV_coverage']>=.8]
    improved=sum(r['common_dV_mae']['d']-r['common_dV_mae']['V']>1e-10 for r in eligible)
    depth=[r for r in confirm if r['cohort']=='depth_holdout'];depth_summary=aggregate(depth)
    metric=depth_summary['clocks']['V']
    findings={'P1':{'passes':all(r['initial_function_max_difference']<=1e-10 and r['initial_validation_function_max_difference']<=1e-10 for r in rows),
                     'max_training_prediction_difference':max(r['initial_function_max_difference'] for r in rows),'max_validation_prediction_difference':max(r['initial_validation_function_max_difference'] for r in rows)},
       'P2':{'scale_holdout_adaptive_paths':len(scale),'eligible':len(eligible),'V_better':improved,'fraction':improved/len(eligible) if eligible else None,'passes':improved>.6*len(eligible) if eligible else False,
             'by_eta_protocol':{p:{'eligible':len(part),'V_better':sum(r['common_dV_mae']['d']-r['common_dV_mae']['V']>1e-10 for r in part)} for p in ['parameter_radius','function_radius'] if (part:=[r for r in eligible if r['eta_protocol']==p])}},
       'P3':{'paths':len(depth),'complete_fraction':metric['complete_fraction'],'median_complete_mae':metric['median_complete_mae'],
             'passes':metric['complete_fraction']>=.9 and metric['median_complete_mae'] is not None and metric['median_complete_mae']<=.01},
       'P4':{'passes':bool(witnesses),'witness_count':len(witnesses),'caution':'fixed-step histories depend on the registered eta matching protocol; this is not pure kernel causality across all updates'}}
    summary={'study':'r002_parameterization','preregistration_commit':receipt['preregistration_commit'],'candidate_paths':len(rows),'reference_paths':len(refby),
      'calibration_groups':len(coefficients),'calibration_paths':sum(r['split']=='calibration' and r['method'] in ADAPTIVE for r in rows),'predictions':findings,
      'reference_nonmonotone_paths':sum(r['nonmonotone'] for r in refby.values()),'reference_complete_paths':sum(r['full'] for r in refby.values()),
      'all_paths':aggregate(rows),'scale_confirmation':aggregate(scale),'depth_confirmation':depth_summary,
      'scale_by_eta_protocol':{p:aggregate([r for r in scale if r['eta_protocol']==p]) for p in ['parameter_radius','function_radius']},
      'scale_by_activation':{a:aggregate([r for r in scale if r['activation']==a]) for a in ['identity','relu']},
      'depth_by_activation':{a:aggregate([r for r in depth if r['activation']==a]) for a in ['identity','relu']},
      'metric_scope':'all fixed0..256 steps; coverage and actual divergence retained; parameter and function clocks have the same 2 fitted coefficients; no confirmation target loss used to fit coefficients'}
    for name,values in [('trajectory_scores',rows),('endpoint_scores',endpoints)]:
        with (RUN/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for value in values:f.write(json.dumps(clean(value),separators=(',',':'))+'\n')
    (RUN/'illustrative_curves.json').write_text(json.dumps(clean(examples),separators=(',',':')))
    (RUN/'kernel_scale_witnesses.json').write_text(json.dumps(clean(witnesses),indent=2))
    (RUN/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n',encoding='utf-8')
    (SOURCE/'r002_summary.json').write_bytes((RUN/'summary.json').read_bytes())
    print(json.dumps(clean({'candidate_paths':len(rows),'predictions':findings,'scale_confirmation':summary['scale_confirmation'],'depth_confirmation':depth_summary}),indent=2))


if __name__=='__main__':main()
