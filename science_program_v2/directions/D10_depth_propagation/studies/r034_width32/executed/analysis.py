import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
STUDY=Path(__file__).resolve().parent.parent

def frozen_time(kernel,residual,eta):
    vals,vecs=np.linalg.eigh(kernel); top=max(float(vals[-1]),1e-30)
    if vals[0] < -1e-10*top: return {'status':'invalid_psd'}
    vals=np.where(vals<=1e-12*top,0,vals); energy=np.einsum('ij,i->j',vecs,residual)**2; energy/=np.dot(residual,residual)
    null=float(energy[vals==0].sum()); q=float(np.dot(energy,vals)); out={'null_energy_fraction':null,'q':q,'lambda_max':top,'positive_rank':int(np.count_nonzero(vals))}
    if eta*top >= 2: return dict(out,status='unstable',t50=None,eta_lambda_max=eta*top)
    if null>=.5: return dict(out,status='unreachable',t50=None)
    factors=(1-eta*vals)**2
    def curve(step): return float(np.dot(energy,factors**step))
    high=1
    while curve(high)>.5 and high<2**60: high*=2
    if curve(high)>.5: return dict(out,status='search_limit',t50=None)
    low=0
    while high-low>1:
        mid=(high+low)//2
        if curve(mid)<=.5: high=mid
        else: low=mid
    return dict(out,status='finite',t50=high,loss_fraction_at=curve(high),loss_fraction_before=curve(high-1))

def comparison(candidate,actual,horizon):
    if candidate['status']=='unstable': return {'decision':'fail','ratio':None}
    if actual is None:
        if candidate['status']=='finite' and candidate['t50']<=horizon/2: return {'decision':'fail','ratio':None}
        return {'decision':'unresolved','ratio':None}
    if candidate['status']=='unreachable': return {'decision':'fail','ratio':None,'ratio_infinite':True}
    if candidate['status']!='finite': return {'decision':'unresolved','ratio':None}
    ratio=candidate['t50']/actual; return {'decision':'pass' if .5<=ratio<=2 else 'fail','ratio':ratio}

def interval(values):
    a=np.array(values,float); mean=float(a.mean()); radius=float(student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))); return {'mean':mean,'t95_seed_interval':[mean-radius,mean+radius]}

def main():
    pre=json.loads((STUDY/'preregistration.json').read_text()); recipe=pre['recipe']; cells=[]
    for depth in recipe['hidden_depths']:
      for seed in recipe['init_seeds']:
        path=STUDY/'results'/f'depth{depth}_seed{seed}.json'; row=json.loads(path.read_text()); npz=path.with_suffix('.npz')
        if hashlib.sha256(npz.read_bytes()).hexdigest()!=row['arrays_sha256']: raise ValueError(f'hash mismatch {path}')
        with np.load(npz) as arr:
          loss=arr['loss']; hits=np.flatnonzero(loss<=loss[0]/2); actual=int(hits[0]) if len(hits) else None; residual=arr['residual_0']; layers=[frozen_time(arr[f'K_0_{l}'],residual,recipe['eta']) for l in range(depth+1)]
          slow={'status':'unstable','t50':None} if any(x['status']=='unstable' for x in layers) else {'status':'unreachable','t50':None} if any(x['status']=='unreachable' for x in layers) else {'status':'finite','t50':max(x['t50'] for x in layers)} if all(x['status']=='finite' for x in layers) else {'status':'unresolved','t50':None}
          total=frozen_time(arr['K_sum_0'],residual,recipe['eta']); cells.append({'id':row['id'],'depth':depth,'seed':seed,'status':row['status'],'parameter_count':row['parameter_count'],'actual_t50':actual,'actual_right_censored':actual is None,'final_relative_loss':float(loss[-1]/loss[0]),'initial_loss':float(loss[0]),'layer_times':layers,'slowest':slow,'sum_kernel':total,'slowest_comparison':comparison(slow,actual,recipe['steps']),'sum_comparison':comparison(total,actual,recipe['steps']),'seconds':row['seconds']})
    depths=[]
    for depth in recipe['hidden_depths']:
      group=[c for c in cells if c['depth']==depth]; entry={'depth':depth,'actual_times':[c['actual_t50'] for c in group]}
      for key in ['slowest','sum']:
        decisions=[c[f'{key}_comparison']['decision'] for c in group]; entry[key]={'passes':decisions.count('pass'),'fails':decisions.count('fail'),'unresolved':decisions.count('unresolved'),'decision':'supported' if decisions.count('pass')>=2 else 'refuted' if decisions.count('fail')>=2 else 'unresolved','ratios':[c[f'{key}_comparison']['ratio'] for c in group]}
        if all(c[f'{key}_comparison']['ratio'] is not None for c in group): entry[key]['ratio_statistics']=interval(entry[key]['ratios'])
      if all(c['actual_t50'] is not None for c in group): entry['actual_statistics']=interval(entry['actual_times'])
      depths.append(entry)
    predictions={}
    slow_ratios=[c['slowest_comparison']['ratio'] for c in cells]
    overestimates=sum(v is not None and v>2 for v in slow_ratios)
    slow_resolved=all(v is not None for v in slow_ratios)
    slow_disproved=any(c['slowest']['status'] in ['unstable','unreachable'] or (c['slowest_comparison']['ratio'] is not None and c['slowest_comparison']['ratio']<=2) for c in cells)
    predictions['P1']={'status':'supported' if overestimates==12 else 'refuted' if slow_resolved or slow_disproved else 'unresolved','ratios_above_two':overestimates,'required':12}
    sum_passes=sum(c['sum_comparison']['decision']=='pass' for c in cells)
    sum_unresolved=sum(c['sum_comparison']['decision']=='unresolved' for c in cells)
    per_depth=all(d['sum']['passes']>=2 for d in depths)
    sum_disproved=sum_passes+sum_unresolved<10 or any(d['sum']['passes']+d['sum']['unresolved']<2 for d in depths)
    predictions['P2']={'status':'supported' if sum_passes>=10 and per_depth else 'refuted' if sum_disproved else 'unresolved','passes':sum_passes,'required_total':10,'each_depth_at_least_two':per_depth}
    baseline_path=STUDY.parent/'r030_eta005/summary.json'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==pre['baseline']['summary_sha256']
    baseline=json.loads(baseline_path.read_text())
    old={c['id']:c for c in baseline['cells']}
    paired=[]
    for c in cells:
      b=old[c['id']]
      paired.append({'id':c['id'],'old_actual_t50':b['actual_t50'],'new_actual_t50':c['actual_t50'],'old_width':16,'new_width':32,'actual_time_difference':c['actual_t50']-b['actual_t50'] if c['actual_t50'] is not None else None,'actual_time_ratio':c['actual_t50']/b['actual_t50'] if c['actual_t50'] is not None else None,'old_sum_ratio':b['sum_comparison']['ratio'],'new_sum_ratio':c['sum_comparison']['ratio'],'old_slowest_ratio':b['slowest_comparison']['ratio'],'new_slowest_ratio':c['slowest_comparison']['ratio']})
    summary={'round':34,'domain':'development','saved_cells':len(cells),'recipe_units':4,'predictions':predictions,'depths':depths,'cells':cells,'paired_width_changes':paired,'training_seconds':sum(c['seconds'] for c in cells),'preregistration_sha256':hashlib.sha256((STUDY/'preregistration.json').read_bytes()).hexdigest()}; (STUDY/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); print(json.dumps({'predictions':predictions,'depths':depths},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
