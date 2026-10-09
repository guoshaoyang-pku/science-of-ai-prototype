from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
STUDY=Path(__file__).resolve().parent; RES=STUDY/'results'; PR=STUDY/'preregistration.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads(PR.read_text()); rows=[]
    for p in sorted(RES.glob('*.json')):
        if p.name=='summary.json': continue
        row=json.loads(p.read_text()); arr=p.with_suffix('.npz')
        if not arr.exists() or sha(arr)!=row['arrays_sha256']: raise RuntimeError('array hash mismatch')
        with np.load(arr) as z:
            r=z['predictions'].astype(float)-z['targets'].astype(float); e=float(np.mean((r-r.mean())**2))
        row['centered_error']=e; rows.append(row)
    funcs=cfg['functions']; labels=list(cfg['recipes']); means=cfg['means']; seeds=cfg['seeds']; pred={}; contrasts=[]
    for fn in funcs:
        for label in labels:
            for seed in seeds:
                es={m:next((r['centered_error'] for r in rows if r['function']==fn and r['label']==label and r['seed']==seed and r['mean']==m),None) for m in means}
                if all(v is not None for v in es.values()): pred[(fn,label,seed)]=es
        for seed in seeds:
            a=pred.get((fn,labels[0],seed)); b=pred.get((fn,labels[1],seed))
            if a and b:
                c0=(a[-3]+a[3])/2-a[0]; c1=(b[-3]+b[3])/2-b[0];
                k0=next(r['log10_condition']['256'] for r in rows if r['function']==fn and r['label']==labels[0] and r['mean']==0 and r['seed']==seed); k1=next(r['log10_condition']['256'] for r in rows if r['function']==fn and r['label']==labels[1] and r['mean']==0 and r['seed']==seed)
                contrasts.append({'function':fn,'seed':seed,'LN_minus_noLN_chord':c0-c1,'LN_minus_noLN_log10_condition':k0-k1,'LN_chord':c0,'noLN_chord':c1})
    if len(contrasts)>=2:
        x=np.array([z['LN_minus_noLN_log10_condition'] for z in contrasts]); y=np.array([z['LN_minus_noLN_chord'] for z in contrasts]); corr=float(pearsonr(x,y).statistic)
    else: corr=None
    result={'study':'r005_gram_condition','status':'measurements_complete' if rows else 'blocked_before_training','saved_cells':len(rows),'planned_cells':120,'rows':rows,'contrasts':contrasts,'pearson_r':corr,'predictions':{'spectrum_direction':'not_evaluated' if not rows else 'computed','chord_mediation':'not_evaluated' if len(contrasts)<2 else ('pass' if corr<=-.5 else 'refuted'),'temporal_alignment':'not_evaluated' if not rows else 'computed'},'source_sha256':{'preregistration.json':sha(PR),'executed/run.py':sha(STUDY/'executed/run.py'),'analysis.py':sha(STUDY/'analysis.py')}}
    (STUDY/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
