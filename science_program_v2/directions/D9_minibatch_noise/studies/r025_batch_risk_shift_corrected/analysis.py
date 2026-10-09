import json, hashlib
from pathlib import Path
import numpy as np
STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[3]
OLD=ROOT/'directions/D2_overfitting_u_curve/studies/r003_noise_sample_scaling/summary.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tstar(a):
 i=int(np.argmin(a)); return i, bool(0<i<len(a)-1)
def main():
 cfg=json.loads((STUDY/'preregistration.json').read_text()); old=json.loads(OLD.read_text()); full={(c['seed'],c['noise_variance']):c['t_star'] for c in old['cells'] if c['n']==64}
 rows=[]
 for p in sorted((STUDY/'results').glob('n64_seed*_B*_var*.json')):
  if p.name=='receipt.json': continue
  row=json.loads(p.read_text()); z=np.load(p.with_suffix('.npz')); risk=z['mean_risk']; ts,inter=tstar(risk); c=row['cell']; base=full[(c['seed'],c['noise_variance'])]; rows.append({'seed':c['seed'],'batch':c['batch'],'variance':c['noise_variance'],'t_star':ts,'full_t_star':base,'ratio':ts/base,'interior':inter,'min_risk':float(risk[ts]),'se_at_min':float(z['standard_error'][ts]),'finite':bool(np.isfinite(risk).all()),'replicates':row['replicates'],'arrays_sha256':row['arrays_sha256']})
 p1={}; p2={}
 for var in cfg['variances']:
  r32=[r for r in rows if r['batch']==32 and r['variance']==var]; r8=[r for r in rows if r['batch']==8 and r['variance']==var]
  p1[str(var)]={'passed':sum(.8<=r['ratio']<=1.2 for r in r32),'total':len(r32),'ratios':[r['ratio'] for r in r32]}
  p2[str(var)]={'passed':sum(r['ratio']<=.9 for r in r8),'total':len(r8),'ratios':[r['ratio'] for r in r8]}
 p1_ok=sum(x['passed']>=3 for x in p1.values())==len(cfg['variances']); p2_ok=sum(x['passed']>=3 for x in p2.values())>=2
 summary={'study':cfg['study'],'round':25,'direction':'D9_minibatch_noise','domain':'development','status':'complete','round_result':'ok','counts':{'planned_cells':len(cfg['seeds'])*len(cfg['batches'])*len(cfg['variances']),'saved_cells':len(rows),'replicates_per_cell':cfg['replicates']},'predictions':{'P1_B32_ratio_0.8_1.2':{'status':'supported' if p1_ok else 'refuted','by_variance':p1},'P2_B8_advance_10pct':{'status':'supported' if p2_ok else 'refuted','by_variance':p2},'P3_finite_interior':{'status':'supported' if all(r['finite'] and r['interior'] for r in rows) else 'refuted'}},'cells':rows,'method':'mean audit MSE over paired fixed-label minibatch trajectories; t_star first global argmin of mean curve','source_hashes':{'old_summary':sha(OLD)}}
 (STUDY/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary['predictions'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
