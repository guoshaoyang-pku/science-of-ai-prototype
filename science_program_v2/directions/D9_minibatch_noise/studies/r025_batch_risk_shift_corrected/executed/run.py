#!/usr/bin/env python3
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
STUDY=Path(__file__).resolve().parents[1]; ROOT=STUDY.parents[3]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def contract():
 c=json.loads((STUDY/'preregistration.json').read_text()); head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); pins={}
 for name in ['executed/run.py','analysis.py','executed/verify.py']:
  p=STUDY/name; ref=head+':'+p.relative_to(ROOT).as_posix(); stored=subprocess.check_output(['git','show',ref],cwd=ROOT)
  if stored!=p.read_bytes(): raise RuntimeError('uncommitted '+name)
  pins[name]=sha(p)
 for name,x in c['source_sha256'].items():
  if pins[name]!=x: raise RuntimeError('source pin '+name)
 for name,x in c['input_sha256'].items():
  if sha(ROOT/name)!=x: raise RuntimeError('input pin '+name)
 return c,{'git_commit':head,'sha256':pins}
def run_cell(d,b,var,cfg,seed):
 x,y,xa,ya=d['train_features'],d['train_y'],d['audit_features'],d['audit_y']
 n,p=x.shape; S=cfg['steps']; R=cfg['replicates']; out=np.empty((R,S+1))
 for r in range(R):
  rng=np.random.default_rng(cfg['batch_seed_base']+100000*seed+1000*b+10*int(var*100)+r)
  target=y+np.sqrt(var)*d['epsilon']; w=np.zeros(p)
  for t in range(S+1):
   z=xa@w-ya
   out[r,t]=np.mean(z*z)
   if t==S:
    break
   idx=np.arange(n) if b==n else rng.choice(n,size=b,replace=False)
   w-=2*cfg['lr']*(x[idx].T@(x[idx]@w-target[idx]))/b
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--max-new-cells',type=int); a=ap.parse_args(); cfg,pins=contract(); rp=STUDY/'results/receipt.json'; rec=json.loads(rp.read_text()) if rp.exists() else {'cells':{},'compute_seconds':0}; start,prev,new=time.monotonic(),rec['compute_seconds'],0
 save(STUDY/'executed/source_manifest.json',{**pins,'numpy':np.__version__,'method':'paired_minibatch_mc'})
 for seed in cfg['seeds']:
  src=ROOT/cfg['data_files'][str(seed)]; local=STUDY/'executed'/src.name
  if local.exists() and sha(local)!=sha(src): raise RuntimeError('data mismatch')
  if not local.exists(): local.write_bytes(src.read_bytes())
  with np.load(local) as z: d={k:z[k] for k in z.files}
  for b in cfg['batches']:
   for var in cfg['variances']:
    label=f'n64_seed{seed}_B{b}_var{var:g}'; p=STUDY/'results'/f'{label}.json'; q=p.with_suffix('.npz')
    if p.exists() or q.exists() or label in rec['cells']:
     if not(p.exists() and q.exists() and label in rec['cells']): raise RuntimeError('partial '+label)
     row=json.loads(p.read_text());
     if sha(p)!=rec['cells'][label] or row['arrays_sha256']!=sha(q) or row['data_sha256']!=sha(local) or row['pins']['sha256']!=pins['sha256']: raise RuntimeError('hash '+label)
     continue
    if prev+time.monotonic()-start>=cfg['compute_budget_seconds']: raise RuntimeError('budget')
    t0=time.monotonic(); out=run_cell(d,b,var,cfg,seed); mean=out.mean(0); se=out.std(0,ddof=1)/np.sqrt(cfg['replicates']); np.savez_compressed(q,replicate_risk=out,mean_risk=mean,standard_error=se)
    save(p,{'cell':{'n':64,'seed':seed,'batch':b,'noise_variance':var},'status':'completed','pins':pins,'data_sha256':sha(local),'arrays_sha256':sha(q),'steps':cfg['steps'],'replicates':cfg['replicates'],'seconds':time.monotonic()-t0}); rec['cells'][label]=sha(p); rec['compute_seconds']=prev+time.monotonic()-start; save(rp,rec); new+=1; print(json.dumps({'saved':label,'completed':len(rec['cells']),'seconds':rec['compute_seconds']}),flush=True)
    if a.max_new_cells and new>=a.max_new_cells: return
if __name__=='__main__': main()
