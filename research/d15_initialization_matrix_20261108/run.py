"""Execute only committed D15 rounds, never overwrite a saved batch."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import design
from engine import train_batch, forward_batch, tensor

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]
DATA=Path(os.environ.get('AIQ_KB_DATA_ROOT',str(REPO.parent/'AIQ_KB_DATA'))).resolve()
RUN=DATA/'runs/d15_initialization_matrix_20261108'
SOURCES=('design.py','engine.py','run.py','preregistration.json')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',required=True,choices=('r001','r002','r003'));ap.add_argument('--frozen-repo',required=True)
    args=ap.parse_args();frozen=Path(args.frozen_repo).resolve()
    commit=subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip()
    frozen_source=frozen/'research'/SOURCE.name
    for name in SOURCES:
        expected=subprocess.check_output(['git','-C',str(frozen),'show',f'HEAD:research/{SOURCE.name}/{name}'])
        if hashlib.sha256(expected).hexdigest()!=sha(SOURCE/name):raise RuntimeError('Source differs from preregistration: '+name)
    if RUN.is_relative_to(REPO):raise RuntimeError('Raw run must be external')
    out=RUN/args.round;out.mkdir(parents=True,exist_ok=True)
    selected=design.cases(args.round);casefile=out/'cases.json'
    if casefile.exists():assert json.loads(casefile.read_text(encoding='utf-8'))==selected
    else:dump(casefile,selected)
    receipt=out/'start.json'
    if not receipt.exists():dump(receipt,dict(preregistration_commit=commit,started_unix=time.time(),source_hashes={n:sha(SOURCE/n) for n in SOURCES},paths=len(selected),torch=torch.__version__,numpy=np.__version__))
    groups=defaultdict(list)
    for c in selected:groups[design.group_key(c)].append(c)
    chunks=[]
    for group in groups.values():chunks.extend(group[i:i+48] for i in range(0,len(group),48))
    torch.set_num_threads(1)
    total_start=time.monotonic();done=0
    for batch_index,chunk in enumerate(chunks):
        target=out/f'batch_{batch_index:04d}.npz';meta=target.with_suffix('.json')
        if target.exists():
            info=json.loads(meta.read_text());assert info['sha256']==sha(target) and info['ids']==[c['id'] for c in chunk]
            done+=len(chunk);continue
        ws=[design.initial_weights(c) for c in chunk]
        weights=[np.stack([w[i] for w in ws]) for i in range(chunk[0]['depth'])]
        datasets=[design.task(c['task'],c['condition'],c['seed']) for c in chunk]
        X,y,Xv,yv=[np.stack([d[j] for d in datasets]) for j in range(4)]
        c=chunk[0]
        with torch.no_grad():
            initial=forward_batch([tensor(w) for w in weights],tensor(X),c['activation'],c['architecture']).numpy()
            baseline=[]
            for case in chunk:
                bc=dict(case,profile='uniform',global_scale=1.)
                baseline.append(design.initial_weights(bc))
            bweights=[tensor(np.stack([w[i] for w in baseline])) for i in range(c['depth'])]
            basepred=forward_batch(bweights,tensor(X),c['activation'],c['architecture']).numpy()
        t=time.monotonic()
        result=train_batch(weights,X,y,Xv,yv,c['activation'],c['architecture'],c['optimizer'],c['lr'],c['steps'])
        arrays={k:v for k,v in result.items() if k!='final_weights'}
        arrays['ids']=np.array([x['id'] for x in chunk]);arrays['initial_prediction']=initial
        arrays['initial_function_delta']=np.max(np.abs(initial-basepred),axis=(1,2))
        arrays['target_train_rms']=np.sqrt(np.mean(y*y,axis=(1,2)))
        arrays['linear_train_floor']=np.array([np.mean((d[0]@np.linalg.lstsq(d[0],d[1],rcond=None)[0]-d[1])**2) for d in datasets])
        for j,w in enumerate(weights):arrays[f'initial_w{j}']=w
        for j,w in enumerate(result['final_weights']):arrays[f'final_w{j}']=w
        with target.open('xb') as handle:np.savez_compressed(handle,**arrays)
        dump(meta,dict(ids=[x['id'] for x in chunk],sha256=sha(target),seconds=time.monotonic()-t))
        done+=len(chunk)
        dump(out/'progress.json',dict(round=args.round,status='running',done=done,total=len(selected),batches=batch_index+1,batch_count=len(chunks),elapsed_seconds=time.monotonic()-total_start))
        print(f'{args.round} {done}/{len(selected)}; batch {batch_index+1}/{len(chunks)}; {time.monotonic()-t:.1f}s',flush=True)
    manifest={p.name:sha(p) for p in sorted(out.glob('batch_*.npz'))}
    dump(out/'manifest.json',dict(round=args.round,status='complete',paths=len(selected),batches=manifest,preregistration_commit=commit,completed_unix=time.time()))
    dump(out/'progress.json',dict(round=args.round,status='complete',done=len(selected),total=len(selected)))


if __name__=='__main__':main()
