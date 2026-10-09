import json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; RES.mkdir(exist_ok=True)
DIMS=4; W=32; STEPS=3000; LR=0.02; SIGMA=0.8
DEPTHS=[1,2,4,8]; SEEDS=[11,22,33]
rng_target=np.random.default_rng(20261007)
Q,_=np.linalg.qr(rng_target.normal(size=(DIMS,DIMS))); T=Q.astype(np.float64)
X=np.random.default_rng(20261007+1).normal(size=(128,DIMS)).astype(np.float64); Y=X@T.T

def init_layers(L, seed):
    rng=np.random.default_rng(seed)
    if L==1: shapes=[(DIMS,DIMS)]
    else: shapes=[(DIMS,W)]+[(W,W)]*(L-2)+[(W,DIMS)]
    return [rng.normal(0,SIGMA/np.sqrt(fin),size=(fin,fout)).astype(np.float64) for fin,fout in shapes]

def train(L, seed):
    layers=init_layers(L,seed); losses=np.empty(STEPS+1);
    for step in range(STEPS+1):
        acts=[X]; z=X
        for A in layers: z=z@A; acts.append(z)
        err=z-Y; losses[step]=np.mean(err*err)
        if step==STEPS: break
        g=2*err/X.shape[0]
        grads=[]
        for j in range(L-1,-1,-1):
            grads.append(acts[j].T@g)
            g=g@layers[j].T
        for A,G in zip(layers,reversed(grads)): A-=LR*G
    return losses

def logistic(t,a,c,k,logtc):
    u=k*(np.log1p(t)-logtc); u=np.clip(u,-700,700)
    return c+(a-c)/(1+np.exp(u))

def fit_a(loss):
    t=np.arange(loss.size,dtype=np.float64); ymax=float(np.max(loss));
    p0=np.array([loss[0],loss[-1],1.0,np.log1p(max(10,np.argmax(loss<((loss[0]+loss[-1])/2))))],float)
    lb=np.array([0.0,0.0,1e-5,0.0]); ub=np.array([max(1,ymax*3),max(1,ymax*3),30.0,np.log1p(1e7)])
    def fun(p): return logistic(t,*p)-loss
    try:
      out=least_squares(fun,np.minimum(np.maximum(p0,lb+1e-9),ub-1e-9),bounds=(lb,ub),max_nfev=4000,xtol=1e-12,ftol=1e-12,gtol=1e-12)
      p=out.x; pred=logistic(t,*p); rmse=float(np.sqrt(np.mean((pred-loss)**2))); nr=rmse/max(float(loss[0]-loss[-1]),1e-15); ss=float(np.sum((loss-loss.mean())**2)); r2=1-float(np.sum((pred-loss)**2))/ss if ss>0 else 0.0
      return {'a':float(p[0]),'c':float(p[1]),'k':float(p[2]),'logtc':float(p[3]),'tc':float(np.expm1(p[3])),'rmse':rmse,'normalized_rmse':nr,'r2':r2,'success':bool(out.success)}
    except Exception as e:
      return {'error':str(e),'success':False}

def fit_b(loss):
    n=loss.size; sm=np.empty(n); half=5
    for i in range(n): sm[i]=np.mean(loss[max(0,i-half):min(n,i+half+1)])
    x=np.log1p(np.arange(n,dtype=np.float64)); d=np.gradient(sm,x); idx=int(np.argmin(d))
    return {'tc':float(idx),'final_loss':float(loss[-1]),'slope':float(d[idx]),'smooth_window':11}

manifest={'target_sha256':hashlib.sha256(T.tobytes()).hexdigest(),'x_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'y_sha256':hashlib.sha256(Y.tobytes()).hexdigest(),'config':{'DIMS':DIMS,'W':W,'steps':STEPS,'lr':LR,'sigma':SIGMA,'depths':DEPTHS,'seeds':SEEDS}}
(ROOT/'executed_manifest.json').write_text(json.dumps(manifest,indent=2))
all_rows=[]
for L in DEPTHS:
  for seed in SEEDS:
    tag=f'L{L}_s{seed}'; outp=RES/f'{tag}.npz'
    if outp.exists():
      losses=np.load(outp)['loss'];
    else:
      losses=train(L,seed); np.savez_compressed(outp,loss=losses,depth=L,seed=seed)
    fa=fit_a(losses); fb=fit_b(losses); row={'depth':L,'seed':seed,'file':outp.name,'loss0':float(losses[0]),'loss_final':float(losses[-1]),'A':fa,'B':fb}; all_rows.append(row)
    (RES/f'{tag}.json').write_text(json.dumps(row,indent=2))
summary={'manifest':manifest,'rows':all_rows,'timestamp':time.time()}; (ROOT/'analysis'/'raw_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
