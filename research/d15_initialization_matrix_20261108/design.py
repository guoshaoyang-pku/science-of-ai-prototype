"""Frozen synthetic tasks and initialization interventions for D15."""
from functools import lru_cache
import hashlib
import itertools
import numpy as np

TASKS = ('linear', 'relu_teacher', 'interaction')
PROFILES = ('uniform', 'front', 'back', 'alternating', 'reverse_alternating',
            'hourglass', 'middle_heavy', 'step', 'reverse_step',
            'single_bottleneck', 'sparse_gates', 'geometric', 'periodic',
            'identity_random', 'head_zero', 'first_zero', 'middle_zero',
            'two_zero', 'zas_embed', 'head_near_zero')
STEPS = 512
DIM, NTRAIN, NVAL = 8, 64, 256
LRS = {'sgd': (0.01, 0.05), 'adam': (0.001, 0.005)}


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)


@lru_cache(None)
def task(name, condition, seed):
    rng=np.random.default_rng(stable_seed('data',seed))
    X=rng.normal(size=(NTRAIN+NVAL,DIM))
    X*=np.sqrt(np.geomspace(1,1/condition,DIM))
    teacher=np.random.default_rng(stable_seed('teacher',seed))
    if name=='linear':
        v=teacher.normal(size=DIM); v/=np.linalg.norm(v); y=X@v
    elif name=='relu_teacher':
        w=teacher.normal(size=(DIM,16))/np.sqrt(DIM)
        a=teacher.normal(size=16)/4
        y=np.maximum(X@w,0)@a
    elif name=='interaction':
        y=X[:,0]*X[:,1]
    else: raise ValueError(name)
    norm=np.sqrt(np.mean(y[:NTRAIN]**2)); y=y/norm
    return X[:NTRAIN],y[:NTRAIN,None],X[NTRAIN:],y[NTRAIN:,None]


def scales(c):
    L=c['depth']; l=np.arange(L); pos=np.linspace(-1,1,L)
    p=c['profile']; z=np.zeros(L)
    if p=='front':z=-pos
    elif p=='back':z=pos
    elif p in ('alternating','reverse_alternating'):
        z=.7*np.where((l+1)%2==0,1,-1)*(1 if p=='alternating' else -1)
    elif p in ('hourglass','middle_heavy'):
        z=2*np.abs(pos)*(1 if p=='hourglass' else -1)
    elif p in ('step','reverse_step'):
        z=np.where(l<L/2,np.log(2),-np.log(2))*(1 if p=='step' else -1)
    elif p=='single_bottleneck':z[L//2]=np.log(.1)
    elif p=='sparse_gates':z[(l%3)==1]=np.log(.1)
    elif p=='geometric':z=l*np.log(.5)
    elif p=='periodic':z=.7*np.cos(2*np.pi*(l+1)/3)
    elif p=='grid':z=c['alpha']*pos+c['beta']*np.where((l+1)%2==0,1,-1)
    return np.exp(z-z.mean())*c.get('global_scale',1.)


def initial_weights(c):
    sizes=[DIM]+[c['width']]*(c['depth']-1)+[1]
    rng=np.random.default_rng(stable_seed('weights',c['seed'],c['depth'],c['width']))
    ws=[rng.normal(size=(a,b))/np.sqrt(a) for a,b in zip(sizes,sizes[1:])]
    p=c['profile']; factor=scales(c)
    if p=='identity_random':
        for j,w in enumerate(ws):
            eye=np.eye(*w.shape)
            mixed=(w+eye)/np.sqrt(2)
            ws[j]=mixed*np.linalg.norm(w)/np.linalg.norm(mixed)
    for j in range(len(ws)):ws[j]=ws[j]*factor[j]
    if p in ('head_zero','zas_embed'):ws[-1]*=0
    if p=='first_zero':ws[0]*=0
    if p=='middle_zero':ws[len(ws)//2]*=0
    if p=='two_zero':ws[0]*=0;ws[-1]*=0
    if p=='head_near_zero':ws[-1]*=.001
    if p=='zas_embed':
        for j,w in enumerate(ws[:-1]):ws[j]=np.eye(*w.shape)
    return ws


def cases(round_id):
    result=[]
    if round_id=='r001':
        grid=itertools.product(TASKS,('linear','relu','tanh'),('sgd','adam'),range(2),range(9301,9305),PROFILES)
        for task_name,act,opt,rate,seed,profile in grid:
            result.append(dict(task=task_name,activation=act,optimizer=opt,rate=rate,seed=seed,
                profile=profile,architecture='plain',depth=4,width=16,condition=1))
    elif round_id=='r002':
        grid=itertools.product(TASKS,('plain','residual_hidden','layernorm_hidden'),(2,4,6),(8,16),
                              ('sgd','adam'),range(2),(1,10),range(9401,9404),
                              ('uniform','front','back','alternating','head_zero','zas_embed'))
        for t,arch,depth,width,opt,rate,condition,seed,p in grid:
            result.append(dict(task=t,activation='relu',optimizer=opt,rate=rate,seed=seed,
                profile=p,architecture=arch,depth=depth,width=width,condition=condition))
    elif round_id=='r003':
        grid=itertools.product(TASKS,('linear','relu','tanh'),('sgd','adam'),range(2),
                              range(9501,9504),(-1.,0.,1.),(-.7,0.,.7),(.3,1.,3.))
        for t,act,opt,rate,seed,a,b,scale in grid:
            result.append(dict(task=t,activation=act,optimizer=opt,rate=rate,seed=seed,profile='grid',
                architecture='plain',depth=4,width=16,condition=1,alpha=a,beta=b,global_scale=scale))
    else:raise ValueError(round_id)
    for i,c in enumerate(result):c.update(id=f'{round_id}_{i:05d}',round=round_id,lr=LRS[c['optimizer']][c['rate']],steps=STEPS)
    return result


def group_key(c):
    return tuple(c[k] for k in ('architecture','activation','optimizer','depth','width','lr'))
