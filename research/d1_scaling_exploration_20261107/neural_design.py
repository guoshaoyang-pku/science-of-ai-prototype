"""Frozen synthetic tasks and function-matched factorizations for round r002."""
from __future__ import annotations
import itertools
import numpy as np

SEEDS=[8101,8102,8201,8202]
CALIBRATION_SEEDS=SEEDS[:2]
CONFIRMATION_SEEDS=SEEDS[2:]
DIM=8
TRAIN_N=64
VALIDATION_N=256
STEPS=256
REFERENCE_STEPS=2048
METHODS=["SGD","Momentum","Adam","AdamW","RMSprop","Adagrad"]


def base_key(case):
    return tuple(case[k] for k in ['teacher','input_condition','activation','width','seed'])


def calibration_group(case):
    return tuple(case[k] for k in ['teacher','input_condition','activation','width'])


def activation_pattern(case):
    # Hold the nonlinear gate count fixed when adding trainable linear factors.
    return [case['activation']]+['identity']*(case['depth']-2)


def cases():
    rows=[]
    bases=list(itertools.product(['linear','relu_teacher'],[1.,10.],['identity','relu'],[8,32],SEEDS))
    for teacher,condition,activation,width,seed in bases:
        base=dict(teacher=teacher,input_condition=condition,activation=activation,width=width,seed=seed)
        for balance,protocol,radius in itertools.product([.1,1.,10.],['parameter_radius','function_radius'],[.25,1.]):
            rows.append(dict(**base,depth=2,balance=balance,eta_protocol=protocol,radius=radius,cohort='core'))
        if seed in CONFIRMATION_SEEDS:
            for balance in [.1,1.,10.]:
                rows.append(dict(**base,depth=4,balance=balance,eta_protocol='function_radius',radius=1.,cohort='depth_holdout'))
    for i,row in enumerate(rows):
        row['case_id']=f'n{i:04d}'
        row['split']='calibration' if row['seed'] in CALIBRATION_SEEDS and row['depth']==2 and row['balance']==1 else ('confirmation' if row['seed'] in CONFIRMATION_SEEDS else 'development_unfitted')
    return rows


def task(case):
    """Only task/covariance/draw affect labels; no optimizer sees validation labels."""
    rng=np.random.default_rng(300000+case['seed'])
    variances=np.geomspace(1/case['input_condition'],1.,DIM)
    x=rng.normal(size=(TRAIN_N,DIM))*np.sqrt(variances)
    xv=rng.normal(size=(VALIDATION_N,DIM))*np.sqrt(variances)
    if case['teacher']=='linear':
        w=rng.normal(size=DIM)/np.sqrt(DIM)
        y=x@w;yv=xv@w
    else:
        w1=rng.normal(size=(DIM,16))*np.sqrt(2/DIM)
        w2=rng.normal(size=16)/4
        y=np.maximum(x@w1,0)@w2;yv=np.maximum(xv@w1,0)@w2
    # No centering: a bias-free linear student must retain its intended model class.
    scale=np.sqrt(np.mean(y*y))
    return x,y/scale,xv,yv/scale


def weights(case):
    """Positive homogeneous balance and inserted identity layers preserve f0."""
    d,w= DIM,case['width']
    rng=np.random.default_rng(400000+case['seed']+w*100)
    if case['activation']=='identity':
        first,rr=np.linalg.qr(rng.normal(size=(w,d)))
        first*=np.sign(np.diag(rr))[None,:]
        # v is independent of width, so all identity models share the initial map.
        v=np.random.default_rng(410000+case['seed']).normal(size=d)
        v*=.1/np.linalg.norm(v)
        last=(v@first.T)[None,:]
    else:
        first=rng.normal(size=(w,d))*np.sqrt(2/d)
        last=rng.normal(size=(1,w))/np.sqrt(w)*.1
    seq=[first]
    seq.extend(np.eye(w) for _ in range(case['depth']-2))
    seq.append(last)
    seq=[a.copy() for a in seq]
    seq[0]*=case['balance'];seq[-1]/=case['balance']
    return seq


def canonical(case):
    return dict(case,depth=2,balance=1.,eta_protocol='function_radius',radius=1.,cohort='reference')
