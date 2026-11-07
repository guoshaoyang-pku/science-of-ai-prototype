"""Small independent nn.Module/optimizer/Jacobian checks; no research training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from neural_engine import simulate_batch,initial_kernel,initial_jvp


def make_optimizer(method,parameters,eta,p):
    if method=='SGD':return torch.optim.SGD(parameters,lr=eta,weight_decay=p['weight_decay'])
    if method=='Momentum':return torch.optim.SGD(parameters,lr=eta,momentum=p['mu'],weight_decay=p['weight_decay'])
    if method=='Adam':return torch.optim.Adam(parameters,lr=eta,betas=(p['beta1'],p['beta2']),eps=p['eps'],weight_decay=p['weight_decay'])
    if method=='AdamW':return torch.optim.AdamW(parameters,lr=eta,betas=(p['beta1'],p['beta2']),eps=p['eps'],weight_decay=p['weight_decay'])
    if method=='RMSprop':return torch.optim.RMSprop(parameters,lr=eta,alpha=p['alpha'],eps=p['eps'],weight_decay=p['weight_decay'])
    return torch.optim.Adagrad(parameters,lr=eta,eps=p['eps'],initial_accumulator_value=p['adagrad_init'],weight_decay=p['weight_decay'])


def forward(parameters,x,pattern):
    for i,w in enumerate(parameters):
        x=torch.nn.functional.linear(x,w)
        if i<len(parameters)-1 and pattern[i]=='relu':x=torch.relu(x)
    return x.squeeze(-1)


def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(19871)
    reports=[];max_error=0.;checks=0
    for depth,pattern in [(2,['identity']),(2,['relu']),(4,['identity']*3),(4,['relu','identity','identity'])]:
        dims=[3]+[4]*(depth-1)+[1]
        w0=[torch.randn(2,b,a)/3 for a,b in zip(dims,dims[1:])]
        x=torch.randn(2,5,3);y=torch.randn(2,5);xv=torch.randn(2,7,3);yv=torch.randn(2,7)
        parameters={'mu':np.array([.5,.9]),'beta1':np.array([.5,.9]),'beta2':np.array([.9,.999]),'alpha':np.array([.9,.99]),'eps':np.array([1e-8,.002]),'weight_decay':np.array([0.,.03]),'adagrad_init':np.array([0.,.1])}
        eta=np.array([.002,.003]);steps=12
        jacobians=[]
        for row in range(2):
            one=tuple(w[row].clone().requires_grad_(True) for w in w0)
            jac=torch.autograd.functional.jacobian(lambda *ww:forward(ww,x[row],pattern),one)
            jac=torch.cat([j.reshape(5,-1) for j in jac],dim=1);jacobians.append(jac)
        expected_kernel=torch.stack([j@j.T for j in jacobians])
        assert torch.allclose(initial_kernel(w0,x,pattern),expected_kernel,rtol=1e-11,atol=1e-13);checks+=1
        dw=[torch.randn_like(w) for w in w0]
        expected_jvp=torch.stack([jacobians[r]@torch.cat([w[r].flatten() for w in dw]) for r in range(2)])
        assert torch.allclose(initial_jvp(w0,x,dw,pattern),expected_jvp,rtol=1e-11,atol=1e-13);checks+=1
        for method in ['SGD','Momentum','Adam','AdamW','RMSprop','Adagrad']:
            result=simulate_batch(w0,x,y,xv,yv,eta,method,pattern,steps,parameters)
            method_error=0.
            for row in range(2):
                actual=[nn.Parameter(w[row].clone()) for w in w0]
                frozen=[nn.Parameter(w[row].clone()) for w in w0]
                spec={k:float(v[row]) for k,v in parameters.items()}
                opt=make_optimizer(method,actual,float(eta[row]),spec)
                const_opt=make_optimizer(method,frozen,float(eta[row]),spec)
                initial_loss=.5*torch.mean((forward(actual,x[row],pattern)-y[row])**2)
                grads=torch.autograd.grad(initial_loss,actual)
                r0=(forward(actual,x[row],pattern)-y[row]).detach()
                theta0=torch.cat([w[row].flatten() for w in w0]);norm0=torch.linalg.norm(theta0)
                for t in range(steps+1):
                    tr=.5*torch.mean((forward(actual,x[row],pattern)-y[row])**2)
                    va=.5*torch.mean((forward(actual,xv[row],pattern)-yv[row])**2)
                    delta=torch.cat([(w-wi[row]).detach().flatten() for w,wi in zip(frozen,w0)])
                    df=jacobians[row]@delta
                    values={'loss':float(tr.detach()),'validation_loss':float(va.detach()),'u_const':float(torch.linalg.norm(delta)),
                            'relative_d':float(torch.linalg.norm(delta)/norm0),'function_norm_V':float(torch.linalg.norm(df)/torch.linalg.norm(r0)),
                            'function_projection_P':float(-r0@df/(r0@r0)),'exact_const_loss':float((.5*torch.mean((forward(frozen,x[row],pattern)-y[row])**2)).detach())}
                    for key,value in values.items():
                        error=abs(result[key][row,t]-value);method_error=max(method_error,error);checks+=1
                        assert np.isclose(result[key][row,t],value,rtol=2e-10,atol=2e-12),(depth,pattern,method,row,t,key,error)
                    if t==steps:break
                    opt.zero_grad();tr.backward();opt.step()
                    for parameter,g in zip(frozen,grads):parameter.grad=g.clone()
                    const_opt.step()
            max_error=max(max_error,method_error)
            reports.append({'depth':depth,'pattern':pattern,'method':method,'paths':2,'steps':steps,'max_abs_error':method_error})
    bad=simulate_batch(w0,x,y,xv,yv,np.array([1e20,.003]),'SGD',pattern,4,{'weight_decay':0.})
    good=simulate_batch([w[1:] for w in w0],x[1:],y[1:],xv[1:],yv[1:],.003,'SGD',pattern,4,{'weight_decay':0.})
    assert bad['diverged_actual'][0] and np.isnan(bad['loss'][0,bad['first_divergence_step_actual'][0]:]).all()
    assert np.allclose(bad['loss'][1],good['loss'][0],rtol=1e-12,atol=1e-14);checks+=2
    out={'status':'passed','checks':checks,'max_abs_error':max_error,'cases':reports,'failure_isolation':'passed',
         'scope':'independent PyTorch optimizers for 48 micro paths, 12 steps; explicit autograd Jacobian/JVP and exact layer Gram kernels; not formal experiment results',
         'engine_sha256':hashlib.sha256((Path(__file__).parent/'neural_engine.py').read_bytes()).hexdigest()}
    (Path(__file__).parent/'neural_engine_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
