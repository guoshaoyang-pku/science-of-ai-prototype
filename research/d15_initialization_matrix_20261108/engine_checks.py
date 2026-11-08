"""Small synthetic numerical checks, not D15 experimental measurements."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch
from engine import train_batch, forward_batch, tensor


class ScalarNetwork(torch.nn.Module):
    """Route-wise independent implementation using nn.Linear, not batched bmm."""
    def __init__(self, weights, activation, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(w.shape[0], w.shape[1], bias=False, dtype=torch.float64)
            for w in weights
        ])
        with torch.no_grad():
            for layer, w in zip(self.layers, weights):
                layer.weight.copy_(tensor(w).T)
        self.activation, self.architecture = activation, architecture

    def forward(self, x):
        fn = {"linear": lambda h: h, "relu": torch.relu, "tanh": torch.tanh}[self.activation]
        for i, layer in enumerate(self.layers):
            if self.architecture == "residual_hidden" and 0 < i < len(self.layers)-1:
                x = x + fn(layer(x))/np.sqrt(max(len(self.layers)-2,1))
            else:
                x = layer(x)
                if i < len(self.layers)-1 and self.architecture != "residual_hidden":
                    if self.architecture == "layernorm_hidden":
                        x = torch.nn.LayerNorm(x.shape[-1], eps=1e-5, elementwise_affine=False, dtype=torch.float64)(x)
                    x = fn(x)
        return x


def run_checks():
    torch.set_num_threads(1)
    rng = np.random.default_rng(156001)
    B,n,d,h = 3, 12, 8, 8
    X=rng.normal(size=(B,n,d)); Xv=rng.normal(size=(B,9,d))
    y=rng.normal(size=(B,n,1)); yv=rng.normal(size=(B,9,1))
    weights=[rng.normal(size=(B,a,b))/np.sqrt(a) for a,b in [(d,h),(h,h),(h,1)]]
    records=[]
    for architecture in ("plain","residual_hidden","layernorm_hidden"):
        for activation in ("linear","relu","tanh"):
            for optimizer in ("SGD","Adam"):
                lr=np.array([0.002,0.004,0.006])
                got=train_batch(weights,X,y,Xv,yv,activation,architecture,optimizer,lr,3,None)
                train_ref=[]; val_ref=[]; grad_ref=[]; weight_ref=[]
                for route in range(B):
                    model=ScalarNetwork([w[route] for w in weights],activation,architecture)
                    optim=(torch.optim.SGD(model.parameters(),lr=float(lr[route])) if optimizer=="SGD"
                           else torch.optim.Adam(model.parameters(),lr=float(lr[route]),betas=(.9,.999),eps=1e-8))
                    lt=[]; lv=[]
                    for t in range(4):
                        loss=(model(tensor(X[route]))-tensor(y[route])).square().mean()
                        lt.append(float(loss.detach()))
                        lv.append(float((model(tensor(Xv[route]))-tensor(yv[route])).square().mean().detach()))
                        if t<3:
                            optim.zero_grad(); loss.backward()
                            if t==0:
                                grad_ref.append([float(p.grad.norm()) for p in model.parameters()])
                            optim.step()
                    train_ref.append(lt); val_ref.append(lv)
                    weight_ref.append([p.detach().numpy().T.copy() for p in model.parameters()])
                errors={
                    "training":float(np.max(np.abs(got['train_loss']-train_ref))),
                    "validation":float(np.max(np.abs(got['val_loss']-val_ref))),
                    "initial_gradient":float(np.max(np.abs(got['initial_grad_norm']-grad_ref))),
                    "final_weights":float(max(np.max(np.abs(got['final_weights'][i][b]-weight_ref[b][i])) for b in range(B) for i in range(3))),
                }
                assert max(errors.values())<2e-11,(architecture,activation,optimizer,errors)
                assert np.all(got['status_code']==0)
                records.append({"architecture":architecture,"activation":activation,"optimizer":optimizer,"max_abs_errors":errors})
    zero=[w.copy() for w in weights]
    zero[0][:]=0;zero[-1][:]=0
    z=train_batch(zero,X,y,Xv,yv,steps=3)
    assert np.all(z['initial_grad_norm']==0)
    assert np.all(z['train_loss']==z['train_loss'][:,[0]])
    zas=[np.broadcast_to(np.eye(8),(B,8,8)).copy(),np.broadcast_to(np.eye(8),(B,8,8)).copy(),np.zeros((B,8,1))]
    zz=train_batch(zas,X,y,Xv,yv,steps=1)
    assert np.all(zz['initial_grad_norm'][:,:-1]==0)
    assert np.all(zz['initial_grad_norm'][:,-1]>0)
    res=train_batch(zero,X,y,Xv,yv,activation='relu',architecture='residual_hidden',steps=1)
    assert np.all(np.isfinite(res['initial_grad_norm']))
    scaled=[w.copy() for w in weights]
    scaled[0]*=10;scaled[-1]/=10
    function_match={}
    for act in ('linear','relu'):
        before=forward_batch([tensor(w) for w in weights],tensor(X),act)
        after=forward_batch([tensor(w) for w in scaled],tensor(X),act)
        error=float(torch.max(torch.abs(before-after)))
        assert error<1e-12
        function_match[act]=error
    # A broken route must not perturb its healthy neighbors.
    corrupt=[w.copy() for w in weights]
    corrupt[0][1]=1e200
    corrupt[-1][1]=1e200
    combined=train_batch(corrupt,X,y,Xv,yv,steps=3)
    baseline=train_batch([w[[0,2]] for w in weights],X[[0,2]],y[[0,2]],Xv[[0,2]],yv[[0,2]],steps=3)
    assert combined['status_code'][1]!=0
    assert np.all(np.isnan(combined['train_loss'][1]))
    assert np.allclose(combined['train_loss'][[0,2]],baseline['train_loss'],rtol=1e-12,atol=1e-12)
    sampled=train_batch(weights,X,y,Xv,yv,steps=3,checkpoints=[2])
    assert np.array_equal(sampled['checkpoint_steps'],[0,2,3])
    probe_start=time.perf_counter()
    bigB=64
    bw=[np.repeat(w[[0]],bigB,axis=0) for w in weights]
    train_batch(bw,np.repeat(X[[0]],bigB,axis=0),np.repeat(y[[0]],bigB,axis=0),np.repeat(Xv[[0]],bigB,axis=0),np.repeat(yv[[0]],bigB,axis=0),steps=32,optimizer='Adam')
    elapsed=time.perf_counter()-probe_start
    result={"status":"passed","independent_route_comparisons":records,"double_zero_plain":"all initial gradients exactly zero; losses unchanged","zas":"only final layer initial gradient nonzero","function_preserving_rebalance_max_error":function_match,"nonfinite_route_isolation":"passed","checkpoint_selection":"passed","smoke_64_routes_32_steps_seconds":elapsed,"torch_version":torch.__version__,"dtype":"float64","threads":torch.get_num_threads()}
    path=Path(__file__).with_name('engine_checks.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({"status":"passed","route_comparison_groups":len(records),"smoke_seconds":elapsed,"artifact":str(path)}))


if __name__=='__main__':
    run_checks()
