from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
PREREG = STUDY / 'preregistration.json'
RESULTS = STUDY / 'results'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def dataset(name, cfg):
    g = torch.Generator().manual_seed(cfg['data_seed'] + cfg['functions'].index(name))
    dim = 12 if name.endswith('12') else 8
    n = cfg['train_n'] + cfg['test_n']
    x = torch.randn(n, dim, generator=g) if dim == 12 else 2 * torch.rand(n, dim, generator=g) - 1
    if name == 'trigonometric8': y = torch.sin(2*x[:,0]) + .7*torch.cos(3*x[:,1]) + .3*x[:,2]*x[:,3]
    elif name == 'quadratic8': y = x[:,0]**2 + .5*x[:,1]**2 - .6*x[:,2]**2 + .2*x[:,3]
    elif name == 'interaction12': y = torch.sin(x[:,0]*x[:,1]) + .5*torch.tanh(x[:,2]+x[:,3]) + .2*x[:,4]*x[:,5]
    elif name == 'radial12': y = torch.exp(-.25*x[:,:6].square().sum(1)) + .1*x[:,6]
    else: raise ValueError(name)
    ntr = cfg['train_n']; center = y[:ntr].double().mean(); scale = y[:ntr].double().std(unbiased=False)
    y = ((y.double()-center)/scale).float().unsqueeze(1)
    return x[:ntr], y[:ntr], x[ntr:], y[ntr:]

class Block(nn.Module):
    def __init__(self, width, use_ln):
        super().__init__(); self.norm = nn.LayerNorm(width) if use_ln else nn.Identity(); self.linear = nn.Linear(width,width); self.act = nn.GELU()
    def forward(self,x): return self.act(self.linear(self.norm(x)))

class Model(nn.Module):
    def __init__(self, input_dim, recipe):
        super().__init__(); w=recipe['width']; self.net=nn.Sequential(nn.Linear(input_dim,w),nn.GELU(),*[Block(w,z) for z in recipe['layer_norm']],nn.Linear(w,1))
    def hidden(self,x):
        h=x
        for layer in list(self.net.children())[:-1]: h=layer(h)
        return h
    def forward(self,x): return self.net(x)

def gram_stat(model,x):
    with torch.inference_mode():
        h=model.hidden(x); h=h-h.mean(0,keepdim=True); g=(h.T@h)/h.shape[0]; eig=torch.linalg.eigvalsh(g).double().cpu().numpy()[::-1]
    pos=np.maximum(eig,1e-12); return eig.tolist(), float(np.log10(pos[0]/pos[-1]))

def run_cell(function,label,mean,seed,cfg,recipe):
    ident=f'{function}_{label}_{mean}_{seed}'; path=RESULTS/(ident+'.json'); arr=RESULTS/(ident+'.npz')
    req={'function':function,'label':label,'mean':mean,'seed':seed,'recipe':recipe,'preregistration_sha256':sha(PREREG)}
    contract=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
    if path.exists():
        row=json.loads(path.read_text())
        if row.get('contract_sha256')!=contract or not arr.exists() or sha(arr)!=row.get('arrays_sha256'): raise RuntimeError('existing result hash mismatch')
        return row,True
    tx,ty,vx,vy=dataset(function,cfg); ty=ty+mean; vy=vy+mean; torch.manual_seed(seed); model=Model(tx.shape[1],recipe)
    opt=torch.optim.SGD(model.parameters(),lr=recipe['lr'],momentum=recipe['momentum'],weight_decay=recipe['weight_decay'])
    cps=set(cfg['checkpoints']); spectra={}; kappas={}
    if 0 in cps:
        spectra['0'],kappas['0']=gram_stat(model,tx)
    t0=time.monotonic()
    for step in range(1,recipe['steps']+1):
        idx=torch.randint(0,tx.shape[0],(recipe['batch_size'],)); loss=((model(tx[idx])-ty[idx])**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step in cps: spectra[str(step)],kappas[str(step)]=gram_stat(model,tx)
    with torch.inference_mode(): pred=model(vx).double().flatten().cpu().numpy()
    np.savez_compressed(arr,predictions=pred,targets=vy.double().flatten().numpy())
    row={**req,'contract_sha256':contract,'arrays_sha256':sha(arr),'spectrum':spectra,'log10_condition':kappas,'seconds':time.monotonic()-t0,'status':'completed'}; save_json(path,row); return row,False

def main():
    torch.set_num_threads(1); cfg=json.loads(PREREG.read_text()); total=len(cfg['functions'])*len(cfg['recipes'])*len(cfg['means'])*len(cfg['seeds']); done=reused=0
    for fn in cfg['functions']:
        for label,recipe in cfg['recipes'].items():
            for mean in cfg['means']:
                for seed in cfg['seeds']:
                    _,cache=run_cell(fn,label,mean,seed,cfg,recipe); done+=1; reused+=int(cache); save_json(STUDY/'current.json',{'status':'running','completed':done,'total':total,'reused':reused,'last_cell':f'{fn}_{label}_{mean}_{seed}'})
    save_json(STUDY/'current.json',{'status':'measurements_complete','completed':done,'total':total,'reused':reused})
if __name__ == '__main__': main()
