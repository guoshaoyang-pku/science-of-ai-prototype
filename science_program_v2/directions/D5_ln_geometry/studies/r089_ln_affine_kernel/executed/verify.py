from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch import nn

STUDY=Path(__file__).resolve().parents[1]
OLD=STUDY.parent/'r077_fixed_kernel_train_chord'

class Block(nn.Module):
    def __init__(self,width,use_ln):
        super().__init__(); self.norm=nn.LayerNorm(width) if use_ln else nn.Identity(); self.linear=nn.Linear(width,width); self.act=nn.GELU()
    def forward(self,x): return self.act(self.linear(self.norm(x)))
class Model(nn.Module):
    def __init__(self,dimension,recipe):
        super().__init__(); width=recipe['width']; self.net=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),*[Block(width,flag) for flag in recipe['layer_norm']],nn.Linear(width,1))
    def forward(self,x): return self.net(x)

def main():
    torch.set_num_threads(1); rng=np.random.default_rng(889); checked=[]; max_kernel=0.; max_eigh=0.; max_vjp=0.
    for path in sorted((STUDY/'results').glob('*.json')):
        row=json.loads(path.read_text()); assert row['status']=='completed'; npz=path.with_suffix('.npz')
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==row['arrays_sha256']
        with np.load(npz) as data:
            jac=data['ln_affine_jacobian']; ka=np.einsum('ip,jp->ij',jac,jac,optimize=False)/jac.shape[0]; kt=data['kernel'].astype(np.float64)+ka
            max_kernel=max(max_kernel,float(np.max(np.abs(ka-data['ln_affine_kernel']))),float(np.max(np.abs(kt-data['total_kernel']))))
            vals,vecs=np.linalg.eigh(kt); v=np.ones(kt.shape[0],dtype=np.float64)
            for _ in range(256): v=v-.002*(vecs@(vals*(vecs.T@v)))
            max_eigh=max(max_eigh,float(np.max(np.abs(v-data['affine_total_propagated_ones_256']))))
            function=row['contract']['function']; seed=row['contract']['seed']; old_path=OLD/'results'/f'{function}_LN010_w64_{seed}.json'; old=json.loads(old_path.read_text())
            assert old['arrays_sha256']==row['contract']['old_result_npz_sha256']
            assert data['kernel_pin']['sha256']==old['kernel_pin']['sha256']
            torch.manual_seed(seed); model=Model(data['train_x'].shape[1],row['contract']['recipe']); named=dict(model.named_parameters()); names=('net.3.norm.weight','net.3.norm.bias'); params=[named[n] for n in names]
            for name,p in named.items(): p.requires_grad_(name in names)
            out=model(torch.from_numpy(data['train_x'])).ravel(); q=torch.from_numpy(rng.standard_normal(out.numel()).astype(np.float32))
            grads=torch.autograd.grad(torch.dot(out,q),params); saved=jac.T@q.numpy().astype(np.float64); actual=np.concatenate([g.detach().numpy().ravel() for g in grads]).astype(np.float64)
            rel=np.linalg.norm(saved-actual)/max(np.linalg.norm(actual),1e-30); max_vjp=max(max_vjp,float(rel))
            assert np.array_equal(out.detach().numpy(),data['prediction_initial_reconstructed'])
        checked.append(path.stem)
    assert max_kernel<1e-12 and max_eigh<1e-9 and max_vjp<2e-5
    result={'status':'passed','cells_verified':len(checked),'cell_names':checked,'max_kernel_abs_error':max_kernel,'max_eigh_vector_abs_error':max_eigh,'max_independent_vjp_relative_error':max_vjp,'checks':['NPZ hashes','affine and total kernel reconstruction','independent eigh propagation','independent random reverse VJP','r077 kernel lineage','initial forward prediction']}
    (STUDY/'executed/saved_evidence_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
