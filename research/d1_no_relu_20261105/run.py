from __future__ import annotations
import argparse, copy, hashlib, json, math, os, shutil, statistics, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

SOURCE_DIR=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get("AIQ_KB_DATA_ROOT",str(REPO.parent/"AIQ_KB_DATA"))).resolve()
STUDY=DATA_ROOT/"runs"/"d1_no_relu_20261105"/"directions"/"D1_u_effective_time"/"studies"/"r001_no_relu"
ETA, EPS, T_MAX = 0.001, 1e-8, 2000
SEEDS=[11,23,37]
Q_VALUES=[0.9,0.5,0.1,0.01]
ARCHS=[{"name":"width64_depth2","width":64,"hidden_layers":2},
       {"name":"width128_depth2","width":128,"hidden_layers":2},
       {"name":"width128_depth4","width":128,"hidden_layers":4}]
METHODS=["SGD","Momentum","Adam","AdamW","RMSprop","Adagrad"]

def contract():
    return {
      "study":"r001_no_relu","direction":"D1_u_effective_time","domain":"development",
      "question":"Does replacing only the student ReLU with Identity recover cross-optimizer U alignment, with the original nonlinear teacher, data and initial Linear tensors held fixed?",
      "predictions":{
        "P1":"At q=0.5, every non-reference optimizer by architecture pair has median relative endpoint error <=5% across the three paired seeds, with all three comparisons uncensored.",
        "P2":"At least one optimizer-by-architecture pair at q=0.1 or q=0.01 has median relative endpoint error >5% or fewer than 3/3 uncensored comparisons."},
      "data":{"task":"synthetic nonlinear scalar regression from a fixed one-hidden-layer ReLU teacher (16-32-1); student is a trainable deep affine network with Identity activations",
        "input_dim":16,"train_examples":256,"validation_examples":256,"target_scaling":"center/scale using training target mean and standard deviation; same transform on validation",
        "seeds":SEEDS,"student_architectures":ARCHS,
        "pairing":"within architecture and seed, all optimizers share data, validation set, initial weights and initial loss"},
      "training":{"device":"CPU","dtype":"float64","batching":"full batch, fixed order, no shuffling","loss":"mean squared error",
        "eta_all_optimizers":ETA,"steps":T_MAX,"thresholds_q":Q_VALUES,
        "optimizer_settings":{"SGD":{"momentum":0,"weight_decay":0},"Momentum":{"momentum":0.9,"dampening":0,"nesterov":False,"weight_decay":0},
          "Adam":{"betas":[0.9,0.999],"eps":EPS,"weight_decay":0,"bias_correction":True},
          "AdamW":{"betas":[0.9,0.999],"eps":EPS,"weight_decay":0.01,"bias_correction":True},
          "RMSprop":{"alpha":0.99,"eps":EPS,"momentum":0,"centered":False,"bias_correction":False},
          "Adagrad":{"initial_accumulator_value":0,"eps":EPS,"lr_decay":0,"weight_decay":0}},
        "determinism":"PyTorch deterministic algorithms; one CPU thread; fixed seeds; no stochastic layers"},
      "U_definition":{"purpose":"empirically test the constant-initial-gradient parameter-displacement calibration without a loss-mode decomposition",
        "globalization":"compute coordinatewise predicted displacement under constant g0, then take its Euclidean norm as scalar U",
        "SGD":"eta*t*||g0||2","Momentum":"eta*[t/(1-mu)-mu*(1-mu^t)/(1-mu)^2]*||g0||2, mu=0.9",
        "Adam":"||eta*t*g0/(abs(g0)+eps)||2",
        "AdamW":"||theta_const(t)-theta0||2 under decoupled shrinkage c=1-eta*wd, including shrinkage in total displacement",
        "RMSprop":"||sum_s eta*g0/(abs(g0)*sqrt(1-alpha^s)+eps)||2; alpha=.99, no bias correction",
        "Adagrad":"||sum_s eta*g0/(sqrt(s*g0^2)+eps)||2; accumulator starts at zero",
        "comparison":"per architecture/seed/q, observed SGD first-crossing defines target U; predict each other method's first t with U(t)>=target and compare to observed first-crossing"},
      "outcomes":{"primary":"relative absolute endpoint error |t_pred-t_obs|/t_obs at q=.5; report each optimizer/architecture median over three paired seeds and uncensored denominator",
        "secondary":"same at q=.9/.1/.01; sustained-to-T endpoints; validation thresholds; saved original loss curves against raw U; affine least-squares floor and threshold attainability diagnostics",
        "censoring":"unreached observed threshold or predicted U crossing by T=2000 remains null; do not impute T",
        "tolerance":"5% inherited from D1; report continuous errors and every denominator"},
      "ablation":{"only_architecture_change":"student hidden ReLU replaced by Identity, preserving state_dict keys and RNG consumption",
        "paired_reference":"d1_nn_u_transfer_r1/r001_nn_u_transfer; original ReLU runs will not be rerun or overwritten",
        "important_difference":"same initial Linear tensors, but initial predictions, L0 and g0 differ after activation removal; U is recomputed",
        "residual_diagnostic":"fit affine OLS on training data with an intercept; report L_affine_star/L0 for every training cell and flag q below this lower bound as class-unattainable",
        "metrics":"store all 2001 per-step loss and U values in separate metrics files, plus 401-point curves",
        "prediction_interpretation":"P1 is a falsifiable alignment hypothesis, not an expectation of success; P2 includes censoring as an evidence limit, not as a measured endpoint error",
        "plotting":"linear raw U; original L/L0 on log y; initial x focus is the largest first U reaching floor_ratio+0.01*(1-floor_ratio), over trajectories that reach it, or full x if none; full trajectories and t=2000 markers retained; no subtraction from primary loss metric"},
      "scope_limits":["development only; synthetic regression is not a benchmark or real-task test",
        "three factorized affine architectures and three paired seeds; seeds do not add architecture conditions; no claim that the joint parameter loss is quadratic",
        "no quadratic-loss mode decomposition is assumed; scalar Euclidean aggregation of coordinate U is an operational choice, not a theorem",
        "one common learning rate and fixed optimizer settings; no hyperparameter tuning or optimizer ranking",
        "no claim about classification, CNNs, transformers, minibatch noise, other data distributions, or learning rates"]}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seal():
    data=contract(); data["execution_source_sha256"]=sha(Path(__file__))
    p=SOURCE_DIR/"preregistration.json"; p.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"preregistration":str(p),"source_sha256":data["execution_source_sha256"]},indent=2))

class Student(nn.Module):
    def __init__(self,width,layers):
        super().__init__(); seq=[]; dim=16
        for _ in range(layers): seq.extend([nn.Linear(dim,width),nn.Identity()]); dim=width
        seq.append(nn.Linear(dim,1)); self.net=nn.Sequential(*seq)
    def forward(self,x): return self.net(x).squeeze(-1)

def make_data(seed):
    gen=torch.Generator().manual_seed(100000+seed); dt=torch.float64
    xtr=torch.randn(256,16,generator=gen,dtype=dt); xva=torch.randn(256,16,generator=gen,dtype=dt)
    w1=torch.randn(16,32,generator=gen,dtype=dt)*math.sqrt(2/16); b1=torch.zeros(32,dtype=dt)
    w2=torch.randn(32,1,generator=gen,dtype=dt)/math.sqrt(32); b2=torch.zeros(1,dtype=dt)
    def teacher(x): return (torch.relu(x@w1+b1)@w2+b2).squeeze(-1)
    ytr,yva=teacher(xtr),teacher(xva); mean,std=ytr.mean(),ytr.std(unbiased=False)
    return xtr,(ytr-mean)/std,xva,(yva-mean)/std

def optimizer(method,model):
    if method=="SGD": return torch.optim.SGD(model.parameters(),lr=ETA)
    if method=="Momentum": return torch.optim.SGD(model.parameters(),lr=ETA,momentum=.9)
    if method=="Adam": return torch.optim.Adam(model.parameters(),lr=ETA,betas=(.9,.999),eps=EPS)
    if method=="AdamW": return torch.optim.AdamW(model.parameters(),lr=ETA,betas=(.9,.999),eps=EPS,weight_decay=.01)
    if method=="RMSprop": return torch.optim.RMSprop(model.parameters(),lr=ETA,alpha=.99,eps=EPS,momentum=0,centered=False)
    if method=="Adagrad": return torch.optim.Adagrad(model.parameters(),lr=ETA,lr_decay=0,weight_decay=0,initial_accumulator_value=0,eps=EPS)
    raise ValueError(method)

def arrays(model):
    theta=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float64)
    grad=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float64)
    return theta,grad

def u_series(method,theta,g):
    u=np.zeros(T_MAX+1,dtype=np.float64)
    if method=="SGD": u[1:]=ETA*np.linalg.norm(g)*np.arange(1,T_MAX+1)
    elif method=="Momentum":
        t=np.arange(1,T_MAX+1,dtype=np.float64); f=t/.1-.9*(1-.9**t)/(.1**2); u[1:]=ETA*np.linalg.norm(g)*f
    elif method=="Adam": u[1:]=ETA*np.arange(1,T_MAX+1)*np.linalg.norm(g/(np.abs(g)+EPS))
    elif method=="AdamW":
        c=1-ETA*.01; step=ETA*g/(np.abs(g)+EPS); t=np.arange(1,T_MAX+1,dtype=np.float64)
        cp=c**t; geom=(1-cp)/(1-c); u[1:]=np.linalg.norm((cp[:,None]-1)*theta[None,:]-geom[:,None]*step[None,:],axis=1)
    else:
        disp=np.zeros_like(g)
        if method=="RMSprop":
            v=np.zeros_like(g)
            for t in range(1,T_MAX+1): v=.99*v+.01*g*g; disp+=ETA*g/(np.sqrt(v)+EPS); u[t]=np.linalg.norm(disp)
        else:
            acc=np.zeros_like(g)
            for t in range(1,T_MAX+1): acc+=g*g; disp+=ETA*g/(np.sqrt(acc)+EPS); u[t]=np.linalg.norm(disp)
    return u

def endpoints(losses):
    out={}
    for q in Q_VALUES:
        first=next((i for i,x in enumerate(losses) if x<=q),None)
        last=max((i for i,x in enumerate(losses) if x>q),default=-1)
        out[str(q)]={"first":first,"sustained_to_T":last+1 if last<len(losses)-1 else None}
    return out

def train_cell(arch,seed,method,init_state,data):
    xtr,ytr,xva,yva=data; model=Student(arch["width"],arch["hidden_layers"]).double(); model.load_state_dict(copy.deepcopy(init_state))
    mse=nn.MSELoss(); init=mse(model(xtr),ytr); init.backward(); theta,g=arrays(model); gnorm=float(np.linalg.norm(g)); model.zero_grad(set_to_none=True)
    us=u_series(method,theta,g); opt=optimizer(method,model); tr=[]; va=[]; curve=[]; start=time.perf_counter()
    with torch.no_grad(): val0=float(mse(model(xva),yva))
    for step in range(T_MAX+1):
        opt.zero_grad(set_to_none=True); loss=mse(model(xtr),ytr); train=float(loss.detach())/float(init.detach())
        with torch.no_grad(): validation=float(mse(model(xva),yva))/val0
        tr.append(train); va.append(validation)
        if step%5==0 or step==T_MAX: curve.append({"step":step,"train_loss_ratio":train,"validation_loss_ratio":validation,"U_initial_gradient":float(us[step])})
        if step==T_MAX: break
        loss.backward()
        if step==0: assert np.allclose(arrays(model)[1],g,rtol=1e-12,atol=1e-14)
        opt.step()
    elapsed=time.perf_counter()-start
    return {"architecture":arch["name"],"width":arch["width"],"hidden_layers":arch["hidden_layers"],"seed":seed,"optimizer":method,
      "parameter_count":int(theta.size),"initial_gradient_norm":gnorm,"initial_gradient_zero_fraction":float(np.mean(g==0)),
      "initial_train_loss":float(init.detach()),"initial_validation_loss":val0,"train_endpoints":endpoints(tr),"validation_endpoints":endpoints(va),
      "final_train_loss_ratio":tr[-1],"final_validation_loss_ratio":va[-1],"elapsed_seconds":elapsed,"curve":curve,"_u":us,"_train":tr,"_validation":va}

def execute(commit):
    head=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if head!=commit: raise RuntimeError(f"HEAD {head} differs from registered commit {commit}")
    if subprocess.check_output(["git","-C",str(REPO),"diff",commit,"--",str(SOURCE_DIR)],text=True).strip(): raise RuntimeError("registered study files must match preregistration commit")
    pp=SOURCE_DIR/"preregistration.json"; pre=json.loads(pp.read_text(encoding="utf-8-sig"))
    if sha(Path(__file__))!=pre["execution_source_sha256"]: raise RuntimeError("source hash mismatch")
    if STUDY.is_relative_to(REPO): raise RuntimeError("results must stay outside the source repository")
    STUDY.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),STUDY/"run.py")
    shutil.copy2(pp,STUDY/"preregistration.json")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.set_default_dtype(torch.float64)
    records=[]; pairing=[]
    for ai,arch in enumerate(ARCHS):
      for seed in SEEDS:
        data=make_data(seed); torch.manual_seed(200000+1000*ai+seed)
        init_model=Student(arch["width"],arch["hidden_layers"]).double(); state=copy.deepcopy(init_model.state_dict())
        torch.manual_seed(200000+1000*ai+seed)
        reference=Student(arch["width"],arch["hidden_layers"]).double()
        for i,layer in enumerate(reference.net):
          if isinstance(layer,nn.Identity): reference.net[i]=nn.ReLU()
        assert all(torch.equal(state[k],reference.state_dict()[k]) for k in state)
        assert not any(isinstance(layer,nn.ReLU) for layer in init_model.modules())
        xtr,ytr,xva,yva=data
        design=torch.cat([xtr,torch.ones((len(xtr),1),dtype=xtr.dtype)],dim=1)
        coef=torch.linalg.lstsq(design,ytr).solution
        floor=float(torch.mean((design@coef-ytr)**2))
        pairing.append({"architecture":arch["name"],"seed":seed,"data_sha256":hashlib.sha256(b"".join(x.numpy().tobytes() for x in data)).hexdigest(),
          "initial_state_sha256":hashlib.sha256(b"".join(v.numpy().tobytes() for v in state.values())).hexdigest(),
          "relu_initial_tensors_equal":True,"affine_ols_training_loss":floor})
        for method in METHODS:
          row=train_cell(arch,seed,method,state,data); us=row.pop("_u"); curve=row.pop("curve")
          full_train=row.pop("_train"); full_validation=row.pop("_validation")
          assert np.all(np.isfinite(full_train)) and np.all(np.isfinite(full_validation))
          mp=STUDY/"metrics"/arch["name"]/f"seed_{seed}"/f"{method}.json"; mp.parent.mkdir(parents=True,exist_ok=True)
          mp.write_text(json.dumps({"steps":list(range(T_MAX+1)),"train_loss_ratio":full_train,"validation_loss_ratio":full_validation,"U_initial_gradient":us.tolist()},separators=(",",":"))+"\n",encoding="utf-8")
          row["metrics_file"]=mp.relative_to(STUDY).as_posix()
          row["affine_ols_training_loss"]=floor
          row["affine_floor_loss_ratio"]=floor/row["initial_train_loss"]
          cp=STUDY/"results"/arch["name"]/f"seed_{seed}"/f"{method}.json"; cp.parent.mkdir(parents=True,exist_ok=True)
          cp.write_text(json.dumps(curve,separators=(",",":"))+"\n",encoding="utf-8"); row["curve_file"]=cp.relative_to(STUDY).as_posix(); row["_u"]=us; records.append(row)
          print(f"finished {arch['name']} seed={seed} optimizer={method}",flush=True)
    lookup={(r["architecture"],r["seed"],r["optimizer"]):r for r in records}; cells=[]
    for arch in ARCHS:
      for seed in SEEDS:
        ref=lookup[(arch["name"],seed,"SGD")]
        for q in Q_VALUES:
          tref=ref["train_endpoints"][str(q)]["first"]; utarget=float(ref["_u"][tref]) if tref is not None else None
          for method in METHODS:
            row=lookup[(arch["name"],seed,method)]; tobs=row["train_endpoints"][str(q)]["first"]
            if method=="SGD": tpred=tref
            elif tref is None: tpred=None
            else:
              crossings=np.flatnonzero(row["_u"]>=utarget); tpred=int(crossings[0]) if crossings.size else None
            err=abs(tpred-tobs)/tobs if tpred is not None and tobs is not None and tobs>0 else None
            cells.append({"architecture":arch["name"],"width":arch["width"],"hidden_layers":arch["hidden_layers"],"seed":seed,"q":q,"optimizer":method,
              "sgd_reference_step":tref,"U_sgd_target":utarget,"predicted_step":tpred,"observed_first_step":tobs,
              "observed_sustained_step":row["train_endpoints"][str(q)]["sustained_to_T"],"relative_absolute_error":err,
              "class_unattainable_threshold":q<row["affine_floor_loss_ratio"],
              "within_5pct":err<=.05 if err is not None else None,"censored":tobs is None or tpred is None})
    grouped={}
    for m in METHODS:
      grouped[m]={}
      for a in ARCHS:
        grouped[m][a["name"]]={}
        for q in Q_VALUES:
          sel=[c for c in cells if c["optimizer"]==m and c["architecture"]==a["name"] and c["q"]==q]
          errs=[c["relative_absolute_error"] for c in sel if c["relative_absolute_error"] is not None]
          grouped[m][a["name"]][str(q)]={"cells":len(sel),"uncensored_comparisons":len(errs),"censored_comparisons":len(sel)-len(errs),
            "within_5pct":sum(e<=.05 for e in errs),"median_relative_error":statistics.median(errs) if errs else None}
    for r in records:r.pop("_u",None)
    outdata={"study":pre["study"],"domain":pre["domain"],"preregistration_commit":commit,"preregistration_sha256":sha(pp),
      "execution_source_sha256":sha(Path(__file__)),"torch_version":torch.__version__,"numpy_version":np.__version__,
      "counts":{"architectures":len(ARCHS),"seeds_per_architecture":len(SEEDS),"optimizers":len(METHODS),"training_cells":len(records),"paired_endpoint_cells":len(cells)},
      "conditions":{"eta":ETA,"T":T_MAX,"q":Q_VALUES,"architectures":ARCHS,"seeds":SEEDS,"optimizers":METHODS},
      "primary_by_optimizer_architecture_q":grouped,"endpoint_cells":cells,"cell_records":records}
    outdata["pairing_and_affine_floor"]=pairing
    (STUDY/"summary.json").write_text(json.dumps(outdata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"summary":str(STUDY/"summary.json"),"counts":outdata["counts"],"primary_q05":{m:{a["name"]:grouped[m][a["name"]]["0.5"] for a in ARCHS} for m in METHODS}},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seal",action="store_true"); ap.add_argument("--execute",action="store_true"); ap.add_argument("--prereg-commit")
    args=ap.parse_args()
    if args.seal:seal()
    elif args.execute:
      if not args.prereg_commit:ap.error("--execute requires --prereg-commit")
      execute(args.prereg_commit)
    else:ap.error("select --seal or --execute")
