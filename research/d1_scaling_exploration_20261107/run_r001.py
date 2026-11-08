"""Preregister and run controlled D1 geometry scans. Never touches old runs."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from quadratic_engine import simulate

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DATA = Path(os.environ.get("AIQ_KB_DATA_ROOT", str(REPO.parent / "AIQ_KB_DATA"))).resolve()
RUN = DATA / "runs/d1_scaling_exploration_20261107/r001_geometry"
METHODS = ["SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad"]
SEEDS = [7101, 7102, 7201, 7202]
STEPS = 512
REFERENCE_STEPS = 4096
RADII = [.25, 1., 4.]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cases():
    out = []
    for dim, kappa, basis, alpha, amplitude, scale, seed in itertools.product(
            [2, 8], [1., 10., 100.], ["axis", "haar"], [.1, .5, .9],
            [.1, 1., 10.], [.01, 1., 100.], SEEDS):
        out.append(dict(cohort="core", dim=dim, kappa=kappa, basis=basis,
                        alpha=alpha, amplitude=amplitude, objective_scale=scale, seed=seed,
                        intervention="baseline", level=0.))
    factors = {
        "mu": [0., .5, .99], "beta1": [0., .5, .99],
        "beta2": [.9, .99, .9999], "alpha_rms": [0., .9, .999],
        "eps_ratio": [0., 1e-3, .1, 1., 100.],
        "adagrad_init_ratio": [.01, 1., 100.],
        "decay_fraction": [.01, .1, .5], "origin_shift": [1., 10., 100.],
        "noise_ratio": [.1, 1.], "batch_size": [1., 8., 64.],
        "noise_cycle": [1., 16.], "loss_floor_ratio": [1., 10.],
    }
    for factor, levels in factors.items():
        for level, seed in itertools.product(levels, SEEDS):
            out.append(dict(cohort="stress",dim=8,kappa=10.,basis="haar",alpha=.5,
                            amplitude=1.,objective_scale=1.,seed=seed,
                            intervention=factor,level=level))
    for i, c in enumerate(out):
        c["case_id"] = f"c{i:05d}"
        c["split"] = "calibration" if c["seed"] in SEEDS[:2] else "confirmation"
        c["alpha_effective"] = float(c["alpha"] + np.random.default_rng(c["seed"]+600000+c["dim"]).uniform(-.025,.025))
    return out


def geometry(case):
    d, seed = case["dim"], case["seed"]
    rng = np.random.default_rng(seed + d * 10000)
    q = np.eye(d)
    if case["basis"] == "haar":
        q, rr = np.linalg.qr(rng.normal(size=(d,d)))
        q *= np.sign(np.diag(rr))[None,:]
    eigen = np.geomspace(1/case["kappa"],1.,d)
    weights = np.r_[np.full(d//2,case["alpha_effective"]/(d//2)), np.full(d-d//2,(1-case["alpha_effective"])/(d-d//2))]
    signs = np.random.default_rng(seed+99).choice([-1.,1.],size=d)
    e = q @ (np.sqrt(weights/eigen)*signs)
    e *= case["amplitude"]/np.linalg.norm(e)
    shift = case["level"] if case["intervention"] == "origin_shift" else 0.
    optimum = shift*e
    theta0 = optimum+e
    h = case["objective_scale"] * (q @ np.diag(eigen) @ q.T)
    return h,theta0,optimum


def batch_inputs(selected, radius, method):
    matrices = [geometry(c) for c in selected]
    h,theta,optimum = [np.stack([x[i] for x in matrices]) for i in range(3)]
    g = np.einsum("nij,nj->ni",h,theta-optimum)
    n,d=g.shape
    rms=np.sqrt(np.mean(g*g,axis=1))
    parameters = {"mu": np.full(n,.9),"beta1":np.full(n,.9),"beta2":np.full(n,.999),
                  "alpha":np.full(n,.99),"eps":np.full(n,1e-8),
                  "weight_decay":np.zeros(n),"adagrad_init":np.zeros(n)}
    noise_ratio=np.zeros(n); batch=np.ones(n); cycle=np.zeros(n,dtype=int)
    floor=np.zeros(n)
    for i,c in enumerate(selected):
        f,v=c["intervention"],c["level"]
        if f in ("mu","beta1","beta2"):parameters[f][i]=v
        if f=="alpha_rms":parameters["alpha"][i]=v
        if f=="eps_ratio":parameters["eps"][i]=v*rms[i]
        if f=="adagrad_init_ratio":parameters["adagrad_init"][i]=v*rms[i]**2
        if f=="noise_ratio":noise_ratio[i]=v
        if f=="batch_size":noise_ratio[i]=.5;batch[i]=v
        if f=="noise_cycle":noise_ratio[i]=.5;cycle[i]=int(v)
        if f=="loss_floor_ratio":floor[i]=v*.5*np.einsum("i,ij,j",theta[i]-optimum[i],h[i],theta[i]-optimum[i])
    # Freeze g0 to choose the learning rate without using any trained loss.
    if method=="SGD": unit=STEPS*g
    elif method=="Momentum":
        mu=parameters["mu"]
        f=STEPS/(1-mu)-mu*(1-mu**STEPS)/(1-mu)**2
        unit=f[:,None]*g
    elif method in ("Adam","AdamW"):
        unit=STEPS*g/(np.abs(g)+parameters["eps"][:,None])
    else:
        unit=np.zeros_like(g)
        acc=np.broadcast_to(parameters["adagrad_init"][:,None],g.shape).copy()
        v=np.zeros_like(g)
        for _ in range(STEPS):
            if method=="RMSprop":
                a=parameters["alpha"][:,None];v=a*v+(1-a)*g*g;denom=np.sqrt(v)
            else:acc+=g*g;denom=np.sqrt(acc)
            unit+=np.divide(g,denom+parameters["eps"][:,None],out=np.zeros_like(g),where=(denom+parameters["eps"][:,None])!=0)
    eta=radius*np.linalg.norm(theta-optimum,axis=1)/np.linalg.norm(unit,axis=1)
    if method=="AdamW":
        for i,c in enumerate(selected):
            if c["intervention"]=="decay_fraction":
                parameters["weight_decay"][i]=(1-(1-c["level"])**(1/STEPS))/eta[i]
    noise=None
    if np.any(noise_ratio):
        noise=np.zeros((STEPS,n,d))
        for i,c in enumerate(selected):
            if noise_ratio[i]==0:continue
            rng=np.random.default_rng(c["seed"]+200000)
            length=cycle[i] or STEPS
            z=rng.normal(size=(length,d))
            noise[:,i]=z[np.arange(STEPS)%length]*rms[i]*noise_ratio[i]/np.sqrt(batch[i])
    return h,theta,optimum,eta,parameters,noise,floor


def seal():
    design=cases()
    document={
      "study":"r001_geometry","domain":"development_and_preassigned_confirmation_draws",
      "question":"When geometry, optimizer memory, and coordinate representation vary, can a shared initial-displacement clock predict cross-optimizer quadratic progress on held-out initializations?",
      "source_hashes":{p.name:digest(p) for p in [Path(__file__),SOURCE/"quadratic_engine.py"]},
      "design":{"cases":len(design),"core_cases":sum(c['cohort']=='core' for c in design),"stress_cases":sum(c['cohort']=='stress' for c in design),
                "methods":METHODS,"steps":STEPS,"reference_steps":REFERENCE_STEPS,"planned_initial_displacement_radii":RADII,
                "calibration_seeds":SEEDS[:2],"confirmation_seeds":SEEDS[2:],"case_manifest":design},
      "scope":"self-built strictly convex 2D/8D quadratics; eigenvalue spectra have largest eigenvalue 1 before objective rescaling; no final benchmark data",
      "pairing":"paired H and modal energies across coordinate rotations; paired true gradient noise by seed across optimizers; optimizer states start at zero except preregistered Adagrad accumulator interventions",
      "initial_draws":"each nominal slow energy share alpha is jittered by a preassigned seed uniform [-.025,.025] amount, stored as alpha_effective; random orthogonal bases and signs vary by seed; no claim that seeds constitute new tasks",
      "learning_rate_rule":"constant eta chosen before training to match radius*||theta0-theta_star|| frozen-gradient displacement at 512 steps; AdamW calibration uses its gradient component, then total U includes registered decay; reference SGD uses radius=1 eta for 4096 steps",
      "noise_model":"optional additive Gaussian gradient noise with fixed initial-coordinate rms scale divided by sqrt(batch_size); fresh IID or explicit repeated cycle; a synthetic noise intervention, not neural minibatch evidence",
      "clocks":{"norm":"tau=U_const/||g0||","projection":"tau=-g0 dot delta_const/||g0||^2","relative_displacement":"d=U_const/||theta0||",
                "calibrated_linear":"tau=A*d, A>=0","calibrated_quadratic":"tau=A*d+B*d^2, A,B>=0"},
      "evaluation":"same-task training seeds fit scalar clocks from equal-loss SGD first passages; freeze before scoring confirmation draws. Report curve MAE and max error normalized by initial excess loss, error relative to actual progress, full-reference coverage, first/sustained endpoints at excess-loss ratios .9/.5/.1/.01, and initial-loss offset sensitivity. Never impute missing endpoints or clip predictions into reference range.",
      "calibration_groups":"group by dim,kappa,basis,alpha,amplitude,objective_scale; share coefficients across Adam/RMSprop/Adagrad and radii. SGD reference step converted to continuous SGD tau=eta_ref*step to remove reference-step units. Stress cases use baseline core group's coefficients without refitting.",
      "predictions":[
        {"id":"P1","type":"analytic_numeric_check","claim":"For frozen nonzero gradients with eps=0 and zero states, Adam/RMSprop/Adagrad directions lie on the sign(g0) ray, and matched displacement produces equal frozen parameters."},
        {"id":"P2","type":"falsifiable","claim":"A projection clock has smaller confirmation MAE than the length clock on more than half of non-SGD core paths with |loss decrease| >=.1 L0 and at least 80% reference coverage for both."},
        {"id":"P3","type":"falsifiable","claim":"The shared nonnegative quadratic calibration has median confirmation MAE <=.01 initial excess loss and >=90% complete reference coverage across all core adaptive paths, while retaining max errors and progress-normalized errors."},
        {"id":"P4","type":"falsifiable","claim":"At least one registered memory or origin intervention changes the true confirmation loss trajectory despite an unchanged frozen length clock or unchanged physical objective; such failures bound universal scalar claims."}],
      "analysis_limits":"geometric identities and coordinate-dependence are not new discoveries; initial norm/slope/curvature are training-before quantities, recorded future gradient drift is diagnostic only; no pooled causal claim from a multivariate correlation",
      "failures":"a nonfinite state or excess loss exceeding 1e12*max(initial excess loss,1e-30) stops that path, retaining earlier values and NaN afterward; constant loss offsets do not change the divergence threshold",
      "compute":"single-thread local CPU, NumPy float64; all outcomes and failures saved externally; no supervisor, paid model, solver, network training or old-run reruns"}
    (SOURCE/"r001_preregistration.json").write_text(json.dumps(document,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"cases":len(design),"training_paths":len(design)*len(METHODS)*len(RADII),"reference_paths":len(design),"hashes":document['source_hashes']}))


def execute(commit):
    pre_path=SOURCE/"r001_preregistration.json"
    pre=json.loads(pre_path.read_text(encoding="utf-8"))
    resolved=subprocess.check_output(["git","-C",str(REPO),"rev-parse",commit],text=True).strip()
    for name,value in pre["source_hashes"].items():
        assert digest(SOURCE/name)==value
        rel=(SOURCE/name).relative_to(REPO).as_posix()
        committed=subprocess.check_output(["git","-C",str(REPO),"show",f"{resolved}:{rel}"])
        assert hashlib.sha256(committed).hexdigest()==value
    rel=pre_path.relative_to(REPO).as_posix()
    assert subprocess.check_output(["git","-C",str(REPO),"show",f"{resolved}:{rel}"])==pre_path.read_bytes()
    assert not RUN.is_relative_to(REPO)
    RUN.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),SOURCE/"quadratic_engine.py",pre_path]:shutil.copy2(p,RUN/p.name)
    start=time.perf_counter();manifest=[]
    all_cases=pre["design"]["case_manifest"]
    (RUN/"case_manifest.json").write_text(json.dumps(all_cases,indent=2))
    for dim in [2,8]:
        selected=[c for c in all_cases if c["dim"]==dim]
        h,theta,optimum,eta,params,_,floor=batch_inputs(selected,1.,"SGD")
        # The reference remains deterministic and unregularized for every stress case.
        reference=simulate(h,theta,optimum,eta,"SGD",params,REFERENCE_STEPS,loss_floor=floor)
        reference.update(eta=eta,theta0=theta,optimum=optimum,H=h,case_ids=np.array([c["case_id"] for c in selected]))
        dest=RUN/f"reference_d{dim}.npz";np.savez_compressed(dest,**reference)
        manifest.append({"file":dest.name,"sha256":digest(dest),"paths":len(selected),"kind":"reference"})
        print(f"reference dim={dim} paths={len(selected)}",flush=True)
        del reference
        for radius in RADII:
            for method in METHODS:
                h,theta,optimum,eta,params,noise,floor=batch_inputs(selected,radius,method)
                result=simulate(h,theta,optimum,eta,method,params,STEPS,noise=noise,loss_floor=floor)
                result.update(eta=eta,theta0=theta,optimum=optimum,H=h,loss_floor=floor,
                              case_ids=np.array([c["case_id"] for c in selected]))
                dest=RUN/f"paths_d{dim}_r{radius:g}_{method}.npz";np.savez_compressed(dest,**result)
                manifest.append({"file":dest.name,"sha256":digest(dest),"paths":len(selected),"kind":"candidate","method":method,"radius":radius})
                print(f"dim={dim} radius={radius:g} method={method} paths={len(selected)}",flush=True)
    receipt={"preregistration_commit":resolved,"preregistration_sha256":digest(pre_path),"elapsed_seconds":time.perf_counter()-start,
             "candidate_paths":len(all_cases)*len(RADII)*len(METHODS),"reference_paths":len(all_cases),"artifacts":manifest}
    (RUN/"execution_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seal",action="store_true");ap.add_argument("--execute",action="store_true");ap.add_argument("--prereg-commit")
    args=ap.parse_args()
    if args.seal:seal()
    elif args.execute and args.prereg_commit:execute(args.prereg_commit)
    else:ap.error("select --seal or --execute --prereg-commit HASH")
