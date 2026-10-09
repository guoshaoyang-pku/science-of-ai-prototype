from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t
import torch
from torch import nn

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return {"mean": mean, "seed_95pct_t_interval": [mean-radius, mean+radius] if radius is not None else None}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    spec = importlib.util.spec_from_file_location("r023_runner", STUDY / "executed/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    commit = runner.committed_contract(config)
    original = REPO / config["reference_study"]
    old_config = json.loads((original / "preregistration.json").read_text())
    torch.set_num_threads(1)
    rows, arrays_by_cell, verified = [], {}, []
    maxima = {}

    def close(actual, expected, label):
        if np.shape(actual) != np.shape(expected):
            raise RuntimeError(f"Shape differs: {label}")
        error = float(np.max(np.abs(actual-expected)))
        maxima[label] = max(maxima.get(label, 0.0), error)
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11):
            raise RuntimeError(f"Evidence recomputation differs: {label}")

    groups = [(original, old_config, [0.0, 1.0], "old"), (STUDY, config, [1.0], "new")]
    for source_study, plan, sigmas, draw in groups:
        expected_names = {f"{fn}_w{w}_s{s}_sigma{sigma:g}" for fn in plan["functions"]
                          for w in plan["widths"] for s in plan["seeds"] for sigma in sigmas}
        paths = sorted((source_study / "results").glob("*.json"))
        for path in paths:
            if path.name.endswith(".failure.json"):
                continue
            if path.stem not in expected_names:
                raise RuntimeError(f"Unexpected result: {path}")
            meta = json.loads(path.read_text())
            request = meta["request"]
            fn, width, seed, sigma = (request[n] for n in ["function", "width", "seed", "sigma"])
            data = runner.make_data(plan, fn, sigma)
            expected_request = {"function": fn, "width": width, "seed": seed, "sigma": sigma,
                                "preregistration_sha256": sha(source_study / "preregistration.json"),
                                "source_sha256": plan["source_sha256"],
                                "inputs_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n,a in data.items()}}
            if (meta["status"] != "success" or meta["cell_id"] != path.stem
                    or request != expected_request or meta["contract_sha256"] != runner.digest(request)
                    or meta["arrays_sha256"] != sha(path.with_suffix(".npz"))):
                raise RuntimeError(f"Hash/contract differs: {path}")
            for name in ["preregistration.json", *plan["source_sha256"]]:
                blob = subprocess.check_output(["git", "show", f"{meta['preregistration_commit']}:{(source_study/name).relative_to(REPO)}"], cwd=REPO)
                if blob != (source_study/name).read_bytes():
                    raise RuntimeError(f"Saved commit differs: {path}")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                a = {n:z[n].copy() for n in z.files}
            if not all(np.isfinite(v).all() for v in a.values()):
                raise RuntimeError("Nonfinite evidence")
            for name, expected in data.items():
                if not np.array_equal(a[name], expected):
                    raise RuntimeError(f"Data not exactly reproduced: {name}")
            if a["checkpoints"].tolist() != plan["checkpoints"]:
                raise RuntimeError("Checkpoints differ")
            train_loss = np.mean((a["train_outputs"]-a["train_y"])**2, axis=2)/2
            clean_loss = np.mean((a["train_outputs"]-a["clean_train_y"])**2, axis=2)/2
            test_loss = np.mean((a["test_outputs"]-a["test_y"])**2, axis=2)/2
            close(train_loss, a["train_loss"][a["checkpoints"]], "checkpoint_loss")
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(plan["data_contract"]["input_dim"], width), nn.SiLU(),
                                  nn.Linear(width,width), nn.SiLU(), nn.Linear(width,1)).double()
            tx, vx = torch.from_numpy(a["train_x"]), torch.from_numpy(a["test_x"])
            close(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy(), a["initial_parameters"], "initial_parameters")
            close(runner.jacobian(model,tx).numpy(), a["initial_jacobian_train"], "initial_jacobian_train")
            close(runner.jacobian(model,vx).numpy(), a["initial_jacobian_test"], "initial_jacobian_test")
            with torch.no_grad():
                close(model(tx).reshape(-1).numpy(), a["train_outputs"][0,0], "initial_real_train")
                close(model(vx).reshape(-1).numpy(), a["test_outputs"][0,0], "initial_real_test")
                offset = 0
                for p in model.parameters():
                    n = p.numel()
                    p.copy_(torch.from_numpy(a["final_parameters"][offset:offset+n]).reshape_as(p))
                    offset += n
                if offset != len(a["final_parameters"]):
                    raise RuntimeError("Final parameter shape differs")
                close(model(tx).reshape(-1).numpy(), a["train_outputs"][-1,0], "endpoint_real_train")
                close(model(vx).reshape(-1).numpy(), a["test_outputs"][-1,0], "endpoint_real_test")
            close(a["train_outputs"][0,0], a["train_outputs"][0,1], "initial_pair_train")
            close(a["test_outputs"][0,0], a["test_outputs"][0,1], "initial_pair_test")
            j,jt = a["initial_jacobian_train"],a["initial_jacobian_test"]
            for name, matrix in [("train",j),("test",jt)]:
                prediction = a[f"{name}_outputs"][0,1]+np.einsum("ip,p->i",matrix,a["final_tangent_delta"],optimize=False)
                close(prediction,a[f"{name}_outputs"][-1,1],f"endpoint_tangent_{name}")
            kernel = np.einsum("ip,jp->ij",j,j,optimize=False)/len(tx)
            cross = np.einsum("ip,jp->ij",jt,j,optimize=False)/len(tx)
            r = a["train_outputs"][0,1]-a["train_y"]
            test_prediction = a["test_outputs"][0,1].copy()
            train_predictions,test_predictions,losses = [],[],[]
            for step in range(plan["recipe"]["steps"]+1):
                losses.append(float(np.mean(r*r)/2))
                if step in plan["checkpoints"]:
                    train_predictions.append(r+a["train_y"])
                    test_predictions.append(test_prediction.copy())
                if step == plan["recipe"]["steps"]:
                    break
                test_prediction -= plan["recipe"]["lr"]*np.einsum("ij,j->i",cross,r,optimize=False)
                r -= plan["recipe"]["lr"]*np.einsum("ij,j->i",kernel,r,optimize=False)
            close(np.array(train_predictions),a["train_outputs"][:,1],"tangent_train_checkpoints")
            close(np.array(test_predictions),a["test_outputs"][:,1],"tangent_test_checkpoints")
            close(np.array(losses),a["train_loss"][:,1],"tangent_all_train_losses")
            late = plan["checkpoints"].index(plan["secondary"]["late_start_step"])
            row = {"cell_id":path.stem,"draw":draw,"function":fn,"width":width,"seed":seed,"sigma":sigma,
                   "metadata_path":str(path.relative_to(REPO)),"seconds":meta["seconds"],
                   "train_loss":train_loss.tolist(),"clean_train_loss":clean_loss.tolist(),"test_loss":test_loss.tolist(),
                   "train_gain":float(train_loss[-1,1]-train_loss[-1,0]),
                   "test_gap":float(test_loss[-1,0]-test_loss[-1,1]),
                   "late_test_change_real":float(test_loss[-1,0]-test_loss[late,0]),
                   "late_test_change_tangent":float(test_loss[-1,1]-test_loss[late,1])}
            rows.append(row)
            arrays_by_cell[(fn,width,seed,sigma,draw)] = a
            verified.append({"metadata_path":str(path.relative_to(REPO)),"metadata_sha256":sha(path),
                             "arrays_sha256":sha(path.with_suffix(".npz")),"preregistration_commit":meta["preregistration_commit"]})
    selected = [r for r in rows if (r["draw"]=="old" and r["sigma"]==0) or r["draw"]=="new"]
    pairs=[]
    for fn in config["functions"]:
        for width in config["widths"]:
            for seed in config["seeds"]:
                matched = [r for r in selected if (r["function"],r["width"],r["seed"])==(fn,width,seed)]
                if len(matched)!=2:
                    continue
                clean = next(r for r in matched if r["sigma"]==0)
                noisy = next(r for r in matched if r["sigma"]==1)
                old_noisy = next(r for r in rows if (r["function"],r["width"],r["seed"],r["sigma"],r["draw"])==(fn,width,seed,1,"old"))
                a0,a1 = arrays_by_cell[(fn,width,seed,0,"old")],arrays_by_cell[(fn,width,seed,1,"new")]
                for n in ["initial_parameters","initial_jacobian_train","initial_jacobian_test","train_x","test_x","clean_train_y","test_y","target_normalization"]:
                    if not np.array_equal(a0[n],a1[n]):
                        raise RuntimeError(f"Clean/noisy pair not exact: {n}")
                pairs.append({"function":fn,"width":width,"seed":seed,
                              "noise_amplification":noisy["test_gap"]-clean["test_gap"],
                              "old_noise_amplification":old_noisy["test_gap"]-clean["test_gap"],
                              "new_minus_old_D":noisy["test_gap"]-old_noisy["test_gap"],
                              "clean_test_gap":clean["test_gap"],"noisy_test_gap":noisy["test_gap"],
                              "clean_train_gain":clean["train_gain"],"noisy_train_gain":noisy["train_gain"],
                              "noise_risk_change_real":noisy["test_loss"][-1][0]-clean["test_loss"][-1][0],
                              "noise_risk_change_tangent":noisy["test_loss"][-1][1]-clean["test_loss"][-1][1]})
    units=[]
    metrics=["noise_amplification","old_noise_amplification","new_minus_old_D","clean_test_gap","noisy_test_gap",
             "clean_train_gain","noisy_train_gain","noise_risk_change_real","noise_risk_change_tangent"]
    for fn in config["functions"]:
        for width in config["widths"]:
            paired = [p for p in pairs if (p["function"],p["width"])==(fn,width)]
            measured = {m:interval([p[m] for p in paired]) for m in metrics} if paired else {}
            hits = sum(p["noise_amplification"]>0 for p in paired)
            passed = bool(measured["noise_amplification"]["mean"]>=config["primary"]["minimum_mean_amplification"] and hits>=config["primary"]["minimum_positive_seeds"]) if len(paired)==len(config["seeds"]) else None
            units.append({"function":fn,"width":width,"paired_seeds":len(paired),"positive_amplification_seeds":hits,"pass":passed,"metrics":measured})
    late_units=[]
    for fn in config["functions"]:
        for width in config["widths"]:
            for sigma in [0,1]:
                matched=[r for r in selected if (r["function"],r["width"],r["sigma"])==(fn,width,sigma)]
                if not matched:
                    continue
                measures={}
                late=config["checkpoints"].index(128)
                for index,model in enumerate(["real","tangent"]):
                    changes=[r["test_loss"][-1][index]-r["test_loss"][late][index] for r in matched]
                    measures[model]={"late_test_change":interval(changes),"positive_late_test_seeds":sum(v>0 for v in changes),
                                     "late_train_change":interval([r["train_loss"][-1][index]-r["train_loss"][late][index] for r in matched]),
                                     "endpoint_clean_train_loss":interval([r["clean_train_loss"][-1][index] for r in matched])}
                late_units.append({"function":fn,"width":width,"sigma":sigma,"metrics":measures})
    complete=len(selected)==config["planned_analysis_cells"]
    passing=sum(u["pass"] is True for u in units)
    summary={"study":config["study"],"round":23,"direction_round":3,"domain":"development","question":config["question"],
             "status":"measurements_complete" if complete else "partial_measurements",
             "saved_new_cells":sum(r["draw"]=="new" for r in rows),"reused_clean_cells":12,"analysis_cells":len(selected),
             "verified_cells_including_old_noisy":len(verified),"new_optimization_trajectories":2*sum(r["draw"]=="new" for r in rows),
             "planned_recipe_units":4,"complete_noise_pairs":len(pairs),
             "prediction":{"id":"P1_noise_amplification","status":("supported" if passing>=3 else "refuted") if complete else "not_evaluated","passing_units":passing if complete else None},
             "units":units,"paired_seed_results":pairs,"rows":selected,"late_units":late_units,
             "endpoint_gap_positive_but_real_test_decreased":[r for r in selected if r["test_gap"]>0 and r["late_test_change_real"]<0],
             "training_seconds":sum(r["seconds"] for r in rows if r["draw"]=="new"),
             "preregistration_commit":commit,"source_sha256":config["source_sha256"],
             "verification":{"status":"passed","maximum_absolute_errors":maxima,"all_reference_hashes_unchanged":True},
             "boundary":"全部development；固定data7331、两个noise draw，各三init不作跨噪声置信结论；H与C分开；不作机制比例、train-only早停或其他recipe外推。"}
    save(STUDY/"summary.json",summary)
    save(STUDY/"executed/saved_evidence_verification.json",{"status":"passed","cells":verified,"maximum_absolute_errors":maxima,"old_reference_files_checked":len(config["reference_files_sha256"]),"method":"无新训练：hash/finite/精确data及配对；初始参数/J重建、终点real重建、einsum切线512步递推及所有检查点复算"})
    print(json.dumps({"status":summary["status"],"new_cells":summary["saved_new_cells"],"analysis_cells":len(selected),"prediction":summary["prediction"],"maximum_absolute_errors":maxima},ensure_ascii=False))


if __name__=="__main__":
    main()
