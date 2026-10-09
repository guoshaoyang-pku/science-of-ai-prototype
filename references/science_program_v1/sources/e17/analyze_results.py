#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
RUN = REPO.parents[2]
BASE = RUN / "jobs/Jfc41e23dc91c/repo"
BASE_RESULTS = REPO / "base_results" if (REPO / "base_results").exists() else BASE / "results"


def interval(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (10,) and np.isfinite(a).all()
    mean, se = float(a.mean()), float(a.std(ddof=1)/np.sqrt(10))
    return {"mean":mean, "se":se, "ci95":[mean-2.262157*se,mean+2.262157*se]}


def main():
    prereg = json.loads((REPO / "preregistration.json").read_text())
    base_analysis = json.loads((BASE / "analysis.json").read_text()) if not (REPO/"e16_analysis.json").exists() else json.loads((REPO/"e16_analysis.json").read_text())
    portable_base = (REPO / "base_manifest.json").exists()
    base_manifest = json.loads((REPO/"base_manifest.json" if portable_base else BASE/"evidence_manifest.json").read_text())["files"]
    base_root = REPO if portable_base else BASE
    sources = json.loads((REPO/"source_manifest.json").read_text())
    for name,pin in sources.items():
        raw=(REPO/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin["sha256"]
    rows=[];groups={};arrays={}
    for path in sorted((REPO/"results").glob("*.json")):
        row=json.loads(path.read_text());record=row["process"];ds=row["dataset"];seed=row["seed"]
        original=json.loads((BASE_RESULTS/f"{ds}_c2.json").read_text())["measurement"]["results"]
        measurement=next(r for r in original if r["candidate"]["optimizer"]["type"]==row["optimizer"])
        pin=measurement["measurement_files"][f"results/process/seed_{seed}.json"]
        original_record=json.loads((base_root/base_manifest[pin["sha256"]]["path"]).read_text())
        assert record["minibatch_stream_sha256"]==original_record["minibatch_stream_sha256"]
        assert record["inputs"]["train_x"]==original_record["inputs"]["train_x"]
        assert record["inputs"]["test_x"]==original_record["inputs"]["test_x"]
        init=record["initial_parameters"];base_init=original_record["initial_parameters"]
        bias_name=list(init)[-1]
        assert all(init[n]==base_init[n] for n in init if row["intervention"]!="matched_bias" or n!=bias_name)
        with np.load(path.with_suffix(".npz")) as a:
            residual=a["predictions"].reshape(-1).astype(float)-a["targets"].reshape(-1).astype(float)
            mse=float(np.mean(residual**2));offset=float(np.mean(residual)**2);centered=mse-offset
            assert np.isclose(mse,row["test_mse"],rtol=2e-7,atol=1e-7)
            expected_shift=row["mean"] if row["intervention"]=="matched_bias" else 0
            pin=measurement["measurement_files"][f"results/process/seed_{seed}.npz"]
            with np.load(base_root/base_manifest[pin["sha256"]]["path"]) as original_arrays:
                assert np.allclose(a["initial_predictions"],original_arrays["initial_predictions"]+expected_shift,atol=5e-7)
            arrays[(ds,row["optimizer"],row["intervention"],row["mean"],seed)]=a["predictions"].copy()
        if row["intervention"]=="frozen_hidden":
            assert all(p["displacement_norm"]==0 for step in record["steps"] for p in step["parameters"] if p["role"]=="hidden")
        short={k:row[k] for k in ("dataset","optimizer","intervention","mean","seed")}
        short.update(test_mse=mse,offset_mse=offset,centered_mse=centered)
        rows.append(short);groups.setdefault((ds,row["intervention"],row["mean"],row["optimizer"]),{})[seed]=short
    assert len(rows)==360
    conditions=[];convexity=[]
    for ds in prereg["datasets"]:
        for intervention in ("matched_bias","frozen_hidden"):
            for mean in prereg["means"]:
                sg=[groups[(ds,intervention,mean,"SGD")][seed] for seed in range(10)]
                ad=[groups[(ds,intervention,mean,"Adam")][seed] for seed in range(10)]
                gap=[s["test_mse"]-a["test_mse"] for s,a in zip(sg,ad)]
                conditions.append({"dataset":ds,"intervention":intervention,"mean":mean,
                                   "SGD":float(np.mean([s["test_mse"] for s in sg])),
                                   "Adam":float(np.mean([a["test_mse"] for a in ad])),
                                   "delta":interval(gap),
                                   "sgd_seed_wins":int(np.sum(np.array(gap)<0)),
                                   "centered_delta":interval([s["centered_mse"]-a["centered_mse"] for s,a in zip(sg,ad)]),
                                   "max_absolute_offset_effect_vs_centered":{o:max(abs(groups[(ds,intervention,mean,o)][seed]["test_mse"]-groups[(ds,intervention,0,o)][seed]["test_mse"]) for seed in range(10)) for o in ("SGD","Adam")}})
            for optimizer in ("SGD","Adam"):
                values=[(groups[(ds,intervention,-3,optimizer)][seed]["test_mse"]+groups[(ds,intervention,3,optimizer)][seed]["test_mse"])/2-groups[(ds,intervention,0,optimizer)][seed]["test_mse"] for seed in range(10)]
                affine=[float(np.max(np.abs(arrays[(ds,optimizer,intervention,-3,seed)]+arrays[(ds,optimizer,intervention,3,seed)]-2*arrays[(ds,optimizer,intervention,0,seed)]))) for seed in range(10)]
                convexity.append({"dataset":ds,"intervention":intervention,"optimizer":optimizer,
                                  "chord_gap":interval(values),
                                  "negative_seeds_below_1e-6":int(np.sum(np.array(values)<-1e-6)),
                                  "max_affine_output_second_difference":max(affine)})
    result={"origin_epoch":17,"seed_runs":len(rows),"conditions":conditions,"convexity":convexity,
            "paired_inputs_and_streams_verified":True,"frozen_hidden_displacement_zero":True,
            "initial_output_shift_verified":True,"source_manifest_verified":True}
    (REPO/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    (REPO/"e16_analysis.json").write_text(json.dumps(base_analysis,indent=2)+"\n")
    with (REPO/"seed_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for c in conditions:print(c["dataset"],c["intervention"],c["mean"],c["delta"],c["sgd_seed_wins"],c["max_absolute_offset_effect_vs_centered"])
    for c in convexity:
        if c["intervention"]=="frozen_hidden" and c["optimizer"]=="SGD":print("convexity",c)


if __name__=="__main__":main()
