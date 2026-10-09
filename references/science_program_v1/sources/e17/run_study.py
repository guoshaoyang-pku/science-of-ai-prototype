#!/usr/bin/env python3
"""Explicit local intervention executable: bias shift or frozen hidden features."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
RUN = REPO.parents[2]
ROOT = next(p for p in REPO.parents if (p / "aiq_rl/tools/kb_jobs.py").is_file())
BASE = RUN / "jobs/Jfc41e23dc91c/repo"
sys.path.insert(0, str(ROOT / "aiq_rl/tools"))
from kb_jobs import update_job, worker_lock


def save(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


@contextlib.contextmanager
def executable(dataset, optimizer, intervention):
    original = json.loads((BASE / "results" / f"{dataset}_c2.json").read_text())["measurement"]["results"]
    measurement = next(r for r in original if r["candidate"]["optimizer"]["type"] == optimizer)
    manifest = json.loads((BASE / "evidence_manifest.json").read_text())["files"]
    directory = REPO / "executed" / optimizer / intervention
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("model.py", "loss.py", "optimizer.py", "train.py"):
        pin = measurement["measurement_files"]["executed/" + name]
        raw = (BASE / manifest[pin["sha256"]]["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        text = raw.decode()
        if name == "train.py":
            marker = "    model = Model().to(run_device)\n"
            assert text.count(marker) == 1
            insert = ("    head = model.net[-1]\n"
                      "    if process[\"intervention\"] == \"matched_bias\":\n"
                      "        with torch.no_grad():\n"
                      "            head.bias.add_(process[\"mean\"])\n"
                      "    elif process[\"intervention\"] == \"frozen_hidden\":\n"
                      "        for name, parameter in model.named_parameters():\n"
                      "            parameter.requires_grad_(name.startswith(f\"net.{len(model.net)-1}.\"))\n"
                      "    else:\n"
                      "        raise ValueError(\"unknown intervention\")\n")
            text = text.replace(marker, marker + insert)
            text += "\nINTERVENTION_IMPLEMENTATION = \"e17_output_bias_or_frozen_hidden_v1\"\n"
        path = directory / name
        if path.exists() and path.read_text() != text:
            raise ValueError("changed intervention source during continuation")
        path.write_text(text)
    old = {name: sys.modules.pop(name, None) for name in ("model", "loss", "optimizer")}
    sys.path.insert(0, str(directory))
    try:
        modspec = importlib.util.spec_from_file_location("e17_intervention_train", directory / "train.py")
        module = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(module)
        yield module, measurement, directory
    finally:
        sys.path.remove(str(directory))
        for name, previous in old.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


def main():
    torch.set_num_threads(1)
    prereg = json.loads((REPO / "preregistration.json").read_text())
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "datasets").mkdir(exist_ok=True)
    with worker_lock(REPO.parent / "job.json") as job:
        t0, rows, pins = time.time(), [], {}
        for dataset in prereg["datasets"]:
            directory = REPO / "datasets" / dataset
            directory.mkdir(exist_ok=True)
            for source in (BASE / "datasets" / dataset).iterdir():
                shutil.copy2(source, directory / source.name)
            train, test = [torch.load(directory / name, weights_only=True) for name in ("train.pt", "test.pt")]
            for optimizer in prereg["optimizers"]:
                for condition in prereg["conditions"]:
                    intervention = condition["id"]
                    with executable(dataset, optimizer, intervention) as (module, original, source_dir):
                        for path in source_dir.iterdir():
                            if path.is_file():
                                raw = path.read_bytes()
                                pins[str(path.relative_to(REPO))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                        for mean in prereg["means"]:
                            shift = float(mean) - float(train["y"].double().mean())
                            ty, vy = train["y"] + shift, test["y"] + shift
                            for seed in prereg["seeds"]:
                                label = f"{dataset}_{optimizer}_{intervention}_{mean}_{seed}"
                                path = REPO / "results" / (label + ".json")
                                if path.exists():
                                    row = json.loads(path.read_text())
                                else:
                                    process = {"version": "regression_mse_v1", "steps": [1,8,32,128,256],
                                               "max_eval_samples": 256, "intervention": intervention, "mean": float(mean)}
                                    result = module.train_and_eval(train["x"], ty, test["x"], vy,
                                        steps=256,batch_size=64,seed=seed,fail_threshold=1e12,process=process)
                                    assert not result["failed"]
                                    diagnostics = result["process_diagnostics"]
                                    record = diagnostics["record"]
                                    arrays = {k:v.detach().cpu().numpy() for k,v in diagnostics["evaluation"].items()}
                                    np.savez_compressed(path.with_suffix(".npz"),**arrays)
                                    curve = np.asarray(result["step_metrics"])
                                    assert curve.shape == (256,) and np.isfinite(curve).all()
                                    row = {"dataset":dataset,"optimizer":optimizer,"intervention":intervention,
                                           "mean":mean,"seed":seed,"applied_label_shift":shift,
                                           "test_mse":result["final_test_mse"],"curve":curve.tolist(),
                                           "process":record,"source_files":{k:v for k,v in pins.items() if k.startswith(str(source_dir.relative_to(REPO)))}}
                                    save(path,row)
                                assert row["process"]["status"] == "completed"
                                if intervention == "frozen_hidden":
                                    assert all(p["displacement_norm"] == 0 for step in row["process"]["steps"]
                                               for p in step["parameters"] if p["role"] == "hidden")
                                rows.append(row)
                            state={"status":"running","job":job["id"],"completed_seed_runs":len(rows),"total_seed_runs":360,
                                   "last_cell":label,"updated_at":time.time()}
                            save(REPO/"current.json",state)
                            print(json.dumps(state),flush=True)
        save(REPO/"source_manifest.json",pins)
        state.update(status="measurements_complete",seconds=time.time()-t0)
        save(REPO/"current.json",state)
        update_job(REPO.parent/"job.json",status="measured",measurements=state,measurements_completed_at=time.time())
        print(json.dumps(state),flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update_job(REPO.parent/"job.json",status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
