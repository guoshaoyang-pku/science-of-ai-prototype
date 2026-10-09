"""Keep normalization forward fixed while intervening on trainable groups."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[1]
sys.path.insert(0,str(STUDY/"executed/host"))
from experiment import run_cell,save,sha
spec=importlib.util.spec_from_file_location("a02_data",ROOT/"studies/A02_ood_residual/run.py")
data_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(data_module)


def main():
 torch.set_num_threads(1)
 config=json.loads((STUDY/"preregistration.json").read_text());previous=json.loads((ROOT/"studies/A02_ood_residual/preregistration.json").read_text())
 with (STUDY/"worker.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  identity=subprocess.check_output(["ps","-p",str(os.getpid()),"-o","lstart=","-o","command="],text=True).strip()
  complete,reused,t0=0,0,time.monotonic()
  for function in config["functions"]:
   data=data_module.dataset(function,previous)
   for mean in config["means"]:
    tensors=data["train_x"],data["train_y"]+mean,data["test_x"],data["test_y"]+mean
    for mode in config["interventions"]:
     for seed in config["seeds"]:
      recipe={**config["recipe"],"input_dim":data["train_x"].shape[1],"optimizer":"SGD","lr":.001}
      cell={"id":f"{function}_{mean}_{mode}_{seed}","dataset":function,"recipe_label":"LN010_w192","mean":mean,"seed":seed,
            "intervention":mode,"recipe":recipe,"preregistration_sha256":sha(STUDY/"preregistration.json")}
      row,cache=run_cell(STUDY,cell,tensors);complete+=1;reused+=cache
      for step in row["process"]["steps"]:
       for p in step["parameters"]:
        norm=".norm." in p["name"];weight=p["name"].endswith("weight") and not norm
        frozen=p["role"]=="head" or (mode=="head_frozen_ln_fixed" and norm) or (mode=="head_frozen_weights_only" and not weight) or (mode=="head_frozen_affine_only" and weight)
        if frozen:assert not p["active"] and p["displacement_norm"]==0
      state={"status":"running","completed":complete,"total":180,"reused":reused,"pid":os.getpid(),"identity":identity,
             "last_cell":cell["id"],"seconds":time.monotonic()-t0,"updated_at":time.time(),"direction":"A","origin_iteration":4}
      save(STUDY/"current.json",state)
  state["status"]="measurements_complete";save(STUDY/"current.json",state);print(json.dumps(state),flush=True)


if __name__=="__main__":main()
