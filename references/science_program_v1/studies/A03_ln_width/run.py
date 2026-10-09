"""Orthogonal width/LN development contrast; immutable study executor snapshot."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY / "executed/host"))
from experiment import run_cell, save, sha
spec = importlib.util.spec_from_file_location("a02_data", ROOT / "studies/A02_ood_residual/run.py")
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)


def main():
    torch.set_num_threads(1)
    config = json.loads((STUDY / "preregistration.json").read_text())
    previous = json.loads((ROOT / "studies/A02_ood_residual/preregistration.json").read_text())
    with (STUDY / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = subprocess.check_output(["ps","-p",str(os.getpid()),"-o","lstart=","-o","command="],text=True).strip()
        complete,reused,t0=0,0,time.monotonic()
        total=4*2*3*2*5
        for function in previous["functions"]:
            data=data_module.dataset(function,previous)
            for label,base in config["recipes"].items():
                for mean in [-3,0,3]:
                    tensors=data["train_x"],data["train_y"]+mean,data["test_x"],data["test_y"]+mean
                    for mode in ["baseline","frozen_head"]:
                        for seed in previous["seeds"]:
                            recipe={**base,"input_dim":data["train_x"].shape[1],"optimizer":"SGD","lr":.001}
                            cell={"id":f"{function}_{label}_{mean}_{mode}_SGD_{seed}","dataset":function,"recipe_label":label,
                                  "mean":mean,"intervention":mode,"seed":seed,"recipe":recipe,"preregistration_sha256":sha(STUDY/"preregistration.json")}
                            row,cache=run_cell(STUDY,cell,tensors)
                            complete+=1;reused+=cache
                            state={"status":"running","completed":complete,"total":total,"pid":os.getpid(),"identity":identity,
                                   "reused":reused,"last_cell":cell["id"],"seconds":time.monotonic()-t0,"updated_at":time.time(),
                                   "direction":"A","origin_iteration":3}
                            save(STUDY/"current.json",state)
        state["status"]="measurements_complete";save(STUDY/"current.json",state);print(json.dumps(state),flush=True)


if __name__ == "__main__": main()
