"""New-function OOD measurement with sealed conditions and atomic seed cells."""
import fcntl
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
import experiment
experiment.ROOT = ROOT
from experiment import run_cell, save, sha, tensor_pin


def dataset(name, prereg):
    path = STUDY / "datasets" / (name + ".pt")
    if path.exists():
        return torch.load(path, weights_only=True)
    dim = 12 if name.endswith("12") else 8
    generator = torch.Generator().manual_seed(prereg["data_seed"] + list(prereg["functions"]).index(name))
    n = prereg["train_n"] + prereg["test_n"]
    x = torch.randn(n, dim, generator=generator) if dim == 12 else 2 * torch.rand(n, dim, generator=generator) - 1
    if name == "trigonometric8":
        y = torch.sin(2 * x[:, 0]) + .7 * torch.cos(3 * x[:, 1]) + .3 * x[:, 2] * x[:, 3]
    elif name == "quadratic8":
        y = x[:, 0] ** 2 + .5 * x[:, 1] ** 2 - .6 * x[:, 2] ** 2 + .2 * x[:, 3]
    elif name == "interaction12":
        y = torch.sin(x[:, 0] * x[:, 1]) + .5 * torch.tanh(x[:, 2] + x[:, 3]) + .2 * x[:, 4] * x[:, 5]
    elif name == "radial12":
        y = torch.exp(-.25 * x[:, :6].square().sum(1)) + .1 * x[:, 6]
    else:
        raise ValueError(name)
    ntrain = prereg["train_n"]
    center, scale = y[:ntrain].double().mean(), y[:ntrain].double().std(unbiased=False)
    y = ((y.double() - center) / scale).float().unsqueeze(1)
    data = {"train_x": x[:ntrain], "test_x": x[ntrain:], "train_y": y[:ntrain], "test_y": y[ntrain:]}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    save(path.with_suffix(".json"), {"generator": name, "data_seed": prereg["data_seed"],
                                    "train_center": float(center), "train_scale": float(scale),
                                    "sha256": sha(path), "tensors": {k: tensor_pin(v) for k,v in data.items()}})
    return data


def main():
    torch.set_num_threads(1)
    prereg = json.loads((STUDY / "preregistration.json").read_text())
    seal = json.loads((STUDY / "seal.json").read_text())
    assert sha(STUDY / "preregistration.json") == seal["preregistration_sha256"]
    assert sha(STUDY / "executed/host/run.py") == seal["run_source_sha256"]
    assert sha(STUDY / "executed/host/experiment.py") == seal["executor_sha256"]
    with (STUDY / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = subprocess.check_output(["ps","-p",str(os.getpid()),"-o","lstart=","-o","command="],text=True).strip()
        complete, reused, t0 = 0,0,time.monotonic()
        total = len(prereg["functions"])*len(prereg["recipes"])*len(prereg["means"])*len(prereg["interventions"])*len(prereg["optimizers"])*len(prereg["seeds"])
        for function in prereg["functions"]:
            data = dataset(function, prereg)
            for label, base in prereg["recipes"].items():
                for mean in prereg["means"]:
                    tensors = data["train_x"], data["train_y"]+mean, data["test_x"],data["test_y"]+mean
                    for mode in prereg["interventions"]:
                        for optimizer, lr in prereg["optimizers"].items():
                            recipe = {**base, "input_dim": data["train_x"].shape[1], "optimizer": optimizer, "lr":lr}
                            for seed in prereg["seeds"]:
                                cell={"id":f"{function}_{label}_{mean}_{mode}_{optimizer}_{seed}",
                                      "dataset":function,"recipe_label":label,"mean":mean,"intervention":mode,
                                      "seed":seed,"recipe":recipe,"preregistration_sha256":seal["preregistration_sha256"]}
                                row, cache = run_cell(STUDY,cell,tensors)
                                complete +=1;reused +=cache
                                state={"status":"running","completed":complete,"total":total,
                                       "reused_this_invocation":reused,"pid":os.getpid(),"identity":identity,
                                       "last_cell":cell["id"],"seconds":time.monotonic()-t0,
                                       "updated_at":time.time(),"direction":"A","origin_iteration":2,
                                       "sealed_commit":seal["commit"]}
                                save(STUDY/"current.json",state)
        state["status"]="measurements_complete"
        save(STUDY/"current.json",state)
        print(json.dumps(state),flush=True)


if __name__ == "__main__":
    main()
