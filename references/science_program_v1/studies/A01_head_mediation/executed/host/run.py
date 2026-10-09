"""Execute preregistered head-freezing contrasts, resuming completed seed cells."""
import argparse
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
sys.path.insert(0, str(ROOT))
from experiment import digest, run_cell, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-new", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prereg = json.loads((STUDY / "preregistration.json").read_text())
    prereg_pin = sha(STUDY / "preregistration.json")
    with (STUDY / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "lstart=", "-o", "command="], text=True).strip()
        rows, new, t0 = [], 0, time.monotonic()
        total = len(prereg["datasets"]) * len(prereg["means"]) * len(prereg["interventions"]) * len(prereg["optimizers"]) * len(prereg["seeds"])
        for dataset in prereg["datasets"]:
            data = ROOT / "sources/e16/datasets" / dataset
            train, test = [torch.load(data / name, weights_only=True) for name in ["train.pt", "test.pt"]]
            for mean in prereg["means"]:
                shift = float(mean) - float(train["y"].double().mean())
                tensors = train["x"], train["y"] + shift, test["x"], test["y"] + shift
                for intervention in prereg["interventions"]:
                    for optimizer, lr in prereg["optimizers"].items():
                        recipe = {**prereg["recipe"], "optimizer": optimizer, "lr": lr}
                        for seed in prereg["seeds"]:
                            cell = {"id": f"{dataset}_{mean}_{intervention}_{optimizer}_{seed}",
                                    "dataset": dataset, "mean": mean, "seed": seed,
                                    "intervention": intervention, "recipe": recipe,
                                    "preregistration_sha256": prereg_pin}
                            row, reused = run_cell(STUDY, cell, tensors)
                            rows.append(row)
                            new += not reused
                            state = {"status": "running", "pid": os.getpid(), "identity": identity,
                                     "completed": len(rows), "total": total, "new_this_invocation": new,
                                     "reused_this_invocation": len(rows) - new, "last_cell": cell["id"],
                                     "seconds": time.monotonic() - t0, "updated_at": time.time(),
                                     "origin_iteration": prereg["iteration"], "direction": "A"}
                            save(STUDY / "current.json", state)
                            if args.limit_new and new >= args.limit_new:
                                state["status"] = "checkpointed"
                                save(STUDY / "current.json", state)
                                print(json.dumps(state), flush=True)
                                return
        state["status"] = "measurements_complete"
        state["failed_runs"] = sum(r["failed"] for r in rows)
        save(STUDY / "current.json", state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
