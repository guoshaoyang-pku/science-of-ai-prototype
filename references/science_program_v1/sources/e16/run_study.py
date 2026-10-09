#!/usr/bin/env python3
"""Execute the frozen e16 label-mean intervention; reuse saved measurements."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parent
RUN = REPO.parents[2]
ROOT = next(p for p in REPO.parents if (p / "aiq_rl/tools/kb_lab.py").is_file())
sys.path.insert(0, str(ROOT / "aiq_rl/tools"))
import kb_lab
from kb_jobs import update_job, worker_lock


def write_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def pin(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def main():
    os.environ["ARCHITECTURE_IQ_SEED_WORKERS"] = "1"
    spec = json.loads((REPO / "preregistration.json").read_text())
    preregistration_hash = pin(REPO / "preregistration.json")["sha256"]
    if preregistration_hash != "d5fdb7ffba75f17272a5e3ee1b4552e551163ff4a9bb04be8ffa8449ba9b6acc":
        raise ValueError("preregistration changed after freezing")
    config = json.loads((RUN / "config.json").read_text())
    pool = Path(config["pool"])
    lab = Path(config["lab"]).parent
    candidates = [{"model": copy.deepcopy(spec["recipe"]["model"]),
                   "optimizer": optimizer, "loss": spec["recipe"]["loss"],
                   "budget": spec["recipe"]["budget"]}
                  for optimizer in spec["recipe"]["optimizers"]]
    (REPO / "results").mkdir(exist_ok=True)
    source_files = ["aiq_rl/tools/kb_lab.py", "aiq_bench_repo/profiles/v1.5.yaml",
                    "aiq_bench_repo/src/architecture_iq/candidates/generator.py",
                    "aiq_bench_repo/src/architecture_iq/ground_truth/runner.py"]
    evidence = {"preregistration_sha256": preregistration_hash,
                "sources": {name: pin(ROOT / name) for name in source_files},
                "datasets": {dataset: {name: pin(kb_lab.DATASETS / dataset / name)
                              for name in ("dataset_spec.json", "synthesize.py", "train.pt", "test.pt")}
                             for dataset in spec["datasets"]},
                "pool_sha256": pin(pool)["sha256"], "lab": str(lab)}
    saved_inputs = REPO / "inputs.json"
    if saved_inputs.exists() and json.loads(saved_inputs.read_text()) != evidence:
        raise ValueError("saved study inputs differ; refusing mixed-source continuation")
    write_json(saved_inputs, evidence)
    with worker_lock(REPO.parent / "job.json") as job:
        started = time.time()
        completed = []
        for dataset in spec["datasets"]:
            for condition_number, transform in enumerate(spec["conditions"]):
                key = dataset.split("/")[-1] + "_c" + str(condition_number)
                path = REPO / "results" / (key + ".json")
                request = {"dataset": dataset, "candidates": candidates,
                           "capture_executable_sources": True, "target_transform": transform,
                           "process": spec["process"],
                           "diagnostic_fail_threshold": spec["diagnostic_fail_threshold"]}
                write_json(REPO / "current.json", {"job": job["id"], "status": "running",
                           "active": key, "completed_requests": len(completed),
                           "total_requests": 12, "updated_at": time.time()})
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous["request"] != request:
                        raise ValueError("saved request changed")
                t0 = time.time()
                measured = kb_lab.run_job(request, workers=2, timeout=1800, lab_dir=lab,
                                          require_allowlist=True, pool=pool)
                result = {"request": request, "measurement": measured, "seconds": time.time() - t0}
                if not path.exists():
                    write_json(path, result)
                rows = measured["results"]
                if len(rows) != 2 or any(row.get("error") or row.get("failed_seeds")
                                        or row.get("n_seeds") != 10 or row.get("base_seed") != 0
                                        or row.get("source_provenance", {}).get("status") != "verified"
                                        or row.get("process_provenance", {}).get("status") != "verified"
                                        for row in rows):
                    raise RuntimeError("incomplete cell; preserve result and inspect " + key)
                completed.append(key)
                print(json.dumps({"event": "condition_saved", "key": key,
                                  "completed_requests": len(completed),
                                  "means": [row["mean"] for row in rows],
                                  "cached": [row["cached"] for row in rows],
                                  "seconds": round(time.time() - t0, 2)}), flush=True)
        final = {"job": job["id"], "status": "measurements_complete",
                 "completed_requests": len(completed), "total_requests": 12,
                 "successful_seed_runs": 240, "seconds": time.time() - started,
                 "updated_at": time.time()}
        write_json(REPO / "current.json", final)
        update_job(REPO.parent / "job.json", status="measured", measurements_completed_at=time.time(),
                   measurements=final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update_job(REPO.parent / "job.json", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
