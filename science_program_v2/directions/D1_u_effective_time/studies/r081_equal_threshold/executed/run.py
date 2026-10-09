import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def committed_contract():
    prereg_path = STUDY / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    receipt = json.loads((STUDY / "executed/preregistration_commit_receipt.json").read_text())
    commit = receipt["preregistration_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=PROGRAM, check=True)
    for name in prereg["pinned_files"]:
        path = STUDY / name
        saved = subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(PROGRAM)}"], cwd=PROGRAM)
        assert saved == path.read_bytes(), name
    assert prereg["execution_source_sha256"] == sha(Path(__file__))
    assert receipt["preregistration_sha256"] == sha(prereg_path)
    return prereg, commit


def request_for(prereg, commit, source):
    return {
        "study": prereg["study"], "domain": prereg["domain"], "q": prereg["threshold"],
        "steps": prereg["steps"], "alpha": source["alpha"], "seed": source["seed"],
        "momentum": source["momentum"], "source_cell_id": source["cell_id"],
        "source_arrays_path": source["arrays_path"], "source_arrays_sha256": source["arrays_sha256"],
        "source_metadata_path": source["metadata_path"], "source_metadata_sha256": source["metadata_sha256"],
        "source_closeout_commit": prereg["source_closeout_commit"],
        "preregistration_commit": commit, "preregistration_sha256": sha(STUDY / "preregistration.json"),
        "execution_source_sha256": sha(Path(__file__)),
    }


def cell_id(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def evaluate(prereg, commit, source):
    request = request_for(prereg, commit, source)
    cid = cell_id(request)
    result = STUDY / "results" / cid
    if result.exists():
        metadata = json.loads((result / "metadata.json").read_text())
        assert metadata["status"] == "success" and metadata["request"] == request
        assert metadata["endpoints_sha256"] == sha(result / "endpoints.json")
        return False, {"cell_id": cid, "endpoints_sha256": metadata["endpoints_sha256"], "metadata_sha256": sha(result / "metadata.json")}
    started = time.time()
    arrays_path = PROGRAM / source["arrays_path"]
    metadata_path = PROGRAM / source["metadata_path"]
    assert sha(arrays_path) == source["arrays_sha256"] and sha(metadata_path) == source["metadata_sha256"]
    original = json.loads(metadata_path.read_text())
    assert original["status"] == "success" and original["arrays_sha256"] == sha(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        loss = arrays["normalized_loss"]
        assert loss.shape == (prereg["steps"] + 1,) and np.isfinite(loss).all()
        assert abs(float(loss[0]) - 1.0) <= 1e-10
        below = np.flatnonzero(loss <= prereg["threshold"])
        above = np.flatnonzero(loss > prereg["threshold"])
        sustained = int(above[-1] + 1) if len(above) else 0
        endpoints = {"first": int(below[0]) if len(below) else None,
                     "sustained_to_T": sustained if sustained <= prereg["steps"] else None}
        diagnostics = {"minimum_distance_to_q": float(np.min(np.abs(loss - prereg["threshold"]))),
                       "loss_at_T": float(loss[-1]),
                       "loss_at_sustained": float(loss[sustained]) if sustained <= prereg["steps"] else None,
                       "loss_before_sustained": float(loss[sustained - 1]) if 0 < sustained <= prereg["steps"] else None}
    result.mkdir(parents=True)
    write_new(result / "endpoints.json", {"cell_id": cid, "source_cell_id": source["cell_id"], "q": prereg["threshold"], "endpoints": endpoints, "diagnostics": diagnostics})
    finished = time.time()
    write_new(result / "metadata.json", {"cell_id": cid, "status": "success", "request": request,
                                        "started_at_epoch": started, "finished_at_epoch": finished,
                                        "elapsed_seconds": finished - started, "endpoints_sha256": sha(result / "endpoints.json"),
                                        "new_training_steps": 0})
    return True, {"cell_id": cid, "endpoints_sha256": sha(result / "endpoints.json"), "metadata_sha256": sha(result / "metadata.json")}


def main():
    prereg, commit = committed_contract()
    created, resumed = [], []
    for source in prereg["source_cells"]:
        is_new, record = evaluate(prereg, commit, source)
        (created if is_new else resumed).append(record)
    for row in prereg["historical_manifest"]:
        path = PROGRAM / row["path"]
        assert sha(path) == row["sha256"] and path.stat().st_mtime_ns == row["mtime_ns"]
    receipt = {"preregistration_commit": commit, "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROGRAM, text=True).strip(),
               "created_successes": created, "resumed_successes": resumed, "new_training_steps": 0,
               "historical_hash_mtime_unchanged": True, "historical_files_checked": len(prereg["historical_manifest"])}
    count = len(list((STUDY / "executed").glob("run_receipt_*.json"))) + 1
    write_new(STUDY / f"executed/run_receipt_{count:03d}.json", receipt)
    print(json.dumps({"created": len(created), "resumed": len(resumed), "new_training_steps": 0}))


if __name__ == "__main__":
    main()
