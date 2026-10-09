import hashlib
import json
import math
import subprocess
from pathlib import Path

STUDY = Path(__file__).resolve().parent
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg = json.loads((STUDY / "preregistration.json").read_text())
    assert sha(Path(__file__)) == prereg["analysis_source_sha256"]
    receipt = json.loads((STUDY / "executed/preregistration_commit_receipt.json").read_text())
    commit = receipt["preregistration_commit"]
    commit_epoch = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit], cwd=PROGRAM))
    cells, manifest = {}, []
    sources = {(s["alpha"], s["seed"], s["momentum"]): s for s in prereg["source_cells"]}
    for path in sorted((STUDY / "results").glob("*/metadata.json")):
        metadata = json.loads(path.read_text()); request = metadata["request"]
        cid = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        assert cid == metadata["cell_id"] == path.parent.name and metadata["status"] == "success"
        assert metadata["new_training_steps"] == 0 and metadata["endpoints_sha256"] == sha(path.parent / "endpoints.json")
        assert request["preregistration_commit"] == commit and request["q"] == prereg["threshold"]
        assert commit_epoch < metadata["started_at_epoch"] <= metadata["finished_at_epoch"]
        key = (request["alpha"], request["seed"], request["momentum"]); assert key in sources
        source = sources[key]
        assert request["source_arrays_sha256"] == source["arrays_sha256"] and request["source_metadata_sha256"] == source["metadata_sha256"]
        output = json.loads((path.parent / "endpoints.json").read_text()); assert output["cell_id"] == cid
        cells[key] = {"cell_id": cid, **output}; manifest += [{"path": str(x.relative_to(PROGRAM)), "sha256": sha(x), "mtime_ns": x.stat().st_mtime_ns} for x in [path, path.parent / "endpoints.json"]]
    assert len(cells) == len(prereg["source_cells"]) == 6
    pairs = []
    for seed in prereg["seeds"]:
        plain, momentum = cells[(prereg["alpha"], seed, 0.0)], cells[(prereg["alpha"], seed, 0.9)]
        for endpoint in ["first", "sustained_to_T"]:
            t0, tm = plain["endpoints"][endpoint], momentum["endpoints"][endpoint]
            assert t0 is not None and tm is not None and tm > 0
            nominal = math.ceil(t0 / 10)
            startup = next(t for t in range(prereg["steps"] + 1) if 10 * t - 90 * (1 - .9 ** t) >= t0)
            pairs.append({"alpha": prereg["alpha"], "seed": seed, "endpoint": endpoint, "plain_cell": plain["cell_id"], "momentum_cell": momentum["cell_id"],
                          "t_plain": t0, "t_momentum": tm, "nominal_prediction": nominal, "startup_prediction": startup,
                          "nominal_relative_abs_error": abs(nominal - tm) / tm, "startup_relative_abs_error": abs(startup - tm) / tm})
    primary = [p for p in pairs if p["endpoint"] == "sustained_to_T"]
    p1 = all(p["startup_relative_abs_error"] > .05 for p in primary)
    summary = {"study": prereg["study"], "round": prereg["round"], "direction_round": prereg["direction_round"], "domain": prereg["domain"],
               "q": prereg["threshold"], "alpha": prereg["alpha"], "preregistration_commit": commit,
               "preregistration_sha256": sha(STUDY / "preregistration.json"), "analysis_source_sha256": sha(Path(__file__)),
               "counts": {"saved_endpoint_cells": len(cells), "reused_training_cells": len(cells), "new_training_cells": 0, "new_training_steps": 0, "coordinate_seeds": len(prereg["seeds"])},
               "predictions": {"P1_alpha_equals_q_startup_gt_5pct": {"status": "supported" if p1 else "refuted", "paired_conditions": 1}},
               "paired_endpoints": pairs, "result_manifest": manifest, "historical_files_unchanged": len(prereg["historical_manifest"]), "boundaries": prereg["boundaries"]}
    target = STUDY / "summary.json"
    if target.exists(): assert json.loads(target.read_text()) == summary
    else: target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
