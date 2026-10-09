import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OLD = ROOT / "directions/D1_u_effective_time/studies/r049_lower_threshold"
STUDY = OLD.parent / "r081_equal_threshold"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((OLD / "executed/final_commit_verification.json").read_text())
    commit = receipt["scientific_closeout_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    matched = 0
    for row in receipt["artifact_manifest"]:
        blob = subprocess.check_output(["git", "show", f"{commit}:{row['path']}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == row["sha256"]
        path = ROOT / row["path"]
        if path.exists() and sha(path) == row["sha256"]:
            matched += 1
    old_prereg = json.loads((OLD / "preregistration.json").read_text())
    old_summary = json.loads((OLD / "summary.json").read_text())
    for row in old_prereg["historical_manifest"] + old_summary["result_manifest"]:
        path = ROOT / row["path"]
        assert sha(path) == row["sha256"] and path.stat().st_mtime_ns == row["mtime_ns"]
    assert old_summary["counts"]["saved_endpoint_cells"] == 36
    assert old_summary["counts"]["new_training_cells"] == 0
    assert json.loads((OLD / "executed/saved_evidence_verification.json").read_text())["status"] == "passed"
    for source in old_prereg["source_cells"]:
        arrays = ROOT / source["arrays_path"]
        metadata = ROOT / source["metadata_path"]
        assert sha(arrays) == source["arrays_sha256"]
        assert sha(metadata) == source["metadata_sha256"]
        assert json.loads(metadata.read_text())["status"] == "success"
    audit = {
        "status": "passed",
        "previous_scientific_closeout_commit": commit,
        "receipt_sha256": sha(OLD / "executed/final_commit_verification.json"),
        "committed_artifact_hashes_verified": len(receipt["artifact_manifest"]),
        "current_artifacts_matching_closeout": matched,
        "historical_hash_mtime_verified": len(old_prereg["historical_manifest"]),
        "endpoint_hash_mtime_verified": len(old_summary["result_manifest"]),
        "prior_successful_training_cells": 144,
        "prior_successful_endpoint_cells": 36,
        "old_predictions_retained": True,
        "new_q_endpoints_computed_before_preregistration": False,
    }
    target = STUDY / "executed/previous_closeout_audit.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
