#!/usr/bin/env python3
"""Read-only audit of saved evidence; no parameter updates or new experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    assert config["status"] == "invalid_draft_do_not_execute"
    audit, cells = [], []
    for rel, pin in config["baseline_files"].items():
        path = ROOT / rel
        committed = subprocess.check_output(["git", "show", f"{pin['snapshot_commit']}:{rel}"], cwd=ROOT)
        audit.append({"path": rel, "sha256_matches": sha(path) == pin["sha256"],
                      "mtime_matches": path.stat().st_mtime_ns == pin["mtime_ns"],
                      "committed_bytes_match": path.read_bytes() == committed})
    assert all(all(x[k] for k in ["sha256_matches", "mtime_matches", "committed_bytes_match"]) for x in audit)
    for slug in ["r027_dimension_rayleigh", "r031_fixed_n8192"]:
        old = ROOT / "directions/D11_width_scaling/studies" / slug
        receipt = json.loads((old / "results/receipt.json").read_text())
        for name, digest in receipt["cells"].items():
            metadata = old / "results" / f"{name}.json"
            arrays = old / "results" / f"{name}.npz"
            meta = json.loads(metadata.read_text())
            request = hashlib.sha256(json.dumps({"cell": meta["cell"], "pins": receipt["pins"]["sha256"]}, sort_keys=True).encode()).hexdigest()
            with np.load(arrays) as data:
                loss = data["loss"]
                threshold = np.flatnonzero(loss / loss[0] <= .01)
                step = int(threshold[0]) if len(threshold) else None
            cell = {"study": slug, "name": name, "metadata_hash_matches": digest == sha(metadata),
                    "npz_hash_matches": meta["arrays_sha256"] == sha(arrays),
                    "request_matches": meta["request_sha256"] == request,
                    "completed": meta["status"] == "completed",
                    "saved_curve_threshold_matches": step == meta["values"]["measured_step"]}
            assert all(cell[k] for k in list(cell)[2:])
            cells.append(cell)
    old_manifest = json.loads((ROOT / "directions/D11_width_scaling/studies/r031_fixed_n8192/executed/baseline_manifest.json").read_text())
    assert all(sha(ROOT / p) == x["sha256"] and (ROOT / p).stat().st_mtime_ns == x["mtime_ns"] for p, x in old_manifest.items())
    previous = json.loads((ROOT / "directions/D11_width_scaling/studies/r031_fixed_n8192/executed/final_commit_verification.json").read_text())
    old_summary = json.loads((ROOT / "directions/D11_width_scaling/studies/r031_fixed_n8192/summary.json").read_text())
    assert previous["scientific_closeout_commit"].startswith("882fd01")
    assert previous["preregistration_commit"].startswith("cc2f552")
    assert previous["preregistration_precedes_all_cells"]
    subprocess.run(["git", "merge-base", "--is-ancestor", previous["preregistration_commit"], previous["scientific_closeout_commit"]], cwd=ROOT, check=True)
    save(STUDY / "executed/prior_evidence_audit.json", {"files": audit, "cells": cells,
         "r027_files": 145, "r031_files": 148, "all_checks_passed": True,
         "r027_scientific_closeout_commit": "37d1ea6794d0481a972c49cd7d2c6a079e465424",
         "r027_original_final_verification_missing": True, "r031_scientific_closeout_commit": previous["scientific_closeout_commit"],
         "r031_current_receipt_matches_snapshot_commit": config["baseline_snapshot_commits"]["r031_fixed_n8192"],
         "r031_preexisting_receipt_checked_files": previous["checked_committed_files"],
         "prior_145_manifest_hash_mtime_unchanged": True,
         "r031_preregistration_precedes_cells": True, "saved_cell_thresholds_recomputed": len(cells),
         "saved_exact_budget_matches": old_summary["exact_budget_matches"],
         "saved_step_difference_range": old_summary["step_difference_range"],
         "saved_max_paired_curve_error": old_summary["max_paired_curve_error"]})
    known = []
    for row in config["known_before_registration"]["integer_reference_table"]:
        d = row["dimension"]
        h = float(np.sum(1 / np.arange(1, d + 1, dtype=np.float64)))
        known.append(row | {"harmonic_number": h,
            "middle_T_over_dh": row["middle"] / (d * h),
            "endpoints_T_over_dh": row["endpoints"] / (d * h)})
    save(STUDY / "summary.json", {"study": config["study"], "round": 43, "direction": "D11_width_scaling",
         "domain": "development", "status": "invalid_no_experiment",
         "counts": {"planned_cells": 66, "saved_cells": 0, "new_training_cells": 0,
                    "new_fitting_cells": 0, "old_cells_audited": len(cells)},
         "predictions": {p["id"]: "not_evaluated" for p in config["draft_predictions"]},
         "compute_seconds": 0, "known_arithmetic_not_new_evidence": known,
         "known_arithmetic_source": "注册前已知线性谱递推；无参数训练、非盲预测、非新理论",
         "old_evidence_files_audited": len(audit), "all_prior_evidence_checks_passed": True,
         "invalidation_evidence": "executed/round_invalidation.json",
         "preregistered_experiment_commit": None, "execution_locked": True,
         "preregistration_sha256": sha(STUDY / "preregistration.json"),
         "execution_source_sha256": sha(STUDY / "executed/run.py"),
         "analysis_source_sha256": sha(Path(__file__)), "boundaries": config["boundaries"]})
    print(json.dumps({"status": "invalid_no_experiment", "old_cells_audited": len(cells),
                      "old_files_audited": len(audit), "all_checks_passed": True}))


if __name__ == "__main__":
    main()
