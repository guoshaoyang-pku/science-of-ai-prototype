#!/usr/bin/env python3
"""Evaluate one committed threshold on existing loss arrays; never train."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
GIT = ["git", f"--git-dir={ROOT / '.git'}", f"--work-tree={ROOT}"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify(commit):
    subprocess.run([*GIT, "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    config = json.loads((STUDY / "preregistration.json").read_text())
    pins = {"preregistration.json": sha(STUDY / "preregistration.json"), **config["source_sha256"]}
    for name, digest in pins.items():
        path = STUDY / name
        saved = subprocess.check_output([*GIT, "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(saved).hexdigest() == sha(path) == digest, name
    manifest = json.loads((STUDY / "executed/input_manifest.json").read_text())
    for name, record in manifest["immutable_artifacts"].items():
        path = ROOT / name
        assert sha(path) == record["sha256"] and path.stat().st_mtime_ns == record["mtime_ns"], name
    source = ROOT / config["source_study"]
    receipt = json.loads((source / "results/receipt.json").read_text())
    names = {f"seed{s}_{t}" for s in config["seeds"] for t in config["targets"]}
    assert set(receipt["cells"]) == names
    expected = {"receipt.json"} | {f"{n}.{ext}" for n in names for ext in ("json", "npz")}
    assert {p.name for p in (source / "results").iterdir()} == expected
    assert receipt["pins"]["preregistration_commit"] == config["original_scientific_commit"]
    for name, digest in receipt["pins"]["sha256"].items():
        path = source / name
        saved = subprocess.check_output([*GIT, "show", f"{config['original_scientific_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert sha(path) == hashlib.sha256(saved).hexdigest() == digest
    for name in sorted(names):
        meta_path = source / f"results/{name}.json"
        meta = json.loads(meta_path.read_text())
        assert sha(meta_path) == receipt["cells"][name]
        assert sha(meta_path.with_suffix(".npz")) == meta["arrays_sha256"]
        assert meta["pins"] == receipt["pins"]
        request = hashlib.sha256(json.dumps({"cell": meta["cell"], "pins": meta["pins"]}, sort_keys=True).encode()).hexdigest()
        assert request == meta["request_sha256"]
    return config, {"preregistration_commit": commit, "sha256": pins}, names


def evaluate(commit):
    config, pins, names = verify(commit)
    source = ROOT / config["source_study"]
    original = json.loads((source / "preregistration.json").read_text())
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    receipt_path = results / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["pins"] == pins and set(receipt["cells"]) <= names
    else:
        receipt = {"pins": pins, "cells": {}, "new_training_cells": 0}
    assert {p.name for p in results.iterdir()} == ({"receipt.json"} if receipt_path.exists() else set()) | {f"{n}.json" for n in receipt["cells"]}
    for name, digest in receipt["cells"].items():
        assert sha(results / f"{name}.json") == digest
        meta = json.loads((results / f"{name}.json").read_text())
        assert meta["pins"] == pins and meta["name"] == name and meta["status"] == "completed"
    before = {str(p.relative_to(STUDY)): {"sha256": sha(p), "mtime_ns": p.stat().st_mtime_ns}
              for p in results.iterdir() if p.is_file()}
    started = datetime.now(timezone.utc).isoformat()
    dump(STUDY / "executed/experiment_start.json" if not receipt["cells"] else STUDY / "executed/resume_start.json",
         {"started_at": started, "pins": pins, "input_manifest_verified": True,
          "already_saved_cells": len(receipt["cells"]), "new_training_cells": 0})
    new = 0
    for seed in config["seeds"]:
        for target in config["targets"]:
            name = f"seed{seed}_{target}"
            if name in receipt["cells"]:
                continue
            tick = time.perf_counter()
            cell_start = datetime.now(timezone.utc).isoformat()
            source_path = source / f"results/{name}.npz"
            with np.load(source_path, allow_pickle=False) as a:
                loss, x = a["loss"], a["features"]
                assert loss.shape == (1001,) and loss.dtype == np.float64 and np.isfinite(loss).all()
                ratio = loss / loss[0]
                hits = np.flatnonzero(ratio <= config["threshold"])
                step = int(hits[0]) if len(hits) else None
                lam = np.arange(1, 10, dtype=float) ** -2
                weights = np.array([float(Fraction(w)) for w in original["targets"][target]])
                reference = np.sum(weights[:, None] * (1 - .5 * lam[:, None]) ** (2 * np.arange(1001)), axis=0)
                error = float(np.max(np.abs(ratio - reference)))
                ref_hits = np.flatnonzero(reference <= config["threshold"])
                ref_step = int(ref_hits[0]) if len(ref_hits) else None
                row = {"name": name, "seed": seed, "target": target, "status": "completed", "pins": pins,
                       "source_npz": str(source_path.relative_to(ROOT)), "source_npz_sha256": sha(source_path),
                       "source_scientific_commit": config["original_scientific_commit"],
                       "source_preregistration_sha256": sha(source / "preregistration.json"),
                       "features_bytes_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                       "threshold": config["threshold"], "threshold_step": step, "censored": step is None,
                       "previous_ratio": float(ratio[step - 1]) if step else None,
                       "hit_ratio": float(ratio[step]) if step is not None else None,
                       "spectral_step": ref_step, "spectral_maxabs": error,
                       "audit_passed": bool(step == ref_step and error <= config["tolerances"]["execution_absolute"]
                                            and abs(float(loss[0]) - .5) <= 1e-12),
                       "started_at": cell_start, "finished_at": datetime.now(timezone.utc).isoformat(),
                       "seconds": time.perf_counter() - tick, "new_training_cells": 0}
            dump(results / f"{name}.json", row)
            receipt["cells"][name] = sha(results / f"{name}.json")
            dump(receipt_path, receipt)
            new += 1
    verify(commit)
    unchanged = all(sha(STUDY / p) == r["sha256"] and (STUDY / p).stat().st_mtime_ns == r["mtime_ns"]
                    for p, r in before.items()) if new == 0 else True
    assert unchanged
    audit = {"new_evaluation_cells": new, "reused_evaluation_cells": len(names) - new, "new_training_cells": 0,
             "inputs_hash_mtime_unchanged": True, "successful_results_hash_mtime_unchanged": unchanged,
             "successful_results_before": before, "pins": pins}
    dump(STUDY / "executed/resume_audit.json" if new == 0 else STUDY / "executed/execution_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("new_evaluation_cells", "reused_evaluation_cells", "new_training_cells")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-commit", required=True)
    evaluate(parser.parse_args().preregistration_commit)
