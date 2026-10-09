#!/usr/bin/env python3
"""Independent scalar/Decimal check; reads original and evaluation results."""
from datetime import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
GIT = ["git", f"--git-dir={ROOT / '.git'}", f"--work-tree={ROOT}"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    source = ROOT / config["source_study"]
    original = json.loads((source / "preregistration.json").read_text())
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    commit = receipt["pins"]["preregistration_commit"]
    assert commit == "f3194fe5dc10d2ed3d617ea6cc4f436fbc5d71a0"
    subprocess.run([*GIT, "merge-base", "--is-ancestor", commit, "HEAD"], check=True, cwd=ROOT)
    for name, digest in receipt["pins"]["sha256"].items():
        path = STUDY / name
        saved = subprocess.check_output([*GIT, "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert sha(path) == hashlib.sha256(saved).hexdigest() == digest
    manifest = json.loads((STUDY / "executed/input_manifest.json").read_text())
    for name, record in manifest["immutable_artifacts"].items():
        path = ROOT / name
        assert sha(path) == record["sha256"] and path.stat().st_mtime_ns == record["mtime_ns"]
    assert set(receipt["cells"]) == {f"seed{s}_{t}" for s in config["seeds"] for t in config["targets"]}
    rows, features = [], {}
    with localcontext() as context:
        context.prec = 60
        for name, digest in sorted(receipt["cells"].items()):
            path = STUDY / f"results/{name}.json"
            assert sha(path) == digest
            result = json.loads(path.read_text())
            assert result["pins"] == receipt["pins"]
            seed, target = result["seed"], result["target"]
            source_path = source / f"results/{name}.npz"
            meta = json.loads(source_path.with_suffix(".json").read_text())
            assert meta["pins"]["preregistration_commit"] == config["original_scientific_commit"]
            assert sha(source_path) == meta["arrays_sha256"] == result["source_npz_sha256"]
            with np.load(source_path, allow_pickle=False) as a:
                loss = a["loss"].copy()
                features[(seed, target)] = a["features"].tobytes()
            observed = next((t for t, l in enumerate(loss) if float(l) / float(loss[0]) <= .001), None)
            rational = [Fraction(w) for w in original["targets"][target]]
            weights = [Decimal(w.numerator) / Decimal(w.denominator) for w in rational]
            decay = [(1 - Decimal(1) / (2 * Decimal(i) ** 2)) ** 2 for i in range(1, 10)]
            state = list(weights)
            decimal_curve = []
            scalar_curve = []
            for t in range(1001):
                decimal_curve.append(sum(state))
                scalar_curve.append(math.fsum(float(w) * (1 - .5 / (i * i)) ** (2 * t)
                                              for i, w in enumerate(rational, 1)))
                state = [v * d for v, d in zip(state, decay)]
            exact_step = next((t for t, r in enumerate(decimal_curve) if r <= Decimal(".001")), None)
            scalar_step = next((t for t, r in enumerate(scalar_curve) if r <= .001), None)
            assert observed == exact_step == scalar_step == result["threshold_step"]
            scalar_error = max(abs(float(l) / float(loss[0]) - r) for l, r in zip(loss, scalar_curve))
            decimal_error = max(abs(float(l) / float(loss[0]) - float(r)) for l, r in zip(loss, decimal_curve))
            assert scalar_error <= 1e-10 and decimal_error <= 1e-10
            rows.append({"name": name, "step": observed, "scalar_maxabs": scalar_error,
                         "decimal_maxabs": decimal_error, "decimal_before": str(decimal_curve[observed - 1]),
                         "decimal_hit": str(decimal_curve[observed])})
    assert all(features[(s, "fast_pair")] == features[(s, "slow_pair")] for s in config["seeds"])
    intervals = config["numeric_prediction"]
    for seed in config["seeds"]:
        fast = next(r["step"] for r in rows if r["name"] == f"seed{seed}_fast_pair")
        slow = next(r["step"] for r in rows if r["name"] == f"seed{seed}_slow_pair")
        assert slow / fast >= config["minimum_step_ratio"]
        assert intervals["fast_steps"][0] <= fast <= intervals["fast_steps"][1]
        assert intervals["slow_steps"][0] <= slow <= intervals["slow_steps"][1]
        assert intervals["step_ratio"][0] <= slow / fast <= intervals["step_ratio"][1]
    epoch = int(subprocess.check_output([*GIT, "show", "-s", "--format=%ct", commit], cwd=ROOT))
    gaps = [datetime.fromisoformat(json.loads((STUDY / f"results/{n}.json").read_text())["started_at"]).timestamp() - epoch
            for n in receipt["cells"]]
    assert min(gaps) > 0
    review = {"kind": "independent_scalar_decimal_saved_evidence", "passed": True, "decimal_precision": 60,
              "verified_evaluation_cells": len(rows), "verified_input_files": len(manifest["immutable_artifacts"]),
              "preregistration_commit": commit, "source_scientific_commit": config["original_scientific_commit"],
              "minimum_seconds_after_freeze": min(gaps), "all3_pairs_same_features_bytes": True,
              "scalar_maxabs": max(r["scalar_maxabs"] for r in rows),
              "decimal_maxabs": max(r["decimal_maxabs"] for r in rows), "rows": rows,
              "new_training_cells": 0, "independent_agent_read_only_review_passed": True}
    (STUDY / "executed/independent_verification.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in review.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
