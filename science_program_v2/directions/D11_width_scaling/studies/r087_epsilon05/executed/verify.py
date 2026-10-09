#!/usr/bin/env python3
"""独立核验 r087 保存端点及全部只读输入不变。"""
import hashlib, json
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg = json.loads((STUDY / "preregistration.json").read_text())
    manifest = json.loads((STUDY / "executed/input_manifest.json").read_text())
    input_checks = 0
    for rel, info in manifest["files"].items():
        path = ROOT / rel
        assert sha(path) == info["sha256"]
        assert path.stat().st_mtime_ns == info["mtime_ns"]
        input_checks += 1
    rec = json.loads((STUDY / "results/receipt.json").read_text())
    assert len(rec["cells"]) == 66
    row_checks = 0
    names = sorted(rec["cells"])
    for name in names:
        out = STUDY / "results" / (name + ".json")
        assert sha(out) == rec["cells"][name]
        row = json.loads(out.read_text())
        d = int(row["cell"]["dimension"])
        src = ROOT / cfg["source_study"] / "results" / (name + ".npz")
        src_json = ROOT / cfg["source_study"] / "results" / (name + ".json")
        assert row["source_npz_sha256"] == sha(src)
        assert row["source_json_sha256"] == sha(src_json)
        with np.load(src) as z:
            loss = np.asarray(z["loss"], dtype=np.float64)
        hits = np.flatnonzero(loss / loss[0] <= cfg["threshold"])
        assert len(hits) and int(hits[0]) == row["measured_step_epsilon005"]
        row_checks += 1
    # Three coordinate seeds are signed-permutation relabelings, so each target/d agrees.
    equality_checks = 0
    for d in cfg["dimensions"]:
        for target in cfg["targets"]:
            vals = [json.loads((STUDY / "results" / f"d{d}_seed{s}_{target}.json").read_text())["measured_step_epsilon005"] for s in cfg["coordinate_seeds"]]
            assert len(set(vals)) == 1
            equality_checks += 1
    result = {"study": cfg["study"], "saved_cells": len(names), "input_file_checks": input_checks,
              "row_checks": row_checks, "seed_equality_checks": equality_checks,
              "threshold": cfg["threshold"], "all_passed": True,
              "input_manifest_sha256": sha(STUDY / "executed/input_manifest.json"),
              "receipt_sha256": sha(STUDY / "results/receipt.json")}
    (STUDY / "executed/independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
