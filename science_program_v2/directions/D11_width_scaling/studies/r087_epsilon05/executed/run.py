#!/usr/bin/env python3
"""Read-only endpoint evaluation of r073 loss curves at epsilon=.05."""
import hashlib, json, os
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[3]
BASE = ROOT / "directions/D11_width_scaling/studies/r073_trace_one"
THRESHOLD = 0.05

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg = json.loads((STUDY / "preregistration.json").read_text())
    receipt = {"study": cfg["study"], "threshold": THRESHOLD, "cells": {}, "pins": cfg["pinned_sources"]}
    for d in cfg["dimensions"]:
        for seed in cfg["coordinate_seeds"]:
            for target in cfg["targets"]:
                name = f"d{d}_seed{seed}_{target}"
                src_json = BASE / "results" / (name + ".json")
                src_npz = BASE / "results" / (name + ".npz")
                out = STUDY / "results" / (name + ".json")
                assert src_json.exists() and src_npz.exists()
                if out.exists():
                    old = json.loads(out.read_text())
                    assert old["source_json_sha256"] == sha(src_json)
                    assert old["source_npz_sha256"] == sha(src_npz)
                    receipt["cells"][name] = sha(out)
                    continue
                meta = json.loads(src_json.read_text())
                with np.load(src_npz) as z:
                    loss = np.asarray(z["loss"], dtype=np.float64)
                    hits = np.flatnonzero(loss / loss[0] <= THRESHOLD)
                    assert len(hits) > 0
                    step = int(hits[0])
                row = {"study": cfg["study"], "cell": meta["cell"],
                       "threshold": THRESHOLD, "measured_step_epsilon005": step,
                       "source_json_sha256": sha(src_json), "source_npz_sha256": sha(src_npz),
                       "source_measured_step_epsilon001": meta["values"]["measured_step"],
                       "loss_length": int(loss.size), "pinned_sources": cfg["pinned_sources"]}
                out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                receipt["cells"][name] = sha(out)
    (STUDY / "results/receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved_cells": len(receipt["cells"]), "threshold": THRESHOLD}))

if __name__ == "__main__":
    main()
