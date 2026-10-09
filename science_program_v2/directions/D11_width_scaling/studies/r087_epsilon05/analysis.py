#!/usr/bin/env python3
"""复算 epsilon=.05 端点并检验预注册误差界。"""
import hashlib, json, math
from pathlib import Path

STUDY = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg = json.loads((STUDY / "preregistration.json").read_text())
    rec = json.loads((STUDY / "results/receipt.json").read_text())
    rows = []
    for name, digest in sorted(rec["cells"].items()):
        out = STUDY / "results" / (name + ".json")
        assert sha(out) == digest
        row = json.loads(out.read_text())
        d = int(row["cell"]["dimension"])
        target = row["cell"]["target"]
        coeff = math.log(20) / (2 if target == "middle" else 1)
        t = int(row["measured_step_epsilon005"])
        h = sum(1.0 / i for i in range(1, d + 1))
        rows.append({**row["cell"], "name": name, "measured_step": t,
                     "harmonic_number": h, "T_over_dh": t / (d * h),
                     "coefficient": coeff,
                     "coefficient_relative_error": t / (d * h * coeff) - 1,
                     "source_json_sha256": row["source_json_sha256"],
                     "source_npz_sha256": row["source_npz_sha256"]})
    aggregate = {}
    for target in cfg["targets"]:
        sub = [r for r in rows if r["target"] == target]
        aggregate[target] = {
            "max_abs_coefficient_relative_error": max(abs(r["coefficient_relative_error"]) for r in sub),
            "d_ge_128_max_abs_coefficient_relative_error": max(abs(r["coefficient_relative_error"]) for r in sub if r["dimension"] >= 128),
            "coefficient_error_range": [min(r["coefficient_relative_error"] for r in sub), max(r["coefficient_relative_error"] for r in sub)]}
    pairs = []
    for d in cfg["dimensions"]:
        for seed in cfg["coordinate_seeds"]:
            middle = next(r for r in rows if r["dimension"] == d and r["seed"] == seed and r["target"] == "middle")
            endpoints = next(r for r in rows if r["dimension"] == d and r["seed"] == seed and r["target"] == "endpoints")
            pairs.append({"dimension": d, "seed": seed, "budget_ratio_endpoints_over_middle": endpoints["measured_step"] / middle["measured_step"]})
    complete = len(rows) == 66
    supported = complete and all(aggregate[t]["max_abs_coefficient_relative_error"] <= cfg["criteria"][t] and aggregate[t]["d_ge_128_max_abs_coefficient_relative_error"] <= cfg["criteria"]["d_ge_128"] for t in cfg["targets"])
    summary = {"study": cfg["study"], "round": 87, "direction": "D11_width_scaling", "domain": "development", "complete": complete,
      "counts": {"saved_cells": len(rows), "target_pairs": len(pairs)},
      "predictions": {"P1": "supported" if complete else "not_evaluated", "P2": "supported" if supported else ("refuted" if complete else "not_evaluated")},
      "rows": rows, "aggregate": aggregate, "target_pairs": pairs, "threshold": 0.05,
      "input_manifest_sha256": cfg["input_manifest_sha256"], "pinned_sources": cfg["pinned_sources"]}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved_cells": len(rows), "predictions": summary["predictions"], "aggregate": aggregate}, ensure_ascii=False))

if __name__ == "__main__":
    main()
