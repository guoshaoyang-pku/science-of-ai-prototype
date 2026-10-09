import json
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from executed.run import ROOT, STUDY, contract, save, sha


def bootstrap(risk, full_t, seed, count):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(risk), size=(count, len(risk)))
    t_stars = np.empty(count, dtype=np.int64)
    for start in range(0, count, 50):
        chosen = indices[start:start + 50]
        t_stars[start:start + len(chosen)] = np.argmin(risk[chosen].mean(1), axis=1)
    return t_stars, np.quantile(t_stars / full_t, [.025, .975])


def main():
    cfg, pins = contract()
    rec = json.loads((STUDY / "results/receipt.json").read_text())
    if len(rec["cells"]) != 2:
        raise RuntimeError("Incomplete cells")
    rows, bootstrap_arrays = [], {}
    for c in cfg["cells"]:
        seed = c["seed"]
        label = f"n64_seed{seed}_eta0.1_B8_var1_r64_127_M64"
        path = STUDY / "results" / (label + ".json")
        meta = json.loads(path.read_text())
        if sha(path) != rec["cells"][label] or sha(path.with_suffix(".npz")) != meta["arrays_sha256"]:
            raise RuntimeError("Hash mismatch")
        with np.load(path.with_suffix(".npz")) as z:
            current = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c["old_block"]) as z:
            old = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c["full"]) as z:
            full_t = int(np.argmin(z["mean_risk"]))
            expected_t = int(np.argmin(z["expected_risk"]))
        if full_t != c["known_full_t_star"]:
            raise RuntimeError("Fixed full denominator changed")
        old_seed = cfg["old_bootstrap_seed_base"] + 100000 * seed + 8100
        new_seed = cfg["bootstrap_seed_base"] + 100000 * seed + 8100
        old_boot, old_ci = bootstrap(old["replicate_risk"], full_t, old_seed, cfg["bootstrap_replicates"])
        new_boot, ci = bootstrap(current["replicate_risk"], full_t, new_seed, cfg["bootstrap_replicates"])
        delta_ci = np.quantile((new_boot - old_boot) / full_t, [.025, .975])
        ts, old_ts = int(np.argmin(current["mean_risk"])), int(np.argmin(old["mean_risk"]))
        rows.append({"seed": seed, "batch": 8, "variance": 1, "eta": .1, "replicates": 64,
                     "replicate_range": [64, 127], "t_star": ts, "full_realized_t_star": full_t,
                     "full_expected_t_star": expected_t, "aligned_ratio": ts / full_t,
                     "legacy_ratio": ts / expected_t, "old_t_star": old_ts,
                     "old_aligned_ratio": old_ts / full_t, "delta_ratio": (ts - old_ts) / full_t,
                     "aligned_ratio_bootstrap_95pct": ci.tolist(), "old_bootstrap_95pct": old_ci.tolist(),
                     "paired_bootstrap_delta_ratio_95pct": delta_ci.tolist(),
                     "delta_interval_crosses_zero": bool(delta_ci[0] <= 0 <= delta_ci[1]),
                     "bootstrap_crosses_two": bool(ci[0] <= 2 <= ci[1]),
                     "bootstrap_fraction_over_two": float(np.mean(new_boot / full_t > 2)),
                     "P1_pass": bool(.5 <= ts / full_t <= 2), "P2_pass": bool(ci[1] < 2),
                     "finite": bool(all(np.isfinite(a).all() for a in current.values())),
                     "interior": bool(0 < ts < cfg["steps"]),
                     "streams_disjoint": bool(not np.intersect1d(current["batch_seeds"], old["batch_seeds"]).size),
                     "min_risk": float(current["mean_risk"][ts]),
                     "se_at_min": float(current["standard_error"][ts]),
                     "replicate_t_stars": np.argmin(current["replicate_risk"], axis=1).tolist()})
        bootstrap_arrays[f"seed{seed}_old_t_star"] = old_boot
        bootstrap_arrays[f"seed{seed}_new_t_star"] = new_boot
    bootstrap_path = STUDY / "results/bootstrap.npz"
    if bootstrap_path.exists():
        with np.load(bootstrap_path) as z:
            if set(z.files) != set(bootstrap_arrays) or not all(np.array_equal(z[k], a) for k, a in bootstrap_arrays.items()):
                raise RuntimeError("Saved bootstrap mismatch; refusing overwrite")
    else:
        np.savez_compressed(bootstrap_path, **bootstrap_arrays)
    audit = json.loads((STUDY / "executed/input_audit.json").read_text())
    unchanged = all(sha(ROOT / p) == v["sha256"] and (ROOT / p).stat().st_mtime_ns == v["mtime_ns"]
                    for p, v in audit["old_files"].items())
    predictions = {}
    for name in ["P1", "P2"]:
        passed = sum(r[name + "_pass"] for r in rows)
        predictions[name] = {"status": "supported" if passed == 2 else "refuted", "passed": passed, "total": 2}
    integrity = unchanged and all(r["finite"] and r["interior"] and r["streams_disjoint"] for r in rows)
    predictions["P3"] = {"status": "supported" if integrity else "refuted", "old_files_unchanged": unchanged}
    summary = {"study": cfg["study"], "round": 41, "direction_round": 4, "direction": "D9_minibatch_noise",
               "domain": "development", "status": "complete", "round_result": "ok", "predictions": predictions,
               "counts": {"recipe_units": 1, "paired_data_seeds": 2, "saved_cells": 2,
                          "new_batch_trajectories": 128, "retained_batch_trajectories": 0,
                          "reused_old_block_trajectories": 128, "reused_full_controls": 2, "new_full_trajectories": 0},
               "compute_seconds": rec["compute_seconds"], "cells": rows, "pins": pins,
               "bootstrap": {"replicates": cfg["bootstrap_replicates"], "arrays_sha256": sha(bootstrap_path),
                             "scope": "固定data/epsilon条件下轨迹percentile描述区间；新旧64条流块分别独立重抽；同数据seed差值不作显著性检验。"},
               "method": "首个argmin(mean audit risk)，新r64..127与旧r0..63不合并；同epsilon分母；不推精确batch期望。"}
    save(STUDY / "summary.json", summary)
    print(json.dumps({"predictions": predictions, "cells": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
