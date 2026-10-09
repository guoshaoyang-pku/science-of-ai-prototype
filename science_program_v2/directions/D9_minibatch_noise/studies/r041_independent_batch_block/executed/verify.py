import json
import os
from pathlib import Path
import subprocess

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from run import ROOT, STUDY, contract, save, sha


def main():
    cfg, pins = contract()
    summary = json.loads((STUDY / "summary.json").read_text())
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    audit = json.loads((STUDY / "executed/input_audit.json").read_text())
    unchanged = all(sha(ROOT / p) == v["sha256"] and (ROOT / p).stat().st_mtime_ns == v["mtime_ns"]
                    for p, v in audit["old_files"].items())
    rows = []
    for c in cfg["cells"]:
        seed = c["seed"]
        label = f"n64_seed{seed}_eta0.1_B8_var1_r64_127_M64"
        path = STUDY / "results" / (label + ".json")
        meta = json.loads(path.read_text())
        row = next(r for r in summary["cells"] if r["seed"] == seed)
        with np.load(path.with_suffix(".npz")) as z:
            current = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c["old_block"]) as z:
            old = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / c["full"]) as z:
            full_t = int(np.argmin(z["mean_risk"]))
        with np.load(STUDY / "executed" / Path(c["data"]).name) as z:
            x, y = z["train_features"], z["train_y"] + z["epsilon"]
            xa, ya = z["audit_features"], z["audit_y"]
        prediction = np.einsum("ap,rp->ra", xa, current["final_heads"], optimize=False)
        final_error = float(np.max(np.abs(np.mean((prediction - ya) ** 2, axis=1) - current["replicate_risk"][:, -1])))
        rng = np.random.default_rng(int(current["batch_seeds"][0]))
        w = np.zeros(x.shape[1])
        replay = []
        for t in range(17):
            residual = np.einsum("ap,p->a", xa, w, optimize=False) - ya
            replay.append(float(np.mean(residual ** 2)))
            if t < 16:
                idx = rng.choice(len(x), size=8, replace=False)
                error = np.einsum("ip,p->i", x[idx], w, optimize=False) - y[idx]
                w -= .2 * np.einsum("ip,i->p", x[idx], error, optimize=False) / 8
        replay_error = float(np.max(np.abs(np.asarray(replay) - current["replicate_risk"][0, :17])))
        boot_ok = True
        boot_ts = {}
        with np.load(STUDY / "results/bootstrap.npz") as boot:
            for block, risks, base in [("old", old["replicate_risk"], cfg["old_bootstrap_seed_base"]),
                                       ("new", current["replicate_risk"], cfg["bootstrap_seed_base"])]:
                rng = np.random.default_rng(base + 100000 * seed + 8100)
                indices = rng.integers(0, 64, size=(cfg["bootstrap_replicates"], 64))
                weights = np.stack([np.bincount(i, minlength=64) for i in indices]) / 64
                ts = np.argmin(np.einsum("bm,mt->bt", weights, risks, optimize=False), axis=1)
                boot_ts[block] = ts
                boot_ok = boot_ok and np.array_equal(ts, boot[f"seed{seed}_{block}_t_star"])
                key = "old_bootstrap_95pct" if block == "old" else "aligned_ratio_bootstrap_95pct"
                boot_ok = boot_ok and np.quantile(ts / full_t, [.025, .975]).tolist() == row[key]
        delta_ci = np.quantile((boot_ts["new"] - boot_ts["old"]) / full_t, [.025, .975]).tolist()
        boot_ok = boot_ok and delta_ci == row["paired_bootstrap_delta_ratio_95pct"]
        commit = meta["pins"]["git_commit"]
        committed = all(subprocess.check_output(["git", "show", commit + ":" + (STUDY / p).relative_to(ROOT).as_posix()], cwd=ROOT)
                        == (STUDY / p).read_bytes() for p in ["preregistration.json", *cfg["source_sha256"]])
        checks = {"receipt_hash": sha(path) == receipt["cells"][label],
                  "arrays_hash": sha(path.with_suffix(".npz")) == meta["arrays_sha256"],
                  "data_hash": sha(ROOT / c["data"]) == meta["data_sha256"],
                  "old_arrays_hash": sha(ROOT / c["old_block"]) == meta["old_arrays_sha256"],
                  "committed_contract": committed, "source_pins": meta["pins"]["sha256"] == pins["sha256"],
                  "shape": current["replicate_risk"].shape == (64, 2305),
                  "replicate_indices": np.array_equal(current["replicate_indices"], np.arange(64, 128)),
                  "batch_seeds": np.array_equal(current["batch_seeds"], 250700 + 100000 * seed + 9000 + np.arange(64, 128)),
                  "streams_disjoint": not np.intersect1d(current["batch_seeds"], old["batch_seeds"]).size,
                  "finite": all(np.isfinite(a).all() for a in current.values()),
                  "mean_risk": np.array_equal(current["mean_risk"], current["replicate_risk"].mean(0)),
                  "standard_error": np.array_equal(current["standard_error"], current["replicate_risk"].std(0, ddof=1) / 8),
                  "argmin": int(np.argmin(current["mean_risk"])) == row["t_star"] and full_t == row["full_realized_t_star"] == c["known_full_t_star"],
                  "old_argmin": int(np.argmin(old["mean_risk"])) == row["old_t_star"],
                  "P1": row["P1_pass"] == bool(.5 <= row["t_star"] / full_t <= 2),
                  "P2": row["P2_pass"] == bool(np.quantile(boot_ts["new"] / full_t, .975) < 2),
                  "final_risk_einsum": final_error < 1e-12, "new_first16_einsum": replay_error < 1e-12,
                  "independent_bootstrap": bool(boot_ok)}
        rows.append({"seed": seed, "checks": checks, "final_risk_max_error": final_error, "replay_max_error": replay_error})
    result = {"all_ok": unchanged and all(all(r["checks"].values()) for r in rows),
              "old_files": len(audit["old_files"]), "old_hash_and_mtime_unchanged": unchanged, "rows": rows}
    save(STUDY / "executed/verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_ok"]:
        raise RuntimeError("Verification failed")


if __name__ == "__main__":
    main()
