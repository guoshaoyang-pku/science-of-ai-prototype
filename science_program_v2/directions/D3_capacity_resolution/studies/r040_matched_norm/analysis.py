#!/usr/bin/env python3
"""Recompute registered endpoints and audits from saved arrays only."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hit(curve, threshold):
    hits = np.flatnonzero(curve <= threshold)
    return int(hits[0]) if len(hits) else None


def bounds(values):
    return [min(values), max(values)] if values else None


def analyze():
    config = json.loads((STUDY / "preregistration.json").read_text())
    prior = json.loads((STUDY / "executed/previous_evidence_audit.json").read_text())
    unchanged = all(sha(ROOT / p) == digest and (ROOT / p).stat().st_mtime_ns == prior["immutable_mtime_ns"][p]
                    for p, digest in prior["immutable_sha256"].items())
    if not unchanged:
        raise RuntimeError("Previous successful evidence changed")
    receipt_path = STUDY / "results/receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cells": {}, "compute_seconds": 0}
    rows, loaded = [], {}
    n = config["data_contract"]["sample_count"]
    eta, steps = config["optimizer"]["eta"], config["optimizer"]["steps"]
    expected_lam = np.arange(1, n + 1, dtype=float) ** -2
    grid = np.arange(steps + 1)
    for seed in config["seeds"]:
        for target in config["targets"]:
            name = f"seed{seed}_{target}"
            if name not in receipt["cells"]:
                continue
            path = STUDY / f"results/{name}.json"
            meta = json.loads(path.read_text())
            if sha(path) != receipt["cells"][name] or sha(path.with_suffix(".npz")) != meta["arrays_sha256"]:
                raise RuntimeError(f"Saved hash mismatch: {name}")
            if meta["pins"] != receipt["pins"] or meta["cell"] != {"seed": seed, "target": target}:
                raise RuntimeError(f"Saved request mismatch: {name}")
            request = hashlib.sha256(json.dumps({"cell": meta["cell"], "pins": meta["pins"]}, sort_keys=True).encode()).hexdigest()
            if request != meta["request_sha256"]:
                raise RuntimeError(f"Request hash mismatch: {name}")
            for p, digest in meta["pins"]["sha256"].items():
                if sha(STUDY / p) != digest:
                    raise RuntimeError(f"Pin mismatch: {p}")
            with np.load(path.with_suffix(".npz")) as archive:
                a = {key: archive[key] for key in archive.files}
            if not all(np.isfinite(x).all() for x in a.values()):
                raise RuntimeError(f"Nonfinite arrays: {name}")
            x, y, theta, loss = a["features"], a["target"], a["theta_history"], a["loss"]
            gram = np.einsum("ij,kj->ik", x, x) / n
            lam, vectors = np.linalg.eigh(gram)
            weights = np.einsum("ij,i->j", vectors, y) ** 2 / np.sum(y ** 2)
            curve = np.sum(weights[:, None] * (1 - eta * lam[:, None]) ** (2 * grid), axis=0)
            residual = np.einsum("ij,tj->ti", x, theta) - y
            recomputed_loss = np.sum(residual ** 2, axis=1) / (2 * n)
            gradient = np.einsum("ij,ti->tj", x, residual[:-1]) / n
            update_error = float(np.max(np.abs(theta[1:] - theta[:-1] + eta * gradient)))
            expected_weights = np.array([float(Fraction(w)) for w in config["targets"][target]])
            expected_curve = np.sum(expected_weights[:, None] * (1 - eta * expected_lam[:, None]) ** (2 * grid), axis=0)
            m1 = float(np.sum(weights * lam))
            m2 = float(np.sum(weights * lam ** 2))
            m3 = float(np.sum(weights * lam ** 3))
            gradient0 = -np.einsum("ij,i->j", x, y) / n
            positive = lam / np.sum(lam)
            projected_y = np.einsum("ij,i->j", vectors, y)
            inverse_y = np.einsum("ij,j->i", vectors, projected_y / lam)
            theta_star = np.einsum("ij,i->j", x, inverse_y) / n
            m_minus1 = float(np.sum(weights / lam))
            descriptors = {"parameter_count": n, "trace": float(np.sum(lam)),
                           "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
                           "L0": float(loss[0]), "m1": m1, "m2": m2,
                           "gradient0_norm_squared": float(np.sum(gradient0 ** 2)),
                           "m_minus1": m_minus1, "theta_star_norm_squared": float(np.sum(theta_star ** 2)),
                           "loss1": float(loss[1]), "loss_drop1": float(loss[0] - loss[1])}
            audit_errors = {
                "gram_maxabs": float(np.max(np.abs(gram - np.diag(expected_lam)))),
                "target_weights_maxabs": float(np.max(np.abs(y ** 2 / np.sum(y ** 2) - expected_weights))),
                "saved_modal_weights_maxabs": float(np.max(np.abs(a["modal_weights"] - expected_weights))),
                "saved_eigenvalues_maxabs": float(np.max(np.abs(a["eigenvalues"] - expected_lam))),
                "initial_theta_maxabs": float(np.max(np.abs(theta[0]))),
                "loss_recompute_maxabs": float(np.max(np.abs(recomputed_loss - loss))),
                "eigh_spectral_curve_maxabs": float(np.max(np.abs(loss / loss[0] - curve))),
                "registered_spectral_curve_maxabs": float(np.max(np.abs(loss / loss[0] - expected_curve))),
                "update_maxabs": update_error,
                "gradient_identity_abs": abs(descriptors["gradient0_norm_squared"] - 2 * loss[0] * m1),
                "loss_drop_identity_abs": abs(descriptors["loss_drop1"] - loss[0] * (2 * eta * m1 - eta ** 2 * m2)),
                "theta_star_residual_maxabs": float(np.max(np.abs(np.einsum("ij,j->i", x, theta_star) - y))),
                "theta_star_norm_identity_abs": abs(descriptors["theta_star_norm_squared"] - 2 * loss[0] * m_minus1),
            }
            observed_hit = hit(loss / loss[0], config["threshold"])
            predicted_hit = hit(curve, config["threshold"])
            registered_hit = hit(expected_curve, config["threshold"])
            row = {"name": name, "seed": seed, "target": target, "descriptors": descriptors,
                   "m3_unmatched": m3, "slowest_mode_energy_unmatched": float(expected_weights[-1]),
                   "observed_threshold_step": observed_hit, "spectral_threshold_step": predicted_hit,
                   "registered_threshold_step": registered_hit, "censored": observed_hit is None,
                   "audit_errors": audit_errors, "preregistration_commit": meta["pins"]["preregistration_commit"]}
            row["audit_passed"] = bool(max(audit_errors.values()) <= config["tolerances"]["execution_absolute"]
                                        and observed_hit == predicted_hit == registered_hit == meta["measured_threshold_step"]
                                        and observed_hit is not None)
            rows.append(row)
            loaded[(seed, target)] = (row, a)
    expected_names = {f"seed{s}_{t}" for s in config["seeds"] for t in config["targets"]}
    if set(receipt["cells"]) - expected_names or len(rows) != len(receipt["cells"]):
        raise RuntimeError("Unexpected receipt cells")
    pairs = []
    for seed in config["seeds"]:
        if (seed, "fast_pair") not in loaded or (seed, "slow_pair") not in loaded:
            continue
        fast, fa = loaded[(seed, "fast_pair")]
        slow, sa = loaded[(seed, "slow_pair")]
        differences = {k: abs(fast["descriptors"][k] - slow["descriptors"][k]) for k in fast["descriptors"]}
        tfast, tslow = fast["observed_threshold_step"], slow["observed_threshold_step"]
        pairs.append({"seed": seed, "same_X_exact": bool(np.array_equal(fa["features"], sa["features"])),
                      "descriptor_abs_differences": differences,
                      "matched": bool(max(differences.values()) <= config["tolerances"]["matching_absolute"]
                                       and np.array_equal(fa["features"], sa["features"])),
                      "fast_step": tfast, "slow_step": tslow,
                      "step_difference": tslow - tfast if tfast is not None and tslow is not None else None,
                      "step_ratio": tslow / tfast if tfast and tslow is not None else None})
    complete = len(rows) == config["counts"]["planned_cells"] and len(pairs) == len(config["seeds"])
    p1 = complete and all(r["audit_passed"] for r in rows) and all(p["matched"] for p in pairs)
    p2 = complete and all(p["step_ratio"] is not None and p["step_ratio"] >= config["minimum_step_ratio"] for p in pairs)
    p3 = complete and all(p["fast_step"] == config["analytic_reference"]["predicted_fast_step"] and p["slow_step"] == config["analytic_reference"]["predicted_slow_step"] for p in pairs)
    predictions = {k: {"status": ("supported" if passed else "refuted") if complete else "not_evaluated"}
                   for k, passed in [("P1", p1), ("P2", p2), ("P3", p3)]}
    summary = {"study": config["study"], "round": config["round"], "direction_round": config["direction_round"],
               "direction": config["direction"], "domain": config["domain"],
               "saved_cells": len(rows), "planned_cells": config["counts"]["planned_cells"],
               "same_seed_pairs": len(pairs), "compute_seconds": receipt["compute_seconds"],
               "previous_evidence_unchanged": unchanged, "preregistration_sha256": sha(STUDY / "preregistration.json"),
               "rows": rows, "pairs": pairs, "predictions": predictions,
               "ranges": {"fast_steps": bounds([p["fast_step"] for p in pairs if p["fast_step"] is not None]),
                          "slow_steps": bounds([p["slow_step"] for p in pairs if p["slow_step"] is not None]),
                          "step_difference": bounds([p["step_difference"] for p in pairs if p["step_difference"] is not None]),
                          "step_ratio": bounds([p["step_ratio"] for p in pairs if p["step_ratio"] is not None]),
                          "maximum_paired_descriptor_difference": max((max(p["descriptor_abs_differences"].values()) for p in pairs), default=None),
                          "maximum_spectral_curve_error": max((r["audit_errors"]["eigh_spectral_curve_maxabs"] for r in rows), default=None)},
               "counting": config["counting"], "boundaries": config["boundaries"]}
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["saved_cells", "predictions", "ranges"]}, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
