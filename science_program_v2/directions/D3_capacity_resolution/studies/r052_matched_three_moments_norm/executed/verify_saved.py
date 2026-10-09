#!/usr/bin/env python3
"""Independent scalar arithmetic on saved results; no training."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
config = json.loads((STUDY / "preregistration.json").read_text())
receipt = json.loads((STUDY / "results/receipt.json").read_text())
rows = []
for name, digest in receipt["cells"].items():
    path = STUDY / f"results/{name}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    meta = json.loads(path.read_text())
    assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == meta["arrays_sha256"]
    with np.load(path.with_suffix(".npz")) as archive:
        x, y, theta, losses = [archive[k].tolist() for k in ["features", "target", "theta_history", "loss"]]
    n, eta = len(y), config["optimizer"]["eta"]
    max_loss_error, max_update_error = 0., 0.
    scalar_losses = []
    for t, parameter in enumerate(theta):
        residual = [math.fsum(x[i][j] * parameter[j] for j in range(n)) - y[i] for i in range(n)]
        loss = math.fsum(r * r for r in residual) / (2 * n)
        scalar_losses.append(loss)
        max_loss_error = max(max_loss_error, abs(loss - losses[t]))
        if t + 1 < len(theta):
            for j in range(n):
                gradient = math.fsum(x[i][j] * residual[i] for i in range(n)) / n
                error = abs(theta[t + 1][j] - parameter[j] + eta * gradient)
                max_update_error = max(max_update_error, error)
    hit = next((t for t, loss in enumerate(scalar_losses) if loss / scalar_losses[0] <= config["threshold"]), None)
    gram_error = max(abs(math.fsum(x[i][j] * x[k][j] for j in range(n)) / n
                         - (1 / (i + 1) ** 2 if i == k else 0)) for i in range(n) for k in range(n))
    interpolant = [math.fsum(x[i][j] * y[i] * (i + 1) ** 2 for i in range(n)) / n for j in range(n)]
    norm_squared = math.fsum(a * a for a in interpolant)
    inverse_moment = math.fsum(y[i] * y[i] * (i + 1) ** 2 for i in range(n)) / n
    interpolation_error = max(abs(math.fsum(x[i][j] * interpolant[j] for j in range(n)) - y[i]) for i in range(n))
    row = {"name": name, "scalar_threshold_step": hit, "metadata_threshold_step": meta["measured_threshold_step"],
           "scalar_loss_drop1": scalar_losses[0] - scalar_losses[1],
           "scalar_loss_maxabs": max_loss_error, "scalar_update_maxabs": max_update_error,
           "scalar_gram_maxabs": gram_error,
           "scalar_theta_star_norm_squared": norm_squared,
           "scalar_norm_identity_abs": abs(norm_squared - inverse_moment),
           "scalar_interpolation_maxabs": interpolation_error,
           "passed": hit == meta["measured_threshold_step"] and max(max_loss_error, max_update_error, gram_error, abs(norm_squared - inverse_moment), interpolation_error) <= 1e-10}
    rows.append(row)
record = {"kind": "posttraining_scalar_saved_evidence_audit", "training_ran": False,
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "completed_cells": len(rows), "all_passed": len(rows) == config["counts"]["planned_cells"] and all(r["passed"] for r in rows),
          "maximum_scalar_loss_error": max(r["scalar_loss_maxabs"] for r in rows),
          "maximum_scalar_update_error": max(r["scalar_update_maxabs"] for r in rows),
          "maximum_scalar_gram_error": max(r["scalar_gram_maxabs"] for r in rows),
          "maximum_scalar_norm_identity_error": max(r["scalar_norm_identity_abs"] for r in rows), "rows": rows}
(STUDY / "executed/independent_verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "rows"}))
assert record["all_passed"]
