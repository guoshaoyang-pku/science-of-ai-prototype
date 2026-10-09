#!/usr/bin/env python3
"""独立 sample-space 检查保存参数；不重新训练。"""
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from run import STUDY, ROOT, contract, sha, save
import numpy as np


def main():
    config, pins = contract()
    receipt = json.loads((STUDY / "results/receipt.json").read_text())
    assert receipt["pins"] == pins
    theta_error = loss_error = update_error = 0.0
    checkpoints = updates = 0
    starts = []
    for name, digest in receipt["cells"].items():
        meta_path = STUDY / f"results/{name}.json"
        meta = json.loads(meta_path.read_text())
        assert sha(meta_path) == digest and meta["pins"] == pins
        npz = STUDY / f"results/{name}.npz"
        assert sha(npz) == meta["arrays_sha256"]
        starts.append(meta["started_at"])
        with np.load(npz) as z:
            a = {key: z[key] for key in z.files}
        d, n, eta = meta["cell"]["dimension"], config["sample_count"], config["optimizer"]["eta"]
        assert meta["cell"]["target"] == "quarter_tail" and meta["status"] == "completed"
        h = math.fsum(1.0 / i for i in range(1, d + 1))
        assert np.max(np.abs(a["eigenvalues"] - 1 / (np.arange(1, d + 1) * h))) < 1e-14
        weights = {d // 4 - 1: 1 / 3, d - 1: 2 / 3}
        def relative_loss(step):
            return math.fsum(w * math.exp(2 * step * math.log1p(-eta / ((i + 1) * h))) for i, w in weights.items())
        lo, hi = 0, math.ceil(5 * d * h)
        while lo < hi:
            mid = (lo + hi) // 2
            if relative_loss(mid) <= config["threshold"]:
                hi = mid
            else:
                lo = mid + 1
        hit = meta["values"]["measured_step"]
        assert hit == lo and relative_loss(hit - 1) > .01 >= relative_loss(hit)
        hits = np.flatnonzero(a["loss"] / a["loss"][0] <= .01)
        assert len(hits) > 0 and int(hits[0]) == hit
        lookup = {int(t): theta for t, theta in zip(a["checkpoint_steps"], a["theta_checkpoints"])}
        for step, theta in lookup.items():
            residual = a["feature_scale"] * a["signs"] * theta[a["permutation"]] - a["target"]
            measured_loss = float(np.sum(residual * residual) / (2 * n))
            loss_error = max(loss_error, abs(measured_loss - a["loss"][step]))
            expected = np.zeros(d)
            for i in weights:
                value = -math.expm1(step * math.log1p(-eta * a["eigenvalues"][i]))
                expected[a["permutation"][i]] = a["target"][i] / (a["feature_scale"][i] * a["signs"][i]) * value
            theta_error = max(theta_error, float(np.max(np.abs(theta - expected))))
            if step + 1 in lookup:
                gradient = np.zeros(d)
                gradient[a["permutation"]] = a["feature_scale"] * a["signs"] * residual / n
                update_error = max(update_error, float(np.max(np.abs(theta - eta * gradient - lookup[step + 1]))))
                updates += 1
            checkpoints += 1
    assert theta_error <= 1e-7 and loss_error <= 1e-10 and update_error <= 1e-10
    commit_time = subprocess.check_output(["git", "show", "-s", "--format=%cI", pins["git_commit"]], cwd=ROOT, text=True).strip()
    from datetime import datetime
    lead = min((datetime.fromisoformat(t) - datetime.fromisoformat(commit_time)).total_seconds() for t in starts)
    assert lead > 0
    value = {"cells": len(receipt["cells"]), "checkpoints": checkpoints, "adjacent_updates": updates,
             "theta_maxabs": theta_error, "loss_maxabs": loss_error, "update_maxabs": update_error,
             "passed": True, "preregistration_commit": pins["git_commit"], "minimum_commit_lead_seconds": lead,
             "receipt_sha256": sha(STUDY / "results/receipt.json"), "method": "独立sample-space残差、梯度与math.fsum谱阈值；0新训练"}
    filename = "independent_verification.json" if len(receipt["cells"]) == 33 else "first_cell_verification.json"
    save(STUDY / "executed" / filename, value)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
