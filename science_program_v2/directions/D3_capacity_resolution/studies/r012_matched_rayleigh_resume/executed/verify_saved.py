#!/usr/bin/env python3
"""Audit saved data with scalar arithmetic; never update parameters."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
ORIGINAL = ROOT / "directions/D3_capacity_resolution/studies/r004_matched_rayleigh"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dot(left, right):
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    start = json.loads((STUDY / "executed/experiment_start.json").read_text())
    receipt = json.loads((ORIGINAL / "results/receipt.json").read_text())
    summary = json.loads((STUDY / "summary.json").read_text())
    commit = start["preregistration_commit"]
    pins = {name: sha(ORIGINAL / name) for name in ("preregistration.json", "executed/run.py", "analysis.py")}
    for relative, expected in start["frozen_paths_sha256"].items():
        path = ROOT / relative
        assert sha(path) == expected
        assert subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) == path.read_bytes()
    expected_labels = {f"p{p}_seed{s}_{t}" for p in config["spectral_exponents"]
                       for s in config["seeds"] for t in config["targets"]}
    assert set(receipt["cells"]) == expected_labels
    assert {p.stem for p in (ORIGINAL / "results").glob("p*.npz")} == expected_labels
    entries = []
    for p in config["spectral_exponents"]:
        for seed in config["seeds"]:
            paired_features = []
            paired_gradient_norms = []
            for target in config["targets"]:
                label = f"p{p}_seed{seed}_{target}"
                path = ORIGINAL / "results" / f"{label}.json"
                row = json.loads(path.read_text())
                cell = {"exponent": p, "seed": seed, "target": target}
                assert row["cell"] == cell and row["status"] == "completed"
                assert row["pins"] == {"git_commit": commit, "sha256": pins}
                assert row["request_sha256"] == hashlib.sha256(json.dumps({"cell": cell, "pins": pins}, sort_keys=True).encode()).hexdigest()
                assert receipt["cells"][label] == sha(path)
                assert row["arrays_sha256"] == sha(path.with_suffix(".npz"))
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                    assert all(np.isfinite(data[key]).all() for key in data.files)
                    x, r, y = data["features"], data["rotation"], data["target"]
                    assert x.shape == r.shape == (9, 9) and y.shape == (9,)
                    assert x.dtype == np.dtype("float64")
                    paired_features.append(x.copy())
                    eigenvalues = [i ** (-p) for i in range(1, 10)]
                    weights = [0.0] * 9
                    if target == "middle":
                        weights[2] = 1.0
                    else:
                        weights[0] = .5 if p == 0 else (eigenvalues[2] - eigenvalues[8]) / (eigenvalues[0] - eigenvalues[8])
                        weights[8] = 1.0 - weights[0]
                    expected_rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(9, 9)))
                    rotation_error = float(np.max(np.abs(r - expected_rotation)))
                    orthogonality_error = max(abs(dot(r[i], r[j]) - float(i == j)) for i in range(9) for j in range(9))
                    structure_error = max(abs(float(x[i, j]) - math.sqrt(9 * eigenvalues[i]) * float(r[j, i])) for i in range(9) for j in range(9))
                    gram_error = max(abs(dot(x[i], x[j]) / 9 - (eigenvalues[i] if i == j else 0.0)) for i in range(9) for j in range(9))
                    target_error = max(abs(float(y[i]) - math.sqrt(9 * weights[i])) for i in range(9))
                    assert max(rotation_error, orthogonality_error, structure_error, gram_error, target_error) <= 1e-12
                    assert max(abs(float(data["eigenvalues"][i]) - eigenvalues[i]) for i in range(9)) <= 1e-12
                    assert max(abs(float(data["modal_weights"][i]) - weights[i]) for i in range(9)) <= 1e-12
                    assert np.count_nonzero(data["initial_theta"]) == 0
                    reference = [math.fsum(weights[i] * (1 - .5 * eigenvalues[i]) ** (2 * t) for i in range(9)) for t in range(1001)]
                    assert data["normalized_loss"].shape == (1001,)
                    curve_error = max(abs(float(a) - b) for a, b in zip(data["normalized_loss"], reference))
                    predicted_error = max(abs(float(a) - b) for a, b in zip(data["predicted_curve"], reference))
                    actual = next((t for t, value in enumerate(data["normalized_loss"]) if value <= .01), None)
                    predicted = next((t for t, value in enumerate(reference) if value <= .01), None)
                    residual = [dot(x[i], data["final_theta"]) - float(y[i]) for i in range(9)]
                    final_error = abs(dot(residual, residual) / dot(y, y) - float(data["normalized_loss"][-1]))
                    assert abs(dot(y, y) / 18 - .5) <= 1e-12
                    initial_gradient = [-dot(x[:, j], y) / 9 for j in range(9)]
                    initial_gradient_norm_squared = dot(initial_gradient, initial_gradient)
                    assert abs(initial_gradient_norm_squared - eigenvalues[2]) <= 1e-12
                    paired_gradient_norms.append(initial_gradient_norm_squared)
                    assert curve_error <= 1e-10 and predicted_error <= 1e-10 and final_error <= 1e-12
                    assert actual == predicted == row["predicted_threshold_step"]
                    assert actual == next(c["threshold_step"] for c in summary["cells"] if c["label"] == label)
                    entries.append({"label": label, "actual_step": actual, "scalar_predicted_step": predicted,
                                    "curve_error": curve_error, "predicted_curve_error": predicted_error,
                                    "final_residual_error": final_error, "gram_error": gram_error,
                                    "feature_structure_error": structure_error, "rotation_seed_error": rotation_error,
                                    "initial_gradient_norm_squared": initial_gradient_norm_squared})
            assert np.array_equal(paired_features[0], paired_features[1])
            assert abs(paired_gradient_norms[0] - paired_gradient_norms[1]) <= 1e-12
    result = {"checked_at": datetime.now().astimezone().isoformat(), "status": "passed",
              "method": "保存数据math.fsum标量dot/Gram/谱曲线/终点残差复算，不调用measure或更新参数；另检查注册旋转seed",
              "verified_cells": len(entries), "preregistration_commit": commit,
              "verification_source_sha256": sha(Path(__file__)),
              "maximum_curve_error": max(e["curve_error"] for e in entries),
              "maximum_gram_error": max(e["gram_error"] for e in entries),
              "cells": entries,
              "contract_wording_correction": "原合同称初始梯度范数未固定，措辞不准确：g0=-X.T@y/n，||g0||²=(||y||²/n)*Rayleigh；匹配L0和Rayleigh在本线性recipe下自动匹配初始参数梯度范数。保存数据标量复算也通过；保持原预注册不改，在报告和KB边界中纠正。目标参数范数与第一步实际loss下降仍未匹配。",
              "warning_assessment": "原matmul警告保留，根因未确定；保存数据的注册结构、finite、标量Gram、1001步损失与阈值和终点残差核验均通过，未发现结果受影响的证据。"}
    (STUDY / "executed/saved_evidence_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary["post_execution_scope_correction"] = result["contract_wording_correction"]
    summary["independent_saved_data_audit"] = {
        "evidence": "directions/D3_capacity_resolution/studies/r012_matched_rayleigh_resume/executed/saved_evidence_verification.json",
        "status": result["status"], "verified_cells": len(entries),
    }
    (STUDY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "verified_cells", "maximum_curve_error", "maximum_gram_error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
