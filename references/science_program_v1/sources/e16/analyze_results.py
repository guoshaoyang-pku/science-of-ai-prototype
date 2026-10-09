#!/usr/bin/env python3
"""Verify paired evidence and decompose MSE into offset and centered error."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
MANIFEST = json.loads((REPO / "evidence_manifest.json").read_text()) if (REPO / "evidence_manifest.json").exists() else {}


def evidence_path(pin):
    portable = MANIFEST.get("files", {}).get(pin["sha256"])
    return REPO / portable["path"] if portable else Path(pin["path"])


def interval(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 10 and np.isfinite(a).all()
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    return {"mean": mean, "se": se, "ci95": [mean - 2.262157 * se, mean + 2.262157 * se]}


def decomposition(predictions, targets):
    residual = predictions.astype(float).reshape(-1) - targets.astype(float).reshape(-1)
    bias = float(residual.mean())
    return {"mean_error": bias, "offset_mse": bias ** 2,
            "centered_mse": float(((residual - bias) ** 2).mean()),
            "mse": float((residual ** 2).mean())}


def norm(parameters, key, role):
    return float(np.sqrt(sum(p.get(key, 0) ** 2 for p in parameters if p["role"] == role)))


def main():
    spec = json.loads((REPO / "preregistration.json").read_text())
    inputs = json.loads((REPO / "inputs.json").read_text())
    paired_keys = {}
    seed_rows, conditions, traces = [], [], []
    for dataset in spec["datasets"]:
        for number, transform in enumerate(spec["conditions"]):
            key = dataset.split("/")[-1] + "_c" + str(number)
            saved = json.loads((REPO / "results" / (key + ".json")).read_text())
            per_optimizer, losses, arrays_by_seed = {}, {}, {}
            for measured in saved["measurement"]["results"]:
                optimizer = measured["candidate"]["optimizer"]["type"]
                for name, evidence in measured["measurement_files"].items():
                    content = evidence_path(evidence).read_bytes()
                    assert hashlib.sha256(content).hexdigest() == evidence["sha256"], name
                    assert len(content) == evidence["bytes"], name
                assert measured["failed_seeds"] == 0
                per_optimizer[optimizer], losses[optimizer] = [], []
                for seed in measured["seed_results"]:
                    sid = seed["seed"]
                    files = measured["measurement_files"]
                    process = json.loads(evidence_path(files[f"results/process/seed_{sid}.json"]).read_text())
                    pin_key = (dataset, sid)
                    pair = {"initial": process["initial_parameters"],
                            "stream": process["minibatch_stream_sha256"],
                            "train_x": process["inputs"]["train_x"],
                            "test_x": process["inputs"]["test_x"]}
                    assert paired_keys.setdefault(pin_key, pair) == pair, (key, sid, optimizer)
                    with np.load(evidence_path(files[f"results/process/seed_{sid}.npz"])) as data:
                        initial = decomposition(data["initial_predictions"], data["targets"])
                        final = decomposition(data["predictions"], data["targets"])
                        train = decomposition(data["train_predictions"], data["train_targets"])
                        labels = data["targets"].copy()
                        train_labels = data["train_targets"].copy()
                    label_key = (dataset, number, sid)
                    label_hashes = (hashlib.sha256(labels.tobytes()).hexdigest(),
                                    hashlib.sha256(train_labels.tobytes()).hexdigest())
                    assert arrays_by_seed.setdefault(label_key, label_hashes) == label_hashes
                    assert np.isclose(final["mse"], seed["final_test_mse"], atol=1e-6)
                    target = seed["target_transform"]
                    assert target["runner_source_sha256"] == inputs["sources"]["aiq_bench_repo/src/architecture_iq/ground_truth/runner.py"]["sha256"]
                    target_mean = target["transformed_labels"]["train_y"]["mean"]
                    if transform["center"]:
                        assert abs(target_mean - transform["offset"]) < 2e-6
                    first, last = process["steps"][0], process["steps"][-1]
                    row = {"dataset": dataset, "condition": number, "optimizer": optimizer,
                           "seed": sid, "train_target_mean": target_mean,
                           "original_train_mean": target["original_labels"]["train_y"]["mean"],
                           "applied_shift": target["applied_offset"],
                           "initial_mse": initial["mse"], "initial_mean_error": initial["mean_error"],
                           "test_mse": final["mse"], "test_mean_error": final["mean_error"],
                           "test_offset_mse": final["offset_mse"],
                           "test_centered_mse": final["centered_mse"],
                           "train_mse": train["mse"],
                           "benchmark_cutoff_exceeded": final["mse"] > 2.0}
                    for role in ("head", "hidden"):
                        row[f"step1_{role}_gradient"] = norm(first["parameters"], "data_gradient_norm", role)
                        row[f"step1_{role}_descent"] = norm(first["parameters"], "descent_norm", role)
                        row[f"terminal_{role}_displacement"] = norm(last["parameters"], "displacement_norm", role)
                    seed_rows.append(row)
                    per_optimizer[optimizer].append(row)
                    losses[optimizer].append(seed["final_test_mse"])
                    for step in process["steps"]:
                        traces.append({"dataset": dataset, "condition": number, "optimizer": optimizer,
                                       "seed": sid, "step": step["step"], **step["metrics"]})
                assert [r["seed"] for r in per_optimizer[optimizer]] == list(range(10))
                with np.load(evidence_path(files["results/curves.npz"])) as curve_file:
                    curves = curve_file["curves"]
                    assert curves.shape == (10, 256)
                    assert np.allclose(curves[:, -1], losses[optimizer], atol=1e-6)
            delta = np.array(losses["SGD"]) - np.array(losses["Adam"])
            winner = "SGD" if delta.mean() < 0 else "Adam"
            summary = {"dataset": dataset, "condition": number, "transform": transform,
                       "target_mean": per_optimizer["SGD"][0]["train_target_mean"],
                       "winner": winner, "delta": interval(delta),
                       "sgd_seed_wins": int((delta < 0).sum()),
                       "prediction": None if number == 0 else ("Adam" if transform["offset"] == 0 else "SGD")}
            for optimizer, rows in per_optimizer.items():
                summary[optimizer] = {name: float(np.mean([r[name] for r in rows]))
                                      for name in rows[0] if isinstance(rows[0][name], (float, bool))}
            for name in ("test_mse", "test_offset_mse", "test_centered_mse", "train_mse"):
                summary[name + "_delta"] = interval([s[name] - a[name] for s, a in
                                                       zip(per_optimizer["SGD"], per_optimizer["Adam"])])
            conditions.append(summary)
    curvature = []
    for dataset in spec["datasets"]:
        for optimizer in ("SGD", "Adam"):
            for metric in ("test_mse", "test_centered_mse"):
                by_condition = {number: [r[metric] for r in seed_rows if r["dataset"] == dataset
                                        and r["optimizer"] == optimizer and r["condition"] == number]
                                for number in (1, 2, 3)}
                chord_gap = (np.array(by_condition[1]) + np.array(by_condition[3])) / 2 - np.array(by_condition[2])
                curvature.append({"dataset": dataset, "optimizer": optimizer, "metric": metric,
                                  "symmetric_chord_gap": interval(chord_gap),
                                  "negative_seed_count": int((chord_gap < 0).sum())})
    result = {"origin_epoch": 16, "paired_groups_verified": len(paired_keys),
              "seed_runs": len(seed_rows), "conditions": conditions,
              "prospective_predictions": 9,
              "correct_predictions": sum(c["winner"] == c["prediction"] for c in conditions if c["prediction"]),
              "benchmark_cutoff_exceedances": sum(r["benchmark_cutoff_exceeded"] for r in seed_rows),
              "offset_chord_tests": curvature,
              "ci_scope": "paired ten training seeds conditional on fixed function and split; unadjusted t9 intervals"}
    (REPO / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, rows in (("seed_metrics.csv", seed_rows), ("process_metrics.csv", traces)):
        with (REPO / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for c in conditions:
        print(c["dataset"], c["condition"], c["winner"], c["sgd_seed_wins"],
              c["delta"], "offset", c["test_offset_mse_delta"]["mean"],
              "centered", c["test_centered_mse_delta"]["mean"])
    print("Prediction accuracy", result["correct_predictions"], "/9; cutoff exceedances", result["benchmark_cutoff_exceedances"])


if __name__ == "__main__":
    main()
