"""Analyze all head mediation cells with paired baselines and retained failures."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from experiment import save, sha


def metric(prediction, target):
    residual = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {"mse": float(np.mean(residual ** 2)),
            "bias_error": float(np.mean(residual) ** 2),
            "centered_error": float(np.mean((residual - residual.mean()) ** 2))}


def interval(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    spread = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {"mean": mean, "ci95": [mean - spread, mean + spread]}


def baseline(dataset, mean):
    condition = {-3: 1, 0: 2, 3: 3}[mean]
    repo = ROOT / "sources/e16"
    saved = json.loads((repo / "results" / f"{dataset}_c{condition}.json").read_text())
    measurement = next(x for x in saved["measurement"]["results"] if x["candidate"]["optimizer"]["type"] == "SGD")
    pins = measurement["measurement_files"]
    manifest = json.loads((repo / "evidence_manifest.json").read_text())["files"]
    result = []
    for seed in range(10):
        candidates = [(k, v) for k, v in pins.items() if k.endswith(".npz") and f"seed_{seed}" in k]
        if len(candidates) != 1:
            raise ValueError(f"Expected one saved seed array: {dataset}/{condition}/{seed}: {[k for k,v in candidates]}")
        pin = candidates[0][1]
        path = repo / manifest[pin["sha256"]]["path"]
        assert sha(path) == pin["sha256"]
        with np.load(path) as arrays:
            result.append(metric(arrays["predictions"], arrays["targets"]))
    return result


def main():
    rows = [json.loads(p.read_text()) for p in sorted((STUDY / "results").glob("*.json"))]
    assert len(rows) == 360
    indexed, failures = {}, []
    for row in rows:
        file = STUDY / "results" / (row["id"] + ".npz")
        assert sha(file) == row["arrays_sha256"]
        with np.load(file) as arrays:
            values = metric(arrays["predictions"], arrays["targets"])
        assert abs(values["mse"] - row["test_mse"]) < 1e-4
        indexed[(row["dataset"], row["mean"], row["intervention"], row["recipe"]["optimizer"], row["seed"])] = values
        if row["failed"]:
            failures.append(row["id"])
    datasets = json.loads((STUDY / "preregistration.json").read_text())["datasets"]
    cells, chords, predictions = [], [], []
    for dataset in datasets:
        base = {m: baseline(dataset, m) for m in [-3, 0, 3]}
        base_chord = np.array([(base[-3][s]["centered_error"] + base[3][s]["centered_error"]) / 2 - base[0][s]["centered_error"] for s in range(10)])
        for mode in ["frozen_head", "frozen_head_weight"]:
            for opt in ["SGD", "Adam"]:
                for mean in [-3, 0, 3]:
                    values = [indexed[(dataset, mean, mode, opt, s)] for s in range(10)]
                    cells.append({"dataset": dataset, "intervention": mode, "optimizer": opt, "mean": mean,
                                  **{k: interval([v[k] for v in values]) for k in values[0]}})
                chord = np.array([(indexed[(dataset, -3, mode, opt, s)]["centered_error"] + indexed[(dataset, 3, mode, opt, s)]["centered_error"]) / 2 - indexed[(dataset, 0, mode, opt, s)]["centered_error"] for s in range(10)])
                entry = {"dataset": dataset, "intervention": mode, "optimizer": opt,
                         "centered_chord": interval(chord), "negative_seeds": int((chord < 0).sum()),
                         "baseline_centered_chord": interval(base_chord)}
                chords.append(entry)
                if mode == "frozen_head_weight" and opt == "SGD":
                    predictions.append({"id": "head_weight_needed", "dataset": dataset,
                                        "predicted": "offset benefit attenuates at least 50%",
                                        "baseline": float(base_chord.mean()), "measured": float(chord.mean()),
                                        "passed": bool(chord.mean() >= .5 * base_chord.mean())})
                if mode == "frozen_head" and opt == "SGD":
                    for mean in [-3, 3]:
                        measured = np.mean([indexed[(dataset, mean, mode, opt, s)]["bias_error"] for s in range(10)])
                        predictions.append({"id": "hidden_can_fit_constant", "dataset": dataset, "mean": mean,
                                            "measured": float(measured), "threshold": .5, "passed": bool(measured < .5)})
    analysis = {"seed_runs": len(rows), "functions": len(datasets), "cells": cells, "chords": chords,
                "prediction_checks": predictions, "failed_runs": failures,
                "seconds": sum(row["seconds"] for row in rows), "new_model_calls": 0,
                "domain": "development; no OOD or solver score evidence"}
    save(STUDY / "analysis.json", analysis)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for mode, color, marker in [("frozen_head", "#0072B2", "o"), ("frozen_head_weight", "#D55E00", "s")]:
            values = [next(c for c in cells if c["dataset"] == dataset and c["intervention"] == mode and c["optimizer"] == "SGD" and c["mean"] == m) for m in [-3,0,3]]
            ys = [v["centered_error"]["mean"] for v in values]
            error = [v["centered_error"]["ci95"][1] - v["centered_error"]["mean"] for v in values]
            ax.errorbar([-3,0,3], ys, yerr=error, color=color, marker=marker, label=mode.replace("_", " "))
        saved = {m: baseline(dataset,m) for m in [-3,0,3]}
        ax.plot([-3,0,3], [np.mean([v["centered_error"] for v in saved[m]]) for m in [-3,0,3]], "k--", label="full network (saved)")
        ax.set_title(dataset.removeprefix("mvar_"));ax.set_xlabel("Target mean")
    axes[0].set_ylabel("Centered test MSE (SGD)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STUDY / "head_mediation.png", dpi=220)
    fig.savefig(STUDY / "head_mediation.pdf")
    print(json.dumps({"runs": len(rows), "seconds": analysis["seconds"], "predictions": predictions}))


if __name__ == "__main__":
    main()
