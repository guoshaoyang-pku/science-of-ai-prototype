"""Compare trainable parameter groups against saved A02 fixed-head cells."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from experiment import save, sha


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    error = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean - error, mean + error]}


def main():
    config = json.loads((STUDY / "preregistration.json").read_text())
    index, sources, failed, high_loss = {}, [], [], 0
    for folder in [ROOT / "studies/A02_ood_residual", STUDY]:
        for path in sorted((folder / "results").glob("*.json")):
            row = json.loads(path.read_text())
            if row["recipe_label"] != "LN010_w192" or row["recipe"]["optimizer"] != "SGD":
                continue
            if row["intervention"] not in ["frozen_head", *config["interventions"]]:
                continue
            assert sha(path.with_suffix(".npz")) == row["arrays_sha256"]
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
            if row["failed"]:
                failed.append(row["id"])
                continue
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                residual = arrays["predictions"].astype(float) - arrays["targets"].astype(float)
                centered = float(np.mean((residual - residual.mean()) ** 2))
            index[(row["dataset"], row["intervention"], row["mean"], row["seed"])]=centered
            if folder == STUDY and row["test_mse"] > 2:
                high_loss += 1
    if failed:
        save(STUDY / "analysis.json", {"failed_cells": failed, "source_files": sources, "prediction_scoring": "incomplete; failures retained"})
        raise RuntimeError("Prediction scoring is incomplete; failed cells retained")
    chords, checks = [], []
    rule = dict(zip(config["interventions"], ["LN_forward_sufficient", "weights_sufficient", "affine_not_sufficient"]))
    for function in config["functions"]:
        values = {}
        for mode in ["frozen_head", *config["interventions"]]:
            chord = np.array([(index[(function,mode,-3,s)] + index[(function,mode,3,s)])/2 - index[(function,mode,0,s)] for s in config["seeds"]])
            values[mode]=chord
            chords.append({"function": function, "mode": mode, "centered_chord": interval(chord), "negative_seeds": int((chord < 0).sum())})
        baseline = values["frozen_head"]
        assert baseline.mean() < 0
        for mode, identifier in rule.items():
            relative = values[mode] / baseline
            ratio = float(values[mode].mean()/baseline.mean())
            success = ratio < .5 if identifier == "affine_not_sufficient" else ratio >= .5
            checks.append({"id": identifier, "function": function, "retained_mean_benefit_fraction": ratio, "paired_seed_fraction": interval(relative), "pass": bool(success)})
    predictions = {name: {"passed": sum(c["pass"] for c in checks if c["id"] == name), "total": sum(c["id"] == name for c in checks)} for name in rule.values()}
    result = {"new_runs": 180, "reused_runs": 60, "functions": 4, "failed_cells": failed, "finite_cutoff_gt2_retained": high_loss, "chords": chords, "checks": checks, "predictions": predictions, "source_files": sources, "domain": "development after A02", "solver_evaluation": "not_run"}
    save(STUDY / "analysis.json", result)
    print(json.dumps({"new_runs": 180, "predictions": predictions, "checks": checks, "finite_cutoff_gt2_retained": high_loss}))


if __name__ == "__main__":
    main()
