#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    seal = json.loads((HERE / "forecast_seal.json").read_text())
    pairs = []
    for path in sorted((HERE / "results").glob("*.json")):
        row = json.loads(path.read_text())
        forecastpath = HERE / "forecasts" / path.name
        if hashlib.sha256(forecastpath.read_bytes()).hexdigest() != seal["forecasts"][str(forecastpath.relative_to(HERE))]:
            raise ValueError("Forecast was changed after sealing")
        if hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != row["arrays_sha256"]:
            raise ValueError("C05 arrays mismatch")
        forecast = json.loads(forecastpath.read_text())
        if row["contract_sha256"] != forecast["contract_sha256"]:
            raise ValueError("C05 prediction/result mismatch")
        pairs.append((row, forecast))
    if len(pairs) != 24:
        raise ValueError("All24recipes required")
    errors, checks, steps, groups = [], [], [], []
    for row, forecast in pairs:
        errors.append(max(abs(a[k] - b[q]) for a, b in zip(row["records"], forecast["records"])
                          for k, q in [("train_signal_mse", "forecast_train_signal_mse"), ("test_signal_mse", "forecast_test_signal_mse")]))
        best = min(forecast["records"], key=lambda p:p["forecast_expected_test_risk"])
        mc_best = min(row["records"], key=lambda p:p["mean_test_risk"])
        step_error = abs(np.log2(max(1, best["step"]) / max(1, mc_best["step"])))
        steps.append(float(step_error))
        selected = []
        for step in [8192, best["step"]]:
            actual = next(r for r in row["records"] if r["step"] == step)
            predicted = next(r for r in forecast["records"] if r["step"] == step)
            delta = actual["mean_test_risk"] - predicted["forecast_expected_test_risk"]
            passed = abs(delta) <= 3 * actual["se_mean_test_risk"] + 1e-8
            checks.append(passed)
            selected.append({"step": step, "actual_minus_forecast_risk": delta, "MC_SE": actual["se_mean_test_risk"], "within3SE": passed})
        groups.append({"cell": row["cell"], "predicted_best_step": best["step"], "MC_best_step": mc_best["step"], "log2step_error": float(step_error),
                       "final_expected_risk": forecast["records"][-1]["forecast_expected_test_risk"], "final_MC_mean_risk": row["records"][-1]["mean_test_risk"],
                       "final_expected_noise_variance": forecast["records"][-1]["forecast_noise_variance"], "checked_forecast": selected})
    lookup = {(r["cell"]["seed"], r["cell"]["scale"], r["cell"]["activation"], r["cell"]["noise_sd"]):(r, f) for r, f in pairs}
    earlier = []
    variances = []
    for seed in [511, 512, 513]:
        for scale in [.015, .15]:
            for activation in ["relu", "silu"]:
                low = lookup[seed, scale, activation, .7][1]["forecast_best_step"]
                high = lookup[seed, scale, activation, 1.4][1]["forecast_best_step"]
                earlier.append(high <= low)
        for sigma in [.7, 1.4]:
            variances.append(lookup[seed, .015, "silu", sigma][1]["records"][-1]["forecast_noise_variance"] <
                             lookup[seed, .015, "relu", sigma][1]["records"][-1]["forecast_noise_variance"])
    verdicts = {"R1":{"pass":max(errors)<1e-8,"max_signal_train_test_curve_error":max(errors)},
                "R2":{"pass":sum(checks)/len(checks)>=.9,"checks_within3_MC_SE":sum(checks),"checks":len(checks)},
                "R3":{"pass":bool(max(r["max_noisy_train_increase"] for r,f in pairs)<1e-10 and np.median(steps)<=2),
                      "max_train_increase":max(r["max_noisy_train_increase"] for r,f in pairs),"median_log2step_error":float(np.median(steps)),
                      "exact_step_matches":sum(v==0 for v in steps),"within_factor2":sum(v<=1 for v in steps)},
                "R4":{"pass":all(earlier) and all(variances),"higher_noise_not_later":sum(earlier),"comparisons":len(earlier),"small_silu_lower_noise_variance":sum(variances)}}
    output = {"cells":len(pairs),"noisy_heads":len(pairs)*16,"clean_signal_heads":len(pairs),"groups":groups,"verdicts":verdicts,
              "training_seconds":sum(r["seconds"] for r,f in pairs),"solver_evaluation":"not_run",
              "forecast_seal_sha256":hashlib.sha256((HERE/"forecast_seal.json").read_bytes()).hexdigest(),
              "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+chr(10))
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.3))
    for ax,sigma in zip(axes,[.7,1.4]):
        for activation,scale,color,label in [("relu",.015,"#2463a8","ReLU"),("silu",.015,"#bc4d20","SiLU0.015"),("silu",.15,"#398352","SiLU0.15")]:
            selected=[lookup[s,scale,activation,sigma] for s in [511,512,513]]
            step=[p["step"] for p in selected[0][0]["records"]]
            actual=np.mean([[p["mean_test_risk"] for p in r["records"]] for r,f in selected],0)
            forecast=np.mean([[p["forecast_expected_test_risk"] for p in f["records"]] for r,f in selected],0)
            ax.plot(step,forecast,"-",color=color,label=label+" forecast")
            ax.plot(step,actual,"o",color=color,ms=3,label=label+" measured")
        ax.set_xscale("symlog",linthresh=1)
        ax.set(xlabel="GD steps",ylabel="Clean-test MSE",title=f"Sealed OOD; noise SD{sigma}")
    axes[0].legend(frameon=False,fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE/"risk_ood_forecast.svg")
    fig.savefig(HERE/"risk_ood_forecast.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"cells":len(pairs),"verdicts":verdicts,"seconds":output["training_seconds"]}))


if __name__=="__main__":
    main()
