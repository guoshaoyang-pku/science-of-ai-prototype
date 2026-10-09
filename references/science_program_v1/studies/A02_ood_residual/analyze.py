"""Score preregistered new-function predictions, including failed transfers."""
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


def metrics(a):
    residual = a["predictions"].astype(float) - a["targets"].astype(float)
    return {"mse": float(np.mean(residual ** 2)), "bias_error": float(residual.mean() ** 2),
            "centered_error": float(np.mean((residual - residual.mean()) ** 2))}


def interval(a):
    a = np.asarray(a, float)
    mean = float(a.mean())
    error = float(t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {"mean": mean, "ci95": [mean-error, mean+error]}


def main():
    rows = [json.loads(p.read_text()) for p in sorted((STUDY / "results").glob("*.json"))]
    assert len(rows) == 960
    prereg = json.loads((STUDY / "preregistration.json").read_text())
    seal = json.loads((STUDY / "seal.json").read_text())
    assert min(r["saved_at"] for r in rows) > seal["sealed_at"]
    index, arrays, failed = {}, {}, []
    for row in rows:
        key = (row["dataset"],row["recipe_label"],row["mean"],row["intervention"],row["recipe"]["optimizer"],row["seed"])
        p = STUDY / "results" / (row["id"] + ".npz")
        assert sha(p) == row["arrays_sha256"]
        with np.load(p) as a:
            values = metrics(a)
            arrays[key] = a["predictions"].copy()
        assert abs(values["mse"]-row["test_mse"]) < 1e-4
        index[key] = values
        if row["failed"]: failed.append(row["id"])
    chords, comparisons, checks = [], [], []
    seeds = prereg["seeds"]
    for function in prereg["functions"]:
        for recipe in prereg["recipes"]:
            for mode in prereg["interventions"]:
                for optimizer in prereg["optimizers"]:
                    def values(mean, key):
                        return np.array([index[(function,recipe,mean,mode,optimizer,s)][key] for s in seeds])
                    centered = (values(-3,"centered_error")+values(3,"centered_error"))/2-values(0,"centered_error")
                    total = (values(-3,"mse")+values(3,"mse"))/2-values(0,"mse")
                    chords.append({"function":function,"recipe":recipe,"mode":mode,"optimizer":optimizer,
                                   "centered_chord":interval(centered),"total_chord":interval(total),
                                   "negative_seeds":int((centered<0).sum())})
                    if mode == "frozen_head" and optimizer == "SGD" and recipe == "LN010_w192":
                        checks.append({"id":"fixed_head_benefit_transfer","function":function,
                                       "centered_chord":float(centered.mean()),"pass":bool(centered.mean()<0)})
                    if mode == "matched_bias":
                        diff = max(abs(index[(function,recipe,m,mode,optimizer,s)]["mse"]-index[(function,recipe,0,mode,optimizer,s)]["mse"]) for m in [-3,3] for s in seeds)
                        checks.append({"id":"translation_transfer","function":function,"recipe":recipe,"optimizer":optimizer,
                                       "max_mse_change":diff,"pass":diff<=.001})
                    if mode == "frozen_hidden" and optimizer == "SGD":
                        affine = max(float(np.max(np.abs(arrays[(function,recipe,-3,mode,optimizer,s)]+arrays[(function,recipe,3,mode,optimizer,s)]-2*arrays[(function,recipe,0,mode,optimizer,s)]))) for s in seeds)
                        checks.append({"id":"fixed_features_transfer","function":function,"recipe":recipe,
                                       "max_affine_second_difference":affine,"min_total_chord":float(total.min()),
                                       "min_centered_chord":float(centered.min()),
                                       "pass":bool(affine<=1e-4 and min(total.min(),centered.min())>=-1e-5)})
            for mean in prereg["means"]:
                delta = np.array([index[(function,recipe,mean,"baseline","SGD",s)]["mse"]-index[(function,recipe,mean,"baseline","Adam",s)]["mse"] for s in seeds])
                comparisons.append({"function":function,"recipe":recipe,"mean":mean,"sgd_minus_adam":interval(delta)})
                if recipe == "LN010_w192":
                    checks.append({"id":"winner_transfer","function":function,"mean":mean,
                                   "measured":float(delta.mean()),"pass":bool(delta.mean()>0 if mean==0 else delta.mean()<0)})
        narrow = next(c for c in chords if c["function"]==function and c["recipe"]=="noLN_w64" and c["mode"]=="frozen_head" and c["optimizer"]=="SGD")["centered_chord"]["mean"]
        wide = next(c for c in chords if c["function"]==function and c["recipe"]=="LN010_w192" and c["mode"]=="frozen_head" and c["optimizer"]=="SGD")["centered_chord"]["mean"]
        checks.append({"id":"LN_boundary","function":function,"noLN_chord":narrow,"LN_chord":wide,
                       "pass":bool(narrow>.5*wide)})
    summary = {name:{"passed":sum(c["pass"] for c in checks if c["id"]==name),
                     "total":sum(c["id"]==name for c in checks)} for name in sorted(set(c["id"] for c in checks))}
    analysis = {"runs":len(rows),"function_units":4,"recipe_conditions":8,"seeds_per_condition":5,
                "seconds":sum(r["seconds"] for r in rows),"predictions":summary,"checks":checks,
                "chords":chords,"comparisons":comparisons,"failed_runs":failed,"cutoff_gt2_retained":sum(r["test_mse"]>2 for r in rows),
                "seal":seal,"first_result_saved_at":min(r["saved_at"] for r in rows),"solver_score":None}
    save(STUDY / "analysis.json",analysis)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    colors={"LN010_w192":"#0072B2","noLN_w64":"#D55E00"}
    for recipe,color in colors.items():
        cs=[next(c for c in chords if c["function"]==f and c["recipe"]==recipe and c["mode"]=="frozen_head" and c["optimizer"]=="SGD") for f in prereg["functions"]]
        y=[c["centered_chord"]["mean"] for c in cs]
        axes[0].plot(range(4),y,"o-",color=color,label=recipe)
    axes[0].axhline(0,color="black",linewidth=.6);axes[0].set_xticks(range(4),prereg["functions"],rotation=25,ha="right");axes[0].set_ylabel("Frozen-head centered chord (SGD)");axes[0].legend(fontsize=8)
    for i,function in enumerate(prereg["functions"]):
        cs=[next(c for c in comparisons if c["function"]==function and c["recipe"]=="LN010_w192" and c["mean"]==m) for m in [-3,0,3]]
        axes[1].plot([-3,0,3],[c["sgd_minus_adam"]["mean"] for c in cs],"o-",label=function)
    axes[1].axhline(0,color="black",linewidth=.6);axes[1].set_xlabel("Target mean");axes[1].set_ylabel("SGD MSE minus Adam MSE");axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(STUDY/"ood_predictions.png",dpi=220);fig.savefig(STUDY/"ood_predictions.pdf")
    print(json.dumps({"runs":len(rows),"predictions":summary,"failed":len(failed)}))


if __name__ == "__main__":
    main()
