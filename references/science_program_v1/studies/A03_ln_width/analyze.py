"""Compare the completed 2x2 LN/width design, reusing A02 saved cells."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT))
from experiment import save,sha


def ci(x):
    x=np.asarray(x,float);m=float(x.mean());e=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {"mean":m,"ci95":[m-e,m+e]}


def main():
    index={};sources=[]
    for folder in [STUDY,ROOT/"studies/A02_ood_residual"]:
        for p in sorted((folder/"results").glob("*.json")):
            r=json.loads(p.read_text())
            if r["recipe"]["optimizer"]!="SGD" or r["intervention"] not in ["baseline","frozen_head"]:continue
            assert sha(p.with_suffix(".npz"))==r["arrays_sha256"]
            with np.load(p.with_suffix(".npz")) as a:
                residual=a["predictions"].astype(float)-a["targets"].astype(float)
                centered=float(np.mean((residual-residual.mean())**2))
            index[(r["dataset"],r["recipe_label"],r["mean"],r["intervention"],r["seed"])]=centered
            sources.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p)})
    functions=["trigonometric8","quadratic8","interaction12","radial12"];seeds=list(range(100,105))
    values={};chords=[];checks=[];pairings=[]
    for f in functions:
        for mode in ["baseline","frozen_head"]:
            for label in ["LN010_w64","noLN_w64","LN010_w192","noLN_w192"]:
                chord=np.array([(index[(f,label,-3,mode,s)]+index[(f,label,3,mode,s)])/2-index[(f,label,0,mode,s)] for s in seeds])
                values[(f,label,mode)]=chord;chords.append({"function":f,"recipe":label,"mode":mode,"chord":ci(chord),"negative_seeds":int((chord<0).sum())})
        for width in [64,192]:
            delta=values[(f,f"LN010_w{width}","frozen_head")]-values[(f,f"noLN_w{width}","frozen_head")]
            pairings.append({"function":f,"width":width,"LN_minus_noLN_chord":ci(delta)})
            checks.append({"id":"LN_more_than_width","function":f,"width":width,"pass":bool(delta.mean()<0)})
        narrow=values[(f,"LN010_w64","frozen_head")].mean();wide=values[(f,"LN010_w192","frozen_head")].mean();no=values[(f,"noLN_w192","frozen_head")].mean()
        checks.append({"id":"LN_suffices_w64","function":f,"chord":float(narrow),"pass":bool(narrow<0)})
        checks.append({"id":"width_noLN_insufficient","function":f,"chord":float(no),"LN_chord":float(wide),"pass":bool(no>.5*wide)})
    result={"new_runs":240,"reused_runs":240,"functions":4,"chords":chords,"same_width_LN_contrasts":pairings,"checks":checks,
            "predictions":{name:{"passed":sum(x["pass"] for x in checks if x["id"]==name),"total":sum(x["id"]==name for x in checks)} for name in sorted({x["id"] for x in checks})},
            "source_files":sources,"domain":"development after A02"}
    save(STUDY/"analysis.json",result)
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
    for ax,width in zip(axes,[64,192]):
        for norm,color in [("LN010","#0072B2"),("noLN","#D55E00")]:
            y=[values[(f,f"{norm}_w{width}","frozen_head")].mean() for f in functions]
            ax.plot(range(4),y,"o-",color=color,label=norm)
        ax.axhline(0,color="black",linewidth=.6);ax.set_title(f"Width {width}");ax.set_xticks(range(4),functions,rotation=25,ha="right");ax.legend()
    axes[0].set_ylabel("SGD fixed-head centered chord");fig.tight_layout();fig.savefig(STUDY/"ln_width.png",dpi=220);fig.savefig(STUDY/"ln_width.pdf")
    print(json.dumps({"new_runs":240,"predictions":result["predictions"],"LN_contrasts":pairings}))


if __name__=="__main__":main()
