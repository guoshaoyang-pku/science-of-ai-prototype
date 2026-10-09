#!/usr/bin/env python3
"""从 r075 JSON 端点复算系数误差和注册判据。"""
import hashlib,json,math
from pathlib import Path
import numpy as np
STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[3]
BASE=ROOT/"directions/D11_width_scaling/studies/r073_trace_one"
EPS=0.02
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads((STUDY/"preregistration.json").read_text()); rec=json.loads((STUDY/"results/receipt.json").read_text()); rows=[]
 for name,digest in sorted(rec["cells"].items()):
  out=STUDY/"results"/(name+".json"); assert sha(out)==digest; row=json.loads(out.read_text()); d=int(row["cell"]["dimension"]); target=row["cell"]["target"]; h=float(np.sum(1/np.arange(1,d+1,dtype=float))); coeff=math.log(50)/(2 if target=="middle" else 1); t=int(row["measured_step_epsilon002"]); rows.append({**row["cell"],"name":name,"measured_step":t,"harmonic_number":h,"T_over_dh":t/(d*h),"coefficient":coeff,"coefficient_relative_error":t/(d*h*coeff)-1,"source_json_sha256":row["source_json_sha256"],"source_npz_sha256":row["source_npz_sha256"]})
 agg={}; pairs=[]
 for target in cfg["targets"]:
  sub=[r for r in rows if r["target"]==target]; agg[target]={"max_abs_coefficient_relative_error":max(abs(r["coefficient_relative_error"]) for r in sub),"d_ge_128_max_abs_coefficient_relative_error":max(abs(r["coefficient_relative_error"]) for r in sub if r["dimension"]>=128),"coefficient_error_range":[min(r["coefficient_relative_error"] for r in sub),max(r["coefficient_relative_error"] for r in sub)]}
 for d in cfg["dimensions"]:
  for seed in cfg["coordinate_seeds"]:
   m=next(r for r in rows if r["dimension"]==d and r["seed"]==seed and r["target"]=="middle"); e=next(r for r in rows if r["dimension"]==d and r["seed"]==seed and r["target"]=="endpoints"); pairs.append({"dimension":d,"seed":seed,"budget_ratio_endpoints_over_middle":e["measured_step"]/m["measured_step"]})
 p2=all(agg[t]["max_abs_coefficient_relative_error"]<=cfg["criteria"][t] and agg[t]["d_ge_128_max_abs_coefficient_relative_error"]<=cfg["criteria"]["d_ge_128"] for t in cfg["targets"])
 summary={"study":cfg["study"],"round":75,"direction":"D11_width_scaling","domain":"development","complete":len(rows)==66,"counts":{"saved_cells":len(rows),"target_pairs":len(pairs)},"predictions":{"P1":"supported" if len(rows)==66 else "not_evaluated","P2":"supported" if p2 else ("refuted" if len(rows)==66 else "not_evaluated")},"rows":rows,"aggregate":agg,"target_pairs":pairs,"threshold":EPS,"source_summary_sha256":cfg["source_summary_sha256"],"source_manifest_sha256":cfg["source_manifest_sha256"],"pinned_sources":cfg["pinned_sources"]}
 (STUDY/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"saved_cells":len(rows),"predictions":summary["predictions"],"aggregate":agg}))
if __name__=="__main__": main()
