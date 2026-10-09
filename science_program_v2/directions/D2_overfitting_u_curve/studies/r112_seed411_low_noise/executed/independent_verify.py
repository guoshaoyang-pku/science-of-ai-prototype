import argparse, json, importlib.util
from pathlib import Path
import numpy as np
STUDY=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',STUDY/'executed/run.py'); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--preregistration-commit',required=True); a=ap.parse_args(); cfg=runner.verify_contract(a.preregistration_commit); assert runner.verify_saved(cfg,a.preregistration_commit)
 with np.load(STUDY/'results'/ (runner.LABEL+'.npz'),allow_pickle=False) as z: d={k:z[k] for k in z.files}
 src=np.load(runner.ROOT/cfg['source_curve'],allow_pickle=False); signal=d['signal_bias']; var=d['variance_unit']; risk=signal+cfg['cell']['noise_variance']*var; base=signal+cfg['baseline_noise_variance']*var
 assert np.max(np.abs(risk-d['expected_risk']))<=1e-14; assert np.max(np.abs(base-d['baseline_risk']))<=1e-14
 result={'status':'verified','max_risk_error':float(np.max(np.abs(risk-d['expected_risk']))),'max_baseline_error':float(np.max(np.abs(base-d['baseline_risk']))),'t_star':int(np.argmin(risk)),'endpoint_minus_minimum':float(risk[-1]-risk.min()),'training_cells':0,'source_arrays_match':bool(np.array_equal(signal,src['signal_bias']) and np.array_equal(var,src['variance_unit']))}
 out=STUDY/'executed/independent_verification.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
