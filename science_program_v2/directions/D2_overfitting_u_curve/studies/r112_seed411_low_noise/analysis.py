"""Analyze the registered n32/seed411 reweighted curve."""
import argparse, hashlib, json, importlib.util
from pathlib import Path
import numpy as np
STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', STUDY/'executed/run.py'); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

def describe(risk, signal, variance):
    t = int(np.argmin(risk)); delta = float(risk[-1]-risk[t])
    return {'t_star':t,'interior':bool(0<t<16384),'initial_risk':float(risk[0]),'minimum_risk':float(risk[t]),'final_risk':float(risk[-1]),'endpoint_minus_minimum':delta,'margin_over_threshold':delta-0.05,'operational_u_shape':bool(0<t<16384 and delta>=0.05),'signal_at_minimum':float(signal[t]),'variance_unit_at_minimum':float(variance[t]),'signal_at_endpoint':float(signal[-1]),'variance_unit_at_endpoint':float(variance[-1])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preregistration-commit',required=True); args=ap.parse_args()
    cfg=runner.verify_contract(args.preregistration_commit); assert runner.verify_saved(cfg,args.preregistration_commit)
    path=STUDY/'results'/ (runner.LABEL+'.npz')
    with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    assert set(arrays)=={'steps','signal_bias','variance_unit','expected_risk','baseline_risk'}
    assert np.array_equal(arrays['steps'],np.arange(16385))
    signal,variance=arrays['signal_bias'],arrays['variance_unit']; risk=signal+cfg['cell']['noise_variance']*variance; baseline=signal+cfg['baseline_noise_variance']*variance
    assert np.array_equal(risk,arrays['expected_risk']); assert np.array_equal(baseline,arrays['baseline_risk'])
    new=describe(risk,signal,variance); old=describe(baseline,signal,variance)
    lo,hi=cfg['predictions'][0]['interval']; passed=bool(lo<=new['endpoint_minus_minimum']<=hi and new['operational_u_shape'])
    summary={'study':cfg['study'],'round':112,'direction_round':8,'direction':cfg['direction'],'domain':'development','status':'completed','preregistration_commit':args.preregistration_commit,'cell':cfg['cell'],'counts':{'new_recipes':1,'seeds':1,'new_evaluation_cells':1,'new_training_cells':0,'saved_controls':1},'new':new,'baseline':{'noise_variance':cfg['baseline_noise_variance'],**old},'paired':{'t_star_ratio':new['t_star']/old['t_star'],'delta_change':new['endpoint_minus_minimum']-old['endpoint_minus_minimum'],'minimum_risk_change':new['minimum_risk']-old['minimum_risk'],'endpoint_risk_change':new['final_risk']-old['final_risk']},'predictions':[{'id':'P1','interval':cfg['predictions'][0]['interval'],'observed':new['endpoint_minus_minimum'],'status':'supported' if passed else 'refuted'}],'risk_arrays_sha256':runner.sha(path),'source_curve_sha256':runner.sha(ROOT_PATH(cfg['source_curve'])),'boundaries':cfg['boundaries']}
    out=STUDY/'summary.json';
    if out.exists(): assert json.loads(out.read_text())==summary, 'refuse overwrite';
    else: runner.save_json(out,summary)
    print(json.dumps({'prediction':summary['predictions'][0],'new':new,'baseline':old,'paired':summary['paired']},ensure_ascii=False))

def ROOT_PATH(rel): return runner.ROOT / rel
if __name__=='__main__': main()
