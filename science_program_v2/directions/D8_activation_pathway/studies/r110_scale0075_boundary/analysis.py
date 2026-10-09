import json

import numpy as np
from scipy.stats import t

from executed.run import STUDY, REPO, cells, gate, label, save, validate_result


def describe(values):
    return {'min':min(values),'max':max(values),'mean':float(np.mean(values))} if values else None


def seed_stats(values):
    a = np.asarray(values)
    half = float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return {**describe(values),'seed_95pct_t_interval':[float(a.mean()-half),float(a.mean()+half)]}


def energy(path):
    with np.load(path,allow_pickle=False) as z:
        return float(np.mean(z['even_centered_raw']**2)/np.mean(z['odd_raw']**2))


def main():
    config,commit,_,_ = gate()
    forecasts = json.loads((STUDY/'executed/numeric_forecasts.json').read_text())
    controls = json.loads((STUDY/'executed/readonly_controls.json').read_text())['rows']
    for row in controls:
        r = energy((REPO/row['metadata_path']).with_suffix('.npz'))
        assert r==row['R'] and r/row['scale']**6==row['Q'], 'Saved control differs'
    rows, pairs = [],[]
    for c in cells(config):
        path = STUDY/'results'/(label(c)+'.json')
        if not path.exists():
            continue
        metadata = validate_result(path,c,config,STUDY/'executed/source_manifest.json')
        prediction = next(r for r in forecasts['rows'] if r['seed']==c['seed'] and r['lambda']==c['lambda'])
        r = energy(path.with_suffix('.npz'))
        q = r/c['scale']**6
        signed = q/prediction['F']-1
        rows.append({**c,'cell_id':label(c),'bias':metadata['request']['bias'],'R':r,'Q':q,'F':prediction['F'],'signed_F_error':signed,'relative_F_error':abs(signed),'predicted_Q':prediction['Q_point'],'Q_point_relative_difference':q/prediction['Q_point']-1,'seconds':metadata['seconds']})
        old = sorted([r for r in controls if r['seed']==c['seed'] and r['lambda']==c['lambda']],key=lambda r:r['scale'])
        qs = [r['Q'] for r in old]+[q]
        old_spread = max(qs[:-1])/min(qs[:-1])-1
        pairs.append({'seed':c['seed'],'lambda':c['lambda'],'scales':config['control_scales']+config['scales'],'Q':qs,'old_spread':old_spread,'spread':max(qs)/min(qs)-1,'spread_increase':max(qs)/min(qs)-1-old_spread,'Q_new_over_old05':q/qs[-2],'Q_new_over_old0125':q/qs[0],'signed_error_new_minus_old05':signed-(qs[-2]/prediction['F']-1)})
    complete = len(rows)==config['planned_cells']
    p1_count = sum(next(f for f in forecasts['rows'] if f['seed']==r['seed'] and f['lambda']==r['lambda'])['absolute_relative_error_interval'][0]<=r['relative_F_error']<=next(f for f in forecasts['rows'] if f['seed']==r['seed'] and f['lambda']==r['lambda'])['absolute_relative_error_interval'][1] and r['signed_F_error']*r['lambda']<0 and r['relative_F_error']>.02 for r in rows)
    p2_count = sum(next(f for f in forecasts['rows'] if f['seed']==p['seed'] and f['lambda']==p['lambda'])['four_scale_spread_interval'][0]<=p['spread']<=next(f for f in forecasts['rows'] if f['seed']==p['seed'] and f['lambda']==p['lambda'])['four_scale_spread_interval'][1] and p['spread']>.02 for p in pairs)
    units = []
    for lam in config['lambdas']:
        selected = [r for r in rows if r['lambda']==lam]
        paired = [p for p in pairs if p['lambda']==lam]
        if selected:
            units.append({'lambda':lam,'seeds':len(selected),'signed_F_error':seed_stats([r['signed_F_error'] for r in selected]),'relative_F_error':describe([r['relative_F_error'] for r in selected]),'spread':seed_stats([p['spread'] for p in paired]),'spread_increase':seed_stats([p['spread_increase'] for p in paired]),'Q_new_over_old05':seed_stats([p['Q_new_over_old05'] for p in paired]),'Q_new_over_old0125':seed_stats([p['Q_new_over_old0125'] for p in paired]),'signed_error_new_minus_old05':seed_stats([p['signed_error_new_minus_old05'] for p in paired])})
    summary = {'study':STUDY.name,'round':110,'direction_round':7,'domain':'development','question':config['question'],'status':'complete' if complete else 'partial','saved_cells':len(rows),'planned_cells':12,'new_lambda_scale_conditions':2,'readonly_control_cells':36,'paired_seed_lambda_units':len(pairs),'training_steps':0,'measurement_seconds':sum(r['seconds'] for r in rows),'preregistration_commit':commit,'predictions':[{'id':'P1','status':('supported' if p1_count==12 else 'refuted') if complete else 'not_evaluated','passed_cells':p1_count,'planned':12},{'id':'P2','status':('supported' if p2_count==12 else 'refuted') if complete else 'not_evaluated','passed_pairs':p2_count,'planned':12}],'original_2pct_criteria':{'F_passed_new_cells':sum(r['relative_F_error']<=.02 for r in rows),'F_failed_new_cells':sum(r['relative_F_error']>.02 for r in rows),'fold_passed_four_scale_pairs':sum(p['spread']<=.02 for p in pairs),'fold_failed_four_scale_pairs':sum(p['spread']>.02 for p in pairs)},'F_range':forecasts['F_range'],'relative_F_error':describe([r['relative_F_error'] for r in rows]),'fold_spread':describe([p['spread'] for p in pairs]),'Q_point_relative_difference':describe([r['Q_point_relative_difference'] for r in rows]),'units':units,'paired_seed_results':pairs,'rows':rows,'verification':{'old_control_recompute':'bitwise equal from saved arrays','historical_overlap_cells':0,'source_commit_input_contract_checks':'pass'},'boundary':config['boundary']}
    output = STUDY/('summary.json' if complete else 'executed/first_cell_summary.json')
    assert not output.exists(), output
    save(output,summary)
    print(json.dumps({k:summary[k] for k in ('saved_cells','predictions','original_2pct_criteria','relative_F_error','fold_spread')},ensure_ascii=False))


if __name__=='__main__':
    main()
