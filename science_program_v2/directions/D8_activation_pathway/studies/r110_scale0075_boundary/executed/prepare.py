import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
OLD = STUDY.parent / 'r092_scale01_boundary'
CONTROL = STUDY.parent / 'r080_lambda_outer_collapse'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    receipt = json.loads((OLD/'executed/final_commit_verification.json').read_text())
    verification = json.loads((OLD/'executed/independent_verification.json').read_text())
    assert receipt['round_result']=='ok' and verification['status']=='pass'
    assert verification['recovery']['new']==0 and verification['recovery']['reused']==12
    for row in receipt['files']:
        blob = subprocess.run(['git','show',receipt['scientific_closeout_commit']+':'+row['path']],cwd=REPO,capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==row['sha256'], row['path']
        if row['path'].startswith(str(OLD.relative_to(REPO))+'/'):
            assert sha(REPO/row['path'])==row['sha256'], row['path']
    protected_old = json.loads((OLD/'executed/old_evidence_manifest.json').read_text())['files']
    for base, rows in ((REPO,protected_old),(OLD,verification['result_files'])):
        for name, row in rows.items():
            path = base/name
            assert sha(path)==row['sha256'] and path.stat().st_mtime_ns==row['mtime_ns'], name
    first = json.loads((OLD/'executed/first_cell_audit.json').read_text())
    assert all(verification['result_files'][name]==row for name,row in first['result_files'].items())
    invalid = json.loads((STUDY.parent/'r048_lambda_collapse/executed/round_invalidation.json').read_text())
    assert invalid['round_result']=='failed' and invalid['valid_scientific_claims_added']==0
    config_old = json.loads((CONTROL/'preregistration.json').read_text())
    root = config_old['bias_root']
    protected, history, grids = {}, [], []
    for old in sorted(STUDY.parent.iterdir()):
        if old==STUDY or not old.is_dir():
            continue
        for path in sorted(old.rglob('*')):
            if path.is_file():
                protected[str(path.relative_to(REPO))]={'sha256':sha(path),'mtime_ns':path.stat().st_mtime_ns}
        config = json.loads((old/'preregistration.json').read_text())
        paths = sorted((old/'results').iterdir())
        metadata = [p for p in paths if p.suffix=='.json']
        assert len(paths)==2*len(metadata) and {p.stem for p in paths if p.suffix=='.npz'}=={p.stem for p in metadata}
        keys = []
        manifest_hash = sha(old/'executed/source_manifest.json')
        for path in metadata:
            row = json.loads(path.read_text())
            assert row['status']=='success' and row['cell_id']==path.stem
            assert row['arrays_sha256']==sha(path.with_suffix('.npz'))
            assert row['request']['manifest_sha256']==manifest_hash
            assert row['contract_sha256']==hashlib.sha256(json.dumps(row['request'],sort_keys=True).encode()).hexdigest()
            cell = row['request']['cell']
            keys.append((cell['seed'],cell['scale'],row['request']['bias']))
            history.append({'study':old.name,'cell_id':path.stem,'seed':cell['seed'],'scale':cell['scale'],'bias':row['request']['bias'],'original_request':row['request'],'contract_sha256':row['contract_sha256'],'arrays_sha256':row['arrays_sha256']})
        if 'lambdas' in config:
            expected = [(s,a,config['bias_root']+lam*a*a) for s,lam,a in itertools.product(config['seeds'],config['lambdas'],config['scales'])]
        elif 'bias_offsets' in config:
            expected = [(s,a,config['bias_root']+delta) for s,delta,a in itertools.product(config['seeds'],config['bias_offsets'],config['scales'])]
        else:
            expected = [(s,a,{'zero':0.,'one':1.,'inflection':root}[b]) for s,b,a in itertools.product(config['seeds'],config['bias_labels'],config['scales'])]
        assert len(keys)==len(expected) and len(set(keys))==len(keys)
        assert all(sum(s==t and a==c and abs(b-d)<1e-14 for t,c,d in keys)==1 for s,a,b in expected), old.name
        grids.append({'study':old.name,'registered_cells':len(expected),'saved_cells':len(keys),'cartesian_complete':True})
    cells = []
    for seed, lam in itertools.product(config_old['seeds'],config_old['lambdas']):
        a = .075
        cell = {'seed':seed,'lambda_index':config_old['lambdas'].index(lam),'lambda':lam,'scale':a,'delta':lam*a**2}
        assert not any(r['seed']==seed and r['scale']==a and abs(r['bias']-(root+cell['delta']))<1e-14 for r in history)
        cells.append({**cell,'bias':root+cell['delta'],'cell_id':f's{seed}_l{cell["lambda_index"]}_a{a:g}','mode':'new_measurement'})
    save(STUDY/'executed/condition_audit.json',{'historical_grids':grids,'historical_requests':history,'historical_request_count':len(history),'new_cells':cells,'planned_cells':12,'overlap_cells':0,'comparison':'same seed/scale and bias tolerance 1e-14; all registered Cartesian and saved requests checked'})
    save(STUDY/'executed/old_evidence_manifest.json',{'files':protected})
    save(STUDY/'executed/prior_closeout_audit.json',{'status':'pass','scientific_closeout_commit':receipt['scientific_closeout_commit'],'receipt_sha256':sha(OLD/'executed/final_commit_verification.json'),'independent_verification_sha256':sha(OLD/'executed/independent_verification.json'),'committed_blob_count':len(receipt['files']),'old896_preserved':len(protected_old),'r092_result_files_preserved':len(verification['result_files']),'first_cell_preserved':True,'r092_recovery_receipt_verified':True,'r048_invalid_zero_claims_preserved':True,'all_old_study_files_protected':len(protected)})
    target = STUDY/'executed/input_snapshot.npz'
    target.write_bytes((OLD/'executed/input_snapshot.npz').read_bytes())
    assert sha(target)==sha(OLD/'executed/input_snapshot.npz')
    forecasts_old = json.loads((CONTROL/'executed/analytic_forecasts.json').read_text())
    controls, predictions = [], []
    for seed, lam in itertools.product(config_old['seeds'],config_old['lambdas']):
        f = next(r['F'] for r in forecasts_old['rows'] if r['seed']==seed and r['lambda']==lam)
        selected = []
        for a in config_old['scales']:
            index = config_old['lambdas'].index(lam)
            path = CONTROL/'results'/f's{seed}_l{index}_a{a:g}.json'
            metadata = json.loads(path.read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                r = float(np.mean(z['even_centered_raw']**2)/np.mean(z['odd_raw']**2))
            q = r/a**6
            row = {'seed':seed,'lambda':lam,'scale':a,'R':r,'Q':q,'F':f,'metadata_path':str(path.relative_to(REPO)),'metadata_sha256':sha(path),'arrays_sha256':metadata['arrays_sha256'],'original_request':metadata['request'],'contract_sha256':metadata['contract_sha256']}
            selected.append(row)
            controls.append(row)
        signed_point = 2.25*(selected[-1]['Q']/f-1)
        point_q = f*(1+signed_point)
        qs = [r['Q'] for r in selected]+[point_q]
        predictions.append({'seed':seed,'lambda':lam,'F':f,'signed_relative_error_point':signed_point,'Q_point':point_q,'Q_interval':[f*(1+signed_point-.002),f*(1+signed_point+.002)],'absolute_relative_error_interval':[abs(signed_point)-.002,abs(signed_point)+.002],'four_scale_spread_point':max(qs)/min(qs)-1,'four_scale_spread_interval':[max(qs)/min(qs)-1-.002,max(qs)/min(qs)-1+.002]})
    save(STUDY/'executed/readonly_controls.json',{'source_study':str(CONTROL.relative_to(REPO)),'source_manifest_sha256':sha(CONTROL/'executed/source_manifest.json'),'source_preregistration_sha256':sha(CONTROL/'preregistration.json'),'source_summary_sha256':sha(CONTROL/'summary.json'),'source_forecast_sha256':sha(CONTROL/'executed/analytic_forecasts.json'),'policy':'只读取保存odd/even与原合同/hash；不调用任何旧激活、训练或旧分析脚本','rows':controls})
    save(STUDY/'executed/numeric_forecasts.json',{'origin':'非盲development：已见旧u解析F和a=.0125/.025/.05的保存Q。新点预测只把a=.05的Q/F−1乘2.25，依据有限尺度首修正随a²的假设；已知a=.1结果；没有拟合或预览a=.075激活。所有区间为事前容许范围，不是置信区间。','source_forecast_sha256':sha(CONTROL/'executed/analytic_forecasts.json'),'F_range':forecasts_old['F_range'],'input_arrays':forecasts_old['input_arrays'],'rows':predictions})
    save(STUDY/'executed/state_before.json',json.loads((REPO/'central/state.json').read_text()))
    print(json.dumps({'prior_audit':'pass','protected_files':len(protected),'history_requests':len(history),'new_cells':len(cells),'overlap':0,'old_controls':len(controls),'predicted_absolute_error_range':[min(abs(r['signed_relative_error_point']) for r in predictions),max(abs(r['signed_relative_error_point']) for r in predictions)]}))


if __name__=='__main__':
    main()
