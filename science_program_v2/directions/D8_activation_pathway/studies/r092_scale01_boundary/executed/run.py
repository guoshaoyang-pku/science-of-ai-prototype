import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def cells(config):
    return [{'seed':s,'lambda_index':i,'lambda':lam,'scale':a,'delta':lam*a**2}
            for s in config['seeds'] for i,lam in enumerate(config['lambdas']) for a in config['scales']]


def label(cell):
    return f's{cell["seed"]}_l{cell["lambda_index"]}_a{cell["scale"]:g}'


def validate_result(path, cell, config, manifest_path):
    row = json.loads(path.read_text())
    request = {'cell':cell,'bias':config['bias_root']+cell['delta'],'manifest_sha256':sha(manifest_path)}
    contract = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    assert row['status']=='success' and row['cell_id']==label(cell)
    assert row['request']==request and row['contract_sha256']==contract
    assert row['arrays_sha256']==sha(path.with_suffix('.npz')) and row['training_steps']==0
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        assert set(z.files)=={'odd_raw','even_centered_raw','feature_center'}
        assert all(np.isfinite(z[k]).all() for k in z.files)
        assert z['odd_raw'].shape==z['even_centered_raw'].shape==(128,64)
        assert z['feature_center'].shape==(64,)
    return row


def gate():
    config = json.loads((STUDY/'preregistration.json').read_text())
    prereg_path = str((STUDY/'preregistration.json').relative_to(REPO))
    commits = subprocess.run(['git','log','--format=%H','--',prereg_path],cwd=REPO,capture_output=True,text=True,check=True).stdout.splitlines()
    assert len(commits)==1, 'Require exactly one immutable preregistration commit'
    commit = commits[0]
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=REPO,check=True)
    for name in ['preregistration.json',*config['source_sha256']]:
        path = STUDY/name
        if name in config['source_sha256']:
            assert sha(path)==config['source_sha256'][name], name
        blob = subprocess.run(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO,capture_output=True,check=True).stdout
        assert blob==path.read_bytes(), name
    baseline = json.loads((STUDY/'executed/old_evidence_manifest.json').read_text())['files']
    for name, row in baseline.items():
        path = REPO/name
        assert sha(path)==row['sha256'] and path.stat().st_mtime_ns==row['mtime_ns'], name
    actual_old = {str(p.relative_to(REPO)) for s in STUDY.parent.iterdir() if s.is_dir() and s!=STUDY for p in s.rglob('*') if p.is_file()}
    assert actual_old==set(baseline), 'Historical files added or removed'
    audit = json.loads((STUDY/'executed/condition_audit.json').read_text())
    planned = cells(config)
    actual = [{**c,'bias':config['bias_root']+c['delta'],'cell_id':label(c),'mode':'new_measurement'} for c in planned]
    assert len(planned)==config['planned_cells']==12 and audit['overlap_cells']==0
    assert actual==audit['new_cells']
    for c in planned:
        assert not any(r['seed']==c['seed'] and r['scale']==c['scale'] and abs(r['bias']-(config['bias_root']+c['delta']))<1e-14 for r in audit['historical_requests']), 'Old activation forbidden'
    controls = json.loads((STUDY/'executed/readonly_controls.json').read_text())
    old = REPO/controls['source_study']
    assert sha(old/'executed/source_manifest.json')==controls['source_manifest_sha256']
    assert sha(old/'preregistration.json')==controls['source_preregistration_sha256']
    assert sha(old/'summary.json')==controls['source_summary_sha256']
    assert sha(old/'executed/analytic_forecasts.json')==controls['source_forecast_sha256']
    for r in controls['rows']:
        p = REPO/r['metadata_path']
        assert sha(p)==r['metadata_sha256'] and sha(p.with_suffix('.npz'))==r['arrays_sha256']
        metadata = json.loads(p.read_text())
        assert metadata['request']==r['original_request'] and metadata['contract_sha256']==r['contract_sha256']
    forecasts = json.loads((STUDY/'executed/numeric_forecasts.json').read_text())
    with np.load(STUDY/'executed/input_snapshot.npz',allow_pickle=False) as z:
        inputs = {k:z[k].copy() for k in z.files}
    assert set(inputs)==set(forecasts['input_arrays'])
    for k,a in inputs.items():
        expected = forecasts['input_arrays'][k]
        assert hashlib.sha256(a.tobytes()).hexdigest()==expected['sha256']
        assert list(a.shape)==expected['shape'] and str(a.dtype)==expected['dtype']
    for s in config['seeds']:
        rebuilt = np.einsum('nd,jd->nj',inputs['train_x'],inputs[f'weight_{s}'])
        assert np.max(np.abs(rebuilt-inputs[f'u_{s}']))<=2e-14
    manifest_path = STUDY/'executed/source_manifest.json'
    expected_manifest = {'preregistration_commit':commit,'preregistration_sha256':sha(STUDY/'preregistration.json'),'source_sha256':config['source_sha256'],'input_arrays':forecasts['input_arrays'],'device':'cpu','threads':1,'training_steps':0}
    result_dir = STUDY/'results'
    paths = list(result_dir.iterdir()) if result_dir.exists() else []
    expected = {label(c):c for c in planned}
    assert all(p.is_file() and p.stem in expected and p.suffix in ('.json','.npz') for p in paths), 'Unknown result files'
    assert {p.stem for p in paths if p.suffix=='.json'}=={p.stem for p in paths if p.suffix=='.npz'}, 'Orphan results'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert all(manifest[k]==v for k,v in expected_manifest.items()), 'Mixed execution manifest'
    else:
        assert not paths, 'Results without manifest'
    # Validate the complete saved set before any new activation.
    for p in paths:
        if p.suffix=='.json':
            validate_result(p,expected[p.stem],config,manifest_path)
    return config, commit, expected_manifest, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    config,commit,manifest,inputs = gate()
    import torch
    torch.set_num_threads(1)
    manifest_path = STUDY/'executed/source_manifest.json'
    if not manifest_path.exists():
        manifest.update({'torch_version':torch.__version__,'numpy_version':np.__version__})
        save(manifest_path,manifest)
    result_dir = STUDY/'results'
    result_dir.mkdir(exist_ok=True)
    new, reused = 0,0
    started = time.monotonic()
    for c in cells(config):
        path = result_dir/(label(c)+'.json')
        if path.exists():
            validate_result(path,c,config,manifest_path)
            reused += 1
        else:
            assert not path.with_suffix('.npz').exists()
            request = {'cell':c,'bias':config['bias_root']+c['delta'],'manifest_sha256':sha(manifest_path)}
            begin = time.monotonic()
            u = torch.from_numpy(inputs[f'u_{c["seed"]}'])
            raw = torch.nn.functional.silu(request['bias']+c['scale']*u)
            reflected = torch.nn.functional.silu(request['bias']-c['scale']*u)
            arrays = {'odd_raw':((raw-reflected)/2).numpy(),'even_centered_raw':((raw+reflected)/2-raw.mean(0)).numpy(),'feature_center':raw.mean(0).numpy()}
            assert all(np.isfinite(a).all() for a in arrays.values())
            np.savez_compressed(path.with_suffix('.npz'),**arrays)
            save(path,{'status':'success','cell_id':label(c),'request':request,'contract_sha256':hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),'arrays_sha256':sha(path.with_suffix('.npz')),'seconds':time.monotonic()-begin,'training_steps':0})
            new += 1
        state = {'completed':new+reused,'new':new,'reused':reused,'planned_cells':12,'last_cell':label(c),'seconds':time.monotonic()-started,'training_steps':0,'preregistration_commit':commit}
        save(STUDY/'current.json',state)
        if args.limit is not None and new>=args.limit:
            break
    assert time.monotonic()-started<=config['execution']['maximum_seconds']
    print(json.dumps(state))


if __name__=='__main__':
    main()
