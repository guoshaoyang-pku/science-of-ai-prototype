"""Independent audit of saved A04 measurements; no training or model calls."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t
import torch

ROOT = Path(__file__).resolve().parents[4]
STUDY = ROOT / 'studies/A04_ln_affine'
PREVIOUS = ROOT / 'studies/A02_ood_residual'
COMMIT = '74bf2b49d2dd13feefb9aa359141a304ea5003aa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + str(path)], cwd=ROOT)


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    error = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean, ci95=[mean-error, mean+error])


def main():
    torch.set_num_threads(1)
    config = json.loads((STUDY/'preregistration.json').read_text())
    analysis = json.loads((STUDY/'analysis.json').read_text())
    history_available = subprocess.run(['git','cat-file','-e',COMMIT+'^{commit}'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    previous_audit = Path(__file__).with_name('independent_audit.json')
    for name in ['preregistration.json', 'run.py', 'executed/host/experiment.py']:
        path = STUDY/name
        if history_available:
            assert git_bytes(COMMIT, path.relative_to(ROOT)) == path.read_bytes()
    if history_available:
        sealed_at = int(subprocess.check_output(['git', 'show', '-s', '--format=%at', COMMIT], cwd=ROOT))
    else:
        prior = json.loads(previous_audit.read_text())
        assert prior['seal_commit']==COMMIT and prior['preregistration_sha256']==sha(STUDY/'preregistration.json')
        assert prior['executor_sha256']==sha(STUDY/'executed/host/experiment.py')
        sealed_at = prior['seal_commit_time']
    a02_seal = json.loads((PREVIOUS/'seal.json').read_text())
    a02_executor = (PREVIOUS/'executed/host/experiment.py').read_bytes()
    assert hashlib.sha256(a02_executor).hexdigest() == a02_seal['executor_sha256']
    data = {}
    for function in config['functions']:
        path = PREVIOUS/'datasets'/f'{function}.pt'
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == metadata['sha256']
        data[function] = torch.load(path, weights_only=True)
    records, arrays, measured, sources, new_rows = {}, {}, {}, [], []
    for folder in [PREVIOUS, STUDY]:
        for path in sorted((folder/'results').glob('*.json')):
            row = json.loads(path.read_text())
            if folder == PREVIOUS and not (row['recipe_label']=='LN010_w192' and row['intervention']=='frozen_head' and row['recipe']['optimizer']=='SGD'):
                continue
            key = row['dataset'], row['intervention'], row['mean'], row['seed']
            assert key not in records
            expected_recipe = dict(config['recipe'], input_dim=data[key[0]]['train_x'].shape[1], optimizer='SGD', lr=.001)
            assert row['recipe'] == expected_recipe
            request = row['contract']
            assert hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest() == row['contract_sha256']
            assert all(row[k]==v for k,v in request['cell'].items())
            expected_prereg = sha(folder/'preregistration.json')
            assert row['preregistration_sha256'] == expected_prereg
            for name,pin in request['executable'].items():
                if name=='experiment.py':
                    raw = a02_executor if folder==PREVIOUS else (STUDY/'executed/host/experiment.py').read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==pin['sha256']
                else:
                    assert sha(folder/'executed/SGD'/name)==pin['sha256']
            record=row['process']
            for name,value in data[key[0]].items():
                transformed = value+key[2] if name.endswith('_y') else value
                expected=hashlib.sha256(transformed.contiguous().numpy().tobytes()).hexdigest()
                assert request['inputs'][name]['sha256']==record['inputs'][name]['sha256']==expected
            assert not row['failed'] and record['status']=='completed'
            assert len(row['curve'])==256 and np.isfinite(row['curve']).all()
            assert [x['step'] for x in record['steps']]==[1,8,32,128,256]
            for step in record['steps']:
                assert step['metrics']['test_mse']==row['curve'][step['step']-1]
                for parameter in step['parameters']:
                    norm='.norm.' in parameter['name']
                    weight=parameter['name'].endswith('weight') and not norm
                    frozen=parameter['role']=='head' or (key[1]=='head_frozen_ln_fixed' and norm) or (key[1]=='head_frozen_weights_only' and not weight) or (key[1]=='head_frozen_affine_only' and weight)
                    assert parameter['active']==(not frozen)
                    if frozen:
                        assert parameter['displacement_norm']==parameter['descent_norm']==0
            array_path=path.with_suffix('.npz')
            assert sha(array_path)==row['arrays_sha256']
            with np.load(array_path,allow_pickle=False) as loaded:
                saved={name:loaded[name].copy() for name in loaded.files}
            for prefix in ['', 'train_']:
                target=data[key[0]][prefix+'y' if prefix else 'test_y'].numpy()+key[2]
                assert np.array_equal(saved[prefix+'targets'],target)
            residual=saved['predictions'].astype(float)-saved['targets'].astype(float)
            centered=float(np.mean((residual-residual.mean())**2))
            mse=float(np.mean(residual**2))
            assert np.isclose(mse,row['test_mse'],atol=1e-7,rtol=3e-7)
            assert np.isclose(mse,centered+float(residual.mean()**2),atol=1e-12)
            records[key]=record; arrays[key]=saved; measured[key]=centered
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            if folder==STUDY:
                new_rows.append(row)
    expected=set(itertools.product(config['functions'],['frozen_head',*config['interventions']],config['means'],config['seeds']))
    assert set(measured)==expected and len(new_rows)==180
    assert sealed_at<min(row['saved_at'] for row in new_rows)
    for key,record in records.items():
        reference=key[0],'frozen_head',key[2],key[3]
        old=records[reference]
        for name in ['initial_parameters','initial_metrics','inputs','minibatch_stream_sha256','optimizer_defaults']:
            assert record[name]==old[name]
        for name in ['initial_predictions','initial_train_predictions']:
            assert np.array_equal(arrays[key][name],arrays[reference][name])
    chords={}
    for function,mode in itertools.product(config['functions'],['frozen_head',*config['interventions']]):
        values=np.array([(measured[(function,mode,-3,s)]+measured[(function,mode,3,s)])/2-measured[(function,mode,0,s)] for s in config['seeds']])
        chords[function,mode]=values
        saved=next(r for r in analysis['chords'] if r['function']==function and r['mode']==mode)
        actual=interval(values)
        assert abs(actual['mean']-saved['centered_chord']['mean'])<1e-12
        assert np.allclose(actual['ci95'],saved['centered_chord']['ci95'],rtol=0,atol=1e-12)
        assert int((values<0).sum())==saved['negative_seeds']
    checks=[]
    for check in analysis['checks']:
        function=check['function']
        mode=dict(zip(['LN_forward_sufficient','weights_sufficient','affine_not_sufficient'],config['interventions']))[check['id']]
        baseline=chords[function,'frozen_head']; value=chords[function,mode]
        ratio=float(value.mean()/baseline.mean()); seed_fraction=interval(value/baseline)
        assert abs(ratio-check['retained_mean_benefit_fraction'])<1e-12
        assert seed_fraction==check['paired_seed_fraction']
        passed=ratio<.5 if check['id']=='affine_not_sufficient' else ratio>=.5
        assert passed==check['pass']
        difference=interval(value-.5*baseline)
        direction_support=difference['ci95'][0]>0 if check['id']=='affine_not_sufficient' else difference['ci95'][1]<0
        checks.append(dict(check,paired_threshold_difference=difference,threshold_direction_ci_excludes_zero=direction_support))
    assert sources==analysis['source_files']
    high_loss=sum(row['test_mse']>2 for row in new_rows)
    assert high_loss==analysis['finite_cutoff_gt2_retained']==40
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit='PASS',study='A04_ln_affine',training=False,model_calls=0,new_runs=180,reused_A02_runs=60,all_contracts_sources_arrays_verified=True,initialization_data_stream_optimizer_pairing=True,frozen_and_active_parameter_groups_verified=True,seal_commit=COMMIT,seal_commit_time=sealed_at,original_git_history_available=history_available,chronology_evidence='Original Git' if history_available else 'Released audit record; original Git absent in this checkout',first_saved_at=min(row['saved_at'] for row in new_rows),preregistration_sha256=sha(STUDY/'preregistration.json'),executor_sha256=sha(STUDY/'executed/host/experiment.py'),seconds_sum=sum(row['seconds'] for row in new_rows),high_loss_retained=high_loss,failed_runs=0,chords=analysis['chords'],checks=checks,qualifiers=['Four A02 targets are now development; no new OOD.', 'Parameter freezing also removes their gradient/decay updates.', 'LN normalization is retained in all three interventions.', 'Affine-only contrasts have CIs crossing zero; no proof of strictly absent effect.', 'Weight-only sufficiency does not separate Jacobian drift from an initial linearization.', 'No solver gain measurement.'])
    Path(__file__).with_name('independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps({k:v for k,v in result.items() if k!='chords'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
