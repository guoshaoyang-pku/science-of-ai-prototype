"""Independent saved-measurement audit; no training or model calls."""
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
STUDY = ROOT / 'studies/A02_ood_residual'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    error = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'ci95': [mean - error, mean + error]}


def main():
    torch.set_num_threads(1)
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    seal = json.loads((STUDY / 'seal.json').read_text())
    analysis = json.loads((STUDY / 'analysis.json').read_text())
    history_available = subprocess.run(['git', 'cat-file', '-e', seal['commit'] + '^{commit}'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    for name, key in [('preregistration.json', 'preregistration_sha256'), ('run.py', 'run_source_sha256')]:
        saved = STUDY / name if name == 'preregistration.json' else STUDY / 'executed/host/run.py'
        assert sha(saved) == seal[key]
        if history_available:
            raw = subprocess.check_output(['git', 'show', seal['commit'] + ':studies/A02_ood_residual/' + name], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == seal[key]
    executor = STUDY / 'executed/host/experiment.py'
    assert sha(executor) == seal['executor_sha256']
    rows = [json.loads(p.read_text()) for p in sorted((STUDY / 'results').glob('*.json'))]
    assert seal['sealed_at'] < min(row['saved_at'] for row in rows)
    data = {}
    generators = {
        'trigonometric8': lambda x: torch.sin(2*x[:,0]) + .7*torch.cos(3*x[:,1]) + .3*x[:,2]*x[:,3],
        'quadratic8': lambda x: x[:,0]**2 + .5*x[:,1]**2 - .6*x[:,2]**2 + .2*x[:,3],
        'interaction12': lambda x: torch.sin(x[:,0]*x[:,1]) + .5*torch.tanh(x[:,2]+x[:,3]) + .2*x[:,4]*x[:,5],
        'radial12': lambda x: torch.exp(-.25*x[:,:6].square().sum(1)) + .1*x[:,6],
    }
    for index, name in enumerate(prereg['functions']):
        generator = torch.Generator().manual_seed(prereg['data_seed'] + index)
        dim = 12 if name.endswith('12') else 8
        n = prereg['train_n'] + prereg['test_n']
        x = torch.randn(n, dim, generator=generator) if dim == 12 else 2*torch.rand(n, dim, generator=generator)-1
        y = generators[name](x)
        nt = prereg['train_n']
        center, scale = y[:nt].double().mean(), y[:nt].double().std(unbiased=False)
        y = ((y.double()-center)/scale).float().unsqueeze(1)
        expected = dict(train_x=x[:nt], test_x=x[nt:], train_y=y[:nt], test_y=y[nt:])
        path = STUDY / 'datasets' / (name + '.pt')
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == metadata['sha256']
        saved = torch.load(path, weights_only=True)
        assert float(center) == metadata['train_center'] and float(scale) == metadata['train_scale']
        for key, tensor in saved.items():
            assert torch.equal(tensor, expected[key])
            assert hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == metadata['tensors'][key]['sha256']
        data[name] = saved
    measured, initial, records = {}, {}, {}
    for row in rows:
        key = (row['dataset'], row['recipe_label'], row['mean'], row['intervention'], row['recipe']['optimizer'], row['seed'])
        assert key not in measured
        assert row['preregistration_sha256'] == seal['preregistration_sha256']
        contract = json.dumps(row['contract'], sort_keys=True, allow_nan=False).encode()
        assert hashlib.sha256(contract).hexdigest() == row['contract_sha256']
        expected_recipe = {**prereg['recipes'][key[1]], 'input_dim': data[key[0]]['train_x'].shape[1], 'optimizer': key[4], 'lr': prereg['optimizers'][key[4]]}
        assert row['recipe'] == expected_recipe
        for name, pin in row['contract']['executable'].items():
            path = executor if name == 'experiment.py' else STUDY / 'executed' / key[4] / name
            assert sha(path) == pin['sha256']
        record = row['process']
        for name, tensor in data[key[0]].items():
            transformed = tensor + key[2] if name.endswith('_y') else tensor
            pin = hashlib.sha256(transformed.contiguous().numpy().tobytes()).hexdigest()
            assert pin == record['inputs'][name]['sha256'] == row['contract']['inputs'][name]['sha256']
        assert not row['failed'] and record['status'] == 'completed'
        assert len(row['curve']) == 256 and np.isfinite(row['curve']).all()
        for step in record['steps']:
            assert step['metrics']['test_mse'] == row['curve'][step['step'] - 1]
            for parameter in step['parameters']:
                frozen = key[3] == 'frozen_head' and parameter['role'] == 'head' or key[3] == 'frozen_hidden' and parameter['role'] == 'hidden'
                if frozen:
                    assert not parameter['active'] and parameter['descent_norm'] == parameter['displacement_norm'] == 0
        array_path = STUDY / 'results' / (row['id'] + '.npz')
        assert sha(array_path) == row['arrays_sha256']
        with np.load(array_path) as arrays:
            residual = arrays['predictions'].astype(float) - arrays['targets'].astype(float)
            mse = float(np.mean(residual**2))
            assert np.isclose(mse, row['test_mse'], rtol=3e-7, atol=1e-7)
            measured[key] = dict(mse=mse, centered=float(np.mean((residual-residual.mean())**2)), predictions=arrays['predictions'].copy())
            initial[key] = arrays['initial_predictions'].copy(), arrays['initial_train_predictions'].copy()
        records[key] = record
    expected = set(itertools.product(prereg['functions'], prereg['recipes'], prereg['means'], prereg['interventions'], prereg['optimizers'], prereg['seeds']))
    assert set(measured) == expected and len(measured) == 960
    for key, record in records.items():
        reference = (key[0], key[1], 0, 'baseline', 'SGD', key[5])
        old = records[reference]
        assert record['minibatch_stream_sha256'] == old['minibatch_stream_sha256']
        head_bias = list(old['initial_parameters'])[-1]
        for name, pin in record['initial_parameters'].items():
            if key[3] != 'matched_bias' or name != head_bias:
                assert pin == old['initial_parameters'][name]
        shift = key[2] if key[3] == 'matched_bias' else 0
        for value, baseline in zip(initial[key], initial[reference]):
            assert np.allclose(value, baseline + shift, rtol=0, atol=5e-7)
    chord_values = {}
    for saved in analysis['chords']:
        key = saved['function'], saved['recipe'], saved['mode'], saved['optimizer']
        for metric, field in [('mse', 'total_chord'), ('centered', 'centered_chord')]:
            values = [(measured[(key[0],key[1],-3,key[2],key[3],s)][metric] + measured[(key[0],key[1],3,key[2],key[3],s)][metric])/2 - measured[(key[0],key[1],0,key[2],key[3],s)][metric] for s in prereg['seeds']]
            actual = interval(values)
            assert abs(actual['mean'] - saved[field]['mean']) < 1e-12
            assert np.allclose(actual['ci95'], saved[field]['ci95'], rtol=0, atol=1e-12)
            chord_values[key + (field,)] = values
    winner_details = []
    for saved in analysis['comparisons']:
        f, recipe, mean = saved['function'], saved['recipe'], saved['mean']
        gaps = [measured[(f,recipe,mean,'baseline','SGD',s)]['mse'] - measured[(f,recipe,mean,'baseline','Adam',s)]['mse'] for s in prereg['seeds']]
        actual = interval(gaps)
        assert abs(actual['mean']-saved['sgd_minus_adam']['mean']) < 1e-12
        assert np.allclose(actual['ci95'], saved['sgd_minus_adam']['ci95'], rtol=0, atol=1e-12)
        winner_details.append(dict(function=f, recipe=recipe, mean=mean, gap=actual, SGD_seed_wins=sum(v<0 for v in gaps)))
    actual_checks = []
    for check in analysis['checks']:
        f, recipe, kind = check['function'], check.get('recipe', 'LN010_w192'), check['id']
        if kind == 'fixed_head_benefit_transfer':
            passed = np.mean(chord_values[(f,recipe,'frozen_head','SGD','centered_chord')]) < 0
        elif kind == 'winner_transfer':
            w = next(x for x in winner_details if x['function']==f and x['recipe']==recipe and x['mean']==check['mean'])
            passed = w['gap']['mean'] > 0 if check['mean'] == 0 else w['gap']['mean'] < 0
        elif kind == 'translation_transfer':
            opt = check['optimizer']
            diff = max(abs(measured[(f,recipe,m,'matched_bias',opt,s)]['mse']-measured[(f,recipe,0,'matched_bias',opt,s)]['mse']) for m in [-3,3] for s in prereg['seeds'])
            assert abs(diff-check['max_mse_change']) < 1e-12
            passed = diff <= .001
        elif kind == 'fixed_features_transfer':
            affine = max(float(np.max(np.abs(measured[(f,recipe,-3,'frozen_hidden','SGD',s)]['predictions']+measured[(f,recipe,3,'frozen_hidden','SGD',s)]['predictions']-2*measured[(f,recipe,0,'frozen_hidden','SGD',s)]['predictions']))) for s in prereg['seeds'])
            assert affine == check['max_affine_second_difference']
            passed = affine <= 1e-4 and min(min(chord_values[(f,recipe,'frozen_hidden','SGD','total_chord')]), min(chord_values[(f,recipe,'frozen_hidden','SGD','centered_chord')])) >= -1e-5
        elif kind == 'LN_boundary':
            narrow = np.mean(chord_values[(f,'noLN_w64','frozen_head','SGD','centered_chord')])
            wide = np.mean(chord_values[(f,'LN010_w192','frozen_head','SGD','centered_chord')])
            assert wide < 0
            passed = narrow > .5*wide
        else:
            raise ValueError(kind)
        assert bool(passed) == check['pass']
        actual_checks.append(dict(id=kind, passed=bool(passed)))
    summary = {kind: dict(passed=sum(c['passed'] for c in actual_checks if c['id']==kind), total=sum(c['id']==kind for c in actual_checks)) for kind in set(c['id'] for c in actual_checks)}
    assert summary == analysis['predictions']
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), study='A02_ood_residual', audit='PASS', training=False, all_960_contracts_sources_arrays_verified=True, data_generators_exact=True, pairing_and_freezing_verified=True, predictions_recomputed=summary, seal=seal, original_git_history_available=history_available, chronology_evidence='Original Git and saved seal' if history_available else 'Saved seal; original Git verification in released audit record', first_saved_at=min(row['saved_at'] for row in rows), high_loss_retained=sum(row['test_mse']>2 for row in rows), failed_runs=sum(row['failed'] for row in rows), frozen_head_chords=[c for c in analysis['chords'] if c['mode']=='frozen_head' and c['optimizer']=='SGD'], baseline_winner_details=winner_details, qualifiers=['LN boundary confounds LN010-to-zero with width192-to64.', 'Four target structures are independent conditions; seeds and recipe variants are dependent.', 'Two failed mean signs are quadratic8 +/-3; both CIs cross zero.', 'Four nonzero baseline LN mean contrasts have CIs crossing zero.', 'Translation/fixed-feature algebra checks do not substitute for winner OOD.', 'Generator metadata stores base seed; actual seed is base+function index.'])
    Path(__file__).with_name('independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({k:v for k,v in result.items() if k not in ['frozen_head_chords', 'baseline_winner_details']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
