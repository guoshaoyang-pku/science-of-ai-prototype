from __future__ import annotations
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, t

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def interval(values):
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values)-1)*np.std(values, ddof=1)/np.sqrt(len(values)))
    return {'mean': mean, 'min': float(min(values)), 'max': float(max(values)),
            'paired_seed_t95': [mean-half, mean+half]}


def main():
    cfg = json.loads((STUDY/'preregistration.json').read_text())
    summary = json.loads((STUDY/'summary.json').read_text())
    receipt = json.loads((STUDY/'executed/receipt.json').read_text())
    commit = receipt['preregistration_commit']
    for relative in ('preregistration.json', 'executed/run.py', 'analysis.py'):
        path = STUDY/relative
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes()
        if relative in cfg['source_sha256']:
            assert sha(path) == cfg['source_sha256'][relative]
    assert receipt['preregistration_sha256'] == sha(STUDY/'preregistration.json')
    assert receipt['commit_epoch'] <= receipt['gate_passed_at_epoch']
    spec = importlib.util.spec_from_file_location('frozen_runner', STUDY/'executed/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    expected = {(fn, label, offset, seed) for fn in cfg['functions'] for label in cfg['recipes']
                for offset in cfg['means'] for seed in cfg['seeds']}
    cells, regenerated = {}, {}
    maximum_gram_error = maximum_head_error = 0.0
    floor_hits = initial_lower = ln_training_lower = 0
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        req = row['contract']
        key = tuple(req[name] for name in ('function', 'label', 'mean', 'seed'))
        assert key in expected and key not in cells
        regenerated_req, inputs = runner.request(*key, cfg, receipt)
        assert req == regenerated_req and digest(req) == row['contract_sha256']
        assert sha(path.with_suffix('.npz')) == row['arrays_sha256']
        assert row['status'] == 'completed'
        assert row['started_at_epoch'] >= receipt['gate_passed_at_epoch']
        seed_key = (key[0], key[1], key[3])
        if seed_key not in regenerated:
            runner.torch.manual_seed(key[3])
            model = runner.Model(inputs['train_x'].shape[1], req['recipe'])
            pins = {name: runner.array_pin(parameter.detach().numpy())
                    for name, parameter in model.named_parameters() if '.norm.' not in name}
            batches = np.stack([runner.torch.randint(0, cfg['train_n'], (req['recipe']['batch_size'],)).numpy()
                                for _ in range(req['recipe']['steps'])])
            regenerated[seed_key] = (pins, batches)
        assert row['initial_linear'] == regenerated[seed_key][0]
        metrics = {}
        with np.load(path.with_suffix('.npz')) as arrays:
            assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
            for name, value in inputs.items():
                assert np.array_equal(arrays[name], value)
            assert np.array_equal(arrays['batch_indices'], regenerated[seed_key][1])
            for role in ('weight', 'bias'):
                assert np.array_equal(arrays[f'head_{role}_initial'], arrays[f'head_{role}_final'])
            for step in cfg['checkpoints']:
                gram = arrays[f'gram_{step}']
                assert gram.dtype == np.float64
                if step in (0, 256):
                    hidden = arrays[f'hidden_{step}']
                    centered = hidden-hidden.mean(0, keepdims=True)
                    recomputed = np.einsum('ni,nj->ij', centered, centered, optimize=False)/len(hidden)
                    error = float(np.max(np.abs(recomputed-gram)))
                    assert np.isfinite(recomputed).all() and error <= 1e-12
                    maximum_gram_error = max(maximum_gram_error, error)
                    predictions = np.einsum('ni,oi->no', hidden, arrays['head_weight_final'].astype(float),
                                            optimize=False)+arrays['head_bias_final']
                    error = float(np.max(np.abs(predictions-arrays[f'train_predictions_{step}'])))
                    assert np.isfinite(predictions).all() and error <= 1e-5
                    maximum_head_error = max(maximum_head_error, error)
                eig = np.linalg.eigvalsh(gram)[::-1]
                assert np.max(np.abs(eig-row['spectrum'][str(step)])) <= 1e-12
                assert eig[-1] >= -1e-12
                floor_hits += int(np.sum(eig < cfg['mediator']['eigenvalue_floor']))
                kappa = float(np.log10(max(eig[0], 1e-12)/max(eig[-1], 1e-12)))
                assert abs(kappa-row['log10_condition'][str(step)]) <= 1e-9
                residual = arrays[f'predictions_{step}'].astype(float)-arrays['test_y'].astype(float)
                metrics[str(step)] = {'log10_condition': kappa,
                    'centered_error': float(np.mean((residual-residual.mean())**2))}
        cells[key] = metrics
    assert set(cells) == expected
    contrasts, chord_pairs = [], []
    for fn in cfg['functions']:
        for offset in cfg['means']:
            deltas = []
            for seed in cfg['seeds']:
                ln = cells[(fn, 'LN010_w64', offset, seed)]
                no = cells[(fn, 'noLN_w64', offset, seed)]
                initial_lower += ln['0']['log10_condition'] < no['0']['log10_condition']
                ln_training_lower += ln['256']['log10_condition'] < ln['0']['log10_condition']
                deltas.append(ln['256']['log10_condition']-no['256']['log10_condition'])
            contrasts.append({'function': fn, 'offset': offset, 'paired_deltas': deltas,
                              'negative_seeds': int(np.sum(np.asarray(deltas) < 0)), **interval(deltas)})
        for seed in cfg['seeds']:
            chord = {}
            for label in cfg['recipes']:
                errors = {offset: cells[(fn, label, offset, seed)]['256']['centered_error']
                          for offset in cfg['means']}
                chord[label] = (errors[-3]+errors[3])/2-errors[0]
            delta_k = cells[(fn, 'LN010_w64', 0, seed)]['256']['log10_condition']-cells[(fn, 'noLN_w64', 0, seed)]['256']['log10_condition']
            chord_pairs.append({'function': fn, 'seed': seed, 'delta_log10_condition': delta_k,
                                'delta_chord': chord['LN010_w64']-chord['noLN_w64']})
    passes = sum(row['negative_seeds'] >= 3 for row in contrasts)
    r = float(pearsonr([row['delta_log10_condition'] for row in chord_pairs],
                       [row['delta_chord'] for row in chord_pairs]).statistic)
    assert passes == summary['predictions']['P1']['passing_function_offset_units']
    assert abs(r-summary['predictions']['P2']['pearson_r']) <= 1e-12
    for original, verified in zip(summary['spectrum_contrasts'], contrasts):
        assert original['function'] == verified['function']
        assert np.array_equal(original['LN_minus_noLN_log10_condition'], verified['paired_deltas'])
    result = {'status': 'passed', 'verified_cells': len(cells), 'method': 'einsum optimize=False独立重构，不调用训练',
        'regenerated_data_initialization_batches': True, 'frozen_source_sha256': cfg['source_sha256'],
        'verification_source_sha256': sha(Path(__file__)), 'preregistration_commit': commit,
        'all_arrays_finite': True, 'maximum_hidden_gram_abs_error': maximum_gram_error,
        'maximum_hidden_head_prediction_abs_error': maximum_head_error,
        'floor_hit_eigenvalue_count': floor_hits, 'P1_pass_units': passes, 'P1_total_units': 12,
        'P1_negative_pairs': sum(row['negative_seeds'] for row in contrasts), 'P1_total_pairs': 60,
        'P2_pearson_r': r, 'P2_status': 'supported' if r >= .5 else 'refuted',
        'initial_condition_LN_lower_pairs': int(initial_lower), 'LN_training_condition_decrease_cells': int(ln_training_lower),
        'spectrum_contrasts': contrasts, 'chord_pairs': chord_pairs,
        'analysis_label_erratum': '原冻结analysis.py把offset存为mean后被统计mean覆盖；原summary保留，本文件offset为正确条件标签。数值、P1/P2与原summary一致。',
        'warning_boundary': '原analysis.py的numpy matmul警告保留于full_analysis.json；本独立复算全部数组有限，Gram误差为0；警告根因未定。'}
    (STUDY/'executed/saved_evidence_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('spectrum_contrasts', 'chord_pairs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
