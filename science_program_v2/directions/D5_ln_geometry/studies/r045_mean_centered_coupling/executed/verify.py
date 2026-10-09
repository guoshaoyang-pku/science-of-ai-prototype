from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', STUDY / 'executed/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
np, torch = runner.np, runner.torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-cell-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'executed/receipt.json').read_text())
    runner.gate(receipt['preregistration_commit'], cfg)
    first = (cfg['functions'][0], next(iter(cfg['recipes'])), cfg['seeds'][0])
    paths = [STUDY / 'results' / f'{first[0]}_{first[1]}_{first[2]}.json'] if args.first_cell_only else sorted((STUDY / 'results').glob('*.json'))
    assert len(paths) == (1 if args.first_cell_only else 40)
    audits, ell_values = [], {}
    for path in paths:
        row = json.loads(path.read_text())
        req = row['contract']
        key = tuple(req[n] for n in ('function', 'label', 'seed'))
        expected, _ = runner.request(*key, cfg, receipt)
        runner.validate(path, expected)
        with np.load(path.with_suffix('.npz')) as a:
            assert all(np.isfinite(a[n]).all() for n in a.files)
            torch.manual_seed(req['seed'])
            model = runner.Model(a['train_x'].shape[1], req['recipe'])
            selected = runner.selected_parameters(model)
            assert row['parameter_names'] == [n for n, _ in selected]
            assert row['parameter_count'] == sum(p.numel() for _, p in selected)
            assert set(a.files) == set(req['input_pins']) | {'prediction_initial', 'kernel_ones', 'centered_kernel_ones'} | {'vjp_ones__'+n for n, _ in selected}
            for n, pin in req['input_pins'].items():
                assert runner.pin(a[n]) == pin
            for n, _ in selected:
                assert runner.pin(a['vjp_ones__'+n]) == row['gradient_pins'][n]
            prediction = model(torch.from_numpy(a['train_x'])).ravel()
            assert np.array_equal(prediction.detach().numpy(), a['prediction_initial'])
            parameters = [p for _, p in selected]
            gradients = torch.autograd.grad(prediction.sum(), parameters, retain_graph=True)
            saved = np.concatenate([a['vjp_ones__'+n].ravel().astype(float) for n, _ in selected])
            rebuilt = np.concatenate([g.detach().numpy().ravel().astype(float) for g in gradients])
            assert np.array_equal(saved, rebuilt)
            # Forward-mode JVP is independent of the two-reverse-pass implementation.
            names = [n for n, _ in selected]
            base = {n: p.detach() for n, p in model.named_parameters()}
            inputs = tuple(base[n] for n in names)
            def function(*values):
                mapping = dict(base)
                mapping.update(zip(names, values))
                return torch.func.functional_call(model, mapping, (torch.from_numpy(a['train_x']),)).ravel()
            product = torch.func.jvp(function, inputs, tuple(g.detach() for g in gradients))[1]
            v = product.detach().numpy().astype(float)/len(prediction)
            u = v-v.mean()
            saved_u = a['centered_kernel_ones']
            assert runner.pin(a['kernel_ones']) == row['kernel_ones_pin']
            assert runner.pin(saved_u) == row['centered_kernel_ones_pin']
            assert np.array_equal(a['kernel_ones']-a['kernel_ones'].mean(), saved_u)
            ell = float(np.mean(saved_u**2))
            assert ell == row['mean_to_centered_coupling'] and ell > 0
            max_abs = float(np.max(np.abs(v-a['kernel_ones'])))
            centered_relative = float(np.linalg.norm(u-saved_u)/np.linalg.norm(saved_u))
            ell_relative = abs(float(np.mean(u**2))-ell)/ell
            assert max_abs <= 3e-6 and centered_relative <= 1e-4 and ell_relative <= 2e-4
            audit = {'function': key[0], 'label': key[1], 'seed': key[2],
                'forward_jvp_max_abs_error': max_abs, 'centered_relative_error': centered_relative,
                'ell_relative_error': ell_relative, 'vjp_byte_equal': True}
            if key == first:
                jacobian = []
                for i in range(len(prediction)):
                    pieces = torch.autograd.grad(prediction[i], parameters, retain_graph=i<len(prediction)-1)
                    jacobian.append(np.concatenate([g.detach().numpy().ravel() for g in pieces]))
                j = np.asarray(jacobian, dtype=float)
                direction = np.sum(j, axis=0)
                explicit_v = np.einsum('ip,p->i', j, direction, optimize=False)/len(prediction)
                explicit_u = explicit_v-explicit_v.mean()
                relative = float(np.linalg.norm(explicit_u-saved_u)/np.linalg.norm(saved_u))
                er = abs(float(np.mean(explicit_u**2))-ell)/ell
                error = float(np.max(np.abs(explicit_v-a['kernel_ones'])))
                assert error <= 3e-6 and relative <= 1e-4 and er <= 2e-4
                audit['explicit_jacobian'] = {'shape': list(j.shape), 'max_abs_error': error,
                    'centered_relative_error': relative, 'ell_relative_error': er}
            ell_values[key] = ell
            audits.append(audit)
    result = {'status': 'passed', 'verified_measurement_cells': len(paths), 'new_training_cells': 0,
              'derivative_checks': audits, 'old_pinned_files_verified': len(json.loads((STUDY/'executed/input_manifest.json').read_text())['files'])}
    if not args.first_cell_only:
        summary = json.loads((STUDY / 'summary.json').read_text())
        rebuilt = []
        for fn in cfg['functions']:
            for seed in cfg['seeds']:
                chords, qs = {}, {}
                for label in cfg['recipes']:
                    variances = []
                    for offset in cfg['offsets']:
                        with np.load(runner.OLD/'results'/f'{fn}_{label}_{offset}_{seed}.npz') as a:
                            variances.append(float(np.var(a['predictions_256'].astype(float)-a['test_y'].astype(float))))
                    chords[label] = .5*(variances[0]+variances[2])-variances[1]
                    qrow = json.loads((runner.RAYLEIGH/'results'/f'{fn}_{label}_{seed}.json').read_text())
                    with np.load(runner.RAYLEIGH/'results'/f'{fn}_{label}_{seed}.npz') as a:
                        gradient = np.concatenate([a['vjp__'+n].astype(float).ravel() for n in qrow['parameter_names']])
                        yc = a['centered_target']
                        qs[label] = float(np.dot(gradient,gradient)/(len(yc)*np.dot(yc,yc)))
                row = {'function': fn, 'seed': seed, 'B': chords['noLN_w64']-chords['LN010_w64'],
                       'X_L': float(np.log10(ell_values[(fn,'noLN_w64',seed)]/ell_values[(fn,'LN010_w64',seed)])),
                       'X_R': float(np.log10(qs['LN010_w64']/qs['noLN_w64']))}
                old = next(r for r in summary['same_seed_pairs'] if r['function']==fn and r['seed']==seed)
                assert all(abs(row[n]-old[n])<1e-10 for n in ('B','X_L','X_R'))
                rebuilt.append(row)
        for centered, label in ((False,'overall'),(True,'function_centered')):
            c = {n: np.asarray([r[n] for r in rebuilt]) for n in ('B','X_L','X_R')}
            if centered:
                for fn in cfg['functions']:
                    mask = np.asarray([r['function']==fn for r in rebuilt])
                    for a in c.values():
                        a[mask] -= a[mask].mean()
            for name, key in (('X_L','coupling_pearson_r'),('X_R','rayleigh_pearson_r')):
                r = float(np.corrcoef(c[name],c['B'])[0,1])
                assert abs(r-summary[label][key])<1e-10
        result['old_training_cells_verified'] = 120
        result['old_rayleigh_cells_verified'] = 40
        result['correlations_independently_verified'] = True
    name = 'first_cell_verification.json' if args.first_cell_only else 'saved_evidence_verification.json'
    runner.save(STUDY / 'executed' / name, result)
    print(json.dumps({'status': result['status'], 'cells': len(paths), 'first_check': audits[0]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
