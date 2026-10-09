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


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x, y = x-x.mean(), y-y.mean()
    return float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y)))


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
    if args.first_cell_only:
        paths = [STUDY / 'results' / f'{first[0]}_{first[1]}_{first[2]}.json']
    else:
        paths = sorted((STUDY / 'results').glob('*.json'))
        assert len(paths) == 40
    metrics, qs = [], {}
    for path in paths:
        row = json.loads(path.read_text())
        req = row['contract']
        key = tuple(req[name] for name in ('function', 'label', 'seed'))
        expected, _ = runner.request(*key, cfg, receipt)
        runner.validate(path, expected)
        with np.load(path.with_suffix('.npz')) as arrays:
            torch.manual_seed(req['seed'])
            model = runner.Model(arrays['train_x'].shape[1], req['recipe'])
            selected = runner.selected_parameters(model)
            assert [name for name, _ in selected] == row['parameter_names']
            x = torch.from_numpy(arrays['train_x'])
            y = arrays['train_y'].ravel().astype(np.float64)
            yc = y-y.mean()
            prediction = model(x).ravel()
            assert np.array_equal(prediction.detach().numpy(), arrays['prediction_initial'])
            saved = np.concatenate([arrays['vjp__'+name].ravel().astype(np.float64)
                                    for name, _ in selected])
            gradient = torch.autograd.grad(torch.sum(prediction.double()*torch.from_numpy(yc)),
                        [p for _, p in selected], retain_graph=(key == first))
            regenerated = np.concatenate([g.detach().numpy().ravel().astype(np.float64) for g in gradient])
            assert np.array_equal(saved, regenerated)
            denominator = len(y)*np.dot(yc, yc)
            q = float(np.dot(saved, saved)/denominator)
            assert abs(q-row['target_rayleigh']) < 1e-14
            qs[key] = q
            audit = {'function': key[0], 'label': key[1], 'seed': key[2],
                     'regenerated_vjp_max_abs_error': float(np.max(np.abs(saved-regenerated))),
                     'saved_q_abs_error': abs(q-row['target_rayleigh'])}
            if key == first:
                jacobian = []
                for i in range(len(y)):
                    pieces = torch.autograd.grad(prediction[i], [p for _, p in selected],
                                                 retain_graph=i < len(y)-1)
                    jacobian.append(np.concatenate([g.detach().numpy().ravel() for g in pieces]))
                jacobian = np.asarray(jacobian, dtype=np.float64)
                explicit = np.einsum('ip,i->p', jacobian, yc, optimize=False)
                error = float(np.max(np.abs(explicit-saved)))
                relative = float(np.linalg.norm(explicit-saved)/np.linalg.norm(saved))
                explicit_q = float(np.dot(explicit, explicit)/denominator)
                q_relative = abs(explicit_q-q)/q
                assert error <= 3e-6 and relative <= 3e-5 and q_relative <= 3e-5
                audit['explicit_jacobian'] = {'shape': list(jacobian.shape),
                    'gradient_max_abs_error': error, 'gradient_l2_relative_error': relative,
                    'q_relative_error': q_relative, 'q': explicit_q}
            metrics.append(audit)
    result = {'status': 'passed', 'verified_measurement_cells': len(paths),
              'new_training_cells': 0, 'derivative_checks': metrics}
    if not args.first_cell_only:
        summary = json.loads((STUDY / 'summary.json').read_text())
        pairs = []
        gram_error = chord_error = 0.0
        for fn in cfg['functions']:
            for seed in cfg['seeds']:
                values = {}
                for label in cfg['recipes']:
                    losses, logk = [], []
                    for offset in cfg['offsets']:
                        path = runner.OLD / 'results' / f'{fn}_{label}_{offset}_{seed}.npz'
                        with np.load(path) as a:
                            residual = a['predictions_256'].astype(np.float64)-a['test_y'].astype(np.float64)
                            losses.append(float(np.var(residual)))
                            h = a['hidden_256']
                            hc = h-np.mean(h, axis=0)
                            g = np.einsum('ni,nj->ij', hc, hc, optimize=False)/len(h)
                            eig = np.linalg.eigvalsh(g)
                            logk.append(float(np.log10(max(eig[-1],1e-12)/max(eig[0],1e-12))))
                    values[label] = {'C': .5*(losses[0]+losses[2])-losses[1],
                                     'G': float(np.mean(logk))}
                ln, no = values['LN010_w64'], values['noLN_w64']
                pair = {'function': fn, 'seed': seed, 'B': no['C']-ln['C'],
                        'X_G': no['G']-ln['G'],
                        'X_R': float(np.log10(qs[(fn,'LN010_w64',seed)]/qs[(fn,'noLN_w64',seed)]))}
                observed = next(r for r in summary['same_seed_pairs'] if r['function']==fn and r['seed']==seed)
                for name in ('B','X_G','X_R'):
                    assert abs(pair[name]-observed[name]) < 1e-10
                gram_error = max(gram_error, abs(pair['X_G']-observed['X_G']))
                chord_error = max(chord_error, abs(pair['B']-observed['B']))
                pairs.append(pair)
        for centered, label in ((False,'overall'), (True,'function_centered')):
            columns = {n:np.asarray([r[n] for r in pairs]) for n in ('X_R','X_G','B')}
            if centered:
                for fn in cfg['functions']:
                    mask = np.asarray([r['function']==fn for r in pairs])
                    for a in columns.values():
                        a[mask] -= a[mask].mean()
            rr, rg = corr(columns['X_R'],columns['B']), corr(columns['X_G'],columns['B'])
            assert abs(rr-summary[label]['rayleigh_pearson_r']) < 1e-10
            assert abs(rg-summary[label]['gram_pearson_r']) < 1e-10
        result['old_training_cells_verified'] = 120
        result['maximum_gram_predictor_error'] = gram_error
        result['maximum_chord_benefit_error'] = chord_error
        result['correlations_verified'] = True
    name = 'first_cell_verification.json' if args.first_cell_only else 'saved_evidence_verification.json'
    runner.save(STUDY / 'executed' / name, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
