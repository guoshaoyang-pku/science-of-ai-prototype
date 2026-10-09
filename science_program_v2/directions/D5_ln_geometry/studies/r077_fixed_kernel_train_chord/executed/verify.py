from pathlib import Path
import argparse
import importlib.util
import json
import time
import run
import numpy as np
import torch

STUDY, ROOT = run.STUDY, run.ROOT


def verify(commit):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((STUDY/'preregistration.json').read_text())
    receipt = run.gate(commit, cfg)
    errors = {'kernel_probe_relative': 0., 'old_kernel_ones_absolute': 0.,
              'old_centered_kernel_ones_relative': 0., 'old_coupling_relative': 0.,
              'spectral_vector_absolute': 0., 'spectral_chord_relative': 0.,
              'real_train_chord_absolute': 0.}
    results = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        req = row['contract']
        expected, _ = run.request(req['function'], req['label'], req['seed'], cfg, receipt)
        assert req == expected
        run.validate(path, req)
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        with np.load(path.with_suffix('.npz')) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            k = z['kernel']
            assert run.pin(k) == row['kernel_pin']
            assert np.max(np.abs(k-k.T)) < 1e-12
            eigenvalues, vectors = np.linalg.eigh(k)
            assert np.max(np.abs(eigenvalues-z['kernel_eigenvalues'])) < 1e-10
            coeff = np.einsum('ij,i->j', vectors, np.ones(256), optimize=False)
            for step in (1, 256):
                v = np.einsum('ij,j->i', vectors, coeff*(1-.002*eigenvalues)**step, optimize=False)
                error = float(np.max(np.abs(v-z[f'propagated_ones_{step}'])))
                errors['spectral_vector_absolute'] = max(errors['spectral_vector_absolute'], error)
                assert error < 1e-10
                centered = v-v.mean()
                chord = float(9*np.mean(centered**2))
                reference = row['fixed_kernel_train_chord'][str(step)]
                relative = abs(chord-reference)/reference
                errors['spectral_chord_relative'] = max(errors['spectral_chord_relative'], relative)
                assert relative < 1e-6
            oldpath = run.COUPLING/'results'/path.name
            oldrow = json.loads(oldpath.read_text())
            with np.load(oldpath.with_suffix('.npz')) as old:
                k1 = np.sum(k, axis=1)
                pk1 = k1-k1.mean()
                aerr = float(np.max(np.abs(k1-old['kernel_ones'])))
                perr = float(np.linalg.norm(pk1-old['centered_kernel_ones'])/np.linalg.norm(old['centered_kernel_ones']))
                lerr = abs(float(np.mean(pk1**2))-oldrow['mean_to_centered_coupling'])/oldrow['mean_to_centered_coupling']
                errors['old_kernel_ones_absolute'] = max(errors['old_kernel_ones_absolute'], aerr)
                errors['old_centered_kernel_ones_relative'] = max(errors['old_centered_kernel_ones_relative'], perr)
                errors['old_coupling_relative'] = max(errors['old_coupling_relative'], lerr)
                assert aerr < 3e-6 and perr < 1e-4 and lerr < 2e-4
            torch.manual_seed(req['seed'])
            model = run.Model(z['train_x'].shape[1], req['recipe'])
            selected = run.selected_parameters(model)
            names = [name for name, _ in selected]
            params = {name: p for name, p in model.named_parameters()}
            x = torch.from_numpy(z['train_x'])
            output = model(x).ravel()
            assert np.array_equal(output.detach().numpy(), z['prediction_initial'])
            probes = np.random.default_rng(7700+req['seed']).normal(size=(2, 256)).astype(np.float32)
            for probe in probes:
                grads = torch.autograd.grad((output*torch.from_numpy(probe)).sum(), [params[name] for name in names], retain_graph=True)
                tangents = {name: torch.zeros_like(p) for name, p in params.items()}
                tangents.update({name: g.detach() for name, g in zip(names, grads)})
                def prediction(p):
                    return torch.func.functional_call(model, p, (x,)).ravel()
                _, jg = torch.func.jvp(prediction, (params,), (tangents,))
                reference = np.einsum('ij,j->i', k, probe.astype(np.float64), optimize=False)
                independent = jg.detach().numpy().astype(np.float64)/256
                relative = float(np.linalg.norm(reference-independent)/np.linalg.norm(reference))
                errors['kernel_probe_relative'] = max(errors['kernel_probe_relative'], relative)
                assert relative < 2e-5
            # Independent direct residual sum of squares from original pinned train arrays.
            energy = {}
            for m in cfg['offsets']:
                source = run.OLD/'results'/f"{req['function']}_{req['label']}_{m}_{req['seed']}.npz"
                with np.load(source) as old:
                    assert np.array_equal(old['train_y'], z[f'train_y__{m}'])
                    assert np.array_equal(old['train_predictions_256'], z[f'train_predictions_256__{m}'])
                    r = old['train_predictions_256'].astype(np.float64).ravel()-old['train_y'].astype(np.float64).ravel()
                    energy[m] = sum(float(v-r.mean())**2 for v in r)/256
            real = .5*(energy[-3]+energy[3])-energy[0]
            results[(req['function'], req['label'], req['seed'])] = real
    spec = importlib.util.spec_from_file_location('analysis', STUDY/'analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.calculate()
    if (STUDY/'summary.json').exists():
        assert summary == json.loads((STUDY/'summary.json').read_text())
    for pair in summary['pairs']:
        delta = results[(pair['function'], 'LN010_w64', pair['seed'])]-results[(pair['function'], 'noLN_w64', pair['seed'])]
        error = abs(delta-pair['real_train_delta_chord'])
        errors['real_train_chord_absolute'] = max(errors['real_train_chord_absolute'], error)
        assert error < 1e-12
    destination = 'first_cell_verification.json' if len(results) == 1 else 'saved_evidence_verification.json'
    run.save(STUDY/'executed'/destination, {'status': 'passed', 'verified_epoch': time.time(),
             'verified_cells': len(results), 'new_training_cells': 0, 'errors': errors,
             'result_pins': {str(p.relative_to(ROOT)): {'sha256': run.sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                 for p in (STUDY/'results').iterdir()},
             'historical_files_unchanged': len(json.loads((STUDY/'executed/input_manifest.json').read_text())['files'])})
    print(json.dumps({'status': 'passed', 'verified_cells': len(results), 'errors': errors}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    verify(parser.parse_args().commit)
