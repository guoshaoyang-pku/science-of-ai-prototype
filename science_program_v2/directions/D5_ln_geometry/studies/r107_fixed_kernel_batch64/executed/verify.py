from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    receipt = json.loads((STUDY/'executed/receipt.json').read_text())
    manifest = json.loads((STUDY/'executed/input_manifest.json').read_text())
    for rel, item in manifest['files'].items():
        path = ROOT/rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        assert path.stat().st_mtime_ns == item['mtime_ns']
    max_step_error = max_chord_error = max_real_error = 0.0
    count = 0
    result_pins = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text())
        request = row['contract']
        assert request['preregistration_commit'] == receipt['preregistration_commit']
        assert row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch']
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == row['arrays_sha256']
        source = request['source']
        with np.load(ROOT/Path(source['kernel_result_json']).with_suffix('.npz'), allow_pickle=False) as old:
            source_kernel = old[source['kernel_array']]
            source_vector = old[source['full_vector_array']]
            source_train = {name: old[name].copy() for name in old.files if name.startswith('train_')}
        with np.load(ROOT/Path(source['batch_result_json']).with_suffix('.npz'), allow_pickle=False) as old:
            source_batches = old['batch_indices'].copy()
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
            assert all(np.isfinite(data[name]).all() for name in data.files)
            assert np.array_equal(data['kernel'], source_kernel)
            assert np.array_equal(data['batch_indices'], source_batches)
            assert np.array_equal(data['full_propagated_ones_256'], source_vector)
            assert all(np.array_equal(data[name], value) for name, value in source_train.items())
            kernel, trajectory = data['kernel'], data['propagated_ones_trajectory']
            assert trajectory.shape == (257, 256) and np.array_equal(trajectory[0], np.ones(256))
            independent = np.ones(256, dtype=np.float64)
            for step, batch in enumerate(data['batch_indices']):
                counts = np.bincount(batch, minlength=256).astype(np.float64)
                weighted = independent * counts
                change = np.sum(kernel * weighted[np.newaxis, :], axis=1)
                predicted = independent - .008 * change
                error = float(np.max(np.abs(predicted-trajectory[step+1])))
                max_step_error = max(max_step_error, error)
                independent = predicted
            final = independent.tolist()
            mean = math.fsum(final)/256
            chord = 9*math.fsum((value-mean)**2 for value in final)/256
            max_chord_error = max(max_chord_error, abs(chord-row['batch64_train_chord_256']))
            assert np.array_equal(data['centered_propagated_ones_256'], trajectory[-1]-trajectory[-1].mean())
            energies = []
            for offset in (-3, 0, 3):
                residual = [float(a)-float(b) for a,b in zip(data[f'train_predictions_256__{offset}'].ravel(), data[f'train_y__{offset}'].ravel())]
                mean_r = math.fsum(residual)/256
                energies.append(math.fsum((value-mean_r)**2 for value in residual)/256)
            scalar_real = (energies[0]+energies[2])/2-energies[1]
            source_energies = []
            for offset in (-3, 0, 3):
                residual = (source_train[f'train_predictions_256__{offset}'].astype(np.float64).ravel()
                            - source_train[f'train_y__{offset}'].astype(np.float64).ravel())
                source_energies.append(float(np.mean((residual-residual.mean())**2)))
            source_real = (source_energies[0]+source_energies[2])/2-source_energies[1]
            max_real_error = max(max_real_error, abs(scalar_real-source_real))
        for p in (path, path.with_suffix('.npz')):
            result_pins[str(p.relative_to(ROOT))] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'mtime_ns': p.stat().st_mtime_ns}
        count += 1
    assert max_step_error <= 1e-12 and max_chord_error <= 1e-12 and max_real_error <= 1e-12
    if not args.allow_partial:
        assert count == 40
        spec = importlib.util.spec_from_file_location('batch_analysis', STUDY/'analysis.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.compute() == json.loads((STUDY/'summary.json').read_text())
    result = {'status': 'passed', 'evaluated_cells': count, 'new_training_cells': 0, 'new_kernel_cells': 0,
              'protected_file_count': len(manifest['files']), 'protected_hash_mtime_unchanged': True,
              'independent_update': 'counts per index then sum over 256 unique columns; duplicates retain multiplicity',
              'max_one_step_absolute_error': max_step_error, 'max_scalar_chord_error': max_chord_error,
              'max_scalar_real_chord_error': max_real_error, 'summary_recomputes': not args.allow_partial,
              'result_pins': result_pins}
    target = 'first_cell_verification.json' if args.allow_partial else 'saved_evidence_verification.json'
    (STUDY/'executed'/target).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'result_pins'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
