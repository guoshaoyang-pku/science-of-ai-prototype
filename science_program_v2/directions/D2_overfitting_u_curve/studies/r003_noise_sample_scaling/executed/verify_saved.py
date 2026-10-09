import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
FREEZE = 'd5fae2cdbf2eaf66f598e74f001684a4be98ba4d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = STUDY / 'executed' / args.output
    assert output.parent == STUDY / 'executed'
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    paths = [STUDY / p for p in ('preregistration.json', 'executed/run.py', 'analysis.py')]
    pins = {p.relative_to(STUDY).as_posix(): sha(p) for p in paths}
    for p in paths:
        assert subprocess.check_output(['git', 'show', f'{FREEZE}:{p.relative_to(ROOT)}'], cwd=ROOT) == p.read_bytes()
    assert pins['executed/run.py'] == prereg['execution_source_sha256']
    assert pins['analysis.py'] == prereg['analysis_source_sha256']
    audit = json.loads((STUDY / 'executed/round011_static_audit.json').read_text())
    call = json.loads((STUDY / 'executed/run_receipt_001.json').read_text())
    freeze_time = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', FREEZE], cwd=ROOT, text=True).strip())
    assert freeze_time < datetime.fromisoformat(audit['checked_at']) < datetime.fromisoformat(call['started_at'])
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    analyzed = {e['label']: e for e in summary['cells']}
    rows = []
    for label, metadata_sha in sorted(receipt['cells'].items()):
        p = STUDY / 'results' / (label + '.json')
        row = json.loads(p.read_text())
        assert sha(p) == metadata_sha
        assert row['status'] == 'completed' and row['finite']
        assert row['pins']['sha256'] == pins
        assert sha(p.with_suffix('.npz')) == row['arrays_sha256']
        assert sha(STUDY / row['data_file']) == row['data_sha256']
        contract = {'cell': row['cell'], 'pins': pins, 'data_sha256': row['data_sha256']}
        assert hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() == row['contract_sha256']
        manifest = json.loads((STUDY / row['data_file']).with_suffix('.json').read_text())
        assert manifest['file_sha256'] == row['data_sha256']
        commit = row['pins']['git_commit']
        subprocess.run(['git', 'merge-base', '--is-ancestor', FREEZE, commit], cwd=ROOT, check=True)
        for source in paths:
            assert subprocess.check_output(['git', 'show', f'{commit}:{source.relative_to(ROOT)}'], cwd=ROOT) == source.read_bytes()
        with np.load(STUDY / row['data_file']) as d, np.load(p.with_suffix('.npz')) as a:
            for key in d.files:
                assert hashlib.sha256(d[key].tobytes()).hexdigest() == manifest['arrays'][key]['sha256']
                assert list(d[key].shape) == manifest['arrays'][key]['shape']
                assert str(d[key].dtype) == manifest['arrays'][key]['dtype']
                assert np.isfinite(d[key]).all()
            assert all(np.isfinite(a[key]).all() for key in a.files)
            x, tx, y, ty = d['train_features'], d['audit_features'], d['train_y'], d['audit_y']
            n, lr, steps = row['cell']['n'], prereg['optimizer']['lr'], prereg['optimizer']['steps']
            gram = np.einsum('ki,kj->ij', tx, tx, optimize=False) / len(tx)
            projection = np.einsum('ki,k->i', tx, ty, optimize=False) / len(tx)
            train_kernel = np.einsum('ik,jk->ij', x, x, optimize=False) / n
            eigen_error = np.max(np.abs(np.einsum('ij,jk->ik', train_kernel, a['eigenvectors'], optimize=False) - a['eigenvectors'] * a['eigenvalues']))
            gram_error = np.max(np.abs(gram - a['audit_gram']))
            projection_error = np.max(np.abs(projection - a['audit_target_projection']))
            spectral_map = np.einsum('ki,kj->ij', x, a['eigenvectors'], optimize=False) / n
            audit_map = np.einsum('ki,ij->kj', tx, spectral_map, optimize=False)
            diagonal = np.mean(audit_map ** 2, axis=0)
            basis_diagonal_error = np.max(np.abs(diagonal - np.diag(a['basis_gram'])))
            q = 1 - 2 * lr * a['eigenvalues']
            f = np.zeros(n)
            for _ in range(steps):
                f = q * f + 2 * lr
            noise_response = np.einsum('ij,kj->ik', audit_map * f, a['eigenvectors'], optimize=False)
            variance_final = np.sum(noise_response ** 2) / len(tx)
            variance_error = abs(variance_final - a['variance_unit'][-1])
            prediction = np.einsum('ki,i->k', tx, a['final_heads'][:, 1], optimize=False)
            direct_bias = np.mean((prediction - ty) ** 2)
            bias_error = abs(direct_bias - a['signal_bias'][-1])
            risk_error = abs(direct_bias + row['cell']['noise_variance'] * variance_final - a['expected_risk'][-1])
            assert max(eigen_error, gram_error, projection_error, basis_diagonal_error) <= 1e-12
            assert max(variance_error, bias_error, risk_error) <= 1e-9
            assert analyzed[label]['P1_pass']
            assert analyzed[label]['t_star'] == int(np.argmin(a['expected_risk']))
        rows.append({'label': label, 'metadata_sha256': metadata_sha, 'arrays_sha256': row['arrays_sha256'],
                     'data_sha256': row['data_sha256'], 'eigen_residual_maxabs': float(eigen_error),
                     'audit_gram_maxabs': float(gram_error), 'audit_projection_maxabs': float(projection_error),
                     'basis_diagonal_maxabs': float(basis_diagonal_error),
                     'final_noise_response_variance_maxabs': float(variance_error),
                     'final_direct_bias_maxabs': float(bias_error), 'final_expected_risk_maxabs': float(risk_error)})
    assert rows and len(rows) == summary['counts']['saved_cells']
    result = {'round': 11, 'status': 'verified', 'checked_at': datetime.now().astimezone().isoformat(),
              'frozen_preregistration_commit': FREEZE, 'freeze_time': freeze_time.isoformat(),
              'pretraining_audit_time': audit['checked_at'], 'first_training_started_at': call['started_at'],
              'verified_cells': len(rows), 'all_cells_saved': len(rows) == prereg['counts']['planned_cells'],
              'summary_sha256': sha(STUDY / 'summary.json'),
              'method': '保存数据的einsum逐项乘积复算；显式train-noise到audit响应矩阵核验方差，无训练、无BLAS matmul。', 'cells': rows}
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write(chr(10))
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
