"""只读保存数组，独立闭式响应复算；不调用训练器。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r019_population_label_normalization'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    prereg = json.loads((STUDY / 'executed/pretraining_commit_verification.json').read_text())
    commit = prereg['preregistration_commit']
    commit_time = datetime.fromisoformat(subprocess.check_output(
        ['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip())
    pins = {p: sha(STUDY / p) for p in ['preregistration.json', 'executed/run.py', 'analysis.py']}
    for relative in pins:
        source = STUDY / relative
        assert subprocess.check_output(['git', 'show', f'{commit}:{source.relative_to(ROOT)}'], cwd=ROOT) == source.read_bytes()
    for relative, expected in config['input_sha256'].items():
        assert sha(ROOT / relative) == expected
    old_files = json.loads((STUDY / 'executed/prior_closeout_audit.json').read_text())['files']
    for row in old_files:
        p = ROOT / row['path']
        assert sha(p) == row['sha256'] and p.stat().st_mtime_ns == row['mtime_ns']
    new_files = sorted(list((STUDY / 'results').glob('*')) + list((STUDY / 'executed').glob('data_*')))
    before = {str(p.relative_to(STUDY)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in new_files}
    rows = []
    for label, expected in sorted(receipt['cells'].items()):
        path = STUDY / 'results' / (label + '.json')
        row = json.loads(path.read_text())
        assert sha(path) == expected and sha(path.with_suffix('.npz')) == row['arrays_sha256']
        assert row['pins']['sha256'] == pins and row['pins']['git_commit'] == commit
        assert commit_time < datetime.fromisoformat(row['started_at'])
        data_path = STUDY / row['data_file']
        assert sha(data_path) == row['data_sha256']
        contract = {'cell': row['cell'], 'pins': pins, 'data_sha256': sha(data_path)}
        assert hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() == row['contract_sha256']
        manifest = json.loads(data_path.with_suffix('.json').read_text())
        source = ROOT / manifest['source_file']
        assert sha(source) == manifest['source_sha256']
        with np.load(path.with_suffix('.npz')) as z, np.load(data_path) as d, np.load(source) as od:
            arrays = {k: z[k] for k in z.files}
            data = {k: d[k] for k in d.files}
            old_data = {k: od[k] for k in od.files}
        assert set(data) == set(config['data_contract']['invariants']) and len(data) == 9
        for key, value in data.items():
            assert np.array_equal(value, old_data[key])
            assert hashlib.sha256(value.tobytes()).hexdigest() == manifest['arrays'][key]['sha256']
        assert all(np.isfinite(v).all() for v in list(arrays.values()) + list(data.values()))
        for split in ['train', 'audit']:
            x0 = data[split + '_x']
            assert np.array_equal((x0[:, 0] + .5*x0[:, 0]*x0[:, 1])/np.sqrt(1.25), data[split + '_y'])
        n, variance = row['cell']['n'], row['cell']['noise_variance']
        x, tx, y, ty = data['train_features'], data['audit_features'], data['train_y'], data['audit_y']
        eta, steps = config['optimizer']['lr'], config['optimizer']['steps']
        kernel = np.einsum('ik,jk->ij', x, x, optimize=False) / n
        ev, u = np.linalg.eigh(kernel)
        mapping = np.einsum('ni,nj->ij', x, u, optimize=False) / n
        audit_map = np.einsum('mi,ij->mj', tx, mapping, optimize=False)
        gram = np.einsum('mi,mj->ij', audit_map, audit_map, optimize=False) / len(tx)
        target = np.einsum('mi,m->i', audit_map, ty, optimize=False) / len(tx)
        signal = np.einsum('ij,i->j', u, y, optimize=False)
        time_grid = np.arange(steps + 1)[:, None]
        alpha = 2*eta*ev
        factors = np.empty((steps + 1, n))
        nonzero = ev != 0
        factors[:, nonzero] = -np.expm1(time_grid*np.log1p(-alpha[nonzero]))/ev[nonzero]
        factors[:, ~nonzero] = 2*eta*time_grid
        coefficients = factors*signal
        bias = np.einsum('ti,ij,tj->t', coefficients, gram, coefficients, optimize=False) - 2*np.einsum('ti,i->t', coefficients, target, optimize=False) + np.mean(ty**2)
        noise = np.einsum('ti,i->t', factors**2, np.diag(gram), optimize=False)
        risk = bias + variance*noise
        response = np.einsum('ij,kj->ik', audit_map*factors[-1], u, optimize=False)
        direct_variance = float(np.sum(response**2)/len(tx))
        final_bias = float(np.mean((np.einsum('mi,i->m', tx, arrays['final_heads'][:, 1], optimize=False) - ty)**2))
        head_map = np.einsum('ij,kj->ik', mapping*factors[-1], u, optimize=False)
        clean_head = np.einsum('ij,j->i', head_map, y, optimize=False)
        noise_head = variance**.5*np.einsum('ij,j->i', head_map, data['epsilon'], optimize=False)
        heads = np.column_stack([clean_head+noise_head, clean_head, noise_head])
        errors = {
            'full_signal_bias_maxabs': float(np.max(np.abs(bias-arrays['signal_bias']))),
            'full_variance_maxabs': float(np.max(np.abs(noise-arrays['variance_unit']))),
            'full_expected_risk_maxabs': float(np.max(np.abs(risk-arrays['expected_risk']))),
            'endpoint_explicit_variance_maxabs': abs(direct_variance-arrays['variance_unit'][-1]),
            'endpoint_explicit_risk_maxabs': abs(final_bias+variance*direct_variance-arrays['expected_risk'][-1]),
            'final_heads_closed_form_maxabs': float(np.max(np.abs(heads-arrays['final_heads']))),
            'noisy_clean_noise_additivity_maxabs': float(np.max(np.abs(arrays['final_heads'][:, 0]-arrays['final_heads'][:, 1]-arrays['final_heads'][:, 2])))
        }
        assert max(errors.values()) <= 1e-9
        t = int(np.argmin(risk))
        saved = next(r for r in summary['cells'] if r['label'] == label)
        assert t == saved['t_star']
        rows.append({'label': label, **row['cell'], 't_star': t, 'errors': errors})
    for relative, expected in before.items():
        p = STUDY / relative
        assert sha(p) == expected['sha256'] and p.stat().st_mtime_ns == expected['mtime_ns']
    assert len(rows) == 8
    maxima = {k: max(r['errors'][k] for r in rows) for k in rows[0]['errors']}
    result = {'status': 'verified', 'round': 51, 'checked_at': datetime.now(timezone.utc).isoformat(), 'preregistration_commit': commit, 'commit_precedes_all_cells': True, 'verified_cells': 8, 'input_pins_verified': 76, 'prior_files_hash_mtime_verified': len(old_files), 'new_success_files_hash_mtime_verified': len(before), 'new_success_files': before, 'summary_sha256': sha(STUDY / 'summary.json'), 'source_sha256': sha(Path(__file__)), 'max_errors': maxima, 'cells': rows, 'method': '独立重建K并eigh；log1p/expm1闭式响应复算全步风险、最终显式噪声矩阵与三head。0训练，不调用measure，不覆盖成功结果。'}
    with (STUDY / 'executed/independent_verification.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write(chr(10))
    print(json.dumps({'status': result['status'], 'max_errors': maxima, 'verified_cells': 8}, ensure_ascii=False))


if __name__ == '__main__':
    main()
