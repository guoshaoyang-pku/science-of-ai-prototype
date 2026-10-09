from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r003_noise_sample_scaling'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    prereg = json.loads((STUDY / 'executed/pretraining_commit_verification.json').read_text())
    commit = prereg['preregistration_commit']
    commit_time = datetime.fromisoformat(subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip())
    for relative, expected in config['input_sha256'].items():
        assert sha(ROOT / relative) == expected
    pins = {p: sha(STUDY / p) for p in ['preregistration.json', 'executed/run.py', 'analysis.py']}
    for relative in pins:
        source = STUDY / relative
        assert subprocess.check_output(['git', 'show', f'{commit}:{source.relative_to(ROOT)}'], cwd=ROOT) == source.read_bytes()
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
        old_path = OLD / 'results' / path.name
        old_row = json.loads(old_path.read_text())
        manifest = json.loads(data_path.with_suffix('.json').read_text())
        with np.load(path.with_suffix('.npz')) as z, np.load(data_path) as d, np.load(OLD / old_row['data_file']) as od, np.load(old_path.with_suffix('.npz')) as oz:
            arrays = {k: z[k] for k in z.files}
            data = {k: d[k] for k in d.files}
            old_data = {k: od[k] for k in od.files}
            old_arrays = {k: oz[k] for k in oz.files}
        assert all(np.isfinite(v).all() for v in list(arrays.values()) + list(data.values()))
        for key, value in data.items():
            assert hashlib.sha256(value.tobytes()).hexdigest() == manifest['arrays'][key]['sha256']
        unchanged = [key for key in ['train_x', 'audit_x', 'epsilon', 'hidden_weight', 'train_features', 'audit_features'] if np.array_equal(data[key], old_data[key])]
        assert len(unchanged) == 6
        for split in ['train', 'audit']:
            raw = data[split + '_x'][:, 0] * (1 + .5 * data[split + '_x'][:, 1])
            assert np.allclose(data[split + '_y'], raw / np.sqrt(1.25), rtol=0, atol=1e-14)
        assert np.array_equal(data['target_mean_rms'], [0., np.sqrt(1.25)])
        n, eta = row['cell']['n'], config['optimizer']['lr']
        x, tx, ty = data['train_features'], data['audit_features'], data['audit_y']
        ev, u = arrays['eigenvalues'], arrays['eigenvectors']
        kernel = np.einsum('ik,jk->ij', x, x, optimize=False) / n
        eigen_error = float(np.max(np.abs(np.einsum('ij,jk->ik', kernel, u, optimize=False) - u * ev)))
        mapping = np.einsum('ni,nj->ij', x, u, optimize=False) / n
        audit_map = np.einsum('mi,ij->mj', tx, mapping, optimize=False)
        signal = np.einsum('ij,i->j', u, data['train_y'], optimize=False)
        q = 1 - 2 * eta * ev
        factors = np.zeros((config['optimizer']['steps'] + 1, n))
        for step in range(len(factors) - 1):
            factors[step + 1] = q * factors[step] + 2 * eta
        gram = np.einsum('mi,mj->ij', audit_map, audit_map, optimize=False) / len(tx)
        target = np.einsum('mi,m->i', audit_map, ty, optimize=False) / len(tx)
        coefficients = factors * signal
        bias = np.einsum('ti,ij,tj->t', coefficients, gram, coefficients, optimize=False) - 2 * np.einsum('ti,i->t', coefficients, target, optimize=False) + np.mean(ty ** 2)
        variance = np.einsum('ti,i->t', factors ** 2, np.diag(gram), optimize=False)
        risk = arrays['signal_bias'] + row['cell']['noise_variance'] * arrays['variance_unit']
        signal_error = float(np.max(np.abs(bias - arrays['signal_bias'])))
        variance_error = float(np.max(np.abs(variance - arrays['variance_unit'])))
        response = np.einsum('ij,kj->ik', audit_map * factors[-1], u, optimize=False)
        direct_variance = float(np.sum(response ** 2) / len(tx))
        endpoint_prediction = np.einsum('mi,i->m', tx, arrays['final_heads'][:, 1], optimize=False)
        direct_bias = float(np.mean((endpoint_prediction - ty) ** 2))
        risk_error = abs(direct_bias + row['cell']['noise_variance'] * direct_variance - risk[-1])
        assert max(eigen_error, signal_error, variance_error, risk_error) <= 1e-9
        assert np.max(np.abs(risk - arrays['expected_risk'])) <= 1e-12
        t = int(np.argmin(risk))
        old_t = int(np.argmin(old_arrays['expected_risk']))
        saved = next(c for c in summary['cells'] if c['label'] == label)
        assert t == saved['t_star'] and old_t == saved['old_t_star']
        rows.append({'label': label, **row['cell'], 'metadata_sha256': expected, 't_star': t, 'old_t_star': old_t, 'invariant_arrays': unchanged, 'eigen_residual_maxabs': eigen_error, 'full_signal_bias_maxabs': signal_error, 'full_variance_maxabs': variance_error, 'endpoint_explicit_risk_maxabs': risk_error})
    pairs = []
    for seed in config['seeds']:
        left, right = sorted([r for r in rows if r['seed'] == seed], key=lambda r: r['n'])
        ratio = right['t_star'] / left['t_star']
        pairs.append({'seed': seed, 'population_ratio': ratio, 'old_ratio': right['old_t_star'] / left['old_t_star'], 'factor2_pass': .5 <= ratio <= 2})
    prediction = next(p for p in pairs if p['seed'] == 414)
    assert len(rows) == 8 and 2 < prediction['population_ratio'] <= 4
    result = {'status': 'verified', 'checked_at': datetime.now().astimezone().isoformat(), 'round': 19, 'preregistration_commit': commit, 'commit_precedes_all_cells': True, 'verified_cells': 8, 'summary_sha256': sha(STUDY / 'summary.json'), 'P2': {'status': 'supported', 'ratio': prediction['population_ratio']}, 'pairs': pairs, 'cells': rows, 'source_sha256': sha(Path(__file__)), 'method': '读取保存数据；独立重建audit谱映射、全步信号/方差和显式终点噪声响应。未调用训练器；未修改成功cell。'}
    with (STUDY / 'executed/independent_verification.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write(chr(10))
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
