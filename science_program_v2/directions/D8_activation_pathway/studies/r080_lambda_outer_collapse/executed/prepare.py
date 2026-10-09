import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
OLD = STUDY.parent / 'r048_lambda_collapse'
SEEDS = list(range(101, 107))
LAMBDAS = [-3, 3]
SCALES = [.0125, .025, .05]
ROOT = 2.3993572805154675


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    invalidation = json.loads((OLD / 'executed/round_invalidation.json').read_text())
    assert invalidation['round_result'] == 'failed'
    assert invalidation['valid_scientific_claims_added'] == 0
    receipt_path = OLD / 'executed/final_commit_verification.json'
    receipt = json.loads(receipt_path.read_text())
    for row in receipt['files']:
        blob = subprocess.run(['git', 'show', receipt['scientific_closeout_commit'] + ':' + row['path']], cwd=REPO, capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == row['sha256'], row['path']
    old_manifest = json.loads((OLD / 'executed/old_evidence_manifest.json').read_text())['files']
    result_manifest = json.loads((OLD / 'executed/independent_verification.json').read_text())['result_files']
    for base, rows in ((REPO, old_manifest), (OLD, result_manifest)):
        for name, row in rows.items():
            p = base / name
            assert sha(p) == row['sha256'] and p.stat().st_mtime_ns == row['mtime_ns'], name
    protected = {}
    history = []
    grids = []
    for old in sorted(STUDY.parent.iterdir()):
        if old == STUDY or not old.is_dir():
            continue
        for p in sorted(old.rglob('*')):
            if p.is_file():
                protected[str(p.relative_to(REPO))] = {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
        config = json.loads((old / 'preregistration.json').read_text())
        rows = [json.loads(p.read_text()) for p in sorted((old / 'results').glob('*.json'))]
        keys = []
        for row in rows:
            assert row['status'] == 'success'
            cell = row['request']['cell']
            keys.append((cell['seed'], cell['scale'], row['request']['bias']))
            history.append({'study': old.name, 'cell_id': row['cell_id'], 'seed': cell['seed'], 'scale': cell['scale'], 'bias': row['request']['bias'], 'original_request': row['request'], 'contract_sha256': row['contract_sha256'], 'arrays_sha256': row['arrays_sha256']})
        if 'lambdas' in config:
            expected = [(s, a, config['bias_root'] + lam*a*a) for s, lam, a in itertools.product(config['seeds'], config['lambdas'], config['scales'])]
        elif 'bias_offsets' in config:
            expected = [(s, a, config['bias_root'] + delta) for s, delta, a in itertools.product(config['seeds'], config['bias_offsets'], config['scales'])]
        else:
            expected = [(s, a, {'zero': 0., 'one': 1., 'inflection': ROOT}[b]) for s, b, a in itertools.product(config['seeds'], config['bias_labels'], config['scales'])]
        assert len(keys) == len(expected) and all(any(s==t and a==c and abs(b-d)<1e-14 for t,c,d in keys) for s,a,b in expected), old.name
        grids.append({'study': old.name, 'registered_cells': len(expected), 'saved_cells': len(keys), 'cartesian_complete': True})
    cells = []
    for seed, lam, a in itertools.product(SEEDS, LAMBDAS, SCALES):
        b = ROOT + lam*a**2
        overlaps = [r for r in history if r['seed']==seed and r['scale']==a and abs(r['bias']-b)<1e-14]
        assert not overlaps, overlaps
        cells.append({'seed': seed, 'lambda_index': LAMBDAS.index(lam), 'lambda': lam, 'scale': a, 'delta': lam*a**2, 'bias': b, 'cell_id': f's{seed}_l{LAMBDAS.index(lam)}_a{a:g}', 'historical_overlap': [], 'mode': 'new_measurement'})
    assert len(cells) == 36 and len({r['cell_id'] for r in cells}) == 36
    save(STUDY / 'executed/old_evidence_manifest.json', {'files': protected})
    save(STUDY / 'executed/prior_closeout_audit.json', {'round48_invalid': True, 'invalid_reason_preserved': invalidation['reason'], 'scientific_closeout_commit': receipt['scientific_closeout_commit'], 'receipt_sha256': sha(receipt_path), 'committed_bytes_verified': len(receipt['files']), 'old599_hash_mtime_verified': len(old_manifest), 'r048_result_hash_mtime_verified': len(result_manifest), 'all_old_study_files_protected': len(protected), 'training_steps': 0})
    save(STUDY / 'executed/condition_audit.json', {'historical_grids': grids, 'historical_requests': history, 'historical_request_count': len(history), 'new_cells': cells, 'planned_cells': 36, 'new_measurement_cells': 36, 'overlap_cells': 0, 'comparison': 'same seed and scale, bias absolute tolerance 1e-14; audit actual requests and complete registered Cartesian grids', 'overlap_policy': '发现任何历史重叠即停止；若研究需要重叠，必须先修订并提交只读/复制旧数组与原request/contract/hash方案，不允许重新调用激活。'})
    source = OLD / 'executed/input_snapshot.npz'
    target = STUDY / 'executed/input_snapshot.npz'
    target.write_bytes(source.read_bytes())
    assert sha(target) == '13a026ccfa1e1745b426af1e9e98411b7afad9954a73f98da43c04b5db81c9ff'
    sigmoid = expit(ROOT)
    q, h = sigmoid*(1-sigmoid), 1-2*sigmoid
    f1 = sigmoid+ROOT*q
    f3 = q*(3*h+ROOT*(h*h-2*q))
    f4 = q*(4*(h*h-2*q)+ROOT*(h**3-8*q*h))
    arrays, forecasts = {}, []
    with np.load(target, allow_pickle=False) as z:
        for name in z.files:
            v = z[name]
            arrays[name] = {'sha256': hashlib.sha256(v.tobytes()).hexdigest(), 'shape': list(v.shape), 'dtype': str(v.dtype)}
        for seed in SEEDS:
            u = z[f'u_{seed}']
            v2, v4 = u*u, u**4
            v2, v4 = v2-v2.mean(0), v4-v4.mean(0)
            moments = {'M2': float(np.mean(u*u)), 'M22': float(np.mean(v2*v2)), 'M24': float(np.mean(v2*v4)), 'M44': float(np.mean(v4*v4))}
            for lam in LAMBDAS:
                even = f3*lam*v2/2+f4*v4/24
                forecasts.append({'seed': seed, 'lambda': lam, 'F': float(np.mean(even*even)/(f1*f1*moments['M2'])), 'moments': moments})
    save(STUDY / 'executed/analytic_forecasts.json', {'provenance': '非盲解析预测；注册前只从旧保存u矩与根导数计算F，不计算新bias激活；阈值沿用失效study的2%，旧失效数值已知且不作为新验证。', 'input_source': str(source.relative_to(REPO)), 'input_source_sha256': sha(source), 'root': ROOT, 'f1': float(f1), 'f3': float(f3), 'f4': float(f4), 'rows': forecasts, 'F_range': [min(r['F'] for r in forecasts), max(r['F'] for r in forecasts)], 'input_arrays': arrays})
    save(STUDY / 'executed/state_before.json', json.loads((REPO / 'central/state.json').read_text()))
    print(json.dumps({'prior_audit': 'pass', 'protected_files': len(protected), 'historical_requests': len(history), 'new_cells': len(cells), 'overlap': 0, 'F_range': [min(r['F'] for r in forecasts), max(r['F'] for r in forecasts)]}))


if __name__ == '__main__':
    main()
