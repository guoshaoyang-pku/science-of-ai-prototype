"""只读审查已保存曲线；不调用执行器、不组合未注册条件。"""
import hashlib
import json
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(
        ['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT
    )


def main():
    target = STUDY / 'executed/peer_review.json'
    assert not target.exists(), '成功审查不覆盖'
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit = subprocess.check_output(
        ['git', 'rev-parse', summary['commit']], cwd=ROOT, text=True
    ).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    freeze_ns = int(subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT
    )) * 10**9
    for rel, value in cfg['source_sha256'].items():
        path = STUDY / rel
        assert sha(path) == value and git_bytes(commit, path) == path.read_bytes(), rel
    prereg = STUDY / 'preregistration.json'
    assert git_bytes(commit, prereg) == prereg.read_bytes()
    assert sha(prereg) == summary['preregistration_sha256']
    for rel, value in cfg['input_sha256'].items():
        path = ROOT / rel
        assert sha(path) == value and git_bytes(commit, path) == path.read_bytes(), rel
    snapshot = json.loads((STUDY / 'executed/prior_snapshot.json').read_text())
    current_old = {str(p.relative_to(ROOT)) for p in STUDY.parent.rglob('*')
                   if p.is_file() and STUDY not in p.parents}
    assert current_old == set(snapshot)
    for rel, value in snapshot.items():
        path = ROOT / rel
        assert sha(path) == value['sha256'] and path.stat().st_mtime_ns == value['mtime_ns'], rel
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    assert receipt['commit'] == summary['commit'] and len(receipt['cells']) == 32
    expected_files = {'receipt.json'}
    rows = []
    risk_errors = []
    starts = []
    for cell in cfg['cells']:
        label = cell['label']
        for name, value in receipt['cells'][label].items():
            expected_files.add(name)
            path = STUDY / 'results' / name
            assert sha(path) == value['sha256'] and path.stat().st_mtime_ns == value['mtime_ns'], name
        metadata = json.loads((STUDY / 'results' / (label + '.json')).read_text())
        assert metadata['status'] == 'completed' and metadata['training_cells'] == 0
        assert metadata['commit'] == summary['commit']
        assert metadata['preregistration_sha256'] == sha(prereg)
        assert metadata['source_sha256'] == sha(ROOT / cell['source'])
        starts.append(metadata['started_time_ns'])
        assert metadata['started_time_ns'] > freeze_ns
        with np.load(ROOT / cell['source'], allow_pickle=False) as source, np.load(
            STUDY / 'results' / (label + '.npz'), allow_pickle=False
        ) as saved:
            b, noise = source['signal_bias'], source['variance_unit']
            assert np.array_equal(saved['signal_bias'], b)
            assert np.array_equal(saved['variance_unit'], noise)
            risk = b + cell['noise_variance'] * noise
            risk_errors.append(float(np.max(np.abs(risk - saved['expected_risk']))))
            t = int(np.argmin(risk))
            registered = next(row for row in summary['cells'] if row['label'] == label)
            assert t == registered['t_star'] and 0 < t < 16384
            rows.append({**cell, 't_star': t})
    assert {p.name for p in (STUDY / 'results').iterdir()} == expected_files
    coeff = {}
    for norm in ['sample', 'population']:
        coeff[norm] = {}
        for n in [32, 64]:
            subset = [row for row in rows if row['normalization'] == norm and row['n'] == n]
            coeff[norm][n] = Fraction(3, 10) * Fraction(
                sum(row['x'] * row['t_star'] for row in subset),
                sum(row['x']**2 for row in subset),
            )
            assert abs(float(coeff[norm][n]) - summary['coefficients'][norm][str(n)]) < 1e-14
    ds = abs(coeff['sample'][64] - coeff['sample'][32])
    dp = abs(coeff['population'][64] - coeff['population'][32])
    ratio = dp / ds
    assert float(ratio) > .5 and summary['P1']['status'] == 'refuted'
    assert abs(float(ratio) - summary['spread_ratio']) < 1e-14
    r112 = STUDY.parent / 'r112_seed411_low_noise'
    old_cfg = json.loads((r112 / 'preregistration.json').read_text())
    old_final = json.loads((r112 / 'executed/final_commit_verification.json').read_text())
    for rel, value in old_cfg['input_sha256'].items():
        assert sha(ROOT / rel) == value
    for rel, value in old_cfg['source_sha256'].items():
        assert sha(r112 / rel) == value
        assert git_bytes(old_final['preregistration_commit'], r112 / rel) == (r112 / rel).read_bytes()
    assert git_bytes(old_final['preregistration_commit'], r112 / 'preregistration.json') == (r112 / 'preregistration.json').read_bytes()
    for rel, value in old_final['artifact_sha256'].items():
        assert hashlib.sha256(git_bytes(old_final['verified_parent_commit'], ROOT / rel)).hexdigest() == value
    old_receipt = json.loads((r112 / 'results/receipt.json').read_text())
    for name, value in old_receipt['artifacts'].items():
        path = r112 / 'results' / name
        assert sha(path) == value['sha256'] and path.stat().st_mtime_ns == value['mtime_ns']
    old_meta = json.loads((r112 / 'results/n32_seed411_var0.0625.json').read_text())
    old_start = datetime.fromisoformat(old_meta['started_at']).timestamp()
    old_freeze = int(subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', old_final['preregistration_commit']], cwd=ROOT
    ))
    result = {
        'status': 'verified', 'review_type': 'independent_saved_artifact_review',
        'review_source_sha256': sha(Path(__file__)), 'freeze_commit': commit,
        'pinned_source_files': len(cfg['source_sha256']), 'input_pins': len(cfg['input_sha256']),
        'old_study_files_hash_mtime_verified': len(snapshot),
        'evaluation_cells': len(rows), 'new_training_cells': 0,
        'risk_reconstruction_maxabs': max(risk_errors),
        'freeze_before_first_cell_seconds': (min(starts) - freeze_ns) / 1e9,
        'exact_coefficients': {norm: {str(n): str(v) for n, v in values.items()}
                               for norm, values in coeff.items()},
        'exact_sample_spread': str(ds), 'exact_population_spread': str(dp),
        'exact_spread_ratio': str(ratio), 'P1': 'refuted',
        'r112_source_and_code_pins_verified': True, 'r112_closeout_git_objects_verified': True,
        'r112_success_hash_mtime_verified': True,
        'r112_freeze_before_evaluation_seconds': old_start - old_freeze,
        'scientific_review': [
            'D=abs(c64-c32) 是注册的聚合系数极差；不是逐seed平均绝对差。',
            'sample四seed的有符号差同时有正负，聚合极差受抵消影响；不能将4.68148倍写为所有seed散布变差。',
            'M=nN与B跨n未相同；s/n的代数因子不足以单独确定t_star。',
            '只支持该合同下归一化未让聚合极差减半；没有独立识别谱因果或跨seed概率。',
        ],
    }
    with target.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write(chr(10))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
