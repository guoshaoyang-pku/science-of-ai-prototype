"""零训练：在冻结提交之后评价一个新噪声方差；成功结果只验证。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
LABEL = 'n32_seed413_var0.0625'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_contract(commit):
    binding = json.loads((STUDY / 'executed/execution_binding.json').read_text())
    assert commit == binding['preregistration_commit'], '唯一提交绑定不匹配'
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    config = json.loads((STUDY / 'preregistration.json').read_text())
    pins = {'preregistration.json': sha(STUDY / 'preregistration.json'), **config['source_sha256']}
    assert pins == binding['sha256']
    for relative, expected in pins.items():
        path = STUDY / relative
        assert sha(path) == expected, relative
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes()
    for relative, expected in config['input_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    audit = json.loads((STUDY / 'executed/prior_closeout_audit.json').read_text())
    for relative, expected in audit['historical_study_files'].items():
        path = ROOT / relative
        assert sha(path) == expected['sha256'], relative
        assert path.stat().st_mtime_ns == expected['mtime_ns'], relative
    return config, pins


def verify_saved(config, pins, commit):
    folder = STUDY / 'results'
    expected_files = {LABEL + '.npz', LABEL + '.json', 'receipt.json'}
    present = {p.name for p in folder.iterdir()} if folder.exists() else set()
    assert not present or present == expected_files, '拒绝孤立、缺失或额外结果文件'
    if not present:
        return False
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['preregistration_commit'] == commit
    assert receipt['sha256'] == pins
    assert receipt['cell'] == config['cell']
    assert set(receipt['artifacts']) == {LABEL + '.npz', LABEL + '.json'}
    for name, expected in receipt['artifacts'].items():
        path = folder / name
        assert sha(path) == expected['sha256']
        assert path.stat().st_mtime_ns == expected['mtime_ns']
    row = json.loads((folder / (LABEL + '.json')).read_text())
    assert row['status'] == 'completed' and row['preregistration_commit'] == commit
    assert row['sha256'] == pins and row['cell'] == config['cell']
    assert row['input_sha256'] == config['input_sha256']
    assert row['arrays_sha256'] == sha(folder / (LABEL + '.npz'))
    assert datetime.fromisoformat(row['started_at']) > datetime.fromisoformat(binding_time(commit))
    return True


def binding_time(commit):
    return subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    args = parser.parse_args()
    config, pins = verify_contract(args.preregistration_commit)
    if verify_saved(config, pins, args.preregistration_commit):
        print(json.dumps({'new_cells': 0, 'reused_cells': 1, 'training_cells': 0}))
        return
    started_at = datetime.now(timezone.utc).isoformat()
    assert datetime.fromisoformat(started_at) > datetime.fromisoformat(binding_time(args.preregistration_commit))
    clock = time.perf_counter()
    with np.load(ROOT / config['source_curve'], allow_pickle=False) as source:
        signal = source['signal_bias'].copy()
        variance = source['variance_unit'].copy()
        baseline = source['expected_risk'].copy()
    assert signal.shape == variance.shape == baseline.shape == (16385,)
    assert all(np.isfinite(x).all() for x in [signal, variance, baseline])
    assert np.max(np.abs(baseline - (signal + .125 * variance))) <= 1e-12
    risk = signal + config['cell']['noise_variance'] * variance
    assert np.isfinite(risk).all()
    folder = STUDY / 'results'
    folder.mkdir(exist_ok=True)
    path = folder / (LABEL + '.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream, steps=np.arange(16385), signal_bias=signal, variance_unit=variance,
                            expected_risk=risk, baseline_risk=baseline)
    row = {
        'status': 'completed', 'cell': config['cell'], 'training_cells': 0,
        'measurement': 'saved_curve_reweighting', 'preregistration_commit': args.preregistration_commit,
        'sha256': pins, 'input_sha256': config['input_sha256'],
        'arrays_sha256': sha(path), 'started_at': started_at,
        'finished_at': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - clock,
        'numpy': np.__version__,
    }
    save_json(folder / (LABEL + '.json'), row)
    artifacts = {p.name: {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                 for p in [path, folder / (LABEL + '.json')]}
    save_json(folder / 'receipt.json', {'cell': config['cell'], 'preregistration_commit': args.preregistration_commit,
                                     'sha256': pins, 'artifacts': artifacts})
    verify_saved(config, pins, args.preregistration_commit)
    print(json.dumps({'new_cells': 1, 'reused_cells': 0, 'training_cells': 0, 'seconds': row['seconds']}))


if __name__ == '__main__':
    main()
