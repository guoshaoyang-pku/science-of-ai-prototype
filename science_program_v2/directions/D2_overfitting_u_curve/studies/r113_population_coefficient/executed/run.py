"""冻结合同后重加权保存曲线；无随机数、特征计算或训练。"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write(chr(10))


def contract(commit):
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for rel, value in {'preregistration.json': sha(STUDY / 'preregistration.json'), **cfg['source_sha256']}.items():
        p = STUDY / rel
        assert sha(p) == value
        assert subprocess.check_output(['git', 'show', f'{commit}:{p.relative_to(ROOT)}'], cwd=ROOT) == p.read_bytes(), rel
    for rel, value in cfg['input_sha256'].items():
        p = ROOT / rel
        assert sha(p) == value, rel
        assert subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=ROOT) == p.read_bytes(), rel
    snapshot = json.loads((STUDY / 'executed/prior_snapshot.json').read_text())
    old_set = {str(p.relative_to(ROOT)) for p in STUDY.parent.rglob('*') if p.is_file() and STUDY not in p.parents}
    assert old_set == set(snapshot), '旧 study 文件集合变化'
    for rel, value in snapshot.items():
        p = ROOT / rel
        assert sha(p) == value['sha256'] and p.stat().st_mtime_ns == value['mtime_ns'], rel
    return cfg


def saved(cfg, commit):
    folder = STUDY / 'results'
    receipt_path = folder / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'commit': commit, 'cells': {}}
    assert receipt['commit'] == commit
    expected = {'receipt.json'} if receipt_path.exists() else set()
    for label, files in receipt['cells'].items():
        assert label in {c['label'] for c in cfg['cells']}
        for name, value in files.items():
            expected.add(name)
            p = folder / name
            assert sha(p) == value['sha256'] and p.stat().st_mtime_ns == value['mtime_ns'], name
        row = json.loads((folder / (label + '.json')).read_text())
        assert row['status'] == 'completed' and row['commit'] == commit
    assert {p.name for p in folder.iterdir()} == expected, '孤立/额外/部分结果'
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', required=True)
    ap.add_argument('--max-new-cells', type=int, default=32)
    args = ap.parse_args()
    cfg = contract(args.commit)
    receipt = saved(cfg, args.commit)
    pending = [c for c in cfg['cells'] if c['label'] not in receipt['cells']]
    if len(receipt['cells']) and pending:
        gate = json.loads((STUDY / 'executed/first_cell_verification.json').read_text())
        assert gate['status'] == 'verified' and gate['commit'] == args.commit
        assert gate['cells'] == 1
    if not receipt['cells']:
        assert args.max_new_cells == 1, '首 cell 必须独立核验后再续跑'
    freeze_ns = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', args.commit], cwd=ROOT)) * 10**9
    new = 0
    for cell in pending[:args.max_new_cells]:
        start = time.time_ns()
        assert start > freeze_ns
        with np.load(ROOT / cell['source'], allow_pickle=False) as z:
            b = z['signal_bias'].copy()
            noise = z['variance_unit'].copy()
        assert b.shape == noise.shape == (16385,)
        risk = b + cell['noise_variance'] * noise
        assert all(np.isfinite(a).all() for a in (b, noise, risk))
        label = cell['label']
        npz = STUDY / 'results' / (label + '.npz')
        with npz.open('xb') as f:
            np.savez_compressed(f, signal_bias=b, variance_unit=noise, expected_risk=risk)
        row = {'status': 'completed', 'commit': args.commit, 'preregistration_sha256': sha(STUDY/'preregistration.json'),
               'cell': cell, 'source_sha256': sha(ROOT/cell['source']), 'training_cells': 0,
               'started_time_ns': start, 'finished_time_ns': time.time_ns(), 'npz_sha256': sha(npz)}
        save(STUDY/'results'/(label+'.json'), row)
        files = [npz, STUDY/'results'/(label+'.json')]
        receipt['cells'][label] = {p.name: {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in files}
        # receipt 是恢复索引；每个已成功 cell 的文件始终 write-once。
        (STUDY/'results/receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+chr(10))
        new += 1
    saved(cfg, args.commit)
    print(json.dumps({'new_evaluation_cells': new, 'reused_cells': len(receipt['cells'])-new, 'training_cells': 0}))


if __name__ == '__main__':
    main()
