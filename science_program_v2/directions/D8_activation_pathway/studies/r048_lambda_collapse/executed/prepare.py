from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
OLD = STUDY.parent / 'r024_near_root_crossover'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    report = STUDY.parents[1] / 'report.md'
    text = report.read_text()
    if '## 规律与公式' not in text:
        title, formula = text.split('## 公式区\n', 1)
        formula, rest = formula.split('## 已测现象与解释\n', 1)
        old_phenomena, rest = rest.split('## 近根测量与转折\n', 1)
        near, rest = rest.split('## 竞争解释与未测边界\n', 1)
        boundaries, evidence = rest.split('## 证据\n', 1)
        old_paragraphs = old_phenomena.strip().split('\n\n')
        method = old_paragraphs.pop(1)
        new = title + '## 规律与公式\n' + formula
        new += '## 现象与解释\n\n' + '\n\n'.join(old_paragraphs) + '\n\n' + near
        new += '## 失败与反例\n\n首轮因预注册提交失败而未测量，三项预测均未评估；此执行失败不是科学反驳。近根测量的三项预测均通过，但正侧 +.001 的 seed104 增长仅 48.047861；不能写成全部超过 100。旧分析警告与首 cell 打印错误保留。\n\n'
        new += '## 方法与条件\n\n' + method + '\n\n' + boundaries
        new += '## 未决问题\n\n固定 λ=δ/a² 后，R/a⁶ 能否在同 seed 的三个小尺度上折叠？未测随机 bias、非对称输入、多层、learned hidden、CE 或干预收益。\n\n'
        new += '## 证据\n' + evidence
        report.write_text(new)
    receipt_path = OLD / 'executed/final_commit_verification.json'
    receipt = json.loads(receipt_path.read_text())
    committed = []
    for row in receipt['files']:
        blob = subprocess.run(['git', 'show', receipt['scientific_closeout_commit'] + ':' + row['path']], cwd=REPO, capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == row['sha256'], row['path']
        committed.append(row['path'])
    baseline = json.loads((OLD / 'executed/old_evidence_manifest.json').read_text())['files']
    for name, row in baseline.items():
        path = REPO / name
        assert sha(path) == row['sha256'] and path.stat().st_mtime_ns == row['mtime_ns'], name
    overlap = []
    first = STUDY.parent / 'r008_bias_inflection'
    zero_index = json.loads((OLD / 'preregistration.json').read_text())['bias_offsets'].index(0)
    for seed in range(101, 107):
        for a in [.025, .05, .1]:
            with np.load(OLD / f'results/s{seed}_d{zero_index:02d}_a{a:g}.npz') as z, np.load(first / f'results/s{seed}_inflection_a{a:g}.npz') as w:
                for key in ('odd_raw', 'even_centered_raw', 'feature_center'):
                    assert np.array_equal(z[key], w[key])
            overlap.append({'seed': seed, 'scale': a, 'arrays_bitwise_equal': True})
    files = dict(baseline)
    for path in sorted(OLD.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(REPO))] = {'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
    save(STUDY / 'executed/old_evidence_manifest.json', {'files': files})
    save(STUDY / 'executed/prior_closeout_audit.json', {'scientific_closeout_commit': receipt['scientific_closeout_commit'], 'receipt_sha256': sha(receipt_path), 'committed_bytes_verified': len(committed), 'old_hash_mtime_verified': len(baseline), 'preserved_files': len(files), 'overlap': overlap, 'saved_cells': 216, 'training_steps': 0})
    source = OLD / 'executed/input_snapshot.npz'
    target = STUDY / 'executed/input_snapshot.npz'
    assert target.read_bytes() == source.read_bytes()
    b = 2.3993572805154675
    s = expit(b)
    q, h = s*(1-s), 1-2*s
    f1 = s+b*q
    f3 = q*(3*h+b*(h*h-2*q))
    f4 = q*(4*(h*h-2*q)+b*(h**3-8*q*h))
    rows, arrays = [], {}
    with np.load(target, allow_pickle=False) as z:
        for name in z.files:
            a = z[name]
            arrays[name] = {'sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'shape': list(a.shape), 'dtype': str(a.dtype)}
        for seed in range(101, 107):
            u = z[f'u_{seed}']
            u2, u4 = u*u, u**4
            v2, v4 = u2-u2.mean(0), u4-u4.mean(0)
            moments = {'M2': float(np.mean(u2)), 'M22': float(np.mean(v2*v2)), 'M24': float(np.mean(v2*v4)), 'M44': float(np.mean(v4*v4))}
            for lam in [-2, -1, 0, 1, 2]:
                e = (f3/2)*lam*v2+(f4/24)*v4
                rows.append({'seed': seed, 'lambda': lam, 'F': float(np.mean(e*e)/(f1*f1*moments['M2'])), 'moments': moments})
    save(STUDY / 'executed/analytic_forecasts.json', {'provenance': '注册前从旧保存 u 矩与已知 Taylor 导数计算；没有计算新 bias 的激活，不是盲新发现。', 'root': b, 'f1': float(f1), 'f3': float(f3), 'f4': float(f4), 'rows': rows, 'F_range': [min(r['F'] for r in rows), max(r['F'] for r in rows)], 'input_arrays': arrays})
    save(STUDY / 'executed/state_before.json', json.loads((REPO / 'central/state.json').read_text()))
    print(json.dumps({'prior_audit': 'pass', 'preserved_files': len(files), 'F_range': [min(r['F'] for r in rows), max(r['F'] for r in rows)]}))


if __name__ == '__main__':
    main()
