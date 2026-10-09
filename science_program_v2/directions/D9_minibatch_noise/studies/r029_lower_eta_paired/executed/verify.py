import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    rows = []
    for label, digest in receipt['cells'].items():
        path = STUDY / 'results' / (label + '.json')
        meta = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(STUDY / 'executed' / f"data_n64_seed{meta['cell']['seed']}.npz") as z:
            data = {k: z[k] for k in z.files}
        c = meta['cell']
        direct = np.mean((data['audit_features'] @ arrays['final_heads'].T - data['audit_y'][:, None]) ** 2, axis=0)
        final_error = float(np.max(np.abs(direct - arrays['replicate_risk'][:, -1])))
        mean_error = float(np.max(np.abs(arrays['replicate_risk'].mean(0) - arrays['mean_risk'])))
        relative = (STUDY / 'preregistration.json').relative_to(ROOT).as_posix()
        prereg_bytes = subprocess.check_output(['git', 'show', meta['pins']['git_commit'] + ':' + relative], cwd=ROOT)
        checks = {'receipt_hash': sha(path) == digest, 'arrays_hash': sha(path.with_suffix('.npz')) == meta['arrays_sha256'],
                  'data_hash': sha(STUDY / 'executed' / f"data_n64_seed{c['seed']}.npz") == meta['data_sha256'],
                  'committed_preregistration': prereg_bytes == (STUDY / 'preregistration.json').read_bytes(),
                  'finite': all(np.isfinite(a).all() for a in arrays.values()),
                  'length': arrays['replicate_risk'].shape == (meta['replicates'], meta['steps'] + 1),
                  'final_risk': final_error < 1e-11, 'mean_risk': mean_error < 1e-13}
        if c['batch'] == 64:
            x = data['train_features']
            e, v = np.linalg.eigh(x @ x.T / 64)
            q = 1 - 2 * c['lr'] * e
            factors = np.zeros((meta['steps'] + 1, 64))
            for i in range(1, len(factors)):
                factors[i] = q * factors[i - 1] + 2 * c['lr']
            transfer = (data['audit_features'] @ x.T @ v) / 64
            signal_pred = transfer @ (factors * (v.T @ data['train_y'])).T
            bias = np.mean((signal_pred - data['audit_y'][:, None]) ** 2, axis=0)
            variance = np.mean(transfer * transfer, axis=0) @ (factors * factors).T
            expectation_error = float(np.max(np.abs(bias + c['noise_variance'] * variance - arrays['expected_risk'])))
            checks['expected_risk'] = expectation_error < 1e-10
        else:
            expectation_error = None
        rows.append({'label': label, 'checks': checks, 'final_risk_error': final_error,
                     'mean_risk_error': mean_error, 'expected_risk_error': expectation_error})
    snapshot = json.loads((STUDY / 'executed/input_audit.json').read_text())
    old_unchanged = all(sha(ROOT / name) == item['sha256'] and (ROOT / name).stat().st_mtime_ns == item['mtime_ns']
                        for name, item in snapshot['files'].items())
    result = {'saved_cells': len(rows), 'old_files': len(snapshot['files']), 'old_hash_and_mtime_unchanged': old_unchanged,
              'all_ok': len(rows) == cfg['counts']['planned_cells'] and old_unchanged and all(all(r['checks'].values()) for r in rows), 'rows': rows}
    (STUDY / 'executed/verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    if not result['all_ok']:
        raise RuntimeError('Verification failed')


if __name__ == '__main__':
    main()
