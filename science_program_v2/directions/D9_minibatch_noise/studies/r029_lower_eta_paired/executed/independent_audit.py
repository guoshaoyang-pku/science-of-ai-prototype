import hashlib
import json
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = ROOT / 'directions/D9_minibatch_noise/studies/r025_batch_risk_shift_corrected'


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    errors, replay_errors, expected_errors = [], [], []
    for meta_path in sorted((STUDY / 'results').glob('n64*.json')):
        meta = json.loads(meta_path.read_text())
        c = meta['cell']
        with np.load(STUDY / 'executed' / f"data_n64_seed{c['seed']}.npz") as z:
            d = {k: z[k] for k in z.files}
        with np.load(meta_path.with_suffix('.npz')) as z:
            a = {k: z[k] for k in z.files}
        pred = np.einsum('ip,rp->ir', d['audit_features'], a['final_heads'], optimize=False)
        errors.append(float(np.max(np.abs(np.mean((pred - d['audit_y'][:, None]) ** 2, axis=0) - a['replicate_risk'][:, -1]))))
        if c['batch'] == 64:
            x = d['train_features']
            kernel = np.einsum('ip,jp->ij', x, x, optimize=False) / 64
            e, v = np.linalg.eigh(kernel)
            factors = np.zeros((meta['steps'] + 1, 64))
            q = 1 - 2 * c['lr'] * e
            for t in range(1, len(factors)):
                factors[t] = q * factors[t - 1] + 2 * c['lr']
            cross = np.einsum('ap,ip->ai', d['audit_features'], x, optimize=False) / 64
            transfer = np.einsum('ai,ik->ak', cross, v, optimize=False)
            target_modes = np.einsum('ik,i->k', v, d['train_y'], optimize=False)
            signal = np.einsum('ak,tk,k->at', transfer, factors, target_modes, optimize=False)
            bias = np.mean((signal - d['audit_y'][:, None]) ** 2, axis=0)
            variance = np.einsum('k,tk->t', np.mean(transfer * transfer, axis=0), factors * factors, optimize=False)
            expected_errors.append(float(np.max(np.abs(bias + c['noise_variance'] * variance - a['expected_risk']))))
        if c['seed'] == 411 and c['batch'] == 8 and c['noise_variance'] == 1:
            w = np.zeros(d['train_features'].shape[1])
            rng = np.random.default_rng(int(a['batch_seeds'][0]))
            target = d['train_y'] + d['epsilon']
            replay = []
            for t in range(17):
                pred = np.einsum('ip,p->i', d['audit_features'], w, optimize=False)
                replay.append(np.mean((pred - d['audit_y']) ** 2))
                if t < 16:
                    idx = rng.choice(64, size=8, replace=False)
                    x = d['train_features'][idx]
                    residual = np.einsum('ip,p->i', x, w, optimize=False) - target[idx]
                    w -= 2 * c['lr'] * np.einsum('ip,i->p', x, residual, optimize=False) / 8
            replay_errors.append(float(np.max(np.abs(np.asarray(replay) - a['replicate_risk'][0, :17]))))
    ratio_error, interval_error = 0., 0.
    for row in summary['cells']:
        seed, b, var = row['seed'], row['batch'], row['variance']
        with np.load(STUDY / 'results' / f'n64_seed{seed}_eta0.1_B{b}_var{var:g}.npz') as z:
            risks = z['replicate_risk'].copy()
        with np.load(STUDY / 'results' / f'n64_seed{seed}_eta0.1_B64_var{var:g}.npz') as z:
            full_ts, expected_ts = int(np.argmin(z['mean_risk'])), int(np.argmin(z['expected_risk']))
        ts = int(np.argmin(risks.mean(0)))
        ratio_error = max(ratio_error, abs(ts / full_ts - row['aligned_ratio']), abs(ts / expected_ts - row['legacy_ratio']))
        rng = np.random.default_rng(290700 + 100000 * seed + 1000 * b + int(var * 100))
        indices = rng.integers(0, 8, size=(2000, 8))
        boot = np.argmin(risks[indices].mean(1), axis=1) / full_ts
        interval_error = max(interval_error, float(np.max(np.abs(np.quantile(boot, [.025, .975]) - row['aligned_ratio_bootstrap_95pct']))))
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    pins_match = all(hashlib.sha256((STUDY / n).read_bytes()).hexdigest() == h for n, h in cfg['source_sha256'].items())
    out = {'cells': len(errors), 'source_pins_match': pins_match, 'max_einsum_final_risk_error': max(errors),
           'max_einsum_expected_risk_error': max(expected_errors), 'first_16_step_einsum_replay_errors': replay_errors,
           'max_ratio_recomputation_error': ratio_error, 'max_bootstrap_interval_recomputation_error': interval_error,
           'warnings_note': 'Pinned run/verify emitted matmul divide/overflow/invalid warnings, all saved arrays finite and independently agree. Root cause not determined; original logs and source retained.',
           'all_ok': pins_match and len(errors) == 48 and max(errors) < 1e-11 and max(expected_errors) < 1e-10 and max(replay_errors) < 1e-11 and ratio_error == 0 and interval_error == 0}
    (STUDY / 'executed/independent_verification.json').write_text(json.dumps(out, indent=2) + chr(10))
    print(json.dumps(out))
    if not out['all_ok']:
        raise RuntimeError('Independent verification failed')


if __name__ == '__main__':
    main()
