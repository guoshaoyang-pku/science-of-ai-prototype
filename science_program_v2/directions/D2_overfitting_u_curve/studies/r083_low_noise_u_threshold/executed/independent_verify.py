"""独立从原data与高精度保存曲线核验；不调用执行器或训练。"""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
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
LABEL = 'n32_seed412_var0.0625'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    binding = json.loads((STUDY / 'executed/execution_binding.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit = binding['preregistration_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    pins = {'preregistration.json': sha(STUDY / 'preregistration.json'), **config['source_sha256']}
    assert pins == binding['sha256'] == summary['sha256']
    for relative, expected in pins.items():
        path = STUDY / relative
        assert sha(path) == expected
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes()
    for relative, expected in config['input_sha256'].items():
        assert sha(ROOT / relative) == expected
    historical = json.loads((STUDY / 'executed/prior_closeout_audit.json').read_text())['historical_study_files']
    for relative, expected in historical.items():
        path = ROOT / relative
        assert sha(path) == expected['sha256'] and path.stat().st_mtime_ns == expected['mtime_ns']
    folder = STUDY / 'results'
    assert {p.name for p in folder.iterdir()} == {LABEL + '.npz', LABEL + '.json', 'receipt.json'}
    before = {p.name: {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in folder.iterdir()}
    receipt = json.loads((folder / 'receipt.json').read_text())
    meta = json.loads((folder / (LABEL + '.json')).read_text())
    assert receipt['preregistration_commit'] == meta['preregistration_commit'] == summary['preregistration_commit'] == commit
    assert receipt['sha256'] == meta['sha256'] == pins
    assert receipt['cell'] == meta['cell'] == summary['cell'] == config['cell']
    assert set(receipt['artifacts']) == {LABEL + '.npz', LABEL + '.json'}
    for name, expected in receipt['artifacts'].items():
        assert before[name] == expected
    assert meta['arrays_sha256'] == before[LABEL + '.npz']['sha256']
    assert meta['input_sha256'] == summary['input_sha256'] == config['input_sha256']
    commit_time = datetime.fromisoformat(subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip())
    started_at = datetime.fromisoformat(meta['started_at'])
    assert commit_time < started_at
    with np.load(folder / (LABEL + '.npz'), allow_pickle=False) as source:
        saved = {key: source[key] for key in source.files}
    assert set(saved) == set(config['evaluation_contract']['saved_arrays'])
    with np.load(ROOT / config['source_curve'], allow_pickle=False) as source:
        original = {key: source[key] for key in ['signal_bias', 'variance_unit', 'expected_risk']}
    with np.load(ROOT / config['source_data'], allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    assert set(data) == set(config['data_contract']['invariants'])
    assert np.array_equal(saved['steps'], np.arange(16385))
    assert np.array_equal(saved['signal_bias'], original['signal_bias'])
    assert np.array_equal(saved['variance_unit'], original['variance_unit'])
    assert np.array_equal(saved['baseline_risk'], original['expected_risk'])
    assert all(np.isfinite(value).all() for value in list(saved.values()) + list(data.values()))
    for split in ['train', 'audit']:
        raw = data[split + '_x']
        assert np.array_equal((raw[:, 0] + .5 * raw[:, 0] * raw[:, 1]) / np.sqrt(1.25), data[split + '_y'])
    x, ax, y, ay = data['train_features'], data['audit_features'], data['train_y'], data['audit_y']
    n, eta = config['cell']['n'], config['optimizer']['lr']
    kernel = np.einsum('ik,jk->ij', x, x, optimize=False) / n
    eigenvalues, vectors = np.linalg.eigh(kernel)
    mapping = np.einsum('ni,nj->ij', x, vectors, optimize=False) / n
    audit_map = np.einsum('mi,ij->mj', ax, mapping, optimize=False)
    gram = np.einsum('mi,mj->ij', audit_map, audit_map, optimize=False) / len(ax)
    target = np.einsum('mi,m->i', audit_map, ay, optimize=False) / len(ax)
    signal = np.einsum('ij,i->j', vectors, y, optimize=False)
    grid = np.arange(16385)[:, None]
    factors = np.empty((16385, n))
    nonzero = eigenvalues != 0
    factors[:, nonzero] = -np.expm1(grid * np.log1p(-2 * eta * eigenvalues[nonzero])) / eigenvalues[nonzero]
    factors[:, ~nonzero] = 2 * eta * grid
    coefficients = factors * signal
    bias = np.einsum('ti,ij,tj->t', coefficients, gram, coefficients, optimize=False) - 2 * np.einsum('ti,i->t', coefficients, target, optimize=False) + np.mean(ay**2)
    variance = np.einsum('ti,i->t', factors**2, np.diag(gram), optimize=False)
    risk = bias + config['cell']['noise_variance'] * variance
    t = int(np.argmin(risk))
    gap = float(risk[-1] - risk[t])
    threshold = config['evaluation_contract']['threshold']
    errors = {
        'signal_bias_maxabs': float(np.max(np.abs(bias - saved['signal_bias']))),
        'variance_unit_maxabs': float(np.max(np.abs(variance - saved['variance_unit']))),
        'expected_risk_maxabs': float(np.max(np.abs(risk - saved['expected_risk']))),
        'endpoint_minus_minimum_error': abs(gap - summary['new']['endpoint_minus_minimum']),
    }
    assert max(errors.values()) <= config['independent_validation']['maxabs_tolerance']
    assert t == summary['new']['t_star']
    assert bool(0 < t < 16384 and gap >= threshold) == summary['new']['operational_u_shape']
    with localcontext() as context:
        context.prec = 60
        sigma = Decimal.from_float(config['cell']['noise_variance'])
        exact = [Decimal.from_float(float(b)) + sigma * Decimal.from_float(float(v))
                 for b, v in zip(original['signal_bias'], original['variance_unit'])]
        exact_t = min(range(len(exact)), key=exact.__getitem__)
        exact_gap = exact[-1] - exact[exact_t]
        exact_rounding_error = max(abs(value - Decimal.from_float(float(double))) for value, double in zip(exact, saved['expected_risk']))
        exact_u = 0 < exact_t < 16384 and exact_gap >= Decimal(str(threshold))
        lo, hi = [Decimal(str(value)) for value in config['predictions'][0]['interval']]
        prediction_status = 'supported' if lo <= exact_gap <= hi and not exact_u else 'refuted'
        assert exact_t == summary['new']['t_star']
        assert exact_u == summary['new']['operational_u_shape']
        assert prediction_status == summary['predictions'][0]['status']
        assert abs(float(exact_gap) - summary['new']['endpoint_minus_minimum']) <= 1e-12
        decimal_result = {'precision': 60, 'first_argmin': exact_t, 'endpoint_minus_minimum': str(exact_gap),
                          'max_float64_rounding_error': str(exact_rounding_error), 'operational_u_shape': exact_u}
    for relative, expected in before.items():
        path = folder / relative
        assert sha(path) == expected['sha256'] and path.stat().st_mtime_ns == expected['mtime_ns']
    for relative, expected in historical.items():
        path = ROOT / relative
        assert sha(path) == expected['sha256'] and path.stat().st_mtime_ns == expected['mtime_ns']
    result = {
        'status': 'verified', 'round': 83, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'preregistration_commit': commit, 'commit_precedes_evaluation_seconds': (started_at - commit_time).total_seconds(),
        'source_sha256': sha(Path(__file__)), 'summary_sha256': sha(STUDY / 'summary.json'),
        'input_pins_verified': len(config['input_sha256']), 'historical_files_hash_mtime_verified': len(historical),
        'success_files_hash_mtime_verified': len(before), 'success_files': before,
        'new_training_cells': 0, 'new_evaluation_cells': 1, 't_star': t,
        'endpoint_minus_minimum': gap, 'max_errors': errors, 'decimal_saved_curve_validation': decimal_result,
        'prediction_status': prediction_status,
        'method': '原data重新K/eigh与log1p/expm1复算全曲线；另以60位Decimal复算保存曲线；无训练、不调用runner、不覆盖结果。',
    }
    with (STUDY / 'executed/independent_verification.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))
    print(json.dumps({'status': 'verified', 't_star': t, 'max_errors': errors, 'decimal': decimal_result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
