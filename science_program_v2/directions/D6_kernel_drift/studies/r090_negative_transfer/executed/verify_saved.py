import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(path, coefficients, width):
    with np.load(path, allow_pickle=False) as z:
        kernels = np.einsum('tip,tjp->tij', z['jacobians'], z['jacobians'], optimize=False) / len(z['train_y'])
        scales = np.trace(kernels, axis1=1, axis2=2) / np.trace(kernels[0])
        values = []
        for v in [z['residuals'][0], z['train_y']]:
            quadratic = np.einsum('i,tij,j->t', v, kernels, v, optimize=False)
            values.append(np.log(quadratic / (scales * quadratic[0])))
        e, q, late = values[0][1], values[1][1], values[0][2]
        features = {'E': [1., e], 'W': [1., e, float(width == 8)], 'T': [1., e, q]}
        return {name: abs(np.dot(features[name], beta) - late) for name, beta in coefficients.items()}


def main():
    config = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    paired_intervals = []
    for unit in summary['paired_data_seed_error_change']:
        function, width = unit['function'], unit['width']
        differences = {name: [] for name in config['frozen_coefficients']}
        for seed in config['seeds']:
            cell = f'{function}_w{width}_s{seed}.npz'
            old = errors(REPO / config['baseline_study'] / 'results' / cell, config['frozen_coefficients'], width)
            new = errors(STUDY / 'results' / cell, config['frozen_coefficients'], width)
            for name in differences:
                differences[name].append(new[name] - old[name])
        verified = {}
        for name, values in differences.items():
            values = np.asarray(values)
            center = np.sum(values) / len(values)
            spread = np.sqrt(np.sum((values - center) ** 2) / (len(values) - 1))
            radius = t.isf(.025, len(values) - 1) * spread / np.sqrt(len(values))
            expected = unit['new_minus_old_absolute_error'][name]
            np.testing.assert_allclose(center, expected['mean'], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose([center - radius, center + radius],
                                       expected['seed_95pct_t_interval'], rtol=1e-10, atol=1e-12)
            verified[name] = {'mean': float(center), 'seed_95pct_t_interval': [float(center - radius), float(center + radius)]}
        paired_intervals.append({'function': function, 'width': width, 'verified_changes': verified})
    files = {str(path.relative_to(REPO)): {'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
             for path in sorted((STUDY / 'results').glob('*')) if path.is_file()}
    assert len(files) == 36 and summary['saved_cells'] == 12
    receipt_path = STUDY / 'executed/saved_evidence_verification.json'
    existed = receipt_path.exists()
    if existed:
        receipt = json.loads(receipt_path.read_text())
        assert receipt['files'] == files, 'Successful cell bytes or mtime changed during resume'
        assert receipt['summary_sha256'] == sha(STUDY / 'summary.json')
    result = {'status': 'passed', 'cells_checked': 12, 'successful_files_checked': len(files),
              'resume_hash_and_mtime_unchanged': existed, 'paired_data_seed_intervals': paired_intervals,
              'summary_sha256': sha(STUDY / 'summary.json'), 'files': files,
              'verification_source_sha256': sha(Path(__file__)),
              'note': '事后独立核验，不是训练源码；从两批保存J重算全部旧新误差变化及seed t区间，0训练/0拟合。'}
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'resume_verified': existed, 'cells_checked': 12}))


if __name__ == '__main__':
    main()
