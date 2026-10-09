import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(loss, q):
    first = next((i for i, value in enumerate(loss) if value <= q), None)
    last_exceed = next((i for i in range(len(loss) - 1, -1, -1) if loss[i] > q), -1)
    sustained = last_exceed + 1
    return {'first': first, 'sustained_to_T': sustained if sustained < len(loss) else None}


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    commit = summary['preregistration_commit']
    commit_epoch = int(subprocess.check_output(['git', '-C', str(PROGRAM), 'show', '-s', '--format=%ct', commit]))
    for name in ['preregistration.json', 'executed/run.py', 'analysis.py']:
        path = STUDY / name
        committed = subprocess.check_output(['git', '-C', str(PROGRAM), 'show', f'{commit}:{path.relative_to(PROGRAM)}'])
        assert committed == path.read_bytes(), name
    assert sha(STUDY / 'executed/run.py') == prereg['execution_source_sha256']
    assert sha(STUDY / 'analysis.py') == prereg['analysis_source_sha256']
    expected = {}
    for condition in prereg['conditions']:
        for seed in prereg['seeds']:
            for momentum in prereg['momenta']:
                request = {**condition, 'seed': seed, 'momentum': momentum,
                           'spectrum': prereg['spectrum'], 'steps': prereg['steps'],
                           'learning_rate': prereg['learning_rate'], 'domain': prereg['domain'],
                           'source_sha256': prereg['execution_source_sha256'],
                           'preregistration_sha256': sha(STUDY / 'preregistration.json'),
                           'preregistration_commit': commit}
                canonical = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
                expected[hashlib.sha256(canonical).hexdigest()[:16]] = request
    assert {p.name for p in (STUDY / 'results').iterdir()} == set(expected)
    cell_audits, endpoint_lookup = [], {}
    summary_audits = {row['cell_id']: row for row in summary['execution_audits']}
    for cell_id, request in expected.items():
        cell = STUDY / 'results' / cell_id
        metadata = json.loads((cell / 'metadata.json').read_text())
        assert metadata['status'] == 'success' and metadata['request'] == request
        assert metadata['arrays_sha256'] == sha(cell / 'arrays.npz')
        assert commit_epoch < metadata['started_at_epoch'] <= metadata['finished_at_epoch']
        with np.load(cell / 'arrays.npz', allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            loss = arrays['normalized_loss']
            assert len(loss) == prereg['steps'] + 1
            endpoint_lookup[cell_id] = {str(q): hits(loss, q) for q in prereg['thresholds']}
            x, r, eigenvalues = arrays['features'], arrays['rotation'], arrays['eigenvalues']
            n = len(eigenvalues)
            error = max(abs(math.fsum(float(x[k, i]) * float(x[k, j]) for k in range(n)) / n
                            - math.fsum(float(r[i, k]) * float(eigenvalues[k]) * float(r[j, k]) for k in range(n)))
                        for i in range(n) for j in range(n))
        assert error < 1e-14
        assert abs(error - summary_audits[cell_id]['hessian_error']) < 1e-14
        cell_audits.append({'cell_id': cell_id, 'arrays_sha256': sha(cell / 'arrays.npz'),
                            'metadata_sha256': sha(cell / 'metadata.json'), 'scalar_hessian_error': error})
    for pair in summary['paired_endpoints']:
        endpoint, q = pair['endpoint'], str(pair['threshold'])
        assert pair['t_plain'] == endpoint_lookup[pair['plain_cell']][q][endpoint]
        assert pair['t_momentum'] == endpoint_lookup[pair['momentum_cell']][q][endpoint]
        t0, tm = pair['t_plain'], pair['t_momentum']
        assert pair['nominal_prediction'] == math.ceil(t0 / 10)
        startup = next(t for t in range(prereg['steps'] + 1)
                       if t / .1 - .9 * (1 - .9 ** t) / .1 ** 2 >= t0)
        assert pair['startup_prediction'] == startup
        assert pair['startup_signed_step_error'] == startup - tm
        assert pair['nominal_signed_step_error'] == pair['nominal_prediction'] - tm
    receipts = [json.loads(path.read_text()) for path in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    assert all(receipt['preregistration_commit'] == commit for receipt in receipts)
    assert len([cell for receipt in receipts for cell in receipt['new_cells']]) == len(expected)
    for receipt in receipts:
        for cell in receipt['resumed_successes']:
            path = STUDY / 'results' / cell['cell_id']
            assert cell['arrays_sha256'] == sha(path / 'arrays.npz')
            assert cell['metadata_sha256'] == sha(path / 'metadata.json')
    early = [p for p in summary['paired_endpoints'] if p['threshold'] == .9 and p['h'] <= .001]
    counterexamples = [p for p in summary['paired_endpoints']
                       if p['threshold'] == .01 and p['endpoint'] == 'sustained_to_T'
                       and abs(p['startup_signed_step_error']) > abs(p['nominal_signed_step_error'])]
    output = {'status': 'passed', 'scope': 'post-run evidence audit; no training or preregistration changes',
              'verification_source_sha256': sha(Path(__file__)), 'summary_sha256': sha(STUDY / 'summary.json'),
              'preregistration_commit': commit, 'verified_cells': len(cell_audits),
              'commit_precedes_all_training': True, 'all_saved_arrays_finite': True,
              'max_scalar_hessian_error': max(row['scalar_hessian_error'] for row in cell_audits),
              'P2_paired_condition_count': len({p['condition'] for p in early}),
              'P2_max_startup_abs_step_error': max(abs(p['startup_signed_step_error']) for p in early),
              'P4_counterexample_paired_condition_count': len({p['condition'] for p in counterexamples}),
              'P4_counterexample_seed_pairs': len(counterexamples), 'cells': cell_audits}
    path = STUDY / 'executed' / 'saved_evidence_verification.json'
    if path.exists():
        assert json.loads(path.read_text()) == output, 'Previous audit differs; inspect before replacing'
    else:
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key != 'cells'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
