import hashlib
import json
import math
import subprocess
from pathlib import Path

STUDY = Path(__file__).resolve().parent
PROGRAM = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg = json.loads((STUDY / 'preregistration.json').read_text())
    assert sha(Path(__file__)) == prereg['analysis_source_sha256']
    receipt = json.loads((STUDY / 'executed/preregistration_commit_receipt.json').read_text())
    commit = receipt['preregistration_commit']
    commit_epoch = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=PROGRAM))
    for name in prereg['pinned_files']:
        path = STUDY / name
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(PROGRAM)}'], cwd=PROGRAM) == path.read_bytes()
    sources = {(s['alpha'], s['seed'], s['momentum']): s for s in prereg['source_cells']}
    cells, result_manifest = {}, []
    elapsed = 0.0
    for path in sorted((STUDY / 'results').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        request = metadata['request']
        cid = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        assert cid == metadata['cell_id'] == path.parent.name
        assert metadata['status'] == 'success' and metadata['new_training_steps'] == 0
        assert metadata['endpoints_sha256'] == sha(path.parent / 'endpoints.json')
        assert request['preregistration_commit'] == commit
        assert request['preregistration_sha256'] == sha(STUDY / 'preregistration.json')
        assert request['execution_source_sha256'] == prereg['execution_source_sha256']
        assert request['q'] == prereg['threshold'] and request['steps'] == prereg['steps']
        assert commit_epoch < metadata['started_at_epoch'] <= metadata['finished_at_epoch']
        key = (request['alpha'], request['seed'], request['momentum'])
        assert key in sources and key not in cells
        source = sources[key]
        for field in ['arrays', 'metadata']:
            assert request[f'source_{field}_path'] == source[f'{field}_path']
            assert request[f'source_{field}_sha256'] == source[f'{field}_sha256']
            assert sha(PROGRAM / source[f'{field}_path']) == source[f'{field}_sha256']
        output = json.loads((path.parent / 'endpoints.json').read_text())
        assert output['cell_id'] == cid and output['source_cell_id'] == source['cell_id']
        cells[key] = {'cell_id': cid, **output}
        elapsed += metadata['elapsed_seconds']
        for artifact in [path, path.parent / 'endpoints.json']:
            result_manifest.append({'path': str(artifact.relative_to(PROGRAM)),
                                    'sha256': sha(artifact), 'mtime_ns': artifact.stat().st_mtime_ns})
    assert set(cells) == set(sources)
    baseline_path = PROGRAM / prereg['baseline_summary']['path']
    assert sha(baseline_path) == prereg['baseline_summary']['sha256']
    baseline = json.loads(baseline_path.read_text())
    pairs = []
    for alpha in prereg['alphas']:
        for seed in prereg['seeds']:
            plain, momentum = cells[(alpha, seed, 0.0)], cells[(alpha, seed, 0.9)]
            for endpoint in ['first', 'sustained_to_T']:
                t0, tm = plain['endpoints'][endpoint], momentum['endpoints'][endpoint]
                if t0 is None or tm is None or tm == 0:
                    raise RuntimeError('Censored or zero endpoint; keep results without extending budget')
                nominal = math.ceil(t0 / 10)
                startup = next(t for t in range(prereg['steps'] + 1) if 10*t - 90*(1 - .9**t) >= t0)
                old = next(p for p in baseline['paired_endpoints']
                           if p['slow_energy_fraction'] == alpha and p['seed'] == seed and p['endpoint'] == endpoint)
                new = {'alpha': alpha, 'seed': seed, 'endpoint': endpoint,
                       'plain_cell': plain['cell_id'], 'momentum_cell': momentum['cell_id'],
                       't_plain': t0, 't_momentum': tm,
                       'nominal_prediction': nominal, 'startup_prediction': startup,
                       'nominal_signed_error': nominal - tm, 'startup_signed_error': startup - tm,
                       'nominal_relative_abs_error': abs(nominal - tm)/tm,
                       'startup_relative_abs_error': abs(startup - tm)/tm}
                keys = ['t_plain', 't_momentum', 'nominal_relative_abs_error', 'startup_relative_abs_error']
                new['saved_q01_control'] = {k: old[k] for k in keys}
                new['paired_change_from_q01'] = {k: new[k] - old[k] for k in keys}
                pairs.append(new)
    primary = [p for p in pairs if p['endpoint'] == 'sustained_to_T']
    interior = [p for p in primary if p['alpha'] > prereg['threshold']]
    equality = [p for p in primary if p['alpha'] == prereg['threshold']]
    p1 = all(p[k] <= .05 for p in interior for k in ['nominal_relative_abs_error', 'startup_relative_abs_error'])
    p2 = all(p['nominal_relative_abs_error'] <= .05 and .05 < p['startup_relative_abs_error'] <= .25 for p in equality)
    aggregate = []
    keys = ['t_plain', 't_momentum', 'nominal_prediction', 'startup_prediction',
            'nominal_relative_abs_error', 'startup_relative_abs_error']
    for alpha in prereg['alphas']:
        group = [p for p in primary if p['alpha'] == alpha]
        aggregate.append({'alpha': alpha, **{k + '_seed_range': [min(p[k] for p in group), max(p[k] for p in group)] for k in keys},
                          'paired_error_change_seed_ranges': {k: [min(p['paired_change_from_q01'][k] for p in group),
                                                                   max(p['paired_change_from_q01'][k] for p in group)]
                                                             for k in keys[-2:]}})
    for row in prereg['historical_manifest']:
        path = PROGRAM / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_mtime_ns == row['mtime_ns']
    run_receipts = [json.loads(p.read_text()) for p in sorted((STUDY / 'executed').glob('run_receipt_*.json'))]
    for run in run_receipts:
        assert run['preregistration_commit'] == commit
        for row in run['resumed_successes']:
            result = STUDY / 'results' / row['cell_id']
            assert sha(result / 'endpoints.json') == row['endpoints_sha256']
            assert sha(result / 'metadata.json') == row['metadata_sha256']
    summary = {'study': prereg['study'], 'round': prereg['round'], 'domain': prereg['domain'],
               'q': prereg['threshold'], 'preregistration_commit': commit,
               'preregistration_sha256': sha(STUDY / 'preregistration.json'),
               'analysis_source_sha256': sha(Path(__file__)),
               'counts': {'saved_endpoint_cells': len(cells), 'reused_training_cells': len(cells),
                          'new_training_cells': 0, 'new_training_steps': 0, 'paired_conditions': len(prereg['alphas']),
                          'optimizer_recipe_units': 2*len(prereg['alphas']), 'coordinate_seeds': len(prereg['seeds'])},
               'evaluation_cell_seconds_sum': elapsed,
               'predictions': {'P1_alpha_above_q': {'status': 'supported' if p1 else 'refuted', 'paired_conditions': 5},
                               'P2_alpha_equals_q': {'status': 'supported' if p2 else 'refuted', 'paired_conditions': 1}},
               'aggregate': aggregate, 'paired_endpoints': pairs, 'result_manifest': result_manifest,
               'historical_files_unchanged': len(prereg['historical_manifest']),
               'resumed_success_count': sum(len(r['resumed_successes']) for r in run_receipts),
               'boundaries': prereg['boundaries']}
    target = STUDY / 'summary.json'
    if target.exists():
        assert json.loads(target.read_text()) == summary, 'Do not overwrite a successful summary'
    else:
        with target.open('x') as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': summary['predictions'], 'counts': summary['counts'], 'aggregate': aggregate}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
