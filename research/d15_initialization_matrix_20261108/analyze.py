"""Recompute D15 matrices from saved arrays, without training or rescoring choices.

Metrics:
* case = one frozen path; cell = all identical design factors except seed/id.
* baseline = same round/context/seed, uniform (r001/2), a=b=0,s0=1 (r003).
* primary ratio includes a failed candidate as +infinity when baseline is
  finite. JSON uses null + ratio_failed/ratio_infinite flags, never NaN/Inf.
  Baseline failure or absence makes the pair unavailable, never a win.
* ratio_median uses those conservative seed values. ratio_median_finite is
  separately labelled as a diagnostic and must not replace the primary.
* first_hit_0_1 and first_hit_0_01 are saved first integer crossings, with -1
  retained. model_unreachable only certifies plain+linear's LS floor > .1.

Default requires three completed manifests and all 13,572 saved paths.
--partial reads hash-verified batches only and writes *_partial outputs.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import design

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DATA = Path(os.environ.get('AIQ_KB_DATA_ROOT', str(REPO.parent / 'AIQ_KB_DATA'))).resolve()
RUN = DATA / 'runs' / SOURCE.name
ROUNDS = ('r001', 'r002', 'r003')
STATUS = {0: 'finite', 1: 'nonfinite_loss', 2: 'nonfinite_gradient', 3: 'nonfinite_update'}
PURE_SCALING = set(design.PROFILES[:13])
NUMERICAL_FLOOR = 1e-10  # Display diagnostic only; never modifies frozen scoring.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    """Native JSON scalars only; all nonfinite values become explicit null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def median(values):
    values = list(values)
    if not values:
        return None
    return float(np.median(values))


def context_key(case, include_seed=True):
    ignored = {'id', 'profile', 'alpha', 'beta', 'global_scale'}
    if not include_seed:
        ignored.add('seed')
    return tuple(sorted((k, v) for k, v in case.items() if k not in ignored))


def is_baseline(case):
    if case['round'] == 'r003':
        return case['alpha'] == 0 and case['beta'] == 0 and case['global_scale'] == 1
    return case['profile'] == 'uniform'


def cell_key(case):
    return tuple(sorted((k, v) for k, v in case.items() if k not in {'id', 'seed'}))


def ratio_value(candidate, baseline):
    """Return conservative ratio, unavailable when baseline did not survive."""
    if baseline is None or baseline['failed']:
        return None
    if candidate['failed']:
        return float('inf')
    a, b = candidate['finalval'], baseline['finalval']
    if b == 0:
        return 1.0 if a == 0 else float('inf')
    return a / b


def read_round(round_id, partial):
    folder = RUN / round_id
    expected = design.cases(round_id)
    manifest_path = folder / 'manifest.json'
    complete = manifest_path.is_file()
    if not complete and not partial:
        raise RuntimeError(f'{round_id}: complete manifest missing; use --partial only for provisional outputs')
    if not (folder / 'cases.json').exists():
        return [], {'round': round_id, 'status': 'not_started', 'expected_n': len(expected), 'loaded_n': 0}
    frozen_cases = json.loads((folder / 'cases.json').read_text(encoding='utf-8'))
    if frozen_cases != expected:
        raise RuntimeError(f'{round_id}: saved case design differs from frozen enumeration')
    case_by_id = {c['id']: c for c in frozen_cases}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if complete else None
    receipt = json.loads((folder / 'start.json').read_text(encoding='utf-8'))
    files = sorted(folder.glob('batch_*.npz'))
    if manifest is not None:
        if manifest['status'] != 'complete' or manifest['paths'] != len(expected):
            raise RuntimeError(f'{round_id}: bad completion manifest')
        if {p.name for p in files} != set(manifest['batches']):
            raise RuntimeError(f'{round_id}: batch list differs from manifest')
    records = []
    seen = set()
    verified_hashes = {}
    for path in files:
        meta_path = path.with_suffix('.json')
        if not meta_path.exists():
            if partial:
                continue  # A running process may still be writing this batch.
            raise RuntimeError(f'{path}: metadata missing')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        actual = digest(path)
        if actual != meta['sha256'] or (manifest and actual != manifest['batches'][path.name]):
            raise RuntimeError(f'{path}: saved-batch SHA-256 mismatch')
        verified_hashes[path.name] = actual
        with np.load(path, allow_pickle=False) as data:
            ids = data['ids'].tolist()
            if ids != meta['ids']:
                raise RuntimeError(f'{path}: metadata/array ids differ')
            steps = data['checkpoint_steps']
            if steps[0] != 0 or steps[-1] != design.STEPS:
                raise RuntimeError(f'{path}: wrong observation window')
            for i, identifier in enumerate(ids):
                if identifier in seen or identifier not in case_by_id:
                    raise RuntimeError(f'{path}: duplicated/unknown case {identifier}')
                seen.add(identifier)
                c = dict(case_by_id[identifier])
                code = int(data['status_code'][i])
                train, val = float(data['train_loss'][i, -1]), float(data['val_loss'][i, -1])
                failed = code != 0 or not np.isfinite(train) or not np.isfinite(val)
                if code == 0 and failed:
                    raise RuntimeError(f'{identifier}: nonfinite endpoint without saved failure status')
                floor = float(data['linear_train_floor'][i])
                rec = dict(c)
                rec.update(
                    finaltrain=train, finalval=val,
                    initialtrain=float(data['initial_loss'][i]),
                    initialval=float(data['initial_val_loss'][i]),
                    initial_allzero=bool(data['initial_zero_gradient'][i]),
                    allzero=bool(data['initial_zero_gradient'][i]),
                    initial_grad_norm=data['initial_grad_norm'][i].tolist(),
                    first_hit_0_1=int(data['first_hit_0_1'][i]),
                    first_hit_0_01=int(data['first_hit_0_01'][i]),
                    nonfinite_code=code, failure_status=STATUS[code],
                    failed=failed, failed_step=int(data['failed_step'][i]),
                    linear_train_floor=floor,
                    model_unreachable=c['architecture'] == 'plain' and c['activation'] == 'linear' and floor > .1,
                    initial_function_delta=float(data['initial_function_delta'][i]),
                    final_weight_relative_change=data['final_weight_relative_change'][i].tolist(),
                    final_prediction_change=float(data['final_prediction_change'][i]),
                    source_batch=str(path.relative_to(RUN)), source_sha256=actual,
                    is_baseline=is_baseline(c),
                )
                records.append(rec)
    if complete and len(seen) != len(expected):
        raise RuntimeError(f'{round_id}: incomplete path denominator')
    info = dict(round=round_id, status='complete' if complete else 'partial',
                expected_n=len(expected), loaded_n=len(records), batch_n=len(verified_hashes),
                preregistration_commit=receipt['preregistration_commit'],
                started_unix=receipt['started_unix'], source_hashes=receipt['source_hashes'],
                verified_batches=verified_hashes)
    return records, info


def pair_baselines(records):
    baselines = {}
    for rec in records:
        if rec['is_baseline']:
            # Metric fields are excluded by constructing keys from design fields.
            key = context_key(design_fields(rec))
            if key in baselines:
                raise RuntimeError(f'duplicated baseline context: {rec["id"]}')
            baselines[key] = rec
    for rec in records:
        baseline = baselines.get(context_key(design_fields(rec)))
        ratio = ratio_value(rec, baseline)
        rec.update(
            baseline_id=baseline['id'] if baseline else None,
            baselineval=baseline['finalval'] if baseline else None,
            baseline_failed=bool(baseline and baseline['failed']),
            baseline_missing=baseline is None,
            baseline_nearzero=bool(baseline and not baseline['failed'] and baseline['finalval'] <= NUMERICAL_FLOOR),
            both_nearzero=bool(baseline and not baseline['failed'] and not rec['failed']
                               and baseline['finalval'] <= NUMERICAL_FLOOR and rec['finalval'] <= NUMERICAL_FLOOR),
            ratio=ratio,
            ratio_failed=bool(rec['failed'] and baseline and not baseline['failed']),
            ratio_infinite=ratio is not None and not np.isfinite(ratio),
            ratio_unavailable=ratio is None,
            improved=bool(ratio is not None and np.isfinite(ratio) and ratio < 1),
            hit01=rec['first_hit_0_1'] >= 0,
            hit001=rec['first_hit_0_01'] >= 0,
        )


DESIGN_KEYS = {'task', 'activation', 'optimizer', 'rate', 'seed', 'profile', 'architecture',
               'depth', 'width', 'condition', 'alpha', 'beta', 'global_scale', 'id', 'round', 'lr', 'steps'}


def design_fields(rec):
    return {k: v for k, v in rec.items() if k in DESIGN_KEYS}


def aggregate(records):
    groups = defaultdict(list)
    for rec in records:
        groups[cell_key(design_fields(rec))].append(rec)
    cells = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: r['seed'])
        values = [r['ratio'] for r in rows]
        available = [v for v in values if v is not None]
        finite = [v for v in available if np.isfinite(v)]
        # Missing/failed baselines make the primary unavailable. Candidate
        # failure is infinity, included in the conservative median.
        primary = median(available) if len(available) == len(rows) else None
        abs_values = [float('inf') if r['failed'] else r['finalval'] for r in rows]
        train_values = [float('inf') if r['failed'] else r['finaltrain'] for r in rows]
        baseline_abs_values = [float('inf') if r['baseline_failed'] else r['baselineval']
                               for r in rows if r['baselineval'] is not None]
        cell = dict(key)
        expected_seed_n = 4 if cell['round'] == 'r001' else 3
        cell.update(
            id=f'cell_{len(cells):05d}', seed_n=len(rows), seeds=[r['seed'] for r in rows],
            expected_seed_n=expected_seed_n, partial_cell=len(rows) != expected_seed_n,
            case_ids=[r['id'] for r in rows],
            ratio_by_seed=values,
            finalval_by_seed=[r['finalval'] for r in rows],
            baselineval_by_seed=[r['baselineval'] for r in rows],
            ratio_failed_by_seed=[r['ratio_failed'] for r in rows],
            ratio_unavailable_by_seed=[r['ratio_unavailable'] for r in rows],
            ratio_median=primary,
            median_failed=primary is not None and not np.isfinite(primary),
            median_unavailable=primary is None,
            ratio_median_finite=median(finite), ratio_finite_n=len(finite),
            failed_n=sum(r['failed'] for r in rows),
            baseline_failed_n=sum(r['baseline_failed'] for r in rows),
            baseline_missing_n=sum(r['baseline_missing'] for r in rows),
            baselineval_median=median(baseline_abs_values),
            baseline_absolutevalmedian=median(baseline_abs_values),
            baseline_nearzero_n=sum(r['baseline_nearzero'] for r in rows),
            both_nearzero_n=sum(r['both_nearzero'] for r in rows),
            improved_n=sum(r['improved'] for r in rows),
            hit01_n=sum(r['hit01'] for r in rows), hit001_n=sum(r['hit001'] for r in rows),
            initial_allzero_n=sum(r['initial_allzero'] for r in rows),
            model_unreachable_n=sum(r['model_unreachable'] for r in rows),
            absolutevalmedian=median(abs_values), absolutetrainmedian=median(train_values),
            val_median_failed=bool(not np.isfinite(median(abs_values))),
            first_hit_0_1_by_seed=[r['first_hit_0_1'] for r in rows],
            first_hit_0_01_by_seed=[r['first_hit_0_01'] for r in rows],
            first_hit_0_1_median_reached=median([r['first_hit_0_1'] for r in rows if r['hit01']]),
            first_hit_0_01_median_reached=median([r['first_hit_0_01'] for r in rows if r['hit001']]),
            initial_function_delta_max=max(r['initial_function_delta'] for r in rows),
            failure_codes=dict(Counter(r['nonfinite_code'] for r in rows)),
        )
        cells.append(cell)
    return cells


def count_metrics(records):
    finite_ratios = [r['ratio'] for r in records if r['ratio'] is not None and np.isfinite(r['ratio'])]
    return dict(n=len(records), failure_codes=dict(Counter(r['nonfinite_code'] for r in records)),
                failed_n=sum(r['failed'] for r in records),
                initial_allzero_n=sum(r['initial_allzero'] for r in records),
                hit01_n=sum(r['hit01'] for r in records), hit001_n=sum(r['hit001'] for r in records),
                model_unreachable_n=sum(r['model_unreachable'] for r in records),
                baseline_n=sum(r['is_baseline'] for r in records),
                comparison_n=sum(not r['is_baseline'] for r in records),
                improved_n=sum(r['improved'] and not r['is_baseline'] for r in records),
                ratio_finite_n=len(finite_ratios),
                baseline_failed_n=sum(r['baseline_failed'] for r in records),
                baseline_missing_n=sum(r['baseline_missing'] for r in records),
                finite_ratio_median_diagnostic=median(finite_ratios))


def universal_front_back(records):
    groups = defaultdict(dict)
    for r in records:
        if r['round'] == 'r001' and r['profile'] in {'front', 'back'}:
            groups[context_key(design_fields(r), include_seed=False)].setdefault(r['seed'], {})[r['profile']] = r
    contexts = []
    for key, seeds in groups.items():
        ratios, wins, unavailable, failures = [], 0, 0, 0
        pairs = []
        for seed, pair in sorted(seeds.items()):
            front, back = pair.get('front'), pair.get('back')
            if front is None or back is None:
                unavailable += 1
                continue
            ratio = ratio_value(front, back)
            if ratio is None:
                unavailable += 1
            else:
                ratios.append(ratio)
                wins += int(np.isfinite(ratio) and ratio < 1)
                failures += int(not np.isfinite(ratio))
            pairs.append(dict(seed=seed, front=front['finalval'], back=back['finalval'],
                              ratio=ratio, front_failed=front['failed'], back_failed=back['failed'],
                              back_nearzero=not back['failed'] and back['finalval'] <= NUMERICAL_FLOOR,
                              both_nearzero=not front['failed'] and not back['failed']
                                            and front['finalval'] <= NUMERICAL_FLOOR and back['finalval'] <= NUMERICAL_FLOOR))
        full = len(seeds) == 4 and len(pairs) == 4 and unavailable == 0
        med = median(ratios) if full else None
        passed = bool(full and med <= .95 and wins >= 3)
        contexts.append(dict(dict(key), seed_n=len(seeds), ratio_median=med, improved_n=wins,
                             unavailable_n=unavailable, candidate_failure_n=failures,
                             front_absolutevalmedian=median([float('inf') if p['front_failed'] else p['front'] for p in pairs]),
                             back_absolutevalmedian=median([float('inf') if p['back_failed'] else p['back'] for p in pairs]),
                             back_nearzero_n=sum(p['back_nearzero'] for p in pairs),
                             both_nearzero_n=sum(p['both_nearzero'] for p in pairs),
                             passes_registered_criterion=passed, pairs=pairs))
    passed_n = sum(c['passes_registered_criterion'] for c in contexts)
    complete = len(contexts) == 36 and all(c['seed_n'] == 4 and len(c['pairs']) == 4 for c in contexts)
    verdict = 'partial'
    if complete:
        verdict = 'supported' if passed_n == 36 else ('mixed_universal_refuted' if passed_n else 'refuted')
    return dict(criterion='Every one of 36 contexts: paired final-validation front/back median <= .95 and >=3/4 seeds improve; failures retained.',
                expected_context_n=36, observed_context_n=len(contexts), passed_context_n=passed_n,
                diagnostic_numerical_floor=dict(threshold=NUMERICAL_FLOOR,
                    contexts_with_any_both_nearzero_n=sum(c['both_nearzero_n'] > 0 for c in contexts),
                    contexts_with_all_pairs_both_nearzero_n=sum(c['both_nearzero_n'] == 4 for c in contexts),
                    passed_contexts_with_any_both_nearzero_n=sum(c['passes_registered_criterion'] and c['both_nearzero_n'] > 0 for c in contexts),
                    note='Absolute near-zero residual diagnostics only; the registered front/back decision is unchanged.'),
                complete=complete, verdict=verdict, contexts=contexts)


def function_fact(records):
    selected = []
    for r in records:
        if r['architecture'] != 'plain' or r['activation'] not in {'linear', 'relu'}:
            continue
        if r['profile'] in PURE_SCALING or (r['profile'] == 'grid' and r['global_scale'] == 1):
            selected.append(r)
    groups = defaultdict(list)
    for r in selected:
        groups[(r['round'], r['activation'])].append(r)
    return dict(
        statement='Positive product-one multipliers preserve the initial plain bias-free linear/ReLU function; saved train-input predictions checked.',
        scope='Initialization identity only, not a new empirical discovery; no invariance claim for tanh/residual/LN or later training.',
        tolerance=1e-10, n=len(selected),
        max_abs_difference=max((r['initial_function_delta'] for r in selected), default=None),
        failed_n=sum(r['initial_function_delta'] > 1e-10 for r in selected),
        groups=[dict(round=k[0], activation=k[1], n=len(rows),
                     max_abs_difference=max(r['initial_function_delta'] for r in rows),
                     failed_n=sum(r['initial_function_delta'] > 1e-10 for r in rows))
                for k, rows in groups.items()])


def make_summary(records, cells, round_info, partial):
    rounds = {}
    for info in round_info:
        rows = [r for r in records if r['round'] == info['round']]
        rounds[info['round']] = dict(info, **count_metrics(rows),
                                    cell_n=sum(c['round'] == info['round'] for c in cells))
    factors = {}
    for factor in ('task', 'activation', 'architecture', 'optimizer', 'depth', 'width', 'condition', 'profile', 'global_scale', 'alpha', 'beta'):
        values = sorted({r[factor] for r in records if factor in r}, key=str)
        factors[factor] = values
    return dict(
        schema_version=1, experiment=SOURCE.name, generated_unix=time.time(),
        status='partial' if partial else 'complete', raw_root=str(RUN),
        counts=count_metrics(records), cell_n=len(cells), rounds=rounds, factor_levels=factors,
        metrics=dict(primary='ratio_median: paired final validation MSE / same-context baseline; conservative failures included.',
                     baseline='r001/r002 uniform; r003 alpha=beta=0 and global_scale=1; same round and seed.',
                     nulls='ratio_failed/median_failed encode +infinity; ratio_unavailable/median_unavailable encode missing or failed baseline.',
                     ratio_median_finite='diagnostic only: omits nonfinite/unavailable pairs and must not replace primary.',
                     absolutevalmedian='all-seed validation MSE median, failed routes counted as +infinity.',
                     first_hits='integer steps, -1 means not reached; reached-only medians always carry success denominator.',
                     model_unreachable='only plain linear with least-squares train floor>.1; no nonlinear reachability claims.'),
        r001_front_vs_back=universal_front_back(records),
        product_preserving_function_check=function_fact(records),
        diagnostic_numerical_floor=dict(
            threshold=NUMERICAL_FLOOR,
            baseline_nearzero_path_n=sum(r['baseline_nearzero'] for r in records),
            both_nearzero_path_n=sum(r['both_nearzero'] for r in records),
            cells_with_any_baseline_nearzero_n=sum(c['baseline_nearzero_n'] > 0 for c in cells),
            cells_with_all_pairs_both_nearzero_n=sum(c['both_nearzero_n'] == c['seed_n'] for c in cells),
            note='Post-audit display diagnostic: ratios with baseline validation MSE <=1e-10 may magnify numerical residual differences. Show absolute candidate/baseline MSE and mark these comparisons; do not suppress raw ratios or alter the preregistered primary score.'),
        r002_baseline_comparisons=[c for c in cells if c['round'] == 'r002'],
        r003_baseline_comparisons=[c for c in cells if c['round'] == 'r003'],
        interpretation='All tasks/seeds/validation data are development synthetic diagnostics; minima are not validated recommendations or evidence of general task inference.',
        architecture_notes=dict(
            residual_hidden='The first projection and final readout are linear. Only internal square layers are residual branches. At depth=2 there are no internal layers: the model is linear, even when activation is labelled relu.',
            layernorm_hidden='Affine-free LayerNorm, eps=1e-5, before each hidden activation; no bias anywhere.',
            zas_embed='Rectangular identity embedding plus square internal identities and zero output. Not a direct reproduction of the square-output ZAS theorem.'),
        source='Frozen case enumeration and SHA-256-verified saved batches; no trajectories trained by this analysis.')


def write_csv(path, records):
    fieldnames = sorted(set().union(*(r.keys() for r in records))) if records else ['id']
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = clean(r)
            for k, value in row.items():
                if isinstance(value, (list, dict)):
                    row[k] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            writer.writerow(row)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='write provisional *_partial outputs from available saved batches')
    args = parser.parse_args()
    records, infos = [], []
    for round_id in ROUNDS:
        rows, info = read_round(round_id, args.partial)
        records.extend(rows)
        infos.append(info)
    pair_baselines(records)
    cells = aggregate(records)
    if not args.partial:
        if any(c['partial_cell'] for c in cells):
            raise RuntimeError('Complete analysis unexpectedly has incomplete seed cells')
        if len(records) != sum(len(design.cases(r)) for r in ROUNDS):
            raise RuntimeError('Complete analysis has wrong frozen denominator')
    suffix = '_partial' if args.partial else ''
    summary = make_summary(records, cells, infos, args.partial)
    dump(SOURCE / f'summary{suffix}.json', summary)
    dump(SOURCE / f'cells{suffix}.json', cells)
    write_csv(SOURCE / f'case_metrics{suffix}.csv', records)
    print(json.dumps(clean(dict(status=summary['status'], cases=len(records), cells=len(cells),
                               rounds={k: v['loaded_n'] for k, v in summary['rounds'].items()},
                               failures=summary['counts']['failed_n'],
                               universal_front=summary['r001_front_vs_back']['verdict'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
