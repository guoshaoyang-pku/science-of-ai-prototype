#!/usr/bin/env python3
"""Finite-sample, fixed-feature GD scaling study. Outputs stay outside the repo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

np.seterr(divide='raise', over='raise', invalid='raise')

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
SPEC_PATH = SOURCE / 'preregistration.json'
SPEC = json.loads(SPEC_PATH.read_text())
DATA_ROOT = Path(os.environ.get('AIQ_KB_DATA_ROOT', SPEC['data_root_default']))
RUN = Path(os.environ['AIQ_KB_DATA_ROOT']) / 'runs' / SPEC['run_name'] if 'AIQ_KB_DATA_ROOT' in os.environ else REPO / 'evidence' / SPEC['run_name']
LEGACY = REPO / 'science_program_v2/directions/D2_overfitting_u_curve'
STUDY = RUN / 'studies' / SPEC['study']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_contract():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    files = [SPEC_PATH, Path(__file__)]
    for path in files:
        stored = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(REPO)}'], cwd=REPO)
        if stored != path.read_bytes():
            raise RuntimeError(f'Uncommitted execution contract: {path}')
    return {'git_commit': commit, 'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in files}}


def targets(x, name):
    a = x[:, 0]
    if name == 'linear':
        y = a
    elif name == 'quadratic':
        y = (a + .5 * (a * a - 1)) / np.sqrt(1.5)
    elif name == 'additive':
        y = (a + .5 * (x[:, 1] ** 2 - 1)) / np.sqrt(1.5)
    elif name == 'interaction':
        y = (a + .5 * a * x[:, 1]) / np.sqrt(1.25)
    elif name == 'product':
        y = a * x[:, 1]
    elif name == 'sine':
        y = np.sin(2 * a) / np.sqrt((1 - np.exp(-8)) / 2) if x.shape[1] == 1 else (np.sin(a) + .5 * np.sin(2 * x[:, 1])) / np.sqrt((1 - np.exp(-2)) / 2 + .25 * (1 - np.exp(-8)) / 2)
    elif name == 'vector':
        return np.column_stack([a, (a + .5 * a * x[:, 1]) / np.sqrt(1.25)])
    else:
        raise ValueError(name)
    return y[:, None]


def make_draw(protocol, n, seed):
    d = SPEC['protocols'][protocol]['input_dim']
    rng = np.random.default_rng(910000 + seed + 10000 * d)
    half = rng.standard_normal((max(SPEC['n']) // 2, d))
    pool = np.stack([half, -half], axis=1).reshape(-1, d)
    val_half = rng.standard_normal((SPEC['validation_rows'] // 2, d))
    vx = np.stack([val_half, -val_half], axis=1).reshape(-1, d)
    x = pool[:n]
    wrng = np.random.default_rng(110000 + seed + 10000 * d)
    weight = wrng.standard_normal((SPEC['width'], d)) / np.sqrt(d)
    bias = wrng.standard_normal(SPEC['width']) if SPEC['protocols'][protocol]['hidden_bias'] else np.zeros(SPEC['width'])
    raw = np.maximum(np.dot(x, weight.T) + bias, 0)
    val_raw = np.maximum(np.dot(vx, weight.T) + bias, 0)
    center = raw.mean(0)
    scale = np.sqrt(np.mean((raw - center) ** 2)) * np.sqrt(SPEC['width'])
    phi = np.column_stack([(raw - center) / scale, np.ones(n)])
    val_phi = np.column_stack([(val_raw - center) / scale, np.ones(len(vx))])
    nrng = np.random.default_rng(210000 + seed + 10000 * d)
    eps = nrng.standard_normal((max(SPEC['n']), 2))[:n]
    val_eps = nrng.standard_normal((len(vx), 2))
    return {'train_x': x, 'val_x': vx, 'phi': phi, 'val_phi': val_phi, 'epsilon': eps, 'val_epsilon': val_eps, 'weight': weight, 'hidden_bias': bias, 'feature_center': center, 'feature_scale': np.array(scale)}


def spectral_model(data):
    phi, vphi = data['phi'], data['val_phi']
    n = len(phi)
    lam, vectors = np.linalg.eigh(np.dot(phi.T, phi) / n)
    tol = max(lam.max() * 1e-12, 1e-14)
    active = lam > tol
    lam = np.where(active, lam, 0)
    q = 1 - 2 * SPEC['eta'] * lam
    if q.min() <= 0 or q.max() > 1 + 1e-10:
        raise RuntimeError('GD factors must be positive and stable')
    hval = np.dot(np.dot(vectors.T, np.dot(vphi.T, vphi) / len(vphi)), vectors)
    decay = -np.log(q) / SPEC['eta']
    return {'lam': lam, 'vectors': vectors, 'q': q, 'active': active, 'rank': int(active.sum()), 'hval': hval, 'decay': decay, 'n': n}


def response(model, t):
    t = np.atleast_1d(t).astype(float)
    lam, active, q = model['lam'], model['active'], model['q']
    powers = np.exp(np.outer(t, np.log(q)))
    f = np.zeros_like(powers)
    f[:, active] = -np.expm1(np.outer(t, np.log(q[active]))) / lam[active]
    return f, powers


def target_terms(model, data, task):
    y, vy = targets(data['train_x'], task), targets(data['val_x'], task)
    g = np.dot(np.dot(model['vectors'].T, data['phi'].T), y) / len(y)
    g[~model['active']] = 0
    vg = np.dot(np.dot(model['vectors'].T, data['val_phi'].T), vy) / len(vy)
    h = model['hval'] * (np.dot(g, g.T) / y.shape[1])
    linear = np.mean(g * vg, axis=1)
    return {'y': y, 'vy': vy, 'g': g, 'h': h, 'linear': linear, 'val_y2': float(np.mean(vy ** 2)), 'train_y2': float(np.mean(y ** 2))}


def expected_curves(model, terms, t):
    f, powers = response(model, t)
    b = np.maximum(np.einsum('ti,ij,tj->t', f, terms['h'], f, optimize=True) - 2 * np.dot(f, terms['linear']) + terms['val_y2'], 0)
    noise_weights = np.diag(model['hval']) * model['lam'] / model['n']
    noise = np.dot(f ** 2, noise_weights)
    energy = np.zeros_like(model['lam'])
    active = model['active']
    energy[active] = np.mean(terms['g'][active] ** 2, axis=1) / model['lam'][active]
    train_bias = np.maximum(terms['train_y2'] - np.sum((1 - powers ** 2) * energy, axis=1), 0)
    train_noise = (model['n'] - model['rank'] + np.sum(powers[:, active] ** 2, axis=1)) / model['n']
    return b, noise, train_bias, train_noise


def derivatives(model, terms, t, sigma2):
    f, powers = response(model, t)
    active = model['active']
    fp, fpp = np.zeros_like(f), np.zeros_like(f)
    fp[:, active] = powers[:, active] * model['decay'][active] / model['lam'][active]
    fpp[:, active] = -fp[:, active] * model['decay'][active]
    hp = np.dot(f, terms['h']) - terms['linear']
    bprime = 2 * np.sum(fp * hp, axis=1)
    bsecond = 2 * (np.sum(fpp * hp, axis=1) + np.einsum('ti,ij,tj->t', fp, terms['h'], fp, optimize=True))
    nw = np.diag(model['hval']) * model['lam'] / model['n']
    nprime = 2 * np.dot(f * fp, nw)
    nsecond = 2 * np.dot(fp ** 2 + f * fpp, nw)
    return bprime, nprime * sigma2, bprime + sigma2 * nprime, bsecond + sigma2 * nsecond


def realized_curves(model, data, terms, t, sigma2):
    y = terms['y']
    outputs = y.shape[1]
    noisy_y = y + np.sqrt(sigma2) * data['epsilon'][:, :outputs]
    noisy_g = np.dot(np.dot(model['vectors'].T, data['phi'].T), noisy_y) / len(y)
    noisy_g[~model['active']] = 0
    f, _ = response(model, t)
    lam = model['lam']
    tr = np.mean(noisy_y ** 2) - 2 * np.dot(f, np.mean(noisy_g ** 2, axis=1)) + np.dot(f ** 2, lam * np.mean(noisy_g ** 2, axis=1))
    ht = model['hval'] * (np.dot(noisy_g, noisy_g.T) / outputs)
    val_projection = np.dot(np.dot(model['vectors'].T, data['val_phi'].T), terms['vy']) / len(data['val_phi'])
    val_linear = np.mean(noisy_g * val_projection, axis=1)
    clean_val = np.einsum('ti,ij,tj->t', f, ht, f, optimize=True) - 2 * np.dot(f, val_linear) + terms['val_y2']
    noisy_vy = terms['vy'] + np.sqrt(sigma2) * data['val_epsilon'][:, :outputs]
    noisy_projection = np.dot(np.dot(model['vectors'].T, data['val_phi'].T), noisy_vy) / len(noisy_vy)
    noisy_val = np.einsum('ti,ij,tj->t', f, ht, f, optimize=True) - 2 * np.dot(f, np.mean(noisy_g * noisy_projection, axis=1)) + np.mean(noisy_vy ** 2)
    return np.maximum(tr, 0), np.maximum(clean_val, 0), np.maximum(noisy_val, 0)


def grid_components(model, data, terms, grid):
    f, powers = response(model, grid)
    b, noise, tb, tn = expected_curves(model, terms, grid)
    bp, np_, _, _ = derivatives(model, terms, grid, 1)
    _, _, _, bsecond = derivatives(model, terms, grid, 0)
    _, _, _, total_second = derivatives(model, terms, grid, 1)
    outputs = terms['y'].shape[1]
    ge = np.dot(np.dot(model['vectors'].T, data['phi'].T), data['epsilon'][:, :outputs]) / model['n']
    ge[~model['active']] = 0
    g = terms['g']
    he = model['hval'] * (np.dot(ge, ge.T) / outputs)
    hc = model['hval'] * (np.dot(g, ge.T) / outputs)
    ve = np.dot(np.dot(model['vectors'].T, data['val_phi'].T), data['val_epsilon'][:, :outputs]) / len(data['val_phi'])
    vy_projection = np.dot(np.dot(model['vectors'].T, data['val_phi'].T), terms['vy']) / len(data['val_phi'])
    val_cross = 2 * (np.einsum('ti,ij,tj->t', f, hc, f, optimize=True) - np.dot(f, np.mean(ge * vy_projection, axis=1)))
    val_noise = np.einsum('ti,ij,tj->t', f, he, f, optimize=True)
    train_cross = 2 * (np.mean(terms['y'] * data['epsilon'][:, :outputs]) - 2 * np.dot(f, np.mean(g * ge, axis=1)) + np.dot(f ** 2, model['lam'] * np.mean(g * ge, axis=1)))
    train_noise_realized = np.mean(data['epsilon'][:, :outputs] ** 2) - 2 * np.dot(f, np.mean(ge ** 2, axis=1)) + np.dot(f ** 2, model['lam'] * np.mean(ge ** 2, axis=1))
    val_label_cross = 2 * (np.mean(terms['vy'] * data['val_epsilon'][:, :outputs]) - np.dot(f, np.mean(g * ve, axis=1)))
    val_label_noise = -2 * np.dot(f, np.mean(ge * ve, axis=1)) + np.mean(data['val_epsilon'][:, :outputs] ** 2)
    return {'b': b, 'noise': noise, 'tb': tb, 'tn': tn, 'bp': bp, 'np': np_, 'bsecond': bsecond, 'nsecond': total_second - bsecond, 'val_cross': val_cross, 'val_noise': val_noise, 'train_cross': train_cross, 'train_noise_realized': train_noise_realized, 'val_label_cross': val_label_cross, 'val_label_noise': val_label_noise}


def measure(model, data, terms, grid, sigma2, base):
    b, noise = base['b'], base['noise']
    risk = b + sigma2 * noise
    j = int(np.argmin(risk))
    minimum = int(grid[j])
    if 0 < j < len(grid) - 1:
        lo, hi = grid[j - 1], grid[j + 1]
        def objective(t):
            bias, cost, _, _ = expected_curves(model, terms, [t])
            return float(bias[0] + sigma2 * cost[0])
        result = minimize_scalar(objective, bounds=(lo, hi), method='bounded', options={'xatol': .05})
        candidates = np.unique(np.clip(np.arange(int(result.x) - 2, int(result.x) + 4), 0, SPEC['max_steps']))
        candidate_b, candidate_n, _, _ = expected_curves(model, terms, candidates)
        minimum = int(candidates[np.argmin(candidate_b + sigma2 * candidate_n)])
    extra = np.setdiff1d([minimum, minimum / 2], grid)
    additional = grid_components(model, data, terms, extra) if len(extra) else {k: np.array([]) for k in base}
    unsorted_t = np.r_[grid, extra]
    order = np.argsort(unsorted_t)
    t = unsorted_t[order]
    combined = {k: np.r_[v, additional[k]][order] for k, v in base.items()}
    b, noise, tb, tn = (combined[k] for k in ('b', 'noise', 'tb', 'tn'))
    risk = b + sigma2 * noise
    idx = int(np.where(t == minimum)[0][0])
    half = int(np.where(t == minimum / 2)[0][0])
    interior = 0 < minimum < SPEC['max_steps']
    tr = np.maximum(tb + np.sqrt(sigma2) * combined['train_cross'] + sigma2 * combined['train_noise_realized'], 0)
    cv = np.maximum(b + np.sqrt(sigma2) * combined['val_cross'] + sigma2 * combined['val_noise'], 0)
    nv = np.maximum(cv + np.sqrt(sigma2) * combined['val_label_cross'] + sigma2 * combined['val_label_noise'], 0)
    bp, np_ = combined['bp'], sigma2 * combined['np']
    rp, rpp = bp + np_, combined['bsecond'] + sigma2 * combined['nsecond']
    after = np.where((t[:-1] > minimum) & (rpp[:-1] > 0) & (rpp[1:] <= 0))[0]
    inflection = None
    if len(after):
        k = int(after[0])
        inflection = float(t[k] - rpp[k] * (t[k + 1] - t[k]) / (rpp[k + 1] - rpp[k]))
    expected_train = tb + sigma2 * tn
    row = {'t_star': minimum, 'U_rise': SPEC['eta'] * minimum, 'interior': interior, 'rank': model['rank'], 'risk_initial': float(risk[0]), 'risk_minimum': float(risk[idx]), 'risk_final': float(risk[-1]), 'noise_at_half_over_at_rise': float(noise[half] / noise[idx]) if noise[idx] > 0 else None, 'expected_train_fraction_at_rise': float(expected_train[idx] / expected_train[0]), 'realized_train_fraction_at_rise': float(tr[idx] / tr[0]), 'noise_risk_at_rise': float(sigma2 * noise[idx]), 'bias_at_rise': float(b[idx]), 'slope_at_rise': float(rp[idx]), 'curvature_at_rise': float(rpp[idx]), 'curvature_inflection_t': inflection, 'curvature_inflection_over_rise': inflection / minimum if inflection and minimum else None}
    arrays = {'t': t, 'U': SPEC['eta'] * t, 'signal_bias': b, 'variance_unit': noise, 'expected_clean_val': risk, 'expected_noisy_val': risk + sigma2, 'expected_noisy_train': expected_train, 'realized_noisy_train': tr, 'realized_clean_val': cv, 'realized_noisy_val': nv, 'signal_slope_U': bp, 'noise_slope_U': np_, 'risk_slope_U': rp, 'risk_curvature_U': rpp}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError('Nonfinite curve')
    return row, arrays


def independent_gate(model, data, terms):
    outputs = terms['y'].shape[1]
    sigma2 = .5
    target = terms['y'] + np.sqrt(sigma2) * data['epsilon'][:, :outputs]
    w = np.zeros((data['phi'].shape[1], outputs))
    checks = []
    for step in range(33):
        if step in (0, 1, 4, 16, 32):
            f, _ = response(model, [step])
            g = np.dot(np.dot(model['vectors'].T, data['phi'].T), target) / len(target)
            g[~model['active']] = 0
            predicted = np.dot(model['vectors'], f[0, :, None] * g)
            error = float(np.max(np.abs(w - predicted)))
            val_loss = float(np.mean((np.dot(data['val_phi'], w) - terms['vy']) ** 2))
            _, val_curve, _ = realized_curves(model, data, terms, [step], sigma2)
            amap = np.dot(np.dot(model['vectors'] * f[0], model['vectors'].T), data['phi'].T) / len(target)
            noise_direct = float(np.sum(np.dot(data['val_phi'], amap) ** 2) / len(data['val_phi']))
            _, noise, _, _ = expected_curves(model, terms, [step])
            checks.append({'step': step, 'head_max_error': error, 'val_mse_error': abs(val_loss - val_curve[0]), 'noise_response_error': abs(noise_direct - noise[0])})
            if max(error, abs(val_loss - val_curve[0]), abs(noise_direct - noise[0])) > 1e-9:
                raise RuntimeError(f'Independent GD gate failed: {checks[-1]}')
        w -= 2 * SPEC['eta'] * np.dot(data['phi'].T, np.dot(data['phi'], w) - target) / len(target)
    return {'status': 'verified', 'checks': checks, 'output_channels': outputs, 'uses_raw_GD': True}


def legacy_summary():
    path = LEGACY / 'studies/r003_noise_sample_scaling/summary.json'
    data = json.loads(path.read_text())
    cells = data['cells']
    x = np.array([c['n'] / c['noise_variance'] for c in cells])
    u = np.array([SPEC['eta'] * c['t_star'] for c in cells])
    c = float(np.exp(np.mean(np.log(u / x))))
    checks = []
    for n in (32, 64, 128):
        train = np.array([cell['n'] != n for cell in cells])
        coefficient = float(np.exp(np.mean(np.log(u[train] / x[train]))))
        factor = np.maximum(u[~train] / (coefficient * x[~train]), coefficient * x[~train] / u[~train])
        checks.append({'held_out_n': n, 'c': coefficient, 'within2': int((factor <= 2).sum()), 'count': int((~train).sum())})
    factor = np.maximum(u / (c * x), c * x / u)
    return {'source': str(path), 'source_sha256': sha(path), 'fixed_exponent_1': {'c': c, 'display_c': round(c, 2), 'within2': int((factor <= 2).sum()), 'count': len(cells)}, 'leave_n_out': checks}


def fit_summary(rows, contract):
    records = []
    for protocol, recipe in SPEC['protocols'].items():
        for task in recipe['tasks']:
            selected = [r for r in rows if r['protocol'] == protocol and r['task'] == task]
            calibration = [r for r in selected if r['n'] in SPEC['calibration']['n'] and r['sigma2'] in SPEC['calibration']['sigma2']]
            holdout = [r for r in selected if r not in calibration]
            valid = [r for r in calibration if r['interior']]
            x = np.array([r['n'] / r['sigma2'] for r in valid])
            y = np.array([r['U_rise'] for r in valid])
            if not len(valid):
                records.append({'protocol': protocol, 'task': task, 'calibration_interior': 0, 'calibration_total': len(calibration), 'models': {}})
                continue
            c = float(np.exp(np.mean(np.log(y / x))))
            b, a = np.polyfit(np.log(x), np.log(y), 1)
            def log_curve(x, log_a, log_b):
                return np.log(np.exp(log_a) * np.log1p(np.exp(log_b) * x))
            pars, _ = curve_fit(log_curve, x, np.log(y), p0=[np.log(np.median(y)), np.log(.05)], bounds=([-20, -20], [20, 20]), maxfev=20000)
            models = {'linear': {'c': c, 'b': 1.0}, 'power': {'c': float(np.exp(a)), 'b': float(b)}, 'log': {'a': float(np.exp(pars[0])), 'b': float(np.exp(pars[1]))}}
            for name, params in models.items():
                for split, sample in [('calibration', calibration), ('expanded', holdout)]:
                    errors = []
                    passed = 0
                    for r in sample:
                        xx = r['n'] / r['sigma2']
                        predicted = params['a'] * np.log1p(params['b'] * xx) if name == 'log' else params['c'] * xx ** params['b']
                        factor = max(r['U_rise'] / predicted, predicted / r['U_rise']) if r['interior'] else None
                        r[f'{name}_prediction_U'] = float(predicted)
                        r[f'{name}_factor_error'] = factor
                        if factor is not None:
                            errors.append(factor)
                            passed += factor <= 2
                    params[split] = {'within2': int(passed), 'total': len(sample), 'interior': len(errors), 'pass_rate': passed / len(sample), 'median_factor_interior': float(np.median(errors)) if errors else None, 'max_factor_interior': float(max(errors)) if errors else None}
            eligible = len(valid) / len(calibration) >= .8
            records.append({'protocol': protocol, 'task': task, 'calibration_interior': len(valid), 'calibration_total': len(calibration), 'eligible': eligible, 'P1': 'supported' if eligible and models['linear']['expanded']['pass_rate'] >= .8 else 'refuted' if eligible else 'not_applicable', 'models': models})
    interiors = [r for r in rows if r['interior']]
    predicates = {'P2': [r['noise_at_half_over_at_rise'] >= .25 for r in interiors], 'P3': [r['expected_train_fraction_at_rise'] <= .1 for r in interiors], 'P4': [r['curvature_at_rise'] > 0 for r in interiors]}
    results = {k: {'passed': int(sum(v)), 'total': len(v), 'rate': sum(v) / len(v), 'status': 'supported' if sum(v) / len(v) >= .9 else 'refuted'} for k, v in predicates.items()}
    calibration_all = [r for r in rows if r['interior'] and r['n'] in SPEC['calibration']['n'] and r['sigma2'] in SPEC['calibration']['sigma2']]
    shared_c = float(np.exp(np.mean([np.log(r['U_rise'] * r['sigma2'] / r['n']) for r in calibration_all])))
    shared_expanded = [r for r in rows if not (r['n'] in SPEC['calibration']['n'] and r['sigma2'] in SPEC['calibration']['sigma2'])]
    shared_passed = sum(r['interior'] and .5 <= r['U_rise'] / (shared_c * r['n'] / r['sigma2']) <= 2 for r in shared_expanded)
    return {'study': SPEC['study'], 'domain': 'expanded validation now inspected; development thereafter', 'contract': contract, 'counts': {'curve_evaluations': len(rows), 'interior_minima': len(interiors), 'boundary_minima': len(rows) - len(interiors), 'task_recipes': len(records)}, 'legacy': legacy_summary(), 'tasks': records, 'shared_coefficient': {'c': shared_c, 'expanded_within2': int(shared_passed), 'expanded_total': len(shared_expanded)}, 'registered_predictions': results, 'train_at_rise_fraction_quantiles': np.quantile([r['expected_train_fraction_at_rise'] for r in interiors], [0, .1, .5, .9, 1]).tolist(), 'noise_before_rise_quantiles': np.quantile([r['noise_at_half_over_at_rise'] for r in interiors], [0, .1, .5, .9, 1]).tolist(), 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-draws', type=int)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    contract = source_contract()
    STUDY.mkdir(parents=True, exist_ok=True)
    results = STUDY / 'results'
    results.mkdir(exist_ok=True)
    manifest = STUDY / 'execution.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['contract'] != contract:
            raise RuntimeError('Frozen source contract changed')
    else:
        save_json(manifest, {'contract': contract, 'started_unix': time.time(), 'numpy': np.__version__, 'spec_sha256': sha(SPEC_PATH)})
        save_json(STUDY / 'preregistration.json', SPEC)
    grid = np.unique(np.r_[np.arange(257), np.rint(np.geomspace(257, SPEC['max_steps'], 4096)).astype(int)])
    started = time.monotonic()
    draws = 0
    for protocol, recipe in SPEC['protocols'].items():
        for n in SPEC['n']:
            for seed in SPEC['seeds']:
                draw_label = f'{protocol}_n{n}_seed{seed}'
                path = results / f'{draw_label}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    for rel, pin in saved['files'].items():
                        if sha(STUDY / rel) != pin:
                            raise RuntimeError(f'Changed saved success: {rel}')
                    continue
                if args.analyze_only:
                    raise RuntimeError(f'Missing draw: {draw_label}')
                if time.monotonic() - started > 1200:
                    raise RuntimeError('Compute budget reached; resume saved results')
                data = make_draw(protocol, n, seed)
                data_path = results / f'{draw_label}_data.npz'
                if data_path.exists():
                    raise RuntimeError('Partial draw requires review, refusing overwrite')
                np.savez_compressed(data_path, **data)
                model = spectral_model(data)
                rows, files = [], {str(data_path.relative_to(STUDY)): sha(data_path)}
                for task in recipe['tasks']:
                    terms = target_terms(model, data, task)
                    if draws == 0:
                        gate = independent_gate(model, data, terms)
                        save_json(STUDY / f'first_draw_gate_{task}.json', gate)
                    base = grid_components(model, data, terms, grid)
                    for sigma2 in SPEC['sigma2']:
                        label = f'{draw_label}_{task}_var{sigma2:g}'
                        row, arrays = measure(model, data, terms, grid, sigma2, base)
                        curve_path = results / f'{label}.npz'
                        if curve_path.exists():
                            raise RuntimeError('Partial curve requires review, refusing overwrite')
                        np.savez_compressed(curve_path, **arrays)
                        rows.append({**row, 'protocol': protocol, 'task': task, 'input_dim': recipe['input_dim'], 'output_dim': terms['y'].shape[1], 'n': n, 'sigma2': sigma2, 'seed': seed, 'curve': str(curve_path.relative_to(STUDY)), 'data': str(data_path.relative_to(STUDY))})
                        files[str(curve_path.relative_to(STUDY))] = sha(curve_path)
                save_json(path, {'draw': draw_label, 'contract': contract, 'rows': rows, 'files': files, 'completed_unix': time.time()})
                draws += 1
                print(json.dumps({'saved_draw': draw_label, 'curves': len(rows), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
                if args.max_draws and draws >= args.max_draws:
                    return
    rows = [r for p in sorted(results.glob('*.json')) for r in json.loads(p.read_text())['rows']]
    summary = fit_summary(rows, contract)
    save_json(STUDY / 'summary.json', summary)
    print(json.dumps({'summary': str(STUDY / 'summary.json'), 'counts': summary['counts'], 'compute_seconds_this_invocation': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
