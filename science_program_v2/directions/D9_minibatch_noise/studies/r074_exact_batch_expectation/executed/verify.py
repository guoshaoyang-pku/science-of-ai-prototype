import itertools
import json
import numpy as np
from run import ROOT, STUDY, contract, save, sha


def multiply(left, right):
    return np.einsum('ij,jk->ik', left, right, optimize=False)


def sample_coordinate_moments(x, xa, labels, audit_y, eta, batch, steps):
    n = len(x)
    rate = 2 * eta
    k = multiply(x, x.T)
    audit = multiply(xa, x.T)
    h_audit = multiply(audit.T, audit) / len(audit)
    l_audit = np.einsum('ij,i->j', audit, audit_y) / len(audit)
    constant = np.mean(audit_y ** 2)
    mean = np.zeros(n)
    second = np.zeros((n, n))
    risks = np.empty(steps + 1)
    p = batch / n
    p_pair = batch * (batch - 1) / (n * (n - 1))
    for t in range(steps + 1):
        risks[t] = constant - 2 * np.dot(l_audit, mean) + np.einsum('ij,ji->', h_audit, second)
        if t == steps:
            break
        k_mean = np.einsum('ij,j->i', k, mean)
        residual_second = multiply(multiply(k, second), k)
        residual_second -= np.outer(k_mean, labels) + np.outer(labels, k_mean)
        residual_second += np.outer(labels, labels)
        selected_second = p_pair * residual_second
        selected_second += (p - p_pair) * np.diag(np.diag(residual_second))
        second = second - rate / n * (multiply(k, second) + multiply(second, k))
        second += rate / n * (np.outer(mean, labels) + np.outer(labels, mean))
        second += rate ** 2 / batch ** 2 * selected_second
        second = (second + second.T) / 2
        mean = mean - rate / n * (k_mean - labels)
    head = np.einsum('ij,i->j', x, mean)
    return risks, head


def enumerated_fixture():
    q = np.array([[.2, -.1], [.4, .3], [-.2, .5], [.1, -.4]])
    labels = np.array([.7, -.2, .1, .3])
    initial = np.array([[.2, -.1], [-.3, .4]])
    weights = np.array([.4, .6])
    n, batch, rate = 4, 2, .2
    mean = np.einsum('i,ij->j', weights, initial)
    centered = initial - mean
    covariance = np.einsum('i,ij,ik->jk', weights, centered, centered)
    h = multiply(q.T, q) / n
    b = np.einsum('ij,i->j', q, labels) / n
    a = np.eye(2) - rate * h
    residual = np.einsum('ij,j->i', q, mean) - labels
    gradient = np.einsum('ij,j->i', h, mean) - b
    diagonal = np.einsum('ij,jk,ik->i', q, covariance, q) + residual ** 2
    term = multiply(q.T * diagonal, q) / n
    term -= multiply(multiply(h, covariance), h) + np.outer(gradient, gradient)
    predicted_mean = np.einsum('ij,j->i', a, mean) + rate * b
    predicted_covariance = multiply(multiply(a, covariance), a.T) + rate ** 2 * (n-batch)/(batch*(n-1)) * term
    outcomes, probabilities = [], []
    for w, probability in zip(initial, weights):
        for selected in itertools.combinations(range(n), batch):
            indices = list(selected)
            residual_batch = np.einsum('ij,j->i', q[indices], w) - labels[indices]
            updated = w - rate / batch * np.einsum('ij,i->j', q[indices], residual_batch)
            outcomes.append(updated)
            probabilities.append(probability / 6)
    outcomes, probabilities = np.array(outcomes), np.array(probabilities)
    exact_mean = np.einsum('i,ij->j', probabilities, outcomes)
    centered = outcomes - exact_mean
    exact_covariance = np.einsum('i,ij,ik->jk', probabilities, centered, centered)
    return {'batch_choices': 6, 'initial_support': 2, 'mean_maxabs_error': float(np.max(np.abs(exact_mean-predicted_mean))),
            'covariance_maxabs_error': float(np.max(np.abs(exact_covariance-predicted_covariance)))}


def main():
    cfg, pins = contract()
    receipt = json.loads((STUDY / 'results/receipt.json').read_text())
    rows = []
    for condition in cfg['cells']:
        seed = condition['seed']
        path = STUDY / 'results' / f'n64_seed{seed}_eta0.1_B8_var1_exact.json'
        metadata = json.loads(path.read_text())
        if sha(path) != receipt['cells'][path.stem] or sha(path.with_suffix('.npz')) != metadata['arrays_sha256']:
            raise RuntimeError('Receipt mismatch')
        with np.load(ROOT / condition['data']) as d:
            x, xa = d['train_features'], d['audit_features']
            labels, audit_y = d['train_y'] + d['epsilon'], d['audit_y']
        independent_risk, mean_head = sample_coordinate_moments(x, xa, labels, audit_y, cfg['eta'], cfg['batch'], cfg['steps'])
        with np.load(path.with_suffix('.npz')) as z:
            main_head = np.einsum('ij,j->i', z['basis'], z['mean_coordinates'][-1])
            curve_error = float(np.max(np.abs(independent_risk - z['expected_risk'])))
            head_error = float(np.max(np.abs(mean_head-main_head)))
            with np.load(ROOT / condition['full']) as full:
                full_error = float(np.max(np.abs(z['full_risk']-full['mean_risk'])))
            span_error = float(np.max(np.abs(x-multiply(multiply(x,z['basis']),z['basis'].T))))
            covariance_zero = float(z['full_covariance_maxabs']) == 0
            ts_match = int(np.argmin(independent_risk)) == int(np.argmin(z['expected_risk']))
            rows.append({'seed': seed, 'independent_raw_second_moment_risk_maxabs': curve_error,
                         'independent_mean_head_maxabs': head_error, 'B_equals_n_saved_full_maxabs': full_error,
                         'span_reconstruction_maxabs': span_error, 'B_equals_n_covariance_zero': covariance_zero,
                         'independent_argmin_matches': ts_match, 'independent_argmin': int(np.argmin(independent_risk)),
                         'passed': bool(curve_error <= 2e-8 and head_error <= 1e-9 and full_error <= 2e-8 and covariance_zero and ts_match)})
    fixture = enumerated_fixture()
    audit = json.loads((STUDY / 'executed/input_audit.json').read_text())
    unchanged = all(sha(ROOT / p) == value['sha256'] and (ROOT / p).stat().st_mtime_ns == value['mtime_ns']
                    for p, value in audit['old_files'].items())
    passed = unchanged and all(row['passed'] for row in rows) and max(fixture['mean_maxabs_error'], fixture['covariance_maxabs_error']) <= 1e-12
    result = {'passed': bool(passed), 'cells': rows, 'enumerated_fixture': fixture,
              'old_files_unchanged': unchanged, 'old_file_count': len(audit['old_files']), 'pins': pins,
              'independent_method': '64维sample coordinates w=X^T alpha；raw second moment、单/双 inclusion概率，未调用主递推或其basis。'}
    save(STUDY / 'executed/verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError('Independent verification failed')


if __name__ == '__main__':
    main()
