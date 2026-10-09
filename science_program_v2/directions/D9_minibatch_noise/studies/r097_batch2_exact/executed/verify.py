import json
from run import ROOT, STUDY, cell_label, contract, save, scan_results
import numpy as np


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


def main():
    cfg, pins = contract()
    receipt = scan_results(cfg, pins)
    if len(receipt['cells']) != len(cfg['cells']):
        raise RuntimeError('Incomplete study')
    rows = []
    for condition in cfg['cells']:
        seed = condition['seed']
        path = STUDY / 'results' / (cell_label(seed) + '.npz')
        with np.load(ROOT / condition['data']) as d:
            x, xa = d['train_features'], d['audit_features']
            labels, audit_y = d['train_y'] + d['epsilon'], d['audit_y']
        independent_risk, mean_head = sample_coordinate_moments(x, xa, labels, audit_y, cfg['eta'], cfg['batch'], cfg['steps'])
        with np.load(path) as z:
            main_head = np.einsum('ij,j->i', z['basis'], z['mean_coordinates'][-1])
            curve_error = float(np.max(np.abs(independent_risk - z['expected_risk'])))
            head_error = float(np.max(np.abs(mean_head-main_head)))
            match = int(np.argmin(independent_risk)) == int(np.argmin(z['expected_risk']))
            span_error = float(np.max(np.abs(x-multiply(multiply(x,z['basis']),z['basis'].T))))
            rows.append({'seed': seed, 'independent_raw_second_moment_risk_maxabs': curve_error,
                         'independent_mean_head_maxabs': head_error, 'span_reconstruction_maxabs': span_error,
                         'independent_argmin_matches': match, 'independent_argmin': int(np.argmin(independent_risk)),
                         'finite': bool(np.isfinite(independent_risk).all() and np.isfinite(mean_head).all()),
                         'passed': bool(curve_error <= 2e-8 and head_error <= 1e-9 and match
                                        and np.isfinite(independent_risk).all())})
    result = {'passed': bool(all(row['passed'] for row in rows)), 'cells': rows, 'pins': pins,
              'old_files_unchanged': True,
              'independent_method': '独立64维样本坐标w=X^T alpha与raw二阶矩；单/双inclusion概率，不调用主矩递推或basis。',
              'full_and_fixture_note': '复用已保存B=n退化/旧枚举证据，不重算旧成功cell。实现差不作严格数值误差证书。'}
    save(STUDY / 'executed/verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise RuntimeError('Independent verification failed')


if __name__ == '__main__':
    main()
