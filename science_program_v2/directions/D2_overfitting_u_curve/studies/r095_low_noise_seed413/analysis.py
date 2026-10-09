"""从新保存的曲线复算最低点与预注册的操作性判据。"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('saved_curve_run', STUDY / 'executed/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def describe(risk, signal, variance):
    t = int(np.argmin(risk))
    delta = float(risk[-1] - risk[t])
    return {
        't_star': t, 'interior': 0 < t < 16384,
        'initial_risk': float(risk[0]), 'minimum_risk': float(risk[t]), 'final_risk': float(risk[-1]),
        'endpoint_minus_minimum': delta, 'margin_over_threshold': delta - .05,
        'operational_u_shape': bool(0 < t < 16384 and delta >= .05),
        'signal_at_minimum': float(signal[t]), 'variance_unit_at_minimum': float(variance[t]),
        'signal_at_endpoint': float(signal[-1]), 'variance_unit_at_endpoint': float(variance[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    args = parser.parse_args()
    config, pins = runner.verify_contract(args.preregistration_commit)
    assert runner.verify_saved(config, pins, args.preregistration_commit)
    path = STUDY / 'results' / (runner.LABEL + '.npz')
    with np.load(path, allow_pickle=False) as source:
        assert set(source.files) == {'steps', 'signal_bias', 'variance_unit', 'expected_risk', 'baseline_risk'}
        arrays = {k: source[k] for k in source.files}
    assert np.array_equal(arrays['steps'], np.arange(16385))
    assert all(np.isfinite(x).all() for x in arrays.values())
    signal, variance = arrays['signal_bias'], arrays['variance_unit']
    risk = signal + .0625 * variance
    baseline = signal + .125 * variance
    assert np.array_equal(risk, arrays['expected_risk'])
    assert np.array_equal(baseline, arrays['baseline_risk'])
    new = describe(risk, signal, variance)
    old = describe(baseline, signal, variance)
    prediction = config['predictions'][0]
    lo, hi = prediction['interval']
    passed = lo <= new['endpoint_minus_minimum'] <= hi and not new['operational_u_shape']
    summary = {
        'study': config['study'], 'round': 95, 'direction_round': 7, 'direction': config['direction'],
        'domain': 'development', 'status': 'completed', 'preregistration_commit': args.preregistration_commit,
        'sha256': pins, 'input_sha256': config['input_sha256'],
        'counts': {'new_recipes': 1, 'seeds': 1, 'new_evaluation_cells': 1, 'new_training_cells': 0, 'saved_controls': 1},
        'cell': config['cell'], 'new': new, 'baseline': {'noise_variance': .125, **old},
        'paired': {'t_star_ratio': new['t_star'] / old['t_star'],
                   'delta_change': new['endpoint_minus_minimum'] - old['endpoint_minus_minimum'],
                   'minimum_risk_change': new['minimum_risk'] - old['minimum_risk'],
                   'endpoint_risk_change': new['final_risk'] - old['final_risk']},
        'predictions': [{'id': 'P1', 'interval': prediction['interval'],
                         'observed': new['endpoint_minus_minimum'], 'status': 'supported' if passed else 'refuted'}],
        'risk_arrays_sha256': runner.sha(path), 'boundaries': config['boundaries'],
    }
    output = STUDY / 'summary.json'
    if output.exists():
        assert json.loads(output.read_text()) == summary, '拒绝改写已有summary'
    else:
        runner.save_json(output, summary)
    print(json.dumps({'prediction': summary['predictions'][0], 'new': new, 'baseline': old, 'paired': summary['paired']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
