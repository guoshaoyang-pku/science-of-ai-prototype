import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
OLD = ROOT.parent / 'r001_chain_tc'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generate(seed):
    target, _ = np.linalg.qr(np.random.default_rng(20261007).normal(size=(4, 4)))
    x = np.random.default_rng(20261008).normal(size=(128, 4))
    y = x @ target.T
    rng = np.random.default_rng(seed)
    shapes = [(4, 32), (32, 32), (32, 32), (32, 4)]
    layers = [rng.normal(0, .8 / np.sqrt(fin), size=(fin, fout)) for fin, fout in shapes]
    return x, y, target, layers


def loss_and_grad(x, y, layers):
    acts = [x]
    for weight in layers:
        acts.append(acts[-1] @ weight)
    err = acts[-1] - y
    loss = float(np.mean(err * err))
    g = 2 * err / len(x)
    grads = [None] * len(layers)
    for j in range(len(layers) - 1, -1, -1):
        grads[j] = acts[j].T @ g
        g = g @ layers[j].T
    return loss, grads


def frozen_contract(freeze):
    contract_path = ROOT / 'preregistration.json'
    contract = json.loads(contract_path.read_text())
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', freeze, 'HEAD'], cwd=REPO)
    rel = str(contract_path.relative_to(REPO))
    additions = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%H', '--', rel], cwd=REPO).decode().splitlines()
    assert additions == [freeze], 'freeze must be the unique commit introducing this contract'
    frozen = subprocess.check_output(['git', 'show', f'{freeze}:{rel}'], cwd=REPO)
    assert frozen == contract_path.read_bytes(), 'contract differs from freeze'
    for rel, expected in contract['source_sha256'].items():
        path = ROOT / rel
        assert sha(path) == expected, f'source hash mismatch: {rel}'
        tracked = subprocess.check_output(['git', 'show', f'{freeze}:{path.relative_to(REPO)}'], cwd=REPO)
        assert tracked == path.read_bytes(), f'source differs from freeze: {rel}'
    old_files = {str(p.relative_to(REPO)) for p in ROOT.parent.rglob('*')
                 if p.is_file() and ROOT not in p.parents}
    assert old_files == set(contract['old_input_pins']), 'historical file set changed'
    state = json.loads((REPO / 'central/state.json').read_text())
    assert state['round'] == contract['state_guard']['round']
    assert {k: v['rounds_done'] for k, v in state['directions'].items()} == contract['state_guard']['directions_rounds_done']
    for rel, pin in contract['old_input_pins'].items():
        path = REPO / rel
        assert sha(path) == pin['sha256'], f'old input changed: {rel}'
        assert path.stat().st_mtime_ns == pin['mtime_ns'], f'old input mtime changed: {rel}'
    return contract, sha(contract_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    contract, contract_sha = frozen_contract(args.freeze)
    result_dir = ROOT / 'results'
    expected = {f'{c["id"]}.{ext}' for c in contract['cells'] for ext in ('npz', 'json')}
    assert {p.name for p in result_dir.iterdir()} <= expected, 'unknown result file'
    expected_fits = {c['id'] + '.json' for c in contract['cells']}
    assert {p.name for p in (ROOT / 'fits').iterdir()} <= expected_fits, 'unknown fit file'
    for p in (ROOT / 'fits').iterdir():
        assert (result_dir / p.name).exists() and (result_dir / p.with_suffix('.npz').name).exists(), 'orphan fit'
    summary_path = ROOT / 'summary.json'
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        assert saved['freeze_commit'] == args.freeze and saved['preregistration_sha256'] == contract_sha
        assert saved['fit_sha256'] == sha(ROOT / 'fits' / (contract['cells'][0]['id'] + '.json'))
    frozen_time = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', args.freeze], cwd=REPO))
    counts = {'new_cells': 0, 'reused_cells': 0}
    for cell in contract['cells']:
        out = result_dir / (cell['id'] + '.npz')
        receipt_path = out.with_suffix('.json')
        assert out.exists() == receipt_path.exists(), 'orphan result or receipt'
        if out.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt['npz_sha256'] == sha(out)
            assert receipt['contract_sha256'] == contract_sha
            assert receipt['freeze_commit'] == args.freeze
            assert receipt['cell'] == cell
            counts['reused_cells'] += 1
            continue
        x, y, target, layers = generate(cell['seed'])
        initial = [a.copy() for a in layers]
        spec = contract['generated_hashes'][str(cell['seed'])]
        assert [array_sha(a) for a in layers] == spec['layers']
        assert {k: array_sha(v) for k, v in [('x', x), ('y', y), ('target', target)]} == spec['data']
        with np.load(REPO / contract['baseline_npz']) as saved_initial:
            for key, value in [('x', x), ('y', y), ('target', target)]:
                assert np.array_equal(value, saved_initial[key])
            for j, value in enumerate(layers):
                assert np.array_equal(value, saved_initial[f'initial_{j}'])
        baseline = np.load(OLD / 'results' / f'L4_s{cell["seed"]}.npz')['loss']
        initial_loss, _ = loss_and_grad(x, y, layers)
        assert abs(initial_loss - baseline[0]) <= 1e-14
        started = time.time()
        assert started > frozen_time
        steps, eta = cell['steps'], cell['eta']
        losses = np.empty(steps + 1)
        checkpoints = sorted(set([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, steps // 2, steps]))
        snapshots = {j: [] for j in range(4)}
        for step in range(steps + 1):
            loss, grads = loss_and_grad(x, y, layers)
            assert np.isfinite(loss)
            losses[step] = loss
            if step in checkpoints:
                for j, weight in enumerate(layers):
                    snapshots[j].append(weight.copy())
            if step != steps:
                for weight, gradient in zip(layers, grads):
                    weight -= eta * gradient
        payload = {'loss': losses, 'checkpoint_steps': np.array(checkpoints), 'x': x, 'y': y, 'target': target}
        for j in range(4):
            payload[f'initial_{j}'] = initial[j]
            payload[f'weights_{j}'] = np.stack(snapshots[j])
        with out.open('xb') as stream:
            np.savez_compressed(stream, **payload)
        receipt = {'cell': cell, 'contract_sha256': contract_sha, 'freeze_commit': args.freeze,
                   'source_sha256': contract['source_sha256'], 'npz_sha256': sha(out),
                   'started_unix': started, 'finished_unix': time.time(), 'elapsed_sec': time.time() - started,
                   'initial_layer_hashes': spec['layers'], 'initial_loss': losses[0], 'final_loss': losses[-1]}
        with receipt_path.open('x') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        counts['new_cells'] += 1
        print(cell['id'], f'{receipt["elapsed_sec"]:.3f}s', flush=True)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
