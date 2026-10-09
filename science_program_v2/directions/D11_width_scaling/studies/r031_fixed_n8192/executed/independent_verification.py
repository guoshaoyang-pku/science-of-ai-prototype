#!/usr/bin/env python3
"""Read-only scalar reconstruction of saved SGD evidence, without training."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

config = json.loads((STUDY / 'preregistration.json').read_text())
receipt = json.loads((STUDY / 'results/receipt.json').read_text())
commit = receipt['pins']['git_commit']
source_matches = {}
for rel in receipt['pins']['sha256']:
    path = STUDY / rel
    source_matches[rel] = subprocess.check_output(['git','show',f'{commit}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT) == path.read_bytes()
n = config['sample_count']
old_study = ROOT / config['baseline_study']
manifest = json.loads((STUDY / 'executed/baseline_manifest.json').read_text())
baseline_checks = {rel: sha(ROOT / rel) == value['sha256'] and (ROOT / rel).stat().st_mtime_ns == value['mtime_ns'] for rel, value in manifest.items()}
rows = []
for name, digest in receipt['cells'].items():
    meta_path = STUDY / 'results' / (name + '.json')
    arrays_path = meta_path.with_suffix('.npz')
    meta = json.loads(meta_path.read_text())
    with np.load(arrays_path) as archive:
        a = {key: archive[key] for key in archive.files}
    d = meta['cell']['dimension']
    threshold_hits = [i for i, loss in enumerate(a['loss']) if float(loss) <= float(a['loss'][0]) * .01]
    measured = threshold_hits[0] if threshold_hits else None
    checkpoint_errors, update_errors, pairs_checked = [], [], 0
    steps = a['checkpoint_steps'].tolist()
    for step, theta in zip(steps, a['theta_checkpoints']):
        residual = np.zeros(n)
        residual[:d] = a['feature_scale'] * a['signs'] * theta[a['permutation']] - a['target']
        scalar_loss = sum(float(r) * float(r) for r in residual) / (2 * n)
        checkpoint_errors.append(abs(scalar_loss - float(a['loss'][step])))
    for index in range(len(steps) - 1):
        if steps[index+1] != steps[index]+1:
            continue
        theta = a['theta_checkpoints'][index]
        sample_residual = np.zeros(n)
        sample_residual[:d] = a['feature_scale'] * a['signs'] * theta[a['permutation']] - a['target']
        gradient = np.zeros(d)
        np.add.at(gradient, a['permutation'], a['feature_scale'] * a['signs'] * sample_residual[:d] / n)
        expected = theta - .5 * gradient
        update_errors.append(float(np.max(np.abs(expected - a['theta_checkpoints'][index+1]))))
        pairs_checked += 1
    rayleigh = sum(float(w) * float(lam) for w, lam in zip(a['modal_weights'], a['eigenvalues']))
    gram_error = max(abs(float(x*x/n) - float(lam)) for x, lam in zip(a['feature_scale'], a['eigenvalues']))
    with np.load(old_study / 'results' / (name + '.npz')) as archive:
        b = {key: archive[key] for key in archive.files}
        old_hits = [i for i, loss in enumerate(b['loss']) if float(loss) <= float(b['loss'][0]) * .01]
        old_step = old_hits[0] if old_hits else None
        common_length = min(len(a['loss']), len(b['loss']))
        curve_difference = float(np.max(np.abs(a['loss'][:common_length] / a['loss'][0] - b['loss'][:common_length] / b['loss'][0])))
        old_checkpoints = dict(zip(b['checkpoint_steps'].tolist(), b['theta_checkpoints']))
        theta_difference = max(float(np.max(np.abs(theta - old_checkpoints[t]))) for t, theta in zip(steps, a['theta_checkpoints']) if t in old_checkpoints)
    rows.append({'name':name,'dimension':d,'seed':meta['cell']['seed'],'target':meta['cell']['target'],
        'old_step':old_step,'budget_difference':None if old_step is None or measured is None else measured-old_step,
        'paired_curve_difference':curve_difference,'paired_theta_difference':theta_difference,
        'hashes_pass':sha(meta_path)==digest and sha(arrays_path)==meta['arrays_sha256'],
        'step':measured,'metadata_step_pass':measured==meta['values']['measured_step'],
        'rayleigh':rayleigh,'initial_loss':float(a['loss'][0]),'gram_error':gram_error,
        'checkpoint_loss_error':max(checkpoint_errors),'update_error':max(update_errors),
        'adjacent_updates_checked':pairs_checked})
fits = {}
for target in config['targets']:
    subset = sorted((r for r in rows if r['target']==target and r['seed']==0),key=lambda r:r['dimension'])
    logx = np.log([r['dimension'] for r in subset]); logy = np.log([r['step'] for r in subset])
    beta = float(np.sum((logx-logx.mean())*(logy-logy.mean())) / np.sum((logx-logx.mean())**2))
    intercept = float(logy.mean()-beta*logx.mean())
    fits[target] = {'coefficient':float(np.exp(intercept)),'exponent':beta,
        'maximum_relative_error':float(np.max(np.abs(np.exp(intercept+beta*logx-logy)-1)))}
lookup = {(r['dimension'],r['seed'],r['target']):r for r in rows}
pairs = []
for d in config['dimensions']:
    for seed in config['coordinate_seeds']:
        m,e = lookup[d,seed,'middle'], lookup[d,seed,'endpoints']
        pairs.append({'dimension':d,'seed':seed,'budget_ratio':e['step']/m['step'],
            'rayleigh_difference':e['rayleigh']-m['rayleigh'],'initial_loss_difference':e['initial_loss']-m['initial_loss']})
passed = all(baseline_checks.values()) and all(source_matches.values()) and len(rows)==66 and all(
    r['hashes_pass'] and r['metadata_step_pass'] and r['budget_difference']==0 and r['paired_curve_difference']<1e-10 and r['paired_theta_difference']<1e-7 and r['gram_error']<1e-10 and
    r['checkpoint_loss_error']<1e-10 and r['update_error']<1e-10 and r['adjacent_updates_checked']>=3
    for r in rows)
summary = {'passed':passed,'preregistration_commit':commit,'source_matches_commit':source_matches,
    'baseline_hash_mtime_checks':baseline_checks,
    'max_paired_theta_difference':max(r['paired_theta_difference'] for r in rows),
    'max_paired_curve_difference':max(r['paired_curve_difference'] for r in rows),
    'cells':rows,'pairs':pairs,'power_fits':fits,
    'max_checkpoint_loss_error':max(r['checkpoint_loss_error'] for r in rows),
    'max_update_error':max(r['update_error'] for r in rows),
    'checkpoint_rows':sum(len(np.load(STUDY/'results'/(r['name']+'.npz'))['checkpoint_steps']) for r in rows),
    'scope':'仅重新读取保存数组，不运行训练；sample-space标量loss与SGD更新重建不同于原parameter-space实现；固定n=8192有n−d零残差行，signed permutation seed只是坐标检查；非零样本仍d，不是新增独立数据。'}
(STUDY/'executed/independent_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':passed,'cells':len(rows),'power_fits':fits,
    'max_checkpoint_loss_error':summary['max_checkpoint_loss_error'],'max_update_error':summary['max_update_error']},ensure_ascii=False))
if not passed:
    raise SystemExit(1)
