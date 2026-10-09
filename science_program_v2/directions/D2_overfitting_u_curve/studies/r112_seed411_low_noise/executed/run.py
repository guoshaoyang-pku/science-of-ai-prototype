"""Zero-training reweighting of one saved n32/seed411 curve."""
import argparse, hashlib, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
LABEL = 'n32_seed411_var0.0625'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def verify_contract(commit):
    cfg = json.loads((STUDY / 'preregistration.json').read_text())
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for rel, expected in cfg['source_sha256'].items():
        path = STUDY / rel
        assert sha(path) == expected, rel
        assert subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT) == path.read_bytes()
    for rel, expected in cfg['input_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    return cfg

def verify_saved(cfg, commit):
    folder = STUDY / 'results'
    expected = {LABEL + '.npz', LABEL + '.json', 'receipt.json'}
    present = {p.name for p in folder.iterdir()} if folder.exists() else set()
    assert not present or present == expected, 'unexpected result files'
    if not present:
        return False
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['preregistration_commit'] == commit
    for name, meta in receipt['artifacts'].items():
        path = folder / name
        assert sha(path) == meta['sha256']
        assert path.stat().st_mtime_ns == meta['mtime_ns']
    row = json.loads((folder / (LABEL + '.json')).read_text())
    assert row['status'] == 'completed' and row['preregistration_commit'] == commit
    return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preregistration-commit', required=True); args = ap.parse_args()
    cfg = verify_contract(args.preregistration_commit)
    if verify_saved(cfg, args.preregistration_commit):
        print(json.dumps({'new_cells': 0, 'reused_cells': 1, 'training_cells': 0})); return
    started = datetime.now(timezone.utc).isoformat(); tic = time.perf_counter()
    src = ROOT / cfg['source_curve']
    with np.load(src, allow_pickle=False) as z:
        signal = z['signal_bias'].copy(); variance = z['variance_unit'].copy(); baseline = z['expected_risk'].copy()
    assert signal.shape == variance.shape == baseline.shape == (16385,)
    assert all(np.isfinite(x).all() for x in (signal, variance, baseline))
    assert np.max(np.abs(baseline - (signal + cfg['baseline_noise_variance'] * variance))) <= 1e-12
    risk = signal + cfg['cell']['noise_variance'] * variance
    assert np.isfinite(risk).all()
    out = STUDY / 'results'; out.mkdir(exist_ok=True)
    npz = out / (LABEL + '.npz')
    with npz.open('xb') as f:
        np.savez_compressed(f, steps=np.arange(16385), signal_bias=signal, variance_unit=variance, expected_risk=risk, baseline_risk=baseline)
    row = {'status':'completed','cell':cfg['cell'],'training_cells':0,'measurement':'saved_curve_reweighting','preregistration_commit':args.preregistration_commit,'source_curve':cfg['source_curve'],'source_curve_sha256':sha(src),'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-tic,'arrays_sha256':sha(npz),'numpy':np.__version__}
    save_json(out / (LABEL + '.json'), row)
    arts = {p.name:{'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns} for p in (npz, out/(LABEL+'.json'))}
    save_json(out / 'receipt.json', {'cell':cfg['cell'],'preregistration_commit':args.preregistration_commit,'artifacts':arts})
    verify_saved(cfg, args.preregistration_commit)
    print(json.dumps({'new_cells':1,'reused_cells':0,'training_cells':0,'seconds':row['seconds']}))

if __name__ == '__main__': main()
