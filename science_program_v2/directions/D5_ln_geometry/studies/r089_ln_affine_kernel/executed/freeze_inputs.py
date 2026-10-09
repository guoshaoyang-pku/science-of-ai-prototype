from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent / 'r077_fixed_kernel_train_chord'
CLOSEOUT = 'f0c2e72824e34f3d3866f43b759dabedd615b119'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}

def main():
    subprocess.run(['git', 'merge-base', '--is-ancestor', CLOSEOUT, 'HEAD'], cwd=ROOT, check=True)
    verification = json.loads((OLD/'executed/final_commit_verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['closeout_commit'] == CLOSEOUT
    assert verification['saved_measurement_cells'] == 40
    assert verification['new_training_cells'] == 0
    fixed = [
        'preregistration.json', 'summary.json', 'analysis.py',
        'executed/run.py', 'executed/verify.py', 'executed/receipt.json',
        'executed/finalization.json', 'executed/final_commit_verification.json',
    ]
    files = {}
    for relative in fixed:
        path = OLD/relative
        files[str(path.relative_to(ROOT))] = {
            'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
    arrays = {}
    for path in sorted((OLD/'results').glob('*.npz')):
        row_path = path.with_suffix('.json')
        row = json.loads(row_path.read_text())
        assert row['status'] == 'completed'
        files[str(path.relative_to(ROOT))] = {
            'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
        files[str(row_path.relative_to(ROOT))] = {
            'sha256': sha(row_path), 'mtime_ns': row_path.stat().st_mtime_ns}
        import numpy as np
        with np.load(path) as data:
            arrays[path.stem] = {name: pin(data[name]) for name in
                ('train_x', 'train_y', 'train_predictions_0', 'train_predictions_256__-3',
                 'train_predictions_256__0', 'train_predictions_256__3', 'kernel')}
    assert len(arrays) == 40 and len(files) == 88
    manifest = {
        'snapshot_commit': CLOSEOUT,
        'source_study': 'directions/D5_ln_geometry/studies/r077_fixed_kernel_train_chord',
        'verified_closeout_artifacts': verification['verified_closeout_artifacts'],
        'files': files, 'arrays': arrays,
        'array_contract': 'SHA-256 is over contiguous saved ndarray bytes; shape and dtype are pinned.',
    }
    out = STUDY/'executed/input_manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'files': len(files), 'cells': len(arrays), 'manifest_sha256': sha(out)}))

if __name__ == '__main__':
    main()
