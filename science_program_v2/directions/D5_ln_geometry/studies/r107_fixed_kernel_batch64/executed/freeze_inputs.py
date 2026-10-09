from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
OLD = STUDY.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(value):
    value = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    assert not (STUDY/'preregistration.json').exists(), 'freeze once only'
    snapshot = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    previous = json.loads((OLD/'r089_ln_affine_kernel/executed/final_commit_verification.json').read_text())
    for commit in (previous['closeout_commit'], previous['preregistration_commit']):
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT, check=True)
    for rel, digest in previous['closeout_artifact_sha256'].items():
        data = subprocess.check_output(['git', 'show', f'{previous["closeout_commit"]}:{rel}'], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == digest
    prior_pins = dict(previous['protected_file_pins'])
    prior_pins.update(previous['r089_result_pins'])
    for rel, item in prior_pins.items():
        path = ROOT/rel
        assert sha(path) == item['sha256'] and path.stat().st_mtime_ns == item['mtime_ns']
    files = {}
    for name in ('r013_gram_condition_review', 'r021_target_rayleigh', 'r045_mean_centered_coupling', 'r077_fixed_kernel_train_chord', 'r089_ln_affine_kernel'):
        for path in sorted((OLD/name).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            assert '.tmp' not in path.name, path
            rel = str(path.relative_to(ROOT))
            digest = sha(path)
            assert hashlib.sha256(subprocess.check_output(['git', 'show', f'{snapshot}:{rel}'], cwd=ROOT)).hexdigest() == digest, rel
            files[rel] = {'sha256': digest, 'mtime_ns': path.stat().st_mtime_ns}
    old_runner_path = OLD/'r013_gram_condition_review/executed/run.py'
    spec = importlib.util.spec_from_file_location('original_training_source', old_runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    generated, original_batches, verified = {}, {}, 0
    for path in sorted((OLD/'r013_gram_condition_review/results').glob('*.json')):
        row = json.loads(path.read_text())
        req = row['contract']
        assert row['status'] == 'completed' and sha(path.with_suffix('.npz')) == row['arrays_sha256']
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
            batches = data['batch_indices'].copy()
            dim = data['train_x'].shape[1]
        key = (dim, req['label'], req['seed'])
        if key not in generated:
            runner.torch.manual_seed(req['seed'])
            model = runner.Model(dim, req['recipe'])
            initial = {name: pin(parameter.detach().numpy()) for name, parameter in model.named_parameters() if '.norm.' not in name}
            reconstructed = np.stack([runner.torch.randint(0,256,(64,)).numpy() for _ in range(256)])
            generated[key] = (initial, reconstructed)
        initial, reconstructed = generated[key]
        assert row['initial_linear'] == initial
        assert np.array_equal(batches, reconstructed) and row['batch_sha256'] == pin(batches)['sha256']
        paired = (req['function'], req['seed'])
        if paired in original_batches:
            assert np.array_equal(original_batches[paired], batches)
        original_batches[paired] = batches
        verified += 1
    assert verified == 120
    cells = {}
    for function in ('trigonometric8', 'quadratic8', 'interaction12', 'radial12'):
        for label in ('LN010_w64', 'noLN_w64'):
            for seed in (100,101,102,103,104):
                kernel_study = 'r089_ln_affine_kernel' if label == 'LN010_w64' else 'r077_fixed_kernel_train_chord'
                path = OLD/kernel_study/'results'/f'{function}_{label}_{seed}.json'
                row = json.loads(path.read_text())
                batch_path = OLD/'r013_gram_condition_review/results'/f'{function}_{label}_0_{seed}.json'
                with np.load(batch_path.with_suffix('.npz'), allow_pickle=False) as data:
                    batches = data['batch_indices'].copy()
                kernel_name = 'total_kernel' if label == 'LN010_w64' else 'kernel'
                vector_name = 'affine_total_propagated_ones_256' if label == 'LN010_w64' else 'propagated_ones_256'
                with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
                    kernel_pin = pin(data[kernel_name])
                    full_pin = pin(data[vector_name])
                    train_pins = {key: pin(data[key]) for key in data.files if key.startswith('train_')}
                assert kernel_pin == row['total_kernel_pin' if label == 'LN010_w64' else 'kernel_pin']
                cells[f'{function}_{label}_{seed}'] = {'kernel_result_json': str(path.relative_to(ROOT)),
                    'kernel_array': kernel_name, 'kernel_pin': kernel_pin, 'full_vector_array': vector_name,
                    'full_vector_pin': full_pin, 'batch_result_json': str(batch_path.relative_to(ROOT)),
                    'batch_pin': pin(batches), 'train_array_pins': train_pins}
    batch_order = {'type': 'saved_actual_training_order', 'source_algorithm': 'torch.manual_seed(seed); original Model construction; 256 torch.randint(0,256,(64,)) calls',
                   'sampling': 'with replacement; repeated indices count individually', 'verified_training_cells': verified,
                   'unique_sequences': len({pin(value)['sha256'] for value in original_batches.values()}),
                   'LN_noLN_and_offsets_same_sequence': True, 'torch': runner.torch.__version__,
                   'training_source_sha256': sha(old_runner_path), 'training_preregistration_commit': '5144cbdf818c7ac3dca76d21fbb85bb5a3a4d036'}
    manifest = {'snapshot_commit': snapshot, 'files': files, 'cells': cells, 'batch_order': batch_order}
    write(STUDY/'executed/input_manifest.json', manifest)
    audit = {'status': 'passed', 'epoch': time.time(), 'snapshot_commit': snapshot,
             'prior_closeout_commit': previous['closeout_commit'], 'prior_preregistration_commit': previous['preregistration_commit'],
             'prior_closeout_artifacts_verified': len(previous['closeout_artifact_sha256']),
             'prior_protected_hash_mtime_verified': len(prior_pins), 'pinned_historical_files': len(files),
             'r089_frozen_verifier_defect_preserved': "verify.py reads missing NPZ kernel_pin; independent verification used instead",
             'batch_order': batch_order, 'new_training_cells': 0, 'new_kernel_cells': 0, 'new_propagations': 0}
    write(STUDY/'executed/precommit_audit.json', audit)
    old_summary = json.loads((OLD/'r089_ln_affine_kernel/summary.json').read_text())
    numerical = [{'function': row['function'], 'seed': row['seed'], 'point': row['fixed_delta_chord_256'],
                  'interval': [row['fixed_delta_chord_256']-.005, row['fixed_delta_chord_256']+.005]}
                 for row in old_summary['pairs']]
    sources = ('executed/run.py', 'analysis.py', 'executed/verify.py', 'executed/freeze_inputs.py')
    cfg = {'study': STUDY.name, 'round': 107, 'direction_round': 7, 'domain': 'development',
           'question': '保持共享Linear+LN affine初始化核，仅换成保存的batch64顺序递推，eta=.001/a=3/T=256的train方向匹配是否改善？',
           'conditions': {'functions': ['trigonometric8','quadratic8','interaction12','radial12'], 'seeds': [100,101,102,103,104],
                          'n':256, 'batch_size':64, 'steps':256, 'eta':.001, 'offset_amplitude':3,
                          'width':64, 'head':'fixed', 'activation':'GELU', 'momentum':0, 'linearized_weight_decay':0,
                          'real_training_weight_decay':1e-4, 'data_seed':48291,
                          'kernel_LN':'r089 total_kernel', 'kernel_noLN':'r077 kernel',
                          'new_training_cells':0, 'new_kernel_cells':0, 'evaluation_cells':40, 'pairs':20},
           'batch_order': batch_order,
           'formulation': {'kernel':'K=J J^T/n, from immutable saved source',
                           'residual_update':'r[t+1]=r[t]-(2*eta*n/b)*K[:,B_t] r[t][B_t]; B_t is saved ordered multiset',
                           'offset_vector':'v[0]=ones(n); v[t+1]=v[t]-.008*K[:,B_t]v[t][B_t]',
                           'chord':'C_batch=9*mean((v[256]-mean(v[256]))^2); same B across -3/0/+3, ideal exact additive offset',
                           'delta':'LN-noLN; real train chord uses immutable saved predictions/labels',
                           'step_unit':'one minibatch update, 256 updates, not 256 epochs',
                           'zero_sign_threshold':1e-12},
           'predictions': {'P1': {'criterion':'M_batch64 == 0/20, equal to saved fullbatch M=0, no direction improvement', 'point_matches':0},
                           'P2': {'criterion':'all 20 abs(delta_batch64-delta_full) <= .005 label_squared', 'max_absolute_shift':.005,
                                  'per_pair_numerical_predictions':numerical}},
           'prediction_provenance': 'development非盲：已知真实train20负/固定全批量20正与小eta；预测batch差与旧full差距离<=.005是未检验猜测，不由计算新递推选出，不称blind或sealed OOD。',
           'analysis': 'same-seed paired values and df4 t95 per function; no iid function inference; no fit/tuning/alternate batches',
           'verification': {'step_error_max':1e-12,'scalar_chord_error_max':1e-12,'scalar_real_chord_error_max':1e-12,
                            'independent_method':'bincount multiplicities followed by sum over all 256 columns',
                            'first_cell_then_complete':True,'recovery_no_overwrite':True,'max_seconds':1200},
           'inputs': {'snapshot_commit':snapshot,'previous_preregistration_commit':previous['preregistration_commit'],
                      'previous_summary_sha256':sha(OLD/'r089_ln_affine_kernel/summary.json')},
           'manifest_sha256':sha(STUDY/'executed/input_manifest.json'),
           'source_sha256':{rel:sha(STUDY/rel) for rel in sources},
           'boundary':'只区分固定核batch算子；同保存实际训练顺序不等于复现训练动力学；不识别核漂移、decay、非线性、test、因果、sealed OOD或跨函数概率。'}
    write(STUDY/'preregistration.json', cfg)
    for rel in sources:
        compile((STUDY/rel).read_text(), str(STUDY/rel), 'exec')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
