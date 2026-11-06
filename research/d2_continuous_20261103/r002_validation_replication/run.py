"""Frozen-time validation replication; no optimization on new validation samples."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib, importlib.util, json, signal, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = json.loads((HERE/'preregistration.json').read_text())
BASE = Path(SPEC['output_root'])
OUT = BASE/SPEC['round_id']
PREV = BASE/SPEC['previous_round']
RAW = Path(SPEC['input_root'])

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def features(data, x):
    raw = np.maximum(x@data['weight'].T+data['hidden_bias'], 0)
    return np.column_stack(((raw-data['feature_center'])/data['feature_scale'], np.ones(len(x))))

def main():
    fd = os.open(BASE/'active.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, json.dumps({'pid':os.getpid(), 'round':SPEC['round_id']}).encode()); os.close(fd)
    created = False
    start = time.monotonic()
    def timeout(signum, frame): raise TimeoutError('30-minute computation limit')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC['compute_seconds'])
    try:
        if OUT.exists(): raise RuntimeError('Output already exists; never overwrite')
        OUT.mkdir(); created = True
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        deps = [HERE/'run.py', HERE/'preregistration.json',
                ROOT/'research/d2_continuous_20261103/r001_alpha_mix/run.py',
                ROOT/'research/d2_continuous_20261103/r001_alpha_mix/preregistration.json',
                ROOT/'research/d2_continuous_20261103/r001_alpha_mix/result_manifest.json',
                ROOT/'research/d2_scaling_20261009/run.py',
                ROOT/'research/d2_scaling_20261009/preregistration.json',
                ROOT/'research/d2_diagonal_20261027/numerics.py']
        pins = {}
        for path in deps:
            rel = str(path.relative_to(ROOT))
            if subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT) != path.read_bytes():
                raise RuntimeError('Uncommitted dependency '+rel)
            pins[rel] = sha(path)
        execution = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                         commit=commit, source_hashes=pins, spec=SPEC,
                         numpy_version=np.__version__)
        save(OUT/'execution.json', execution)
        previous_manifest = json.loads(deps[4].read_text())
        for item in previous_manifest['artifacts']:
            p = PREV/item['path']
            if sha(p) != item['sha256'] or p.stat().st_size != item['bytes']:
                raise RuntimeError('Previous artifact mismatch '+str(p))
        previous_execution = json.loads((PREV/'execution.json').read_text())
        for rel, digest in previous_execution['input_hashes'].items():
            if sha(RAW/rel) != digest: raise RuntimeError('Input mismatch '+rel)
        execution['input_hashes'] = previous_execution['input_hashes']
        execution['previous_manifest_hash'] = sha(deps[4]); save(OUT/'execution.json', execution)
        progress = json.loads((BASE/'progress.json').read_text())
        progress.update(status='running', active_round=SPEC['round_id']); save(BASE/'progress.json', progress)
        r1 = load('d2_round1', deps[2])
        frozen = [r for r in json.loads((PREV/'rows.json').read_text()) if r['sigma2']==0]
        assert len(frozen)==SPEC['conditions']
        eligible = [r for r in frozen if r['interior'] and r['late_within_window']]
        assert len(eligible)==SPEC['eligible']
        save(OUT/'frozen_conditions.json', frozen)
        weights, draws, checks, feature_errors = [], {}, [], []
        for row in eligible:
            key = (row['n'],row['seed'])
            if key not in draws:
                rel = f'evidence/d2_scaling_20261009/studies/n_sigma_tasks/results/d4_homogeneous_n{key[0]}_seed{key[1]}_data.npz'
                with np.load(RAW/rel) as z: data = {k:z[k] for k in z.files}
                model = r1.legacy.spectral_model(data)
                draws[key] = (data, model)
                for xkey, pkey in [('train_x','phi'),('val_x','val_phi')]:
                    error = float(np.max(np.abs(features(data,data[xkey])-data[pkey])))
                    feature_errors.append(error)
                    if error>1e-12: raise RuntimeError('Feature reconstruction failed')
            data, model = draws[key]
            terms = r1.terms_for(model,data,row['alpha'])
            f = r1.legacy.response(model,[row['t_star'],row['t_late']])[0]
            w = (model['vectors']@(f*terms['g'][:,0]).T).T
            risks = np.mean((data['val_phi']@w.T-terms['vy'])**2,axis=0)
            error = float(np.max(np.abs(risks-[row['R_star'],row['R_star']+row['delta_R']])))
            checks.append(error)
            if error>1e-8*max(1,row['R0']): raise RuntimeError('Original endpoint mismatch')
            weights.append(w)
        weights = np.asarray(weights)
        np.savez_compressed(OUT/'frozen_weights.npz',weights=weights)
        # Both times and weights are fixed and saved before any new validation input exists.
        rng = np.random.Generator(np.random.PCG64(SPEC['rng_seed']))
        inputs = rng.standard_normal((SPEC['batches'],SPEC['batch_rows'],SPEC['input_dimension']))
        np.save(OUT/'validation_inputs.npy', inputs)
        means = np.empty((len(eligible),SPEC['batches'],3))
        for batch, x in enumerate(inputs):
            for key, (data, _) in draws.items():
                indices = [i for i,r in enumerate(eligible) if (r['n'],r['seed'])==key]
                phi = features(data,x)
                for i in indices:
                    a = eligible[i]['alpha']
                    y = (x[:,0]+a*x[:,0]*x[:,1])/np.sqrt(1+a*a)
                    sq = (phi@weights[i].T-y[:,None])**2
                    means[i,batch] = [sq[:,0].mean(),sq[:,1].mean(),(sq[:,1]-sq[:,0]).mean()]
        if not np.isfinite(means).all(): raise RuntimeError('Nonfinite validation result')
        np.savez_compressed(OUT/'batch_means.npz', means=means,
                            columns=np.array(['early_mse','late_mse','paired_delta']))
        critical = float(student_t.ppf(1-SPEC['family_alpha']/(2*SPEC['eligible']),SPEC['batches']-1))
        results=[]; index=0
        for old in frozen:
            row = dict(n=old['n'],seed=old['seed'],alpha=old['alpha'],sigma2=0,
                       t_star=old['t_star'],t_late=old['t_late'],old_R0=old['R0'],
                       old_delta=old['delta_R'],train_delta=old['expected_noisy_train_delta'])
            if not old['interior'] or not old['late_within_window']:
                row.update(status='excluded_boundary',reason='t_star=T, 2T outside registered window')
            else:
                stats=means[index]; delta=float(stats[:,2].mean())
                se=float(stats[:,2].std(ddof=1)/np.sqrt(SPEC['batches']))
                lo,hi=delta-critical*se,delta+critical*se
                category='significant_positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved'
                row.update(status='evaluated',weight_index=index,early_mse=float(stats[:,0].mean()),
                           late_mse=float(stats[:,1].mean()),delta=delta,se=se,ci_lower=lo,ci_upper=hi,
                           delta_over_old_R0=delta/old['R0'],category=category,
                           effect_threshold=SPEC['effect_fraction']*old['R0'],
                           primary_pass=lo>SPEC['effect_fraction']*old['R0'])
                index+=1
            results.append(row)
        selected=[r for r in results if r['status']=='evaluated']
        def aggregate(rows):
            return dict(evaluated=len(rows),passed=sum(r['primary_pass'] for r in rows),
                        point_positive=sum(r['delta']>0 for r in rows),
                        categories={k:sum(r['category']==k for r in rows) for k in ['significant_positive','unresolved','nonpositive']})
        agg=aggregate(selected)
        summary=dict(question=SPEC['question'],conditions=len(results),excluded=len(results)-len(selected),
                     **agg,required_passes=SPEC['required_passes'],critical_value=critical,
                     hypothesis_status='supported_on_fixed_development_models' if agg['passed']>=SPEC['required_passes'] else 'not_supported',
                     by_alpha=[dict(alpha=a,**aggregate([r for r in selected if r['alpha']==a])) for a in sorted({r['alpha'] for r in results})],
                     by_seed=[dict(seed=s,**aggregate([r for r in selected if r['seed']==s])) for s in sorted({r['seed'] for r in results})])
        save(OUT/'rows.json',results);save(OUT/'summary.json',summary)
        save(OUT/'verification.json',dict(previous_artifacts_verified=len(previous_manifest['artifacts']),
            original_inputs_verified=len(previous_execution['input_hashes']),original_endpoint_checks=2*len(checks),
            max_endpoint_error=max(checks),feature_reconstruction_checks=len(feature_errors),max_feature_error=max(feature_errors),
            validation_input_sha256=sha(OUT/'validation_inputs.npy'),frozen_weights_sha256=sha(OUT/'frozen_weights.npz')))
        execution.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
        save(OUT/'execution.json',execution)
        save(OUT/'manifest.json',{'artifacts':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()]})
        progress.update(status='awaiting_interpretation',active_round=None)
        progress['rounds'].append({'id':SPEC['round_id'],'status':'computed','commit':commit})
        save(BASE/'progress.json',progress)
        print(json.dumps(summary,ensure_ascii=False))
    except Exception:
        if created:
            save(OUT/'failure.json',{'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start})
            progress=json.loads((BASE/'progress.json').read_text())
            progress.update(status='blocked_failure',active_round=None,failure_round=SPEC['round_id'])
            save(BASE/'progress.json',progress)
        raise
    finally:
        signal.alarm(0);(BASE/'active.lock').unlink(missing_ok=True)

if __name__=='__main__': main()
