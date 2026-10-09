from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.array(values, dtype=np.float64)
    mean = float(values.mean())
    quantile = np.sqrt(2) * .95 / np.sqrt(1 - .95**2)
    radius = float(quantile * values.std(ddof=1) / np.sqrt(3))
    return {'mean': mean, 'seed_95pct_t_interval': [mean-radius, mean+radius]}


def main():
    config = json.loads((STUDY/'preregistration.json').read_text())
    summary = json.loads((STUDY/'summary.json').read_text())
    old_files = json.loads((STUDY/'executed/old_files_before.json').read_text())
    for relative, saved in old_files.items():
        path = REPO/relative
        assert sha(path) == saved['sha256']
        assert path.stat().st_mtime_ns == saved['mtime_ns']
    files = {str(p.relative_to(REPO)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
             for p in sorted((STUDY/'results').glob('*'))}
    assert len(files) == 48
    expected = {f'{fn}_w{w}_s{s}_sigma{sigma:g}' for fn in config['functions']
                for w in config['widths'] for s in config['seeds'] for sigma in [0,1]}
    assert {p.stem for p in (STUDY/'results').glob('*.json')} == expected
    commit = summary['preregistration_commit']
    for name in ['preregistration.json', *config['source_sha256']]:
        path = STUDY/name
        assert subprocess.check_output(['git','show',f'{commit}:{path.relative_to(REPO)}'],cwd=REPO) == path.read_bytes()
    gate = json.loads((STUDY/'executed/pre_execution_audit.json').read_text())
    started = json.loads((STUDY/'executed/first_cell_execution.json').read_text())['started_at']
    commit_time = subprocess.check_output(['git','show','-s','--format=%cI',commit],cwd=REPO,text=True).strip()
    assert datetime.fromisoformat(commit_time) <= datetime.fromisoformat(started)
    assert datetime.fromisoformat(commit_time) <= datetime.fromisoformat(gate['verified_at'])
    gate_ns = int(datetime.fromisoformat(gate['verified_at']).timestamp()*1e9)
    assert gate_ns <= min(item['mtime_ns'] for item in files.values())
    pairs, units = [], []
    max_summary_difference = 0.0
    maxima = {'new_tangent_train_checkpoints': 0.0, 'new_tangent_test_checkpoints': 0.0,
              'new_tangent_all_train_losses': 0.0}

    def load(source, cell):
        path = source/'results'/f'{cell}.json'
        m = json.loads(path.read_text())
        assert m['status']=='success' and m['cell_id']==cell
        assert m['arrays_sha256']==sha(path.with_suffix('.npz'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            return {n:z[n].copy() for n in z.files}

    def risk(a):
        return np.square(a['test_outputs']-a['test_y']).mean(axis=-1)/2

    old_clean = REPO/config['reference_study']
    old_noisy = REPO/config['previous_study']
    for fn in config['functions']:
        for width in config['widths']:
            values, differences = [], []
            for seed in config['seeds']:
                stem = f'{fn}_w{width}_s{seed}'
                a0,a1 = [load(STUDY,stem+f'_sigma{v}') for v in [0,1]]
                b0,b1 = load(old_clean,stem+'_sigma0'),load(old_noisy,stem+'_sigma1')
                for n in ['train_x','test_x','clean_train_y','test_y','target_normalization','noise_base',
                          'initial_parameters','initial_jacobian_train','initial_jacobian_test']:
                    assert np.array_equal(a0[n],a1[n])
                for n in ['initial_parameters','noise_base']:
                    assert np.array_equal(a1[n],b1[n])
                new_h = [float(np.diff(risk(a)[-1][::-1])[0]) for a in [a0,a1]]
                old_h = [float(np.diff(risk(a)[-1][::-1])[0]) for a in [b0,b1]]
                delta = new_h[1]-new_h[0]
                difference = delta-(old_h[1]-old_h[0])
                values.append(delta); differences.append(difference)
                c = float(risk(a1)[-1,0]-risk(a1)[a1['checkpoints'].tolist().index(128),0])
                row = {'function':fn,'width':width,'seed':seed,'D':delta,'H0':new_h[0],'H1':new_h[1],
                       'previous_D':old_h[1]-old_h[0],'new_minus_old_D':difference,'noisy_real_C':c}
                pairs.append(row)
                reported = next(p for p in summary['paired_seed_results'] if (p['function'],p['width'],p['seed'])==(fn,width,seed))
                for actual,key in [(delta,'noise_amplification'),(difference,'new_minus_old_D'),(new_h[0],'clean_test_gap'),(new_h[1],'noisy_test_gap')]:
                    error=abs(actual-reported[key]); max_summary_difference=max(max_summary_difference,error)
                    assert error<1e-12
                for a in [a0,a1]:
                    j,jtest = a['initial_jacobian_train'],a['initial_jacobian_test']
                    kernel=np.einsum('ip,jp->ij',j,j,optimize=False)/32
                    cross=np.einsum('ip,jp->ij',jtest,j,optimize=False)/32
                    r=a['train_outputs'][0,1]-a['train_y']
                    test=a['test_outputs'][0,1].copy()
                    losses=[]; pred=[]; predtest=[]
                    for step in range(513):
                        losses.append(np.square(r).mean()/2)
                        if step in a['checkpoints']:
                            pred.append(r+a['train_y']); predtest.append(test.copy())
                        if step<512:
                            test-=.05*np.einsum('ij,j->i',cross,r,optimize=False)
                            r-=.05*np.einsum('ij,j->i',kernel,r,optimize=False)
                    for key,actual,saved in [('new_tangent_train_checkpoints',pred,a['train_outputs'][:,1]),
                                             ('new_tangent_test_checkpoints',predtest,a['test_outputs'][:,1]),
                                             ('new_tangent_all_train_losses',losses,a['train_loss'][:,1])]:
                        error=float(np.abs(np.array(actual)-saved).max())
                        assert error<1e-11
                        maxima[key]=max(maxima[key],error)
            observed={'function':fn,'width':width,'D':stats(values),'new_minus_old_D':stats(differences),
                      'positive_seeds':sum(v>0 for v in values)}
            observed['pass']=observed['D']['mean']>=.02 and observed['positive_seeds']>=2
            units.append(observed)
            reported=next(u for u in summary['units'] if (u['function'],u['width'])==(fn,width))
            assert reported['pass']==observed['pass']
            for actual,key in [(observed['D'],'noise_amplification'),(observed['new_minus_old_D'],'new_minus_old_D')]:
                for field in ['mean','seed_95pct_t_interval']:
                    error=float(np.abs(np.array(actual[field])-np.array(reported['metrics'][key][field])).max())
                    assert error < (1e-10 if field=='seed_95pct_t_interval' else 1e-12)
                    max_summary_difference=max(max_summary_difference,error)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(REPO/'.cache'),TORCH_HOME=str(REPO/'.cache/torch'))
    resumed=subprocess.run(['python3','-B',str(STUDY/'executed/run.py'),'--max-new-cells','0'],cwd=REPO,env=env,capture_output=True,text=True)
    assert resumed.returncode==0, resumed.stderr
    entries=[json.loads(line) for line in resumed.stdout.splitlines()]
    assert len(entries)==24 and all(e.get('resumed') is True for e in entries)
    for relative, saved in files.items():
        path=REPO/relative
        assert sha(path)==saved['sha256'] and path.stat().st_mtime_ns==saved['mtime_ns']
    assert sum(u['pass'] for u in units)==summary['prediction']['passing_units']==4
    result={'status':'passed','checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'old_files_hash_and_mtime_checked':len(old_files),'new_files_hash_and_mtime_checked':len(files),
            'same_commit_contract_and_sources':True,'commit_before_training':True,'gate_before_first_saved_results':True,
            'independent_primary_passing_units':4,'independent_pairs':pairs,'independent_units':units,
            'maximum_summary_difference':max_summary_difference,'new_only_maximum_absolute_errors':maxima,
            'resume_new_training_cells':0,'resume_stdout':resumed.stdout,'resume_stderr':resumed.stderr,
            'new_success_files':files,'interval_method':'df2 Student-t 97.5% quantile analytic inverse; same-init pairs only'}
    (STUDY/'executed/independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','old_files_hash_and_mtime_checked','new_files_hash_and_mtime_checked','independent_primary_passing_units','maximum_summary_difference','new_only_maximum_absolute_errors','resume_new_training_cells']}))


if __name__=='__main__':
    main()
