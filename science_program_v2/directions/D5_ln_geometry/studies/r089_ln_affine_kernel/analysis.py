from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import t

STUDY = Path(__file__).resolve().parent
OLD = STUDY.parent / 'r077_fixed_kernel_train_chord'

def stats(values):
    x = np.asarray(values, dtype=np.float64); mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return {'mean': mean, 'min': float(x.min()), 'max': float(x.max()), 'paired_seed_t95': [mean-half, mean+half]}
def sign(x): return 1 if x > 1e-12 else -1 if x < -1e-12 else 0

def compute():
    cells = {}
    for path in sorted((STUDY/'results').glob('*.json')):
        row = json.loads(path.read_text()); assert row['status'] == 'completed'
        npz = path.with_suffix('.npz'); assert hashlib.sha256(npz.read_bytes()).hexdigest() == row['arrays_sha256']
        req = row['contract']
        with np.load(npz) as data:
            energies = {}
            for m in (-3, 0, 3):
                residual = data[f'train_predictions_256__{m}'].astype(np.float64).ravel() - data[f'train_y__{m}'].astype(np.float64).ravel()
                residual -= residual.mean(); energies[str(m)] = float(np.mean(residual**2))
            real = (energies['-3'] + energies['3'])/2 - energies['0']
            fixed = {str(step): float(9*np.mean(data[f'affine_total_centered_propagated_ones_{step}']**2)) for step in (1,256)}
            assert all(abs(fixed[str(step)] - row['fixed_kernel_train_chord'][str(step)]) < 1e-14 for step in (1,256))
        cells[(req['function'], req['seed'])] = {'real': real, 'energies': energies, 'fixed': fixed, 'seconds': row['seconds'], 'lambda_max': row['kernel_eigenvalue_max']}
    pairs=[]; functions=('trigonometric8','quadratic8','interaction12','radial12')
    for function in functions:
        for seed in (100,101,102,103,104):
            ln=cells[(function,seed)]; old_path=OLD/'results'/f'{function}_noLN_w64_{seed}.json'; old=json.loads(old_path.read_text())
            with np.load(old_path.with_suffix('.npz')) as data:
                energies={}
                for m in (-3,0,3):
                    residual=data[f'train_predictions_256__{m}'].astype(np.float64).ravel()-data[f'train_y__{m}'].astype(np.float64).ravel(); residual-=residual.mean(); energies[str(m)]=float(np.mean(residual**2))
                no_real=(energies['-3']+energies['3'])/2-energies['0']
            real_delta=ln['real']-no_real; fixed_delta=ln['fixed']['256']-old['fixed_kernel_train_chord']['256']
            pairs.append({'function':function,'seed':seed,'real_train_delta_chord':real_delta,'fixed_delta_chord_256':fixed_delta,'real_train_chord_LN':ln['real'],'real_train_chord_noLN':no_real,'fixed_chord_LN_total_256':ln['fixed']['256'],'fixed_chord_noLN_linear_256':old['fixed_kernel_train_chord']['256'],'direction_match':sign(real_delta)!=0 and sign(fixed_delta)==sign(real_delta)})
    matches=sum(p['direction_match'] for p in pairs); per=[]
    for function in functions:
        subset=[p for p in pairs if p['function']==function]
        per.append({'function':function,'real_train_delta_chord':stats([p['real_train_delta_chord'] for p in subset]),'fixed_delta_chord_256':stats([p['fixed_delta_chord_256'] for p in subset]),'direction_matches':sum(p['direction_match'] for p in subset)})
    return {'study':'r089_ln_affine_kernel','round':89,'direction_round':6,'domain':'development','status':'complete' if len(cells)==20 else 'partial','saved_measurement_cells':len(cells),'new_training_cells':0,'reused_training_cells':120,'pair_count':len(pairs),'measurement_seconds':sum(v['seconds'] for v in cells.values()),'prediction':{'P1':'supported' if matches>=16 else 'refuted' if len(pairs)==20 else 'not_evaluated'},'direction_match_count':matches,'direction_match_rate':matches/len(pairs) if pairs else None,'pairs':pairs,'functions':per,'fixed_delta_range_256':[min(p['fixed_delta_chord_256'] for p in pairs),max(p['fixed_delta_chord_256'] for p in pairs)] if pairs else [],'real_train_delta_range':[min(p['real_train_delta_chord'] for p in pairs),max(p['real_train_delta_chord'] for p in pairs)] if pairs else [],'max_eta_lambda':.001*max((v['lambda_max'] for v in cells.values()),default=0),'boundary':'仅检验固定初始化核加入LN affine参数，不识别核漂移、小批量、decay或test动力学。'}

if __name__=='__main__':
    result=compute(); (STUDY/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','saved_measurement_cells','direction_match_count','prediction')},ensure_ascii=False))
