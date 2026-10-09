"""Independent public evidence review: contracts, saved arrays and analytic checks."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_sha(request, compact=False):
    options = dict(sort_keys=True)
    if compact:
        options['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(request, **options).encode()).hexdigest()


def load(path):
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def close(a, b, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert error < tolerance, error
    return error


def sigmoid(x):
    return 1/(1+np.exp(-x))


def b_data(condition):
    rng = np.random.default_rng(condition['data_seed'])
    shape = condition['input_dim']
    xs = [rng.uniform(-np.sqrt(3),np.sqrt(3),size=(n,shape)) if condition.get('distribution')=='uniform' else rng.normal(size=(n,shape)) for n in [condition['train_samples'],condition['test_samples']]]
    functions = {
        'smooth_sum':lambda x:np.tanh(x[:,0])+.5*np.tanh(x[:,1]),
        'product':lambda x:np.tanh(x[:,0]*x[:,1]),
        'mixed_sine':lambda x:np.sin(1.7*x[:,0])+.4*np.sin(x[:,1]*x[:,2]),
        'radial':lambda x:np.exp(-(x*x).sum(1)/shape),
        'triple_product':lambda x:np.tanh(x[:,0]*x[:,1]*x[:,2]),
        'soft_bump':lambda x:np.exp(-.7*((x[:,:3]-.3)**2).sum(1))+.2*np.tanh(x[:,3]),
    }
    ys = [functions[condition['function']](x) for x in xs]
    center,scale = ys[0].mean(),ys[0].std()
    return dict(train_x=xs[0],test_x=xs[1],train_y=(ys[0]-center)/scale,test_y=(ys[1]-center)/scale)


def mlp_output(x, vector, width):
    shapes=[(width,x.shape[1]),(width,),(width,width),(width,),(1,width),(1,)]
    parameters=[]; offset=0
    for shape in shapes:
        size=int(np.prod(shape)); parameters.append(vector[offset:offset+size].reshape(shape)); offset+=size
    assert offset==len(vector)
    h=x@parameters[0].T+parameters[1]; h=h*sigmoid(h)
    h=h@parameters[2].T+parameters[3]; h=h*sigmoid(h)
    return (h@parameters[4].T+parameters[5]).reshape(-1)


def audit_b():
    outputs={}; b02_groups={}; b03_groups={}; saturation={}
    counts={'B01':192,'B02':72,'B03':24}
    for name,folder in [('B01','B01_effective_time'),('B02','B02_nonlinear_clock'),('B03','B03_kernel_direction')]:
        study=ROOT/'studies'/folder; config=read(study/'preregistration.json')
        source=config.get('execution_source_sha256',config.get('run_source_sha256'))
        assert sha(study/'run.py')==source
        if name!='B01':
            assert sha(study/'executed/experiment.py')==config['model_source_sha256']
            prediction=read(study/'ood_predictions.json')
            assert prediction['preregistration_sha256']==sha(study/'preregistration.json')
            assert prediction['development_analysis_sha256']==sha(study/'development_analysis.json')
        starts=[]; seconds=0.; n=0; adam_error=0.; mlp_error=0.; probe_error=0.
        for phase in ['development','ood']:
            seen=set()
            for path in sorted((study/'results'/phase).glob('*.json')):
                row=read(path); request=row['request']; arrays=load(path)
                assert row['status']=='success' and all(np.isfinite(x).all() for x in arrays.values())
                assert sha(path.with_suffix('.npz'))==row['arrays_sha256']
                assert cell_sha(request,name=='B01')[:16]==row['cell_id']==path.stem
                assert request['preregistration_sha256']==sha(study/'preregistration.json')
                assert request['condition'] in config[phase+'_conditions']
                if name=='B01':
                    assert request['source_sha256']==source and request['coordinate'] in config['coordinates']
                    assert request['recipe'] in config['recipes'] and request['steps']==config['steps']
                    key=request['condition']['name'],request['coordinate'],request['recipe']['name']
                    close(arrays['loss'],np.sum(arrays['mode_errors']**2,axis=1)/2)
                    if request['recipe']['optimizer']=='Adam':
                        eigen=arrays['eigenvalues']; rotation=arrays['rotation']; target=arrays['target_modes']; recipe=request['recipe']
                        gradient=-(rotation@(np.sqrt(eigen)*target))
                        expected=-target+np.sqrt(eigen)*(rotation.T@(-recipe['lr']*gradient/(abs(gradient)+recipe['eps'])))
                        adam_error=max(adam_error,close(expected,arrays['mode_errors'][1]))
                    if phase=='ood' and request['coordinate']=='diagonal' and request['recipe']['name']=='sgd_small' and request['condition']['spectrum']=='geometric':
                        loss=arrays['loss']; future_max=np.maximum.accumulate(loss[::-1])[::-1]
                        times=np.flatnonzero(future_max<=.01*loss[0])
                        saturation[request['condition']['alignment']]=dict(saturation_step=int(times[0]) if len(times) else None,final_relative_loss=float(loss[-1]/loss[0]))
                else:
                    width=request['width']; assert request['seed'] in config['seeds']
                    assert request['condition'] in config[phase+'_conditions']
                    widths=config['widths'] if name=='B02' else config[phase+'_widths']; assert width in widths
                    key=request['condition']['name'],width,request['seed']
                    assert request['run_source_sha256' if name=='B02' else 'execution_sha256']==source
                    assert request['model_source_sha256' if name=='B02' else 'model_sha256']==config['model_source_sha256']
                    for field,value in b_data(request['condition']).items():
                        close(arrays[field],value,1e-12)
                    if name=='B02':
                        recipe=request['recipe']; assert recipe in config['recipes'] and request['steps']==config['steps']
                        key+=recipe['name'],
                        mlp_error=max(mlp_error,close(mlp_output(arrays['train_x'],arrays['initial_parameters'],width),arrays['initial_train_output']))
                        for prefix in ['train','test']:
                            for kind in ['true','tangent']:
                                loss=float(np.mean((arrays['final_'+kind+'_'+prefix+'_output']-arrays[prefix+'_y'])**2)/2)
                                reported=arrays[('tangent_' if kind=='tangent' else '')+'train_half_mse'][-1] if prefix=='train' else arrays['records'][-1,3 if kind=='true' else 4]
                                close(loss,reported)
                        if phase=='ood' and recipe['optimizer']=='SGD':
                            group=request['condition']['name'],width,recipe['name']
                            gain=(arrays['tangent_train_half_mse'][-1]-arrays['train_half_mse'][-1])/arrays['train_half_mse'][0]
                            test_gap=arrays['records'][-1,3]-arrays['records'][-1,4]
                            b02_groups.setdefault(group,[]).append([float(gain),float(test_gap)])
                    else:
                        for prefix,parameter in [('initial',arrays['initial_parameters']),('learned',arrays['learned_parameters'])]:
                            predicted=mlp_output(arrays['train_x'],parameter,width)-arrays['train_y']
                            reported=arrays['pretrain_loss'][0 if prefix=='initial' else -1]
                            mlp_error=max(mlp_error,close(np.mean(predicted**2)/2,reported))
                        residual=arrays['probe_train_residual']; k0=arrays['initial_kernel']; k1=arrays['learned_kernel']; scale=float(arrays['scale'])
                        close(np.trace(scale*k0),np.trace(k1));
                        ray=lambda k,r:float(r@k@r/(r@r))
                        observed=ray(k1,residual)/ray(scale*k0,residual)
                        perm=[ray(k1,r)/ray(scale*k0,r) for r in [np.random.default_rng(s).permutation(residual) for s in config['permutation_seeds']]]
                        close(arrays['rayleigh_ratios'],[observed,*perm])
                        for label,k in [('initial',k0),('initial_equal_trace',scale*k0),('learned',k1)]:
                            eigen,v=np.linalg.eigh(k); modes=v.T@residual; steps=np.arange(config['probe_steps']+1)[:,None]
                            expected=((1-float(arrays['probe_lr'])*eigen)**(2*steps)*modes**2).sum(1)/(2*len(residual))
                            probe_error=max(probe_error,close(expected,arrays[label+'_probe_train_loss']))
                        if phase=='ood':
                            gain=(arrays['initial_equal_trace_probe_train_loss'][-1]-arrays['learned_probe_train_loss'][-1])/arrays['learned_probe_train_loss'][0]
                            b03_groups.setdefault((request['condition']['name'],width),[]).append([float(gain),observed,observed/np.mean(perm)])
                assert key not in seen; seen.add(key); n+=1; seconds+=row['seconds']
                if phase=='ood':
                    start=row.get('started_at',row.get('saved_at',0)-row['seconds'])
                    assert start>(config['created_at'] if name=='B01' else prediction['created_at']); starts.append(start)
            expected=len(config[phase+'_conditions'])*(len(config['coordinates'])*len(config['recipes']) if name=='B01' else len(config['widths'])*len(config['recipes'])*len(config['seeds']) if name=='B02' else len(config[phase+'_widths'])*len(config['seeds']))
            assert len(seen)==expected
        assert n==counts[name]
        outputs[name]=dict(saved_cells=n,failed_cells=0,request_source_data_verified=True,seconds_sum=seconds,run_source_sha256=source,prediction_saved_at=config['created_at'] if name=='B01' else prediction['created_at'],first_ood_started_at=min(starts),chronology_margin_seconds=min(starts)-(config['created_at'] if name=='B01' else prediction['created_at']))
        if name=='B01': outputs[name].update(independent_Adam_first_step_max_error=adam_error,saturation=saturation,boundary='Known quadratic optimization calibration; rotations are coordinate controls, not new targets.')
        if name=='B02': outputs[name].update(initial_output_manual_forward_max_error=mlp_error,boundary='72 nonlinear and 72 matched tangent optimizer trajectories; training gain is separate from test gain.')
        if name=='B03': outputs[name].update(manual_parameter_forward_max_error=mlp_error,independent_closed_form_probe_error=probe_error,boundary='Fixed learned kernel substitution, equal trace/residual/lr; not actual nonlinear continuation.')
    b02=[dict(condition=k[0],width=k[1],recipe=k[2],mean_train_gain_relative_initial=float(np.mean(v,0)[0]),mean_true_minus_tangent_test_loss=float(np.mean(v,0)[1])) for k,v in sorted(b02_groups.items())]
    assert len(b02)==8 and all(v['mean_train_gain_relative_initial']>.05 for v in b02)
    outputs['B02'].update(ood_SGD_units=b02,train_gain_passed=8,test_harm_units=sum(v['mean_true_minus_tangent_test_loss']>0 for v in b02))
    b03=[dict(condition=k[0],width=k[1],mean=np.mean(v,0).tolist(),seed_gains=[r[0] for r in v]) for k,v in sorted(b03_groups.items())]
    assert len(b03)==4 and all(r['mean'][0]>.05 and r['mean'][1]>1 and r['mean'][2]>1 for r in b03)
    outputs['B03'].update(ood_units=b03,correlated_checks_passed=12,independent_conditions=4)
    return outputs


def c01_features(xs,seed,scale,activation,width):
    weight=torch.randn(width,xs[0].shape[1],generator=torch.Generator().manual_seed(seed),dtype=torch.float64).numpy()/np.sqrt(xs[0].shape[1])
    raw=[np.maximum(scale*x@weight.T,0) if activation=='relu' else (scale*x@weight.T)*sigmoid(scale*x@weight.T) for x in xs]
    center=raw[0].mean(0); norm=np.sqrt(np.mean((raw[0]-center)**2))
    return [(a-center)/norm/np.sqrt(width) for a in raw]


def audit_c():
    outputs={}; row_sets={}
    counts={'C01_activation_scale':216,'C02_parity_ood':192,'C03_hidden_boundary':128,'C04_noise_generalization':64,'C05_risk_ood':24}
    for folder,count in counts.items():
        study=ROOT/'studies'/folder; rows=[]; source=read(study/'source_manifest.json')['executable']
        for name,pin in source.items(): assert sha(study/'executed'/name)==pin
        for path in sorted((study/'results').glob('*.json')):
            row=read(path); arrays=load(path); assert row['finite'] and row['status']=='completed'
            assert all(np.isfinite(value).all() for value in arrays.values())
            assert sha(path.with_suffix('.npz'))==row['arrays_sha256']; rows.append((path,row,arrays))
        assert len(rows)==count
        row_sets[folder]=rows; summary=read(study/'summary.json')
        outputs[folder[:3]]=dict(recipes=count,head_trajectories=count*(17 if folder.startswith('C05') else 3 if folder.startswith('C04') else 1),failed_cells=0,seconds_sum=sum(r['seconds'] for _,r,_ in rows),source_pins=source,failed_predictions=[k for k,v in summary['verdicts'].items() if not v['pass']])
    c02=row_sets['C02_parity_ood']; study=ROOT/'studies/C02_parity_ood'; config=read(study/'preregistration.json')
    feature_error=0.; risk_error=0.; effect={}; lookup={}
    for path,row,a in c02:
        cell=row['cell']; data=load(study/f"data_{cell['distribution']}.npz")
        xs=[data['train_x'],data['test_x']]; width=config['architecture']['width']; scale=cell['scale']
        weight=torch.randn(width,xs[0].shape[1],generator=torch.Generator().manual_seed(cell['seed']),dtype=torch.float64).numpy()/np.sqrt(xs[0].shape[1])
        zs=[scale*x@weight.T for x in xs]; odd=[z/2 for z in zs]
        even=[abs(z)/2 if cell['activation']=='relu' else z*np.tanh(z/2)/2 for z in zs]
        center=even[0].mean(0); even=[e-center for e in even]
        reference=abs(zs[0])/2; reference-=reference.mean(0)
        multiplier=np.sqrt(np.mean(reference**2)/np.mean(even[0]**2)) if cell['intervention']=='parity_balanced' else 1.
        raw=[o+multiplier*e for o,e in zip(odd,even)]; norm=np.sqrt(np.mean(raw[0]**2)); phi=[r/norm/np.sqrt(width) for r in raw]
        feature_error=max(feature_error,close(phi[0],a['train_features']),close(phi[1],a['test_features']))
        close(multiplier,row['features']['even_multiplier'])
        for split in ['train','test']:
            close(a[split+'_target'],data[split+'_y_'+cell['target']])
        risk=float(np.mean((a['test_features']@a['final_head']-a['test_target'])**2))
        risk_error=max(risk_error,close(risk,row['records'][-1]['test_mse']))
        lookup[tuple(cell[k] for k in ['distribution','seed','scale','activation','intervention','target'])]=risk
    for distribution in config['data']['distributions']:
        effects=[]; plain=[]; balanced=[]
        for seed in config['seeds']:
            p=lookup[distribution,seed,.03,'silu','plain','interaction']; b=lookup[distribution,seed,.03,'silu','parity_balanced','interaction']
            plain.append(p); balanced.append(b); effects.append(b-p)
        assert all(x<0 for x in effects)
        effect[distribution]=dict(plain_mean=float(np.mean(plain)),balanced_mean=float(np.mean(balanced)),paired_mean_difference=float(np.mean(effects)),seed_wins=sum(v<0 for v in effects))
    outputs['C02'].update(independent_even_odd_feature_error=feature_error,independent_endpoint_MSE_error=risk_error,effect=effect,boundary='Intervention uses train inputs and random weights only; antithetic data, zero bias, fixed features; test endpoint was preregistered.')
    for name,seal_time in [('C02',config['sealed_at']),('C05',read(ROOT/'studies/C05_risk_ood/forecast_seal.json')['sealed_at'])]:
        folder='C02_parity_ood' if name=='C02' else 'C05_risk_ood'; study=ROOT/'studies'/folder
        if (study/'worker.log').exists():
            logs=[json.loads(line) for line in (study/'worker.log').read_text().splitlines() if line.startswith('{')]
            first=next(r for r in logs if r.get('completed')==1)
            process_start=first['updated_at']-first['seconds']; first_mtime=min(p.stat().st_mtime for p,_,_ in row_sets[folder]); first_cell_start=min(p.stat().st_mtime-r['seconds'] for p,r,_ in row_sets[folder])
            assert seal_time<process_start<=first_cell_start<first_mtime
            outputs[name]['chronology']=dict(sealed_at=seal_time,first_process_start_from_saved_log=process_start,earliest_cell_start_from_mtime_and_duration=first_cell_start,first_saved_mtime=first_mtime,earliest_saved_data_mtime=min(p.stat().st_mtime for p in study.glob('data*.npz')),evidence='Original filesystem timestamps and saved progress log; normalized here before public copy. Not externally attested.')
        else:
            prior=read(Path(__file__).with_name('release_audit.json'))['C'][name]
            assert prior['source_pins']==outputs[name]['source_pins'] and prior['chronology']['sealed_at']==seal_time
            outputs[name]['chronology']=dict(prior['chronology'],evidence='Released independent audit record; original progress logs were excluded. Numeric/source checks rerun, original time evidence not reverified.')
    c03=row_sets['C03_hidden_boundary']; study=ROOT/'studies/C03_hidden_boundary'; values={}
    for path,row,a in c03:
        cell=row['cell']; data=load(study/f"data_{cell['distribution']}.npz")
        risk=float(np.mean((a['final_test_prediction']-data['test_y_'+cell['target']])**2))
        close(risk,row['records'][-1]['test_mse']); values[tuple(cell[k] for k in ['distribution','seed','scale','activation','hidden_mode','target'])]=risk
    comparison=[]
    for scale,activation,mode in itertools.product([.03,.3],['relu','silu'],['frozen','learned']):
        risks=[values['symmetric_gaussian',s,scale,activation,mode,'interaction'] for s in [311,312,313,314]]
        comparison.append(dict(scale=scale,activation=activation,hidden_mode=mode,mean_test_mse=float(np.mean(risks))))
    outputs['C03'].update(independent_endpoint_comparison=comparison,boundary='Head starts zero; first hidden gradient is zero. Later energy growth is a diagnostic, not proof of mediation.')
    c04=row_sets['C04_noise_generalization']; decomp_error=0.
    for path,row,a in c04:
        signal=a['final_test_predictions'][:,1]-a['test_clean']; noise=a['final_test_predictions'][:,2]
        actual=np.mean((a['final_test_predictions'][:,0]-a['test_clean'])**2)
        expected=np.mean(signal**2)+np.mean(noise**2)+2*np.mean(signal*noise)
        decomp_error=max(decomp_error,close(actual,expected)); close(actual,row['records'][-1]['test_clean_mse'])
        close(a['final_heads'][:,0],a['final_heads'][:,1]+a['final_heads'][:,2])
    outputs['C04'].update(independent_final_decomposition_error=decomp_error,boundary='Fixed-feature bias/variance identity; N4 passes the preregistered median, with one seed optimum earlier than ReLU.')
    c05=row_sets['C05_risk_ood']; study=ROOT/'studies/C05_risk_ood'; config=read(study/'preregistration.json'); seal=read(study/'forecast_seal.json')
    forecast_error=0.; final_error=0.; matches=0; within2=0; checks=0; unique_checks=set(); zero_optima=0
    for path,row,a in c05:
        cell=row['cell']; forecast=read(study/'forecasts'/path.name); data=load(study/f"data_seed{cell['seed']}.npz")
        assert seal['forecasts']['forecasts/'+path.name]==sha(study/'forecasts'/path.name)==row['forecast_sha256']
        assert forecast['contract_sha256']==row['contract_sha256'] and forecast['saved_at']<=seal['sealed_at']
        matrices=c01_features([data['train_x'],data['test_x']],cell['seed'],cell['scale'],cell['activation'],config['architecture']['width'])
        x,tx=[np.column_stack([m,np.ones(len(m))]) for m in matrices]; n=len(x); eigen,v=np.linalg.eigh(x@x.T/n)
        q=1-2*config['optimizer']['lr']*eigen; target=v.T@data['train_y']; basis=tx@x.T@v/n
        for saved in forecast['records']:
            step=saved['step']; factors=np.divide(1-q**step,eigen,out=np.full_like(eigen,2*config['optimizer']['lr']*step),where=eigen>1e-12)
            signal=float(np.mean((basis@(factors*target)-data['test_y'])**2)); variance=float(cell['noise_sd']**2*np.sum(np.mean(basis**2,axis=0)*factors**2))
            train=float(np.sum(target**2*q**(2*step))/n)
            forecast_error=max(forecast_error,close(signal,saved['forecast_test_signal_mse'],1e-8),close(variance,saved['forecast_noise_variance'],1e-8),close(train,saved['forecast_train_signal_mse'],1e-8))
        final=a['final_test_predictions']; final_risks=np.mean((final[:,1:]-data['test_y'][:,None])**2,axis=0)
        final_error=max(final_error,close(final_risks,row['records'][-1]['test_risk_draws']),close(tx@a['final_heads'],final))
        pred_best=min(forecast['records'],key=lambda r:r['forecast_expected_test_risk']); mc_best=min(row['records'],key=lambda r:r['mean_test_risk'])
        assert pred_best['step']==forecast['forecast_best_step']
        zero_optima+=int(pred_best['step']==0 or mc_best['step']==0); matches+=int(pred_best['step']==mc_best['step']); within2+=int(abs(np.log2(max(1,pred_best['step'])/max(1,mc_best['step'])))<=1)
        for step in [8192,pred_best['step']]:
            actual=next(r for r in row['records'] if r['step']==step); predicted=next(r for r in forecast['records'] if r['step']==step)
            noise_risks=np.array(actual['test_risk_draws']); close(noise_risks.mean(),actual['mean_test_risk']); close(noise_risks.std(ddof=1)/4,actual['se_mean_test_risk'])
            checks+=int(abs(noise_risks.mean()-predicted['forecast_expected_test_risk'])<=3*noise_risks.std(ddof=1)/4+1e-8); unique_checks.add((path.stem,step))
    assert matches==17 and within2==24 and checks==48
    outputs['C05'].update(independent_conditional_forecast_max_error=forecast_error,independent_final_arrays_risk_error=final_error,best_checkpoint_exact=matches,best_checkpoint_within_factor2=within2,checks_within_3_MC_SE=checks,distinct_recipe_checkpoint_checks=len(unique_checks),zero_checkpoint_optima=zero_optima,configuration_sealed_at=config['sealed_at'],first_forecast_saved_at=min(read(study/p)['saved_at'] for p in seal['forecasts']),last_forecast_saved_at=max(read(study/p)['saved_at'] for p in seal['forecasts']),boundary='Forecast reads new initial features, train and test inputs, clean train and clean test labels, and noise variance. No observed trained loss is used. Not a blind task prediction; optima are on a doubling checkpoint grid, SE from 16 draws.')
    return outputs


def main():
    torch.set_num_threads(1)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit='PASS',training=False,model_calls=0,existing_direction_verifiers='B/C both passed; B checks source/J/K/recurrence and C checks all source/data/array/contracts; independent checks below add request consistency, numerical claims and chronology limits.',B=audit_b(),C=audit_c())
    result['counts']=dict(B_optimizer_trajectories=360,B_nonlinear_MLP_trajectories=96,B_quadratic_calibration_trajectories=192,B_matched_tangent_trajectories=72,B_analytic_kernel_probes=72,C_recipes=624,C_head_trajectories=1136,B_C_saved_cells=912)
    path=Path(__file__).with_name('release_audit.json'); path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
    print(json.dumps(dict(audit=result['audit'],counts=result['counts'],B02_test_harm_units=result['B']['B02']['test_harm_units'],C02_effect=result['C']['C02']['effect'],C05=result['C']['C05']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
