"""Independent saved-array audit; never calls training or modifies raw files."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import design

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DATA = Path(os.environ.get("AIQ_KB_DATA_ROOT", str(REPO.parent / "AIQ_KB_DATA"))).resolve()
RUN = DATA / "runs/d15_initialization_matrix_20261108"
COMMIT = "25981bf9e8ce7616a679e803a6527216bdf25d08"
SOURCE_FILES = ("design.py", "engine.py", "run.py", "preregistration.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activation(x, kind):
    if kind == "linear":
        return x
    if kind == "relu":
        return np.maximum(x, 0)
    if kind == "tanh":
        return np.tanh(x)
    raise AssertionError(kind)


def activation_derivative(x, kind):
    if kind == "linear":
        return np.ones_like(x)
    if kind == "relu":
        return (x > 0).astype(x.dtype)
    if kind == "tanh":
        return 1 - np.tanh(x) ** 2
    raise AssertionError(kind)


def numpy_forward(weights, X, act, architecture, cache=False):
    """NumPy-only implementation, independent of engine.forward_batch."""
    h = X
    tape = []
    L = len(weights)
    for j, w in enumerate(weights):
        old = h
        z = old @ w
        residual = architecture == "residual_hidden" and 0 < j < L-1
        hidden = j < L-1 and architecture != "residual_hidden"
        if residual:
            h = old + activation(z, act) / np.sqrt(max(L-2, 1))
            normed, invstd = None, None
        elif hidden:
            if architecture == "layernorm_hidden":
                centered = z - z.mean(axis=-1, keepdims=True)
                invstd = 1 / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
                normed = centered * invstd
                h = activation(normed, act)
            else:
                normed, invstd = None, None
                h = activation(z, act)
        else:
            normed, invstd = None, None
            h = z
        if cache:
            tape.append((old, z, residual, hidden, normed, invstd))
    return (h, tape) if cache else h


def numpy_initial_gradients(weights, X, y, act, architecture):
    """Analytic MSE reverse pass, including affine-free LayerNorm derivative."""
    prediction, tape = numpy_forward(weights, X, act, architecture, cache=True)
    delta = 2 * (prediction-y) / (y.shape[1] * y.shape[2])
    gradients = [None] * len(weights)
    for j in reversed(range(len(weights))):
        old, z, residual, hidden, normed, invstd = tape[j]
        skip = delta if residual else None
        if residual:
            dz = delta * activation_derivative(z, act) / np.sqrt(max(len(weights)-2, 1))
        elif hidden:
            if architecture == "layernorm_hidden":
                dn = delta * activation_derivative(normed, act)
                dz = invstd * (dn - dn.mean(axis=-1, keepdims=True)
                               - normed * (dn * normed).mean(axis=-1, keepdims=True))
            else:
                dz = delta * activation_derivative(z, act)
        else:
            dz = delta
        gradients[j] = old.swapaxes(1, 2) @ dz
        delta = dz @ weights[j].swapaxes(1, 2)
        if residual:
            delta = delta + skip
    return gradients


class Auditor:
    def __init__(self):
        self.comparisons = 0
        self.numeric_elements = 0
        self.records = []
        self.errors = defaultdict(lambda: {"elements": 0, "max_absolute_error": 0.,
                                           "max_relative_error": 0., "max_saved_magnitude": 0.})

    def require(self, condition, label):
        self.comparisons += 1
        if not condition:
            raise AssertionError(label)

    def close(self, actual, expected, label, metric):
        a, e = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == e.shape, label + ": shape")
        self.require(np.array_equal(np.isnan(a), np.isnan(e)), label + ": NaN mask")
        self.require(np.array_equal(np.isposinf(a), np.isposinf(e)), label + ": +inf mask")
        self.require(np.array_equal(np.isneginf(a), np.isneginf(e)), label + ": -inf mask")
        finite = np.isfinite(a) & np.isfinite(e)
        aa, ee = a[finite], e[finite]
        if aa.size:
            error = np.abs(aa-ee)
            tolerance = 2e-11 + 1e-9*np.abs(ee)
            self.require(bool(np.all(error <= tolerance)), label + ": numeric tolerance")
            record = self.errors[metric]
            record["elements"] += int(aa.size)
            record["max_absolute_error"] = max(record["max_absolute_error"], float(error.max()))
            record["max_saved_magnitude"] = max(record["max_saved_magnitude"], float(np.abs(aa).max()))
            positive = np.abs(ee) > 2e-11
            if positive.any():
                record["max_relative_error"] = max(record["max_relative_error"], float((error[positive]/np.abs(ee[positive])).max()))
            self.numeric_elements += int(aa.size)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_round(round_id, audit, commit_time):
    root = RUN / round_id
    manifest = load_json(root / "manifest.json")
    audit.require(manifest["status"] == "complete", round_id+": complete manifest")
    start = load_json(root / "start.json")
    expected = design.cases(round_id)
    saved = load_json(root / "cases.json")
    audit.require(saved == expected, round_id+": cases match frozen design exactly")
    audit.require(manifest["preregistration_commit"] == start["preregistration_commit"] == COMMIT,
                  round_id+": preregistration commit")
    audit.require(commit_time < start["started_unix"] <= manifest["completed_unix"], round_id+": chronology")
    audit.require(start["paths"] == manifest["paths"] == len(expected), round_id+": path denominator")
    for name in SOURCE_FILES:
        payload = subprocess.check_output(["git", "-C", str(REPO), "show", f"{COMMIT}:research/{SOURCE.name}/{name}"])
        frozen_hash = hashlib.sha256(payload).hexdigest()
        audit.require(start["source_hashes"][name] == frozen_hash == digest(SOURCE/name),
                      round_id+": source frozen " + name)
    groups = defaultdict(list)
    for case in expected:
        key = tuple(case[k] for k in ("architecture", "activation", "optimizer", "depth", "width", "lr"))
        groups[key].append(case)
    chunks = []
    for group in groups.values():
        chunks.extend(group[i:i+48] for i in range(0, len(group), 48))
    files = sorted(root.glob("batch_*.npz"))
    audit.require(len(files) == len(chunks) == len(manifest["batches"]), round_id+": batch denominator")
    audit.require(set(manifest["batches"]) == {f.name for f in files}, round_id+": manifest filenames")
    seen = []
    status_counts = defaultdict(int)
    zero_count = 0
    max_initial_func_delta = 0.
    high_loss_count = 0
    for batch_index, (path, cases) in enumerate(zip(files, chunks)):
        label = f"{round_id}/{path.name}"
        actual_sha = digest(path)
        meta = load_json(path.with_suffix(".json"))
        ids = [c["id"] for c in cases]
        audit.require(actual_sha == meta["sha256"] == manifest["batches"][path.name], label+": SHA256")
        audit.require(meta["ids"] == ids, label+": metadata IDs")
        c = cases[0]
        with np.load(path, allow_pickle=False) as z:
            audit.require(z["ids"].tolist() == ids, label+": NPZ IDs")
            seen.extend(ids)
            checkpoint = z["checkpoint_steps"]
            audit.require(np.array_equal(checkpoint, np.arange(c["steps"]+1)), label+": full integer checkpoints")
            audit.require(z["train_loss"].shape == (len(cases), c["steps"]+1), label+": loss shape")
            initial = [z[f"initial_w{j}"] for j in range(c["depth"])]
            final = [z[f"final_w{j}"] for j in range(c["depth"])]
            generated = [design.initial_weights(case) for case in cases]
            for j in range(c["depth"]):
                audit.require(np.array_equal(initial[j], np.stack([w[j] for w in generated])), label+": initial weight exact")
            datasets = [design.task(case["task"], case["condition"], case["seed"]) for case in cases]
            X, y, Xv, yv = [np.stack([d[j] for d in datasets]) for j in range(4)]
            predictions = numpy_forward(initial, X, c["activation"], c["architecture"])
            predictions_v = numpy_forward(initial, Xv, c["activation"], c["architecture"])
            audit.close(z["initial_prediction"], predictions, label+": initial predictions", "initial_prediction")
            audit.close(z["train_loss"][:,0], np.mean((predictions-y)**2, axis=(1,2)), label+": initial training MSE", "initial_loss")
            audit.close(z["val_loss"][:,0], np.mean((predictions_v-yv)**2, axis=(1,2)), label+": initial validation MSE", "initial_loss")
            audit.close(z["initial_loss"], z["train_loss"][:,0], label+": initial_loss copy", "stored_copies")
            audit.close(z["initial_val_loss"], z["val_loss"][:,0], label+": initial_val_loss copy", "stored_copies")
            gradients = numpy_initial_gradients(initial, X, y, c["activation"], c["architecture"])
            norms = np.stack([np.linalg.norm(g.reshape(len(cases),-1), axis=1) for g in gradients], 1)
            audit.close(z["initial_grad_norm"], norms, label+": initial gradients", "initial_gradient_norm")
            iszero = np.all(norms == 0, axis=1)
            audit.require(np.array_equal(z["initial_zero_gradient"], iszero), label+": exact zero-gradient flags")
            zero_count += int(iszero.sum())
            status = z["status_code"]
            failure = z["failed_step"]
            audit.require(set(status.tolist()) <= {0,1,2,3}, label+": status values")
            for code in status:
                status_counts[int(code)] += 1
            healthy = status == 0
            audit.require(np.all(failure[healthy] == -1), label+": healthy no failed step")
            audit.require(np.all((failure[~healthy] >= 0) & (failure[~healthy] <= c["steps"])), label+": failed step domain")
            for k in np.flatnonzero(~healthy):
                first_missing = int(failure[k]) + (1 if status[k] == 2 else 0)
                audit.require(np.all(np.isnan(z["train_loss"][k,first_missing:])), label+": failed training mask")
                audit.require(np.all(np.isnan(z["val_loss"][k,first_missing:])), label+": failed validation mask")
            audit.require(np.isfinite(z["train_loss"][healthy]).all() and np.isfinite(z["val_loss"][healthy]).all(), label+": healthy finite trajectories")
            for q, key in ((0.1,"first_hit_0_1"),(0.01,"first_hit_0_01")):
                hit = z["train_loss"] <= q
                expected_hit = np.where(hit.any(1), hit.argmax(1), -1)
                audit.require(np.array_equal(z[key], expected_hit), label+": all first crossings "+key)
            # All final weights in failed routes are NaN; forward is evaluated
            # only on healthy routes to avoid interpreting a masked route.
            for j, w in enumerate(final):
                audit.require(np.isnan(w[~healthy]).all(), label+": failed final weight mask")
                audit.require(np.isfinite(w[healthy]).all(), label+": healthy final weights")
            if healthy.any():
                fw = [w[healthy] for w in final]
                pred_end = numpy_forward(fw, X[healthy], c["activation"], c["architecture"])
                pred_end_v = numpy_forward(fw, Xv[healthy], c["activation"], c["architecture"])
                audit.close(z["train_loss"][healthy,-1], np.mean((pred_end-y[healthy])**2,axis=(1,2)), label+": final training MSE", "final_loss")
                audit.close(z["val_loss"][healthy,-1], np.mean((pred_end_v-yv[healthy])**2,axis=(1,2)), label+": final validation MSE", "final_loss")
                high_loss_count += int(np.sum(z["train_loss"][healthy,-1] > 1e6))
                norm0 = np.stack([np.linalg.norm(w[healthy].reshape(healthy.sum(),-1),axis=1) for w in initial],1)
                delta = np.stack([np.linalg.norm((w[healthy]-v[healthy]).reshape(healthy.sum(),-1),axis=1) for w,v in zip(final,initial)],1)
                relative = np.divide(delta,norm0,out=np.full_like(delta,np.nan),where=norm0>0)
                audit.close(z["initial_weight_norm"][healthy],norm0,label+": initial parameter norms","parameter_diagnostics")
                audit.close(z["final_weight_delta_norm"][healthy],delta,label+": parameter displacement","parameter_diagnostics")
                audit.close(z["final_weight_relative_change"][healthy],relative,label+": relative parameter displacement","parameter_diagnostics")
                yv_norm = np.linalg.norm(yv[healthy].reshape(healthy.sum(),-1),axis=1)
                pred_change = np.linalg.norm((pred_end_v-predictions_v[healthy]).reshape(healthy.sum(),-1),axis=1)/yv_norm
                audit.close(z["final_prediction_change"][healthy],pred_change,label+": output displacement","prediction_change")
            for key in ("final_weight_delta_norm","final_weight_relative_change","final_prediction_change"):
                audit.require(np.isnan(z[key][~healthy]).all(),label+": failed diagnostics mask "+key)
            audit.close(z["target_train_rms"], np.sqrt(np.mean(y*y,axis=(1,2))), label+": target normalization", "target_rms")
            floor = np.array([np.mean((d[0]@np.linalg.lstsq(d[0],d[1],rcond=None)[0]-d[1])**2) for d in datasets])
            audit.close(z["linear_train_floor"], floor, label+": least-squares floor", "linear_floor")
            baseline = [design.initial_weights(dict(case,profile="uniform",global_scale=1.)) for case in cases]
            bw = [np.stack([w[j] for w in baseline]) for j in range(c["depth"])]
            bp = numpy_forward(bw,X,c["activation"],c["architecture"])
            func_delta = np.max(np.abs(predictions-bp),axis=(1,2))
            audit.close(z["initial_function_delta"],func_delta,label+": initial function change","initial_function_change")
            max_initial_func_delta = max(max_initial_func_delta,float(func_delta.max()))
            for k, case in enumerate(cases):
                audit.records.append(dict(case, finaltrain=float(z["train_loss"][k,-1]),
                    finalval=float(z["val_loss"][k,-1]), failed=bool(status[k]!=0),
                    first_hit_0_1=int(z["first_hit_0_1"][k]), first_hit_0_01=int(z["first_hit_0_01"][k]),
                    initial_allzero=bool(iszero[k]),
                    initial_function_delta=float(z['initial_function_delta'][k]),
                    model_unreachable=case['architecture']=='plain' and case['activation']=='linear' and float(z['linear_train_floor'][k])>.1))
        if (batch_index+1) % 40 == 0:
            print(f"audit {round_id}: {batch_index+1}/{len(files)} batches",flush=True)
    audit.require(len(seen)==len(set(seen))==len(expected),round_id+": unique exhaustive route denominator")
    audit.require(set(seen)=={c["id"] for c in expected},round_id+": all frozen IDs")
    return dict(paths=len(expected),batches=len(files),status_counts=dict(status_counts),
                exact_zero_gradient_paths=zero_count,high_finite_terminal_training_loss_over_1e6=high_loss_count,
                max_initial_function_delta=max_initial_func_delta,
                commit_to_start_seconds=start["started_unix"]-commit_time,
                cases_sha256=digest(root/"cases.json"),manifest_sha256=digest(root/"manifest.json"))


def audit_public_artifacts(audit, partial):
    """Independently recompute published case/cell endpoints and denominators."""
    suffix = "_partial" if partial else ""
    csvpath = SOURCE/f"case_metrics{suffix}.csv"
    cellpath = SOURCE/f"cells{suffix}.json"
    summarypath = SOURCE/f"summary{suffix}.json"
    if not all(p.exists() for p in (csvpath,cellpath,summarypath)):
        return dict(status="not_available", scope="Public artifacts not yet all present")
    with csvpath.open(encoding="utf-8-sig",newline="") as handle:
        csvrows=list(csv.DictReader(handle))
    cells=load_json(cellpath)
    summary=load_json(summarypath)
    byid={r['id']:r for r in csvrows}
    audit.require(len(byid)==len(csvrows),"public CSV unique IDs")
    rounds={r['round'] for r in audit.records}
    published=[r for r in csvrows if r['round'] in rounds]
    audit.require({r['id'] for r in published}=={r['id'] for r in audit.records},"public CSV exhaustive audited rounds")
    basefields=('round','task','activation','architecture','depth','width','condition','optimizer','rate','seed','lr','steps')
    def basekey(r):
        return tuple(r[k] for k in basefields)
    def baseline(r):
        return (r['alpha']==0 and r['beta']==0 and r['global_scale']==1) if r['round']=='r003' else r['profile']=='uniform'
    bases={basekey(r):r for r in audit.records if baseline(r)}
    for r in audit.records:
        b=bases.get(basekey(r))
        r['baseline_id']=b['id'] if b else None
        r['baseline_failed']=bool(b and b['failed'])
        r['baseline_missing']=b is None
        r['baseline_nearzero']=bool(b and not b['failed'] and b['finalval']<=1e-10)
        r['both_nearzero']=bool(r['baseline_nearzero'] and not r['failed'] and r['finalval']<=1e-10)
        r['ratio']=(None if b is None or b['failed'] else
                    float('inf') if r['failed'] else
                    (1. if r['finalval']==0 else float('inf')) if b['finalval']==0 else r['finalval']/b['finalval'])
        r['improved']=r['ratio'] is not None and np.isfinite(r['ratio']) and r['ratio']<1
        row=byid[r['id']]
        for field in ('finaltrain','finalval'):
            number=float(row[field]) if row[field] else float('nan')
            audit.close(np.array(number),np.array(r[field]),r['id']+': public '+field,'public_case_endpoint')
        for field in ('first_hit_0_1','first_hit_0_01'):
            audit.require(int(row[field])==r[field],r['id']+': public '+field)
        for field in ('failed','initial_allzero','model_unreachable','baseline_failed','baseline_missing','improved'):
            audit.require((row[field]=='True')==bool(r[field]),r['id']+': public '+field)
        for field in ('baseline_nearzero','both_nearzero'):
            if field in row:
                audit.require((row[field]=='True')==r[field],r['id']+': public '+field)
        audit.require(row['baseline_id']==(r['baseline_id'] or ''),r['id']+': public paired baseline ID')
        # CSV serializes absent and nonfinite ratios as empty cells; explicit
        # flags preserve which case applies.
        raw=row['ratio']
        if r['ratio'] is not None and np.isfinite(r['ratio']):
            audit.close(np.array(float(raw)),np.array(r['ratio']),r['id']+': public ratio','public_case_ratio')
        else:
            audit.require(raw in ('','inf','Infinity'),r['id']+': nonfinite/absent ratio representation')
        audit.require((row['ratio_unavailable']=='True')==(r['ratio'] is None),r['id']+': unavailable flag')
        audit.require((row['ratio_infinite']=='True')==(r['ratio'] is not None and not np.isfinite(r['ratio'])),r['id']+': infinite flag')
    rawbyid={r['id']:r for r in audit.records}
    auditedcells=[c for c in cells if c['round'] in rounds]
    factorfields={round_id:tuple(design.cases(round_id)[0]) for round_id in rounds}
    memberships=[]
    for c in auditedcells:
        label='public '+c['id']
        ids=c['case_ids'];memberships.extend(ids)
        audit.require(all(k in rawbyid for k in ids),label+': known cases')
        rows=sorted([rawbyid[k] for k in ids],key=lambda r:r['seed'])
        audit.require(c['seed_n']==len(rows)==len(set(r['seed'] for r in rows)),label+': seed denominator')
        audit.require(c['seeds']==[r['seed'] for r in rows],label+': seed list')
        for name in factorfields[c['round']]:
            if name not in ('id','seed'):
                audit.require(all(r[name]==c[name] for r in rows),label+': cell factor '+name)
        ratios=[r['ratio'] for r in rows]
        primary=None if any(v is None for v in ratios) else float(np.median(ratios))
        missing=primary is None; infinite=primary is not None and not np.isfinite(primary)
        audit.require(c['median_unavailable']==missing and c['median_failed']==infinite,label+': primary status')
        if primary is not None and np.isfinite(primary):
            audit.close(np.array(c['ratio_median']),np.array(primary),label+': median ratio','public_cell_ratio')
        else:
            audit.require(c['ratio_median'] is None,label+': null primary')
        expected_fields={
            'failed_n':sum(r['failed'] for r in rows),
            'baseline_failed_n':sum(r['baseline_failed'] for r in rows),
            'baseline_missing_n':sum(r['baseline_missing'] for r in rows),
            'improved_n':sum(r['improved'] for r in rows),
            'hit01_n':sum(r['first_hit_0_1']>=0 for r in rows),
            'hit001_n':sum(r['first_hit_0_01']>=0 for r in rows),
            'initial_allzero_n':sum(r['initial_allzero'] for r in rows),
            'model_unreachable_n':sum(r['model_unreachable'] for r in rows),
        }
        for field,key in (('baseline_nearzero_n','baseline_nearzero'),('both_nearzero_n','both_nearzero')):
            if field in c:
                expected_fields[field]=sum(r[key] for r in rows)
        for field,value in expected_fields.items():
            audit.require(c[field]==value,label+': '+field)
        for field,key in (('first_hit_0_1_by_seed','first_hit_0_1'),('first_hit_0_01_by_seed','first_hit_0_01')):
            audit.require(c[field]==[r[key] for r in rows],label+': threshold seed list')
    audit.require(len(memberships)==len(set(memberships))==len(audit.records),'public cells disjoint exhaustive audited cases')
    for round_id in rounds:
        info=summary['rounds'][round_id]
        n=sum(r['round']==round_id for r in audit.records)
        audit.require(info['loaded_n']==info['expected_n']==n,round_id+': public round denominator')
    if not partial:
        audit.require(summary['counts']['n']==len(audit.records),'public summary total paths')
        audit.require(summary['cell_n']==len(auditedcells),'public summary total cells')
        for field,value in {
            'failed_n':sum(r['failed'] for r in audit.records),
            'initial_allzero_n':sum(r['initial_allzero'] for r in audit.records),
            'hit01_n':sum(r['first_hit_0_1']>=0 for r in audit.records),
            'hit001_n':sum(r['first_hit_0_01']>=0 for r in audit.records),
            'model_unreachable_n':sum(r['model_unreachable'] for r in audit.records),
            'baseline_n':sum(baseline(r) for r in audit.records),
            'baseline_failed_n':sum(r['baseline_failed'] for r in audit.records),
            'baseline_missing_n':sum(r['baseline_missing'] for r in audit.records),
            'comparison_n':sum(not baseline(r) for r in audit.records),
            'improved_n':sum(r['improved'] and not baseline(r) for r in audit.records),
            'ratio_finite_n':sum(r['ratio'] is not None and np.isfinite(r['ratio']) for r in audit.records),
        }.items():
            audit.require(summary['counts'][field]==value,'public summary count '+field)
    if 'r001' in rounds:
        groups=defaultdict(dict)
        for r in audit.records:
            if r['round']=='r001' and r['profile'] in ('front','back'):
                key=tuple(r[k] for k in basefields if k!='seed')
                groups[key].setdefault(r['seed'],{})[r['profile']]=r
        passes=0
        for key,seeds in groups.items():
            ratio=[]
            for pair in seeds.values():
                f,b=pair['front'],pair['back']
                if b['failed']:
                    ratio.append(None)
                elif f['failed']:
                    ratio.append(float('inf'))
                elif b['finalval']==0:
                    ratio.append(1. if f['finalval']==0 else float('inf'))
                else:
                    ratio.append(f['finalval']/b['finalval'])
            passed=len(ratio)==4 and all(v is not None for v in ratio)
            if passed:
                passed=np.median(ratio)<=.95 and sum(np.isfinite(v) and v<1 for v in ratio)>=3
            passes+=int(passed)
        audit.require(len(groups)==36,'r001 registered front/back contexts')
        audit.require(summary['r001_front_vs_back']['passed_context_n']==passes,'r001 registered criterion pass count')
        expected_verdict='supported' if passes==36 else 'mixed_universal_refuted' if passes else 'refuted'
        audit.require(summary['r001_front_vs_back']['verdict']==expected_verdict,'r001 registered verdict')
    if not partial:
        selected=[r for r in audit.records if r['architecture']=='plain' and r['activation'] in ('linear','relu')
                  and (r['profile'] in design.PROFILES[:13] or (r['profile']=='grid' and r['global_scale']==1))]
        fact=summary['product_preserving_function_check']
        audit.require(fact['n']==len(selected),'function identity check denominator')
        audit.require(fact['failed_n']==sum(r['initial_function_delta']>1e-10 for r in selected),'function identity check failures')
        audit.close(np.array(fact['max_abs_difference']),np.array(max(r['initial_function_delta'] for r in selected)),
                    'function identity maximum','public_function_fact')
    return dict(status='passed',audited_rounds=sorted(rounds),case_rows=len(audit.records),cells=len(auditedcells),
                full_summary_total_counts_audited=not partial,
                hashes={p.name:digest(p) for p in (csvpath,cellpath,summarypath)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial",action="store_true",help="Audit only rounds with complete manifests; never claim all-round completion.")
    args=parser.parse_args()
    begin=time.monotonic()
    commit_time=int(subprocess.check_output(["git","-C",str(REPO),"show","-s","--format=%ct",COMMIT],text=True).strip())
    rounds=("r001","r002","r003")
    complete=[r for r in rounds if (RUN/r/"manifest.json").is_file()]
    if not args.partial and complete != list(rounds):
        raise RuntimeError("Full audit requires complete manifests for all rounds; use --partial for a bounded audit")
    if not complete:
        raise RuntimeError("No completed rounds available")
    audit=Auditor()
    result={r:validate_round(r,audit,commit_time) for r in complete}
    public=audit_public_artifacts(audit,args.partial)
    report=dict(status="partial_passed" if args.partial else "passed",audit_scope="Every saved route and batch in listed completed rounds; NumPy-only forward and analytic initial reverse pass; no retraining",
                preregistration_commit=COMMIT,rounds=result,all_rounds_complete=len(complete)==3,
                comparisons=audit.comparisons,numeric_elements=audit.numeric_elements,
                public_artifacts=public,
                numeric_tolerance=dict(absolute=2e-11,relative=1e-9),numeric_error_ranges=dict(audit.errors),
                actual_elapsed_seconds=time.monotonic()-begin,
                omissions="Intermediate network states are not saved; intermediate losses are audited for masks and crossings, not reconstructed by retraining. Public-artifact audit scope is separately recorded.")
    target=SOURCE/("verification_partial.json" if args.partial else "verification.json")
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(status=report["status"],rounds=list(result),paths=sum(r["paths"] for r in result.values()),comparisons=audit.comparisons,elapsed=report["actual_elapsed_seconds"],report=str(target)),ensure_ascii=False))


if __name__=="__main__":
    main()
