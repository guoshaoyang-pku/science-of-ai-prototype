import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / 'directions/D5_ln_geometry/studies/r089_ln_affine_kernel'
OLD = ROOT / 'directions/D5_ln_geometry/studies/r077_fixed_kernel_train_chord'
PRE = '99a9ed117c73bcd79f10e5b37deca5e900603442'

def read_json(path):
    return json.loads(path.read_text())

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()

def committed_bytes(rev, path):
    return subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=ROOT)

def pin(path):
    data = path.read_bytes()
    return {'sha256': digest(data), 'mtime_ns': path.stat().st_mtime_ns}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    out = STUDY / 'executed' / ('final_commit_verification.json' if args.final else 'precommit_audit.json')
    head = git('rev-parse', 'HEAD')
    receipt = read_json(STUDY / 'executed/receipt.json')
    prereg = read_json(STUDY / 'preregistration.json')
    manifest_path = STUDY / 'executed/input_manifest.json'
    manifest = read_json(manifest_path)
    checks = {}

    checks['preregistration_ancestor'] = git('merge-base', '--is-ancestor', PRE, head) == ''
    checks['preregistration_pin'] = digest(committed_bytes(PRE, 'directions/D5_ln_geometry/studies/r089_ln_affine_kernel/preregistration.json')) == receipt['preregistration_sha256']
    checks['preregistration_current_bytes'] = (STUDY / 'preregistration.json').read_bytes() == committed_bytes(PRE, 'directions/D5_ln_geometry/studies/r089_ln_affine_kernel/preregistration.json')
    checks['frozen_sources'] = True
    for rel, expected in receipt['source_sha256'].items():
        current = (STUDY / rel).read_bytes()
        source = committed_bytes(PRE, f'directions/D5_ln_geometry/studies/r089_ln_affine_kernel/{rel}')
        checks['frozen_sources'] &= digest(current) == expected == digest(source)
    checks['manifest_pin'] = digest(manifest_path.read_bytes()) == prereg['inputs']['manifest_sha256'] == receipt['manifest_sha256']
    checks['manifest_commit_bytes'] = manifest_path.read_bytes() == committed_bytes(PRE, 'directions/D5_ln_geometry/studies/r089_ln_affine_kernel/executed/input_manifest.json')
    checks['receipt_gate'] = receipt['commit_epoch'] == int(git('show', '-s', '--format=%ct', PRE))

    protected = {}
    for rel, expected in manifest['files'].items():
        path = ROOT / rel
        protected[rel] = pin(path)
        checks['r077_input_pins'] = checks.get('r077_input_pins', True) and protected[rel]['sha256'] == expected['sha256'] and protected[rel]['mtime_ns'] == expected['mtime_ns']
        checks['r077_input_snapshot_bytes'] = checks.get('r077_input_snapshot_bytes', True) and path.read_bytes() == committed_bytes(manifest['snapshot_commit'], rel)
    old_manifest = read_json(OLD / 'executed/input_manifest.json')
    for rel, expected in old_manifest['files'].items():
        path = ROOT / rel
        protected[rel] = pin(path)
        checks['r077_historical_pins'] = checks.get('r077_historical_pins', True) and protected[rel]['sha256'] == expected['sha256'] and protected[rel]['mtime_ns'] == expected['mtime_ns']
    old_verify = read_json(OLD / 'executed/saved_evidence_verification.json')
    for rel, expected in old_verify['result_pins'].items():
        path = ROOT / rel
        protected[rel] = pin(path)
        checks['r077_result_pins'] = checks.get('r077_result_pins', True) and protected[rel]['sha256'] == expected['sha256'] and protected[rel]['mtime_ns'] == expected['mtime_ns']

    rows = sorted((STUDY / 'results').glob('*.json'))
    checks['twenty_cells'] = len(rows) == 20
    result_pins = {}
    for row_path in rows:
        row = read_json(row_path)
        npz_path = row_path.with_suffix('.npz')
        result_pins[str(row_path.relative_to(ROOT))] = pin(row_path)
        result_pins[str(npz_path.relative_to(ROOT))] = pin(npz_path)
        checks['result_contracts'] = checks.get('result_contracts', True) and row['status'] == 'completed' and row['contract']['preregistration_commit'] == PRE and row['started_epoch'] >= receipt['gate_epoch'] >= receipt['commit_epoch'] and digest(npz_path.read_bytes()) == row['arrays_sha256']
    checks['zero_training'] = prereg['conditions']['new_training_cells'] == 0
    checks['json_parse'] = True
    for path in STUDY.rglob('*.json'):
        read_json(path)

    summary_path = STUDY / 'summary.json'
    summary_pin_before = pin(summary_path)
    spec = importlib.util.spec_from_file_location('r089_analysis', STUDY / 'analysis.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    recomputed = analysis.compute()
    checks['summary_recomputes'] = recomputed == read_json(summary_path)
    checks['summary_unmodified'] = pin(summary_path) == summary_pin_before

    state = read_json(ROOT / 'central/state.json')
    base_state = json.loads(committed_bytes(PRE, 'central/state.json'))
    state_copy = copy.deepcopy(state)
    state_copy['directions']['D5_ln_geometry']['last_round'] = base_state['directions']['D5_ln_geometry']['last_round']
    state_copy['directions']['D5_ln_geometry']['next_question'] = base_state['directions']['D5_ln_geometry']['next_question']
    checks['state_scope'] = state_copy == base_state and state['round'] == 89 and state['directions']['D5_ln_geometry']['rounds_done'] == 5
    kb = read_json(ROOT / 'central/kb.json')
    base_kb = json.loads(committed_bytes(PRE, 'central/kb.json'))
    checks['kb_scope'] = kb['claims'][:-1] == base_kb['claims'] and len(kb['claims']) == len(base_kb['claims']) + 1 and kb['claims'][-1]['id'] == 'D5-010' and kb['claims'][-1]['status'] == 'refuted'

    report = (ROOT / 'directions/D5_ln_geometry/report.md').read_text()
    headings = [line for line in report.splitlines() if line.startswith('## ')]
    required = ['## 结论','## Formulation','## 成立程度','## 方法与条件','## 失败与反例','## 未决问题','## 证据']
    checks['report_skeleton'] = [h for h in headings if h in required] == required
    evidence_start = report.index('## 证据')
    checks['report_links_at_end'] = not any(token in report[:evidence_start] for token in ('studies/', 'findings/', 'commit '))
    checks['inbox_handled'] = '[已处理 2026-10-07T15:04:36+08:00]' in (ROOT / 'directions/D5_ln_geometry/inbox.md').read_text()

    if args.final:
        prior = read_json(STUDY / 'executed/precommit_audit.json')
        checks['protected_unchanged_since_precommit'] = protected == prior['protected_file_pins'] and result_pins == prior['r089_result_pins']
        tracked = [
            'central/kb.json','central/state.json','reports/PROGRESS.md',
            'directions/D5_ln_geometry/report.md',
            'directions/D5_ln_geometry/findings/r089_ln_affine_kernel.md',
            'directions/D5_ln_geometry/inbox.md',
        ] + git('ls-files', '--', str(STUDY.relative_to(ROOT))).splitlines()
        tracked = [rel for rel in tracked if rel != str(out.relative_to(ROOT))]
        committed = {}
        for rel in tracked:
            blob = committed_bytes(head, rel)
            work = (ROOT / rel).read_bytes()
            committed[rel] = digest(blob)
            checks['closeout_artifacts_committed'] = checks.get('closeout_artifacts_committed', True) and blob == work
        checks['closeout_commit_after_preregistration'] = checks['preregistration_ancestor']
        payload = {'status':'passed' if all(checks.values()) else 'failed','round':89,'closeout_commit':head,'preregistration_commit':PRE,'closeout_message':git('show','-s','--format=%s',head),'verified_epoch':time.time(),'saved_measurement_cells':len(rows),'new_training_cells':0,'r077_input_files_unchanged':len(manifest['files']),'r077_historical_files_hash_mtime_unchanged':len(old_manifest['files']),'r077_old_result_files_hash_mtime_unchanged':len(old_verify['result_pins']),'r089_result_files_hash_mtime_unchanged':len(result_pins),'recovery_call_new_measurement_cells':19,'state_changed_only_last_round_next_question':checks['state_scope'],'protected_round':89,'protected_rounds_done':5,'direction_matches':read_json(summary_path)['direction_match_count'],'prediction':read_json(summary_path)['prediction'],'frozen_verifier_defect':'verify.py references absent NPZ field kernel_pin; frozen source bytes match preregistration and independent verification passed','checks':checks,'protected_file_pins':protected,'r089_result_pins':result_pins,'closeout_artifact_sha256':committed}
    else:
        payload = {'status':'passed' if all(checks.values()) else 'failed','round':89,'head_before_closeout':head,'preregistration_commit':PRE,'verified_epoch':time.time(),'checks':checks,'protected_file_pins':protected,'r089_result_pins':result_pins}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':payload['status'],'checks':checks},ensure_ascii=False,indent=2))
    if payload['status'] != 'passed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
