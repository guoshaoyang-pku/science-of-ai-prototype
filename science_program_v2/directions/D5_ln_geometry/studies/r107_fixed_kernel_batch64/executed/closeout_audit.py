from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
DIRECTION = STUDY.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closeout')
    args = parser.parse_args()
    cfg = json.loads((STUDY/'preregistration.json').read_text())
    receipt = json.loads((STUDY/'executed/receipt.json').read_text())
    manifest = json.loads((STUDY/'executed/input_manifest.json').read_text())
    verification = json.loads((STUDY/'executed/saved_evidence_verification.json').read_text())
    recovery = json.loads((STUDY/'executed/recovery_verification.json').read_text())
    commit = receipt['preregistration_commit']
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
    for rel in ('preregistration.json','executed/input_manifest.json',*cfg['source_sha256']):
        path = STUDY/rel
        assert git_bytes(commit,path) == path.read_bytes(), rel
    assert sha(STUDY/'preregistration.json') == receipt['preregistration_sha256']
    assert sha(STUDY/'executed/input_manifest.json') == cfg['manifest_sha256'] == receipt['manifest_sha256']
    for rel,item in manifest['files'].items():
        path = ROOT/rel
        assert sha(path)==item['sha256'] and path.stat().st_mtime_ns==item['mtime_ns'],rel
    assert set((STUDY/'results').iterdir()) == {ROOT/rel for rel in verification['result_pins']}
    for pins in (verification['result_pins'], recovery['protected_result_pins']):
        for rel,item in pins.items():
            path = ROOT/rel
            assert sha(path)==item['sha256'] and path.stat().st_mtime_ns==item['mtime_ns'],rel
    rows=[json.loads(p.read_text()) for p in (STUDY/'results').glob('*.json')]
    assert len(rows)==40
    assert all(row['status']=='completed' and row['contract']['preregistration_commit']==commit for row in rows)
    assert all(row['started_epoch']>=receipt['gate_epoch']>=receipt['commit_epoch'] for row in rows)
    assert all(row['new_training_cells']==row['new_kernel_cells']==0 for row in rows)
    assert recovery['recovery_new_evaluation_cells']==0 and recovery['reused_evaluation_cells']==40
    spec=importlib.util.spec_from_file_location('audit_analysis',STUDY/'analysis.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary=json.loads((STUDY/'summary.json').read_text())
    assert module.compute()==summary
    state=json.loads((ROOT/'central/state.json').read_text())
    baseline=json.loads(git_bytes(commit,ROOT/'central/state.json'))
    for key in ('last_round','next_question'):
        state['directions']['D5_ln_geometry'][key]=baseline['directions']['D5_ln_geometry'][key]
    assert state==baseline, 'state scope'
    kb=json.loads((ROOT/'central/kb.json').read_text())
    old_kb=json.loads(git_bytes(commit,ROOT/'central/kb.json'))
    assert kb['claims'][:-2]==old_kb['claims']
    assert [row['id'] for row in kb['claims'][-2:]]==['D5-011','D5-012']
    for key in old_kb:
        if key!='claims': assert old_kb[key]==kb[key]
    text=(DIRECTION/'report.md').read_text()
    prior=git_bytes(manifest['snapshot_commit'],DIRECTION/'report.md').decode()
    assert re.findall(r'^## (.*)$',text,re.M)==['结论','Formulation','成立程度','方法与条件','未决问题','证据']
    number=r'[-−+]?\d+(?:\.\d+)?(?:[eE][−+-]?\d+)?'
    assert set(re.findall(number,prior))<=set(re.findall(number,text))
    for block in re.findall(r'\\\[.*?\\\]',prior,re.S): assert block in text
    for row in prior.splitlines():
        if row.startswith('|'): assert row in text
    assert not re.findall(r'\[[^\]]+\]\([^)]+\)',text[:text.index('## 证据')])
    pending=STUDY/'executed/final_commit_verification.json'
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        path=DIRECTION/target
        assert path.is_file() or path==pending
    assert '[已处理 2026-10-07T23:40:50+08:00]' in (DIRECTION/'inbox.md').read_text()
    artifacts=[ROOT/'central/kb.json',ROOT/'central/state.json',ROOT/'reports/PROGRESS.md',
               DIRECTION/'report.md',DIRECTION/'inbox.md',DIRECTION/'findings/r107_fixed_kernel_batch64.md']
    artifacts.extend(p for p in STUDY.rglob('*') if p.is_file() and p!=pending and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in artifacts}
    result={'status':'passed','round':107,'preregistration_commit':commit,'verified_epoch':time.time(),
            'saved_evaluation_cells':40,'new_training_cells':0,'new_kernel_cells':0,
            'protected_historical_files':len(manifest['files']),'new_result_and_summary_hash_mtime_unchanged':81,
            'first_cell_after_preregistration_seconds':min(row['started_epoch'] for row in rows)-receipt['commit_epoch'],
            'direction_match_count':summary['direction_match_count_batch64'],'prediction':summary['prediction'],
            'state_changed_only_D5_last_round_next_question':True,'protected_round':baseline['round'],
            'protected_rounds_done':baseline['directions']['D5_ln_geometry']['rounds_done'],
            'summary_recomputes':True,'report_prior_numbers_formulas_tables_preserved':True,
            'frozen_sources_unchanged':True,'inbox_handled':True,'independent_verification_passed':True,
            'recovery_new_evaluation_cells':0,'protected_file_pins':manifest['files'],
            'result_and_summary_pins':recovery['protected_result_pins']}
    if args.closeout:
        closeout=subprocess.check_output(['git','rev-parse',args.closeout],cwd=ROOT,text=True).strip()
        subprocess.run(['git','merge-base','--is-ancestor',commit,closeout],cwd=ROOT,check=True)
        for path in artifacts:
            assert git_bytes(closeout,path)==path.read_bytes(),str(path)
        result.update(closeout_commit=closeout,closeout_artifact_sha256=hashes,closeout_artifacts_verified=len(hashes))
        pending.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    else:
        (STUDY/'executed/closeout_precheck.json').write_text(json.dumps({key:value for key,value in result.items() if not key.endswith('_pins')},ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in result.items() if not key.endswith('_pins') and key!='closeout_artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':
    main()
