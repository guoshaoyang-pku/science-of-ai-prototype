#!/usr/bin/env python3
"""Read-only hash, denominator, source and upstream checks. No model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='require all release payload files')
    args = parser.parse_args()
    checked = missing_payload = 0
    for path in sorted((ROOT / 'evidence/manifests').glob('*.json')):
        manifest = json.loads(path.read_text())
        for row in manifest['files']:
            target = ROOT / row['path']
            if not target.exists() and row['delivery'] == 'release' and not args.full:
                missing_payload += 1
                continue
            if not target.is_file() or digest(target) != row['sha256']:
                raise RuntimeError('Missing or changed evidence: ' + row['path'])
            checked += 1
    state = json.loads((ROOT / 'science_program_v2/central/state.json').read_text())
    budget = sum(d['rounds_done'] for d in state['directions'].values())
    counted = sum(float(h.get('elapsed_sec') or 0) >= 120 for h in state['history'])
    if (state['round'], budget, counted) != (113, 93, 92):
        raise RuntimeError('Historical accounting snapshot changed')
    claims = json.loads((ROOT / 'science_program_v2/central/kb.json').read_text())
    if isinstance(claims, dict):
        claims = claims.get('claims', claims.get('kb', []))
    if len(claims) != 128:
        raise RuntimeError('Historical KB denominator changed')
    study = ROOT / 'evidence/d2_scaling_20261009/studies/n_sigma_tasks'
    summary = json.loads((study / 'summary.json').read_text())
    c = summary['counts']
    if (c['curve_evaluations'], c['interior_minima'], c['boundary_minima'], c['task_recipes']) != (3240, 3137, 103, 12):
        raise RuntimeError('D2 counts changed')
    if any(t['P1'] != 'refuted' for t in summary['tasks']):
        raise RuntimeError('D2 failed predictions changed')
    if sum(t['models']['linear']['expanded']['total'] for t in summary['tasks']) != 2700:
        raise RuntimeError('D2 expanded denominator changed')
    lock = json.loads((ROOT / 'upstream.lock.json').read_text())
    pin = subprocess.check_output(['git', '-C', str(ROOT / lock['path']), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != lock['commit']:
        raise RuntimeError('ArchitectureIQ submodule changed')
    print(json.dumps({'status': 'verified', 'files_checked': checked,
                      'release_files_not_downloaded': missing_payload,
                      'v2_rounds': [113, 93, 92], 'claims': 128,
                      'd2_counts': c, 'upstream_commit': pin}, ensure_ascii=False))


if __name__ == '__main__':
    main()
