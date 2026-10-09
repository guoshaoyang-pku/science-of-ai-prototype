#!/usr/bin/env python3
"""Create a parked, independent research run; never starts an agent."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'science_program_v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--direction', required=True, choices=sorted(p.name for p in (SOURCE / 'directions').iterdir() if p.is_dir()))
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name) or args.rounds < 1:
        parser.error('run-name must be a path-free name and rounds must be positive')
    data = Path(os.environ.get('AIQ_KB_DATA_ROOT', str(ROOT.parent / 'AIQ_KB_DATA'))).resolve()
    run = data / 'runs' / args.run_name
    if run.is_relative_to(ROOT) or run.exists():
        parser.error('use a new run directory outside the source checkout')
    run.mkdir(parents=True)
    (run / 'central').mkdir()
    shutil.copy2(SOURCE / 'central/supervisor.py', run / 'central/supervisor.py')
    shutil.copytree(SOURCE / 'central/style', run / 'central/style')
    shutil.copy2(SOURCE / 'central/kb.json', run / 'central/kb.json')
    shutil.copy2(SOURCE / 'GOAL.md', run / 'GOAL.md')
    shutil.copy2(SOURCE / 'AGENTS.md', run / 'AGENTS.md')
    (run / 'references').symlink_to(ROOT / 'references', target_is_directory=True)
    (run / 'history').symlink_to(ROOT / 'evidence/history', target_is_directory=True)
    directions = run / 'directions'
    directions.mkdir()
    for source in (SOURCE / 'directions').iterdir():
        if not source.is_dir():
            continue
        target = directions / source.name
        target.mkdir()
        (target / 'studies').mkdir()
        for file in source.iterdir():
            if file.is_file() and file.suffix == '.md':
                if file.name == 'inbox.md':
                    shutil.copy2(file, target / 'inbox_historical.md')
                    (target / 'inbox.md').write_text('# New run instructions' + chr(10))
                else:
                    text = file.read_text().replace('../../../references/science_program_v1/', '../../references/science_program_v1/')
                    (target / file.name).write_text(text)
        if (source / 'findings').is_dir():
            shutil.copytree(source / 'findings', target / 'findings')
        if (source / 'figs').is_dir():
            (target / 'figs').symlink_to(source / 'figs', target_is_directory=True)
        if (source / 'studies').is_dir():
            for old_study in (source / 'studies').iterdir():
                (target / 'studies' / old_study.name).symlink_to(old_study, target_is_directory=old_study.is_dir())
    shutil.copytree(SOURCE / 'sources', run / 'sources')
    (run / 'latest_d2').symlink_to(ROOT / 'evidence/d2_scaling_20261009', target_is_directory=True)
    if args.direction == 'D2_overfitting_u_curve':
        (directions / args.direction / 'inbox.md').write_text(
            '# Current D2 evidence' + chr(10) + chr(10)
            + 'Read latest_d2/report.md and latest_d2/studies/n_sigma_tasks/summary.json before selecting a question. '
            + 'The 2026-10-09 expanded scan supersedes the older report and central KB for D2. '
            + 'Do not treat the fixed-exponent law or inverse-noise minimum as established. '
            + 'Preserve the old r113 aggregate failure and the new boundary denominators.' + chr(10))
    cfg = json.loads((SOURCE / 'central/config.json').read_text())
    cfg.update(rounds_per_direction=args.rounds, round_overrides={}, max_total_rounds=args.rounds,
               direction_order=[args.direction], python_bin=sys.executable)
    (run / 'central/config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + chr(10))
    state = {'program': 'science_program_v2', 'status': 'parked', 'round': 0,
             'started_at': None, 'supervisor_pid': None, 'history': [],
             'directions': {args.direction: {'status': 'active', 'rounds_done': 0,
                                             'last_round': None, 'next_question': None}}}
    (run / 'central/state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + chr(10))
    (run / 'central/STOP').write_text('Review config and remove STOP manually before opt-in.' + chr(10))
    (run / 'reports').mkdir()
    (run / 'reports/PROGRESS.md').write_text('# New research run' + chr(10))
    ignored = ['central/rounds/**/output.log', '__pycache__/', '.cache/', 'central/supervisor.lock',
               'references', 'history', 'latest_d2']
    ignored.extend(str(p.relative_to(run)) for p in run.rglob('*') if p.is_symlink())
    (run / '.gitignore').write_text(chr(10).join(sorted(set(ignored))) + chr(10))
    receipt = {'source': str(ROOT), 'selected_direction': args.direction, 'round_budget': args.rounds,
               'historical_state': 'source science_program_v2/central/state.json; not imported as new accounting',
               'historical_evidence': 'symlinks; read only by research contract', 'models_started': 0}
    (run / 'handoff.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + chr(10))
    subprocess.run(['git', 'init', '-b', 'main', str(run)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', '-C', str(run), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(run), 'commit', '-m', 'Prepare parked science run from handoff'], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({'run': str(run), 'direction': args.direction, 'rounds': args.rounds,
                      'models_started': 0, 'next': 'Review config, remove central/STOP, then run SCIENCE_ENABLE_AGENT=1 python central/supervisor.py from the new run.'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
