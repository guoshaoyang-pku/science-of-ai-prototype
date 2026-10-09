#!/usr/bin/env python3
"""Prepare an external AIQ compatibility checkout from the pinned main commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    data=Path(os.environ.get('AIQ_KB_DATA_ROOT',str(ROOT.parent/'AIQ_KB_DATA'))).resolve()
    destination=(args.output or data/'dependencies/ArchitectureIQ-kb').resolve()
    if destination.exists() or destination.is_relative_to(ROOT):
        parser.error('use a new external checkout path')
    spec=json.loads((ROOT/'benchmark_compat/manifest.json').read_text())
    patch=ROOT/'benchmark_compat/architectureiq-kb-lab.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=spec['patch_sha256']:
        raise RuntimeError('Changed compatibility patch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(ROOT/'external/ArchitectureIQ'),str(destination)],check=True)
    subprocess.run(['git','-C',str(destination),'checkout','--detach',spec['upstream_commit']],check=True)
    subprocess.run(['git','-C',str(destination),'apply','--check',str(patch)],check=True)
    subprocess.run(['git','-C',str(destination),'apply',str(patch)],check=True)
    for row in spec['files']:
        if hashlib.sha256((destination/row['path']).read_bytes()).hexdigest()!=row['extension_sha256']:
            raise RuntimeError('Compatibility output mismatch: '+row['path'])
    (destination/'KB_COMPATIBILITY.json').write_text(json.dumps(spec,indent=2)+chr(10))
    print(json.dumps({'bench_root':str(destination),'upstream':spec['upstream_commit'],
                      'patched_files':len(spec['files']),'models_started':0,
                      'next':'export AIQ_BENCH_ROOT='+str(destination)},ensure_ascii=False))


if __name__=='__main__':
    main()
