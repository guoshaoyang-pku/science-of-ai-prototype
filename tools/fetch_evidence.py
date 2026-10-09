#!/usr/bin/env python3
"""Verify release archives and extract only allowlisted, immutable payloads."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--group', choices=('v1', 'v2', 'd2'), help='verify/extract one bundle')
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((ROOT / 'evidence/handoff_manifest.json').read_text())
    for item in manifest['archives']:
        if args.group and not item['manifest'].endswith('/' + args.group + '.json'):
            continue
        path = args.archives / item['name']
        if not path.exists() and item.get('parts'):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.assembling')
            if partial.exists():
                raise RuntimeError('Partial reassembly requires review: ' + partial.name)
            try:
                with partial.open('xb') as assembled:
                    for part in item['parts']:
                        segment = args.archives / part['name']
                        if not segment.is_file() or segment.stat().st_size != part['size'] or digest(segment) != part['sha256']:
                            raise RuntimeError('Missing or changed archive part: ' + part['name'])
                        with segment.open('rb') as source:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                assembled.write(block)
                if digest(partial) != item['sha256']:
                    raise RuntimeError('Reassembled archive hash mismatch: ' + item['name'])
                partial.replace(path)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        if not path.is_file() or digest(path) != item['sha256']:
            raise RuntimeError('Missing or changed archive: ' + item['name'])
        group = json.loads((ROOT / item['manifest']).read_text())
        allowed = {r['path']: r for r in group['files'] if r['delivery'] == 'release'}
        seen = set()
        with tarfile.open(path, mode='r|gz') as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                        or member.name not in allowed or member.name in seen):
                    raise RuntimeError('Invalid archive member: ' + member.name)
                seen.add(member.name)
                source = archive.extractfile(member)
                h = hashlib.sha256()
                destination = output / member.name
                if not destination.resolve().is_relative_to(output):
                    raise RuntimeError('Output path escapes root: ' + member.name)
                existing = destination.exists()
                temporary = None
                if not args.verify_only and not existing:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + '.extracting')
                    sink = temporary.open('xb')
                else:
                    sink = None
                try:
                    with source:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            h.update(block)
                            if sink:
                                sink.write(block)
                finally:
                    if sink:
                        sink.close()
                row = allowed[member.name]
                if h.hexdigest() != row['sha256'] or member.size != row['bytes']:
                    if temporary:
                        temporary.unlink(missing_ok=True)
                    raise RuntimeError('Changed payload: ' + member.name)
                if existing and not args.verify_only and digest(destination) != row['sha256']:
                    raise RuntimeError('Refusing to overwrite changed file: ' + member.name)
                if temporary:
                    temporary.replace(destination)
        if seen != set(allowed):
            raise RuntimeError('Archive membership mismatch: ' + item['name'])
        print(json.dumps({'archive': item['name'], 'files': len(seen), 'verified': True,
                          'extracted': not args.verify_only}), flush=True)


if __name__ == '__main__':
    main()
