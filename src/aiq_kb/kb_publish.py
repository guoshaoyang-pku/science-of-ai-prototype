#!/usr/bin/env python3
"""Publish saved KB/research checkpoints using the existing experiment viewer."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

if __package__:
    from .config import KBConfig
    from .build_run_pages import make_data, render
    from .kb_jobs import process_alive, process_identity
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import KBConfig
    from build_run_pages import make_data, render
    from kb_jobs import process_alive, process_identity

CONFIG = KBConfig.from_env()
SETTING_PATH = Path(__file__).with_name("KB_SETTING.md")

ORIGIN = 'https://guoshaoyang-pku.github.io/'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def git(blog, *args):
    env = os.environ.copy()
    if args[0] == 'push':
        env.setdefault('GIT_SSH_COMMAND', 'ssh -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2')
    return subprocess.run(['git', '-C', str(blog), *args], text=True,
                          capture_output=True, check=True, timeout=60, env=env).stdout.strip()


def validate_live(run, pool):
    proof = read_json(run / 'publication_provenance.json')
    if (proof.get('kind') != 'real-experiment' or proof.get('runtime_verified') is not True
            or proof.get('pool_sha256') != sha256(pool.read_bytes())
            or proof.get('config_sha256') != sha256((run / 'config.json').read_bytes())):
        raise ValueError('publication provenance does not match verified live inputs/config')
    if not (run / 'kb_live.json').is_file() and not list((run / 'kb').glob('kb_*.json')):
        raise ValueError('a real saved KB snapshot is required for publication')
    manifest_path = pool.parent / 'manifest.json'
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (proof.get('release_manifest_sha256') != sha256(manifest_path.read_bytes())
                or manifest.get('files', {}).get(pool.name) != sha256(pool.read_bytes())):
            raise ValueError('release manifest does not match verified publication input')
    if proof.get('lab_manifest_path'):
        lab_manifest = Path(proof['lab_manifest_path'])
        if proof.get('lab_manifest_sha256') != sha256(lab_manifest.read_bytes()):
            raise ValueError('lab manifest does not match verified publication input')
        lab = read_json(lab_manifest)
        if lab.get('pool', {}).get('sha256') != sha256(pool.read_bytes()):
            raise ValueError('lab manifest references a different release pool')
        for name, fingerprint in lab.get('files', {}).items():
            if Path(name).name != name or sha256((lab_manifest.parent / name).read_bytes()) != fingerprint:
                raise ValueError('lab artifact does not match verified publication input')
        for source in lab.get('splits', []):
            if sha256(Path(source['path']).read_bytes()) != source['sha256']:
                raise ValueError('lab split does not match verified publication input')


def ensure_publisher(run: Path, pool: Path, blog: Path, interval: float = 1800) -> dict:
    """Attach to the exact detached watcher or start one after a verified stop."""
    run, pool, blog = Path(run).resolve(), Path(pool).resolve(), Path(blog).resolve()
    validate_live(run, pool)
    if interval < 0:
        raise ValueError('publication interval must be nonnegative')
    descriptor = run / 'publisher_process.json'
    expected = {'run': str(run), 'pool': str(pool), 'blog': str(blog), 'interval': interval}
    with (run / 'publisher_manager.lock').open('a') as manager:
        fcntl.flock(manager, fcntl.LOCK_EX)
        previous = read_json(descriptor) if descriptor.exists() else {}
        alive = process_alive(previous.get('identity'))
        if alive is not False:
            if alive is None:
                return {**previous, 'status': 'observation_unavailable'}
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError('the surviving publisher has different inputs or interval')
            return {**previous, 'status': 'running'}
        for lock_name in ('publication_watch.lock', 'publication.lock'):
            with (run / lock_name).open('a') as worker_lock:
                try:
                    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return {'status': 'unidentified_publisher_survives', **expected}
        command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run), '--pool', str(pool),
                   '--blog', str(blog), '--interval', str(interval), '--watch', '--push']
        with (run / 'publisher.log').open('a') as log:
            process = subprocess.Popen(command, cwd=run, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        identity = process_identity(process.pid)
        state = {**expected, 'pid': process.pid, 'identity': identity, 'started_at': time.time(),
                 'restart_count': previous.get('restart_count', -1) + 1,
                 'status': 'started' if identity else 'stopped'}
        write_json(descriptor, state)
        return state


def generate(run, pool, temporary, *, include_test=True):
    data = make_data(run, pool, include_test=include_test)
    name = data['run']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('invalid run name for public paths')
    prefix = Path('blogs/kb_site/runs')
    files = {}
    source_map = {}
    for report in data.get('reports', {}).values():
        files[str(prefix / report['url'])] = report['text'].encode()
        for artifact in report.get('files', {}).values():
            content = artifact.pop('text').encode()
            original_url = artifact['url']
            fingerprint = sha256(content)
            url = '_sources/' + fingerprint + Path(original_url).suffix
            if len(content) > 256 * 1024:
                url += '.gz'
                if str(prefix / url) not in files:
                    files[str(prefix / url)] = gzip.compress(content, mtime=0)
                artifact['compression'] = 'gzip'
            else:
                files[str(prefix / url)] = content
            artifact.update(url=url, bytes=len(content), sha256=fingerprint)
            source_map[original_url] = dict(artifact)
    files[str(prefix / name / 'source-files.json')] = (
        json.dumps(source_map, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    source_path = run / 'publication_source.json'
    if source_path.is_file():
        source = read_json(source_path)
        commit = source['source_commit']
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('invalid source archive commit')
        for suffix in ('zip', 'json'):
            filename = 'source-' + commit[:7] + '.' + suffix
            item = source['files'][filename]
            path = Path(item['path'])
            if path.name != filename:
                raise ValueError('invalid source archive filename')
            raw = path.read_bytes()
            if {'sha256': sha256(raw), 'bytes': len(raw)} != {key: item[key] for key in ('sha256', 'bytes')}:
                raise ValueError('source archive bytes do not match saved hashes')
            files[str(prefix / name / filename)] = raw
        data['sourceArchive'] = {'commit': commit, 'zipURL': name + '/source-' + commit[:7] + '.zip',
                                 'manifestURL': name + '/source-' + commit[:7] + '.json'}
    page = temporary / (name + '.html')
    render(data, page)
    files[str(prefix / page.name)] = page.read_bytes()
    files[str(prefix / (name + '.json'))] = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    setting_path = run / 'publication_setting.md'
    setting = (setting_path if setting_path.is_file() else SETTING_PATH).read_bytes()
    files[str(prefix / name / 'setting.md')] = setting
    return data, files


def artifact_digest(files):
    return sha256('\n'.join(path + ':' + sha256(raw) for path, raw in sorted(files.items())).encode())


def deploy_bytes(blog, files):
    for relative, content in files.items():
        target = blog / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() == content:
            continue
        temporary = target.with_name('.' + target.name + '.' + str(os.getpid()) + '.tmp')
        temporary.write_bytes(content)
        os.replace(temporary, target)


def verify_online(files, timeout=240, verified=None):
    verified = verified if verified is not None else {}
    expected = {path: {'sha256': sha256(raw), 'bytes': len(raw)} for path, raw in files.items()}
    pending = {path for path, meta in expected.items() if verified.get(path) != meta}
    for path in pending:
        verified.pop(path, None)
    deadline = time.monotonic() + max(0, timeout)

    def check(relative):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        request = urllib.request.Request(ORIGIN + urllib.parse.quote(relative),
            headers={'Cache-Control': 'no-cache', 'User-Agent': 'KB-checkpoint-publisher',
                     'Accept-Encoding': 'gzip'})
        try:
            with urllib.request.urlopen(request, timeout=min(15, remaining)) as response:
                if response.status != 200 or time.monotonic() >= deadline:
                    return False
                headers = getattr(response, 'headers', {})
                decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS)
                           if headers.get('Content-Encoding', '').lower() == 'gzip' else None)
                if hasattr(response, 'read1'):
                    digest, length = hashlib.sha256(), 0
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False
                        connection = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
                        if connection is not None:
                            connection.settimeout(min(15, remaining))
                        chunk = response.read1(min(64 * 1024, expected[relative]['bytes'] - length + 1))
                        if not chunk:
                            if decoder is not None:
                                if not decoder.eof:
                                    return False
                                tail = decoder.flush()
                                length += len(tail)
                                digest.update(tail)
                            return length == expected[relative]['bytes'] and digest.hexdigest() == expected[relative]['sha256']
                        if decoder is not None:
                            chunk = decoder.decompress(chunk, expected[relative]['bytes'] - length + 1)
                            if decoder.unconsumed_tail:
                                return False
                        length += len(chunk)
                        if length > expected[relative]['bytes']:
                            return False
                        digest.update(chunk)
                raw = response.read()
                if decoder is not None:
                    raw = decoder.decompress(raw, expected[relative]['bytes'] + 1)
                    if not decoder.eof or decoder.unconsumed_tail:
                        return False
                    raw += decoder.flush()
                return len(raw) == expected[relative]['bytes'] and sha256(raw) == expected[relative]['sha256']
        except (OSError, urllib.error.URLError, zlib.error):
            return False

    with cf.ThreadPoolExecutor(max_workers=6, thread_name_prefix='kb-http-check') as workers:
        while pending and time.monotonic() < deadline:
            paths = iter(sorted(pending))
            active = {}
            exhausted = False
            while active or not exhausted:
                while len(active) < 6 and not exhausted and time.monotonic() < deadline:
                    relative = next(paths, None)
                    if relative is None:
                        exhausted = True
                        break
                    active[workers.submit(check, relative)] = relative
                if not active:
                    break
                remaining = max(0, deadline - time.monotonic())
                done, _ = cf.wait(active, timeout=remaining, return_when=cf.FIRST_COMPLETED)
                if not done:
                    break
                for future in done:
                    relative = active.pop(future)
                    if future.result():
                        verified[relative] = expected[relative]
                        pending.remove(relative)
                if time.monotonic() >= deadline:
                    break
            for future, relative in active.items():
                if not future.cancel() and future.result():
                    verified[relative] = expected[relative]
                    pending.remove(relative)
            if pending:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(40, remaining))
    return not pending


def committed_files(blog, commit, manifest):
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not manifest:
        raise ValueError('invalid pending publication commit or manifest')
    paths = sorted(manifest)
    for path in paths:
        relative = Path(path)
        if (relative.is_absolute() or '..' in relative.parts or '\n' in path
                or not path.startswith('blogs/kb_site/runs/')):
            raise ValueError('invalid pending publication path')
    request = ''.join(commit + ':' + path + '\n' for path in paths).encode()
    result = subprocess.run(['git', '-C', str(blog), 'cat-file', '--batch'], input=request,
                            capture_output=True, check=True, timeout=60)
    stream, files = io.BytesIO(result.stdout), {}
    for path in paths:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('missing pending publication bytes')
        raw = stream.read(int(header[2]))
        if (stream.read(1) != b'\n'
                or {'sha256': sha256(raw), 'bytes': len(raw)} != manifest[path]):
            raise ValueError('pending publication bytes do not match saved hashes')
        files[path] = raw
    return files


def push_main(blog):
    try:
        git(blog, 'push', 'origin', 'main')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        remote = git(blog, 'remote', 'get-url', 'origin')
        match = re.fullmatch(r'(?:git@github\.com:|ssh://git@(?:github\.com|ssh\.github\.com)(?::443)?/)'
                             r'([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', remote)
        if not match:
            raise
        repository = match[1].removesuffix('.git')
        env = os.environ.copy()
        env.update(GIT_TERMINAL_PROMPT='0', GIT_ASKPASS='/usr/bin/false')
        subprocess.run(['git', '-C', str(blog), '-c', 'credential.interactive=false',
                        '-c', 'http.connectTimeout=10', '-c', 'http.lowSpeedLimit=1000',
                        '-c', 'http.lowSpeedTime=15', 'push',
                        'https://github.com/' + repository + '.git', 'HEAD:refs/heads/main'],
                       text=True, capture_output=True, check=True, timeout=60, env=env)


def push_files(blog, files, run_name, on_commit=None):
    git_dir = Path(git(blog, 'rev-parse', '--absolute-git-dir'))
    with (git_dir / 'kb-publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        staged = set(git(blog, 'diff', '--cached', '--name-only').splitlines())
        if staged - set(files):
            raise ValueError('git index contains unrelated staged files')
        if git(blog, 'branch', '--show-current') != 'main':
            raise ValueError('public blog checkout must be on main')
        deploy_bytes(blog, files)
        with tempfile.NamedTemporaryFile(prefix='kb-publish-paths-') as paths:
            paths.write(b''.join(path.encode() + b'\0' for path in sorted(files)))
            paths.flush()
            pathspec = ('--pathspec-from-file=' + paths.name, '--pathspec-file-nul')
            git(blog, 'add', *pathspec)
            if git(blog, 'diff', '--cached', '--name-only'):
                git(blog, 'commit', '-m', 'blog(kb_site): update saved checkpoint for ' + run_name,
                    '--only', *pathspec)
        commit = git(blog, 'rev-parse', 'HEAD')
        if on_commit is not None:
            on_commit(commit)
        push_main(blog)
    return commit


def publish_once(run, pool, blog, interval=1800, push=False, verify_timeout=240):
    run, pool, blog = run.resolve(), pool.resolve(), blog.resolve()
    state_path = run / 'publication_state.json'
    state = read_json(state_path) if state_path.exists() else {}
    if push:
        validate_live(run, pool)
    elif (blog / '.git').exists():
        raise ValueError('offline export requires a preview directory; use --push for a blog checkout')
    pending = state.get('pending_publication') if push else None
    if pending:
        files = committed_files(blog, pending['commit'], pending['file_manifest'])
        git_dir = Path(git(blog, 'rev-parse', '--absolute-git-dir'))
        with (git_dir / 'kb-publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if git(blog, 'branch', '--show-current') != 'main':
                raise ValueError('public blog checkout must be on main')
            git(blog, 'merge-base', '--is-ancestor', pending['commit'], 'HEAD')
            push_main(blog)
            pushed_head = git(blog, 'rev-parse', 'HEAD')
        manifest = pending['file_manifest']
        verified_files = {path: meta for path, meta in state.get('verified_files', {}).items()
                          if manifest.get(path) == meta}
        state.update(pushed_digest=pending['digest'], pushed_at=time.time(), git_commit=pending['commit'],
                     pushed_head=pushed_head, deployment_verified=False, file_manifest=manifest,
                     verified_files=verified_files)
        state.pop('pending_publication')
        write_json(state_path, state)
        verified = verify_online(files, verify_timeout, verified_files)
        state.update(deployment_verified=verified, verified_files=verified_files)
        write_json(state_path, state)
        return {'status': 'deployed' if verified else 'build_pending', 'run': run.name,
                'changed': False, 'resumed_push': True, 'commit': pending['commit'],
                'url': ORIGIN + 'blogs/kb_site/runs/' + run.name + '.html'}
    if push and state.get('file_manifest') and time.time() - state.get('pushed_at', 0) < interval:
        if state.get('deployment_verified') is False:
            manifest = state['file_manifest']
            verified_files = {path: meta for path, meta in state.get('verified_files', {}).items()
                              if manifest.get(path) == meta}
            unverified = {path: meta for path, meta in manifest.items()
                          if verified_files.get(path) != meta}
            files = committed_files(blog, state['git_commit'], unverified) if unverified else {}
            verify_online(files, verify_timeout, verified_files)
            state.update(verified_files=verified_files,
                         deployment_verified=all(verified_files.get(path) == meta for path, meta in manifest.items()))
            write_json(state_path, state)
        return {'status': 'throttled', 'run': run.name, 'changed': False,
                'verification_pending': len(state['file_manifest']) - len(state.get('verified_files', {}))}
    with tempfile.TemporaryDirectory(prefix='kb-publish-') as temp:
        data, files = generate(run, pool, Path(temp))
    current = artifact_digest(files)
    manifest = {path: {'sha256': sha256(raw), 'bytes': len(raw)} for path, raw in files.items()}
    verified_files = {path: meta for path, meta in state.get('verified_files', {}).items()
                      if manifest.get(path) == meta}
    last = state.get('pushed_digest' if push else 'exported_digest')
    if current == last:
        if push and state.get('deployment_verified') is False:
            verified = verify_online(files, verify_timeout, verified_files)
            state.update(deployment_verified=verified, verified_files=verified_files, file_manifest=manifest)
            write_json(state_path, state)
            return {'status': 'deployed' if verified else 'build_pending', 'run': data['run'], 'changed': False}
        return {'status': 'unchanged', 'run': data['run'], 'changed': False}
    if push and time.time() - state.get('pushed_at', 0) < interval:
        if state.get('deployment_verified') is False and state.get('file_manifest'):
            published = state['file_manifest']
            prior_verified = {path: meta for path, meta in state.get('verified_files', {}).items()
                              if published.get(path) == meta}
            pending_files = {}
            for path, meta in published.items():
                if prior_verified.get(path) == meta:
                    continue
                relative = Path(path)
                if relative.is_absolute() or '..' in relative.parts:
                    continue
                target = blog / relative
                if target.is_file():
                    raw = target.read_bytes()
                    if {'sha256': sha256(raw), 'bytes': len(raw)} == meta:
                        pending_files[path] = raw
            verify_online(pending_files, verify_timeout, prior_verified)
            state.update(verified_files=prior_verified,
                         deployment_verified=all(prior_verified.get(path) == meta for path, meta in published.items()))
            write_json(state_path, state)
        return {'status': 'throttled', 'run': data['run'], 'changed': True}
    if not push:
        deploy_bytes(blog, files)
    state.update(exported_digest=current, exported_at=time.time(), files=sorted(files))
    result = {'status': 'exported', 'run': data['run'], 'changed': True,
              'files': sorted(files), 'snapshots': len(data['snaps']), 'reports': len(data.get('reports', {}))}
    if push:
        def save_pending(commit):
            state['pending_publication'] = {'commit': commit, 'digest': current,
                                           'file_manifest': manifest, 'created_at': time.time()}
            write_json(state_path, state)
        commit = push_files(blog, files, data['run'], on_commit=save_pending)
        state.pop('pending_publication', None)
        state.update(pushed_digest=current, pushed_at=time.time(), git_commit=commit, deployment_verified=False,
                     file_manifest=manifest, verified_files=verified_files)
        write_json(state_path, state)
        verified = verify_online(files, verify_timeout, verified_files)
        state.update(deployment_verified=verified, verified_files=verified_files)
        result.update(status='deployed' if verified else 'build_pending', commit=commit,
                      url=ORIGIN + 'blogs/kb_site/runs/' + data['run'] + '.html')
    write_json(state_path, state)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--blog', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1800)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--verify-timeout', type=float, default=240)
    args = parser.parse_args(argv)
    if args.interval < 0 or args.verify_timeout < 0:
        parser.error('interval and verify timeout must be nonnegative')
    if not args.run.is_dir():
        parser.error('run directory does not exist')
    lock_name = 'publication_watch.lock' if args.watch else 'publication.lock'
    with (args.run / lock_name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status': 'another_publisher_is_running'}), flush=True)
            return 0
        while True:
            try:
                if args.watch:
                    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(args.run),
                               '--pool', str(args.pool), '--blog', str(args.blog), '--once',
                               '--interval', str(args.interval), '--verify-timeout', str(args.verify_timeout)]
                    if args.push:
                        command.append('--push')
                    subprocess.run(command, check=False)
                else:
                    result = publish_once(args.run, args.pool, args.blog, args.interval,
                                          args.push, args.verify_timeout)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                result = {'status': 'retryable_error', 'error_type': type(error).__name__}
                print(json.dumps(result), flush=True)
                if not args.watch:
                    return 1
            else:
                if not args.watch:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.watch:
                return 0
            time.sleep(min(40, max(1, args.interval)))


if __name__ == '__main__':
    raise SystemExit(main())
