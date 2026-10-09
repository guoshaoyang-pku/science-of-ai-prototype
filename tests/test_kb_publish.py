import importlib.util
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'src/aiq_kb/kb_publish.py'
from aiq_kb import kb_publish as publish


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'synthetic_test_only'
        self.blog = self.root / 'blog'
        self.run.mkdir()
        self.blog.mkdir()
        (self.blog / 'index.html').write_text('personal homepage sentinel')
        self.pool = self.root / 'questions.jsonl'
        self.pool.write_text(json.dumps({'question_id': 'q1', 'messages': [{'role': 'user', 'content': 'Synthetic fixture'}]}) + '\n')
        config = {'run_name': self.run.name, 'model': 'test-model', 'progressive_disclosure': True,
                  'api_key': 'PRIVATE_VALUE', 'nested': {'authorization': 'PRIVATE_VALUE'}}
        (self.run / 'config.json').write_text(json.dumps(config))
        repo = self.run / 'jobs/J1/repo'
        repo.mkdir(parents=True)
        subprocess.run(['git', 'init', '-b', 'research'], cwd=repo, check=True, capture_output=True)
        (repo / 'report.md').write_text('# Continuing science\nCompeting explanations remain unresolved.')
        (repo / 'measure.py').write_text('print("synthetic_test_only")\n')
        (repo / 'secret.json').write_text('{"api_key":"PRIVATE_VALUE"}')
        subprocess.run(['git', 'add', '.'], cwd=repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@local', 'commit', '-m', 'test-only'],
                       cwd=repo, check=True, capture_output=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        report_dir = self.run / 'reports/R1'
        report_dir.mkdir(parents=True)
        (report_dir / (commit + '.md')).write_text((repo / 'report.md').read_text())
        report = {'id': 'R1', 'title': 'Test report', 'topic': 'synthetic_test_only', 'claim_ids': ['K1001'],
                  'repo_commit': commit, 'origin_epoch': 3, 'job_id': 'J1', 'repo': str(repo),
                  'path': str(report_dir / (commit + '.md'))}
        snapshot = {'epoch': 3, 'claims': [{'id': 'K1001', 'text': 'Observed comparison', 'support_count': 1,
                     'failure_count': 0, 'created_epoch': 3, 'report_ids': ['R1']}], 'reports': {'R1': report}}
        (self.run / 'kb_live.json').write_text(json.dumps(snapshot))
        snapshots = self.run / 'solver_snapshots'
        snapshots.mkdir()
        (snapshots / 'frozen_e0004.json').write_text(json.dumps(snapshot))
        job = {'id': 'J1', 'kind': 'research', 'origin_epoch': 3, 'current_epoch': 4, 'member_epoch': 4,
               'label': 'from e3', 'status': 'running', 'payload': {'api_key': 'PRIVATE_VALUE'},
               'session_id': 'PRIVATE_VALUE', 'error': 'PRIVATE_VALUE'}
        (self.run / 'jobs/J1/job.json').write_text(json.dumps(job))
        epoch = self.run / 'epochs/e0004'
        (epoch / 'solves').mkdir(parents=True)
        record = {'question_id': 'q1', 'source': 'arch170', 'score': 1, 'solver_snapshot': 'frozen_e0004',
                  'answer': 'A', 'prediction': 'A', 'solution': 'A', 'cited': ['K1001'],
                  'discover_trace': [{'content': 'done', 'calls': [{'name': 'python', 'arguments': '{"api_key":"PRIVATE_VALUE"}'}],
                    'results': ['Authorization=PRIVATE_VALUE'], 'usage': {}}]}
        (epoch / 'records.jsonl').write_text(json.dumps(record) + '\n' + '{"partial":')
        (self.run / 'commits.jsonl').write_text(json.dumps({'commit': 'C1', 'epoch': 4, 'origin_epoch': 3,
                 'applied_epoch': 4, 'op': 'async_apply', 'payload': {'operation': {'op': 'report', 'reason': 'test'}}}) + '\n')

    def tearDown(self):
        self.temp.cleanup()

    def test_excluding_test_never_reads_test_results_or_writes_run_state(self):
        from aiq_kb import build_run_pages

        test_report = self.run / 'test_report.json'
        test_report.write_text('unreadable final-test sentinel')
        evaluation = self.run / 'eval'
        for label in ('test_kb', 'test_nokb', 'kb_0004'):
            folder = evaluation / label
            folder.mkdir(parents=True)
            (folder / '_summary.json').write_text(
                'unreadable final-test sentinel' if label.startswith('test')
                else json.dumps({'label': label, 'mean': 0.75}))
        before = {p.relative_to(self.run): p.read_bytes()
                  for p in self.run.rglob('*') if p.is_file()}
        original_read = build_run_pages.read_json

        def guarded_read(path, default=None):
            if path == test_report or path.parent.name.startswith('test'):
                raise AssertionError('final-test result was read')
            return original_read(path, default)

        output = self.root / 'readonly_output'
        output.mkdir()
        with patch.object(build_run_pages, 'read_json', side_effect=guarded_read):
            data, files = publish.generate(self.run, self.pool, output, include_test=False)
        self.assertEqual(data['test'], {})
        self.assertEqual(set(data['evals']), {'kb_0004'})
        self.assertIn('blogs/kb_site/runs/synthetic_test_only.html', files)
        self.assertNotIn('unreadable final-test sentinel',
                         files['blogs/kb_site/runs/synthetic_test_only.json'].decode())
        after = {p.relative_to(self.run): p.read_bytes()
                 for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

        test_report.write_text(json.dumps({'test_only': True}))
        for label in ('test_kb', 'test_nokb'):
            (evaluation / label / '_summary.json').write_text(json.dumps({'label': label}))
        default_data = build_run_pages.make_data(self.run, self.pool)
        self.assertEqual(default_data['test'], {'test_only': True})
        self.assertEqual(set(default_data['evals']), {'test_kb', 'test_nokb', 'kb_0004'})

    def test_export_preserves_reports_and_frozen_snapshots_without_private_state(self):
        result = publish.publish_once(self.run, self.pool, self.blog)
        self.assertEqual(result['status'], 'exported')
        raw = (self.blog / 'blogs/kb_site/runs/synthetic_test_only.json').read_text()
        self.assertNotIn('PRIVATE_VALUE', raw)
        self.assertNotIn('session_id', raw)
        self.assertNotIn('"payload": {"api_key"', raw)
        data = json.loads(raw)
        self.assertIn('frozen_e0004', {s['label'] for s in data['snaps']})
        self.assertEqual(len(data['epochs'][0]['records']), 1)
        report = next(iter(data['reports'].values()))
        self.assertIn('Competing explanations', report['text'])
        self.assertIn('measure.py', report['files'])
        self.assertNotIn('secret.json', report['files'])
        self.assertTrue((self.blog / 'blogs/kb_site/runs' / report['url']).is_file())
        self.assertEqual((self.blog / 'index.html').read_text(), 'personal homepage sentinel')
        second = publish.publish_once(self.run, self.pool, self.blog)
        self.assertEqual(second['status'], 'unchanged')

    def test_push_requires_live_provenance_and_missing_report_keeps_previous(self):
        publish.publish_once(self.run, self.pool, self.blog)
        with self.assertRaises(FileNotFoundError):
            publish.publish_once(self.run, self.pool, self.blog, push=True)
        (self.run / 'publication_provenance.json').write_text(json.dumps({'kind': 'synthetic_test_only', 'runtime_verified': True}))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            publish.publish_once(self.run, self.pool, self.blog, push=True)
        existing = (self.blog / 'blogs/kb_site/runs/synthetic_test_only.html').read_bytes()
        next((self.run / 'reports/R1').glob('*.md')).unlink()
        with self.assertRaisesRegex(ValueError, 'missing report'):
            publish.publish_once(self.run, self.pool, self.blog)
        self.assertEqual((self.blog / 'blogs/kb_site/runs/synthetic_test_only.html').read_bytes(), existing)

    def test_pinned_source_archive_is_published_without_rebuilding_bytes(self):
        commit = 'a' * 40
        directory = self.root / 'saved_source'
        directory.mkdir()
        files = {}
        for suffix, raw in (('zip', b'exact saved source archive'), ('json', b'{"source_commit":"saved"}')):
            filename = 'source-' + commit[:7] + '.' + suffix
            path = directory / filename
            path.write_bytes(raw)
            files[filename] = {'path': str(path), 'sha256': publish.sha256(raw), 'bytes': len(raw)}
        publish.write_json(self.run / 'publication_source.json', {'source_commit': commit, 'files': files})
        result = publish.publish_once(self.run, self.pool, self.blog)
        page_data = json.loads((self.blog / ('blogs/kb_site/runs/' + self.run.name + '.json')).read_text())
        self.assertEqual(page_data['sourceArchive']['commit'], commit)
        self.assertEqual(page_data['sourceArchive']['zipURL'], self.run.name + '/source-aaaaaaa.zip')
        self.assertNotIn(str(directory), json.dumps(page_data))
        for filename, item in files.items():
            relative = 'blogs/kb_site/runs/' + self.run.name + '/' + filename
            self.assertIn(relative, result['files'])
            self.assertEqual((self.blog / relative).read_bytes(), Path(item['path']).read_bytes())
        state = publish.read_json(self.run / 'publication_state.json')
        old_digest = state['exported_digest']
        source = Path(files['source-aaaaaaa.zip']['path'])
        source.write_bytes(b'changed after saved source hash')
        with self.assertRaisesRegex(ValueError, 'saved hashes'):
            publish.publish_once(self.run, self.pool, self.blog)
        self.assertEqual(publish.read_json(self.run / 'publication_state.json')['exported_digest'], old_digest)
        self.assertEqual((self.blog / 'index.html').read_text(), 'personal homepage sentinel')

    def test_changed_content_respects_push_interval(self):
        publish.publish_once(self.run, self.pool, self.blog)
        state = publish.read_json(self.run / 'publication_state.json')
        state.update(pushed_digest=state['exported_digest'], pushed_at=publish.time.time())
        publish.write_json(self.run / 'publication_state.json', state)
        snapshot = publish.read_json(self.run / 'kb_live.json')
        snapshot['claims'][0]['text'] += ' revised'
        (self.run / 'kb_live.json').write_text(json.dumps(snapshot))
        with patch.object(publish, 'validate_live'), patch.object(publish, 'push_files') as push:
            result = publish.publish_once(self.run, self.pool, self.blog, push=True)
        self.assertEqual(result['status'], 'throttled')
        push.assert_not_called()

    def test_git_push_stages_only_intended_files_to_local_remote(self):
        remote = self.root / 'local-only.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
        subprocess.run(['git', 'init', '-b', 'main'], cwd=self.blog, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.blog, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@local'], cwd=self.blog, check=True)
        subprocess.run(['git', 'add', 'index.html'], cwd=self.blog, check=True)
        subprocess.run(['git', 'commit', '-m', 'base'], cwd=self.blog, check=True, capture_output=True)
        subprocess.run(['git', 'remote', 'add', 'origin', str(remote)], cwd=self.blog, check=True)
        (self.blog / 'unrelated.txt').write_text('leave untracked')
        files = {'blogs/kb_site/runs/test-only.md': b'synthetic_test_only\n'}
        commit = publish.push_files(self.blog, files, 'synthetic_test_only')
        tracked = subprocess.check_output(['git', 'ls-tree', '--name-only', '-r', 'HEAD'], cwd=self.blog, text=True)
        self.assertIn('index.html', tracked)
        self.assertIn('blogs/kb_site/runs/test-only.md', tracked)
        self.assertNotIn('unrelated.txt', tracked)
        remote_head = subprocess.check_output(['git', '--git-dir', str(remote), 'rev-parse', 'main'], text=True).strip()
        self.assertEqual(commit, remote_head)
        subprocess.run(['git', 'add', 'unrelated.txt'], cwd=self.blog, check=True)
        before = publish.git(self.blog, 'rev-parse', 'HEAD')
        with self.assertRaisesRegex(ValueError, 'unrelated staged'):
            publish.push_files(self.blog, files, 'synthetic_test_only')
        self.assertEqual(publish.git(self.blog, 'rev-parse', 'HEAD'), before)

    def test_large_publication_uses_a_nul_pathspec_file(self):
        remote = self.root / 'local-only.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
        subprocess.run(['git', 'init', '-b', 'main'], cwd=self.blog, check=True, capture_output=True)
        publish.git(self.blog, 'config', 'user.name', 'Test')
        publish.git(self.blog, 'config', 'user.email', 'test@local')
        publish.git(self.blog, 'add', 'index.html')
        publish.git(self.blog, 'commit', '-m', 'base')
        publish.git(self.blog, 'remote', 'add', 'origin', str(remote))
        files = {f"blogs/kb_site/runs/_sources/{i:05d}-" + 'a' * 220 + '.md': b'fixture\n'
                 for i in range(4500)}
        files['blogs/kb_site/runs/name with spaces\nand newline.md'] = b'exact path\n'
        self.assertGreater(sum(len(path.encode()) + 1 for path in files), 1024 * 1024)
        (self.blog / 'unrelated.txt').write_text('leave untracked')
        commit = publish.push_files(self.blog, files, 'synthetic_large_publication')
        tracked = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', 'HEAD'],
                                          cwd=self.blog).split(b'\0')
        self.assertTrue({path.encode() for path in files}.issubset(set(tracked)))
        self.assertNotIn(b'unrelated.txt', tracked)
        remote_head = subprocess.check_output(['git', '--git-dir', str(remote), 'rev-parse', 'main'],
                                              text=True).strip()
        self.assertEqual(commit, remote_head)

    def test_failed_push_resumes_saved_commit_before_generating_new_content(self):
        remote = self.root / 'local-only.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
        subprocess.run(['git', 'init', '-b', 'main'], cwd=self.blog, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.blog, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@local'], cwd=self.blog, check=True)
        subprocess.run(['git', 'add', 'index.html'], cwd=self.blog, check=True)
        subprocess.run(['git', 'commit', '-m', 'base'], cwd=self.blog, check=True, capture_output=True)
        subprocess.run(['git', 'remote', 'add', 'origin', str(remote)], cwd=self.blog, check=True)
        publish.git(self.blog, 'push', 'origin', 'main')
        viewer = 'blogs/kb_site/runs/' + self.run.name + '.html'
        files = {viewer: b'saved first generation\n'}
        data = {'run': self.run.name, 'snaps': [], 'reports': {}}
        original_git = publish.git
        def disconnected(blog, *args):
            if args[0] == 'push':
                raise subprocess.CalledProcessError(128, ['git', 'push'])
            return original_git(blog, *args)
        with patch.object(publish, 'validate_live'), patch.object(publish, 'generate', return_value=(data, files)), \
                patch.object(publish, 'git', side_effect=disconnected):
            with self.assertRaises(subprocess.CalledProcessError):
                publish.publish_once(self.run, self.pool, self.blog, interval=0, push=True)
        state = publish.read_json(self.run / 'publication_state.json')
        saved_commit = state['pending_publication']['commit']
        self.assertNotIn('pushed_digest', state)
        self.assertEqual(publish.committed_files(self.blog, saved_commit, state['pending_publication']['file_manifest']), files)
        (self.blog / 'unrelated.txt').write_text('concurrent publication sentinel')
        publish.git(self.blog, 'add', 'unrelated.txt')
        publish.git(self.blog, 'commit', '-m', 'concurrent content')
        newest = publish.git(self.blog, 'rev-parse', 'HEAD')
        with patch.object(publish, 'validate_live'), patch.object(publish, 'generate') as generate, \
                patch.object(publish, 'verify_online', return_value=True) as verify:
            result = publish.publish_once(self.run, self.pool, self.blog, interval=0, push=True)
        generate.assert_not_called()
        self.assertEqual(verify.call_args.args[0], files)
        self.assertTrue(result['resumed_push'])
        after = publish.read_json(self.run / 'publication_state.json')
        self.assertNotIn('pending_publication', after)
        self.assertEqual(after['git_commit'], saved_commit)
        self.assertEqual(after['pushed_head'], newest)
        self.assertEqual(after['pushed_digest'], publish.artifact_digest(files))
        self.assertEqual(publish.git(self.blog, 'rev-parse', 'HEAD'), newest)
        remote_head = subprocess.check_output(['git', '--git-dir', str(remote), 'rev-parse', 'main'], text=True).strip()
        self.assertEqual(remote_head, newest)
        self.assertEqual((self.blog / 'unrelated.txt').read_text(), 'concurrent publication sentinel')

    def test_pending_commit_hash_corruption_blocks_push_and_keeps_checkpoint(self):
        subprocess.run(['git', 'init', '-b', 'main'], cwd=self.blog, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.blog, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@local'], cwd=self.blog, check=True)
        viewer = 'blogs/kb_site/runs/' + self.run.name + '.html'
        publish.deploy_bytes(self.blog, {viewer: b'saved'})
        publish.git(self.blog, 'add', viewer)
        publish.git(self.blog, 'commit', '-m', 'saved')
        commit = publish.git(self.blog, 'rev-parse', 'HEAD')
        pending = {'commit': commit, 'digest': 'saved',
                   'file_manifest': {viewer: {'sha256': 'wrong', 'bytes': 5}}}
        publish.write_json(self.run / 'publication_state.json', {'pending_publication': pending})
        with patch.object(publish, 'validate_live'), patch.object(publish, 'generate') as generate:
            with self.assertRaisesRegex(ValueError, 'saved hashes'):
                publish.publish_once(self.run, self.pool, self.blog, push=True)
        generate.assert_not_called()
        self.assertEqual(publish.read_json(self.run / 'publication_state.json')['pending_publication'], pending)

    def test_git_push_has_bounded_timeout_and_no_permanent_transport_change(self):
        result = subprocess.CompletedProcess([], 0, stdout='ok\n', stderr='')
        with patch.object(publish.subprocess, 'run', return_value=result) as run, \
                patch.dict(publish.os.environ, {}, clear=True):
            self.assertEqual(publish.git(self.blog, 'push', 'origin', 'main'), 'ok')
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs['timeout'], 60)
            self.assertIn('ConnectTimeout=10', kwargs['env']['GIT_SSH_COMMAND'])
            self.assertNotIn('GIT_SSH_COMMAND', publish.os.environ)

    def test_github_ssh_failure_falls_back_to_exact_repo_without_prompting(self):
        def github(blog, *args):
            if args[0] == 'push':
                raise subprocess.TimeoutExpired(['git', 'push'], 60)
            return 'ssh://git@ssh.github.com:443/example/saved-blog.git'
        with patch.object(publish, 'git', side_effect=github), \
                patch.object(publish.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            publish.push_main(self.blog)
        self.assertEqual(run.call_args.args[0][-2:],
                         ['https://github.com/example/saved-blog.git', 'HEAD:refs/heads/main'])
        self.assertEqual(run.call_args.kwargs['timeout'], 60)
        self.assertEqual(run.call_args.kwargs['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(run.call_args.kwargs['env']['GIT_ASKPASS'], '/usr/bin/false')
        def other_host(blog, *args):
            if args[0] == 'push':
                raise subprocess.CalledProcessError(128, ['git', 'push'])
            return 'ssh://git@example.invalid/team/blog.git'
        with patch.object(publish, 'git', side_effect=other_host), patch.object(publish.subprocess, 'run') as run:
            with self.assertRaises(subprocess.CalledProcessError):
                publish.push_main(self.blog)
        run.assert_not_called()

    def test_online_verification_requires_matching_bytes(self):
        class Response:
            status = 200
            def __init__(self, value):
                self.value = value
            def read(self):
                return self.value
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        files = {'test-only.md': b'current'}
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response(b'older')):
            self.assertFalse(publish.verify_online(files, timeout=.05))
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response(b'current')):
            self.assertTrue(publish.verify_online(files, timeout=.05))
        with patch.object(publish.urllib.request, 'urlopen') as request:
            self.assertFalse(publish.verify_online(files, timeout=0))
        request.assert_not_called()

    def test_gzip_transport_verifies_decoded_bytes_and_rejects_bad_streams(self):
        raw = b'current saved measurement\n' * 10000
        encoded = publish.gzip.compress(raw, mtime=0)
        class Response:
            status = 200
            headers = {'Content-Encoding': 'gzip'}
            def __init__(self, value):
                self.value, self.offset = value, 0
            def read1(self, size):
                chunk = self.value[self.offset:self.offset + min(size, 7)]
                self.offset += len(chunk)
                return chunk
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        def response(request, timeout):
            self.assertEqual(request.get_header('Accept-encoding'), 'gzip')
            return Response(encoded)
        with patch.object(publish.urllib.request, 'urlopen', side_effect=response):
            self.assertTrue(publish.verify_online({'measurement.md': raw}, timeout=.5))
        for bad in (encoded[:-8], b'invalid gzip', publish.gzip.compress(raw + b'extra', mtime=0)):
            with patch.object(publish.urllib.request, 'urlopen', side_effect=lambda request, timeout: Response(bad)):
                self.assertFalse(publish.verify_online({'measurement.md': raw}, timeout=.01))

    def test_gzip_transport_without_read1_compares_saved_bytes(self):
        raw = b'current scientific report'
        class Response:
            status = 200
            headers = {'Content-Encoding': 'gzip'}
            def read(self):
                return publish.gzip.compress(raw, mtime=0)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response()):
            self.assertTrue(publish.verify_online({'report.md': raw}, timeout=.05))

    def test_many_file_verification_respects_deadline_and_joins_workers(self):
        files = {f'file-{i}.md': b'current' for i in range(1000)}
        calls = []
        active = 0
        peak = 0
        lock = threading.Lock()
        def stalled(request, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append(timeout)
            try:
                time.sleep(timeout)
                raise TimeoutError('synthetic timeout')
            finally:
                with lock:
                    active -= 1
        started = time.monotonic()
        with patch.object(publish.urllib.request, 'urlopen', side_effect=stalled):
            self.assertFalse(publish.verify_online(files, timeout=.08))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, .5, 'verification traversed all1000 requests beyond deadline')
        self.assertTrue(0 < len(calls) <= 6)
        self.assertTrue(all(0 < timeout <= .08 for timeout in calls))
        self.assertLessEqual(peak, 6)
        self.assertEqual(active, 0)
        self.assertFalse(any(thread.name.startswith('kb-http-check') for thread in threading.enumerate()))

    def test_partial_verification_persists_and_retries_only_pending_bytes(self):
        class Response:
            status = 200
            def __init__(self, raw):
                self.raw = raw
            def read(self):
                return self.raw
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        files = {'immutable.md': b'saved', 'viewer.html': b'latest'}
        requests = []
        def first(request, timeout):
            requests.append(request.full_url)
            return Response(b'saved' if request.full_url.endswith('immutable.md') else b'older')
        verified = {}
        with patch.object(publish.urllib.request, 'urlopen', side_effect=first):
            self.assertFalse(publish.verify_online(files, timeout=.05, verified=verified))
        self.assertEqual(verified, {'immutable.md': {'sha256': publish.sha256(b'saved'), 'bytes': 5}})
        requests.clear()
        def second(request, timeout):
            requests.append(request.full_url)
            return Response(b'latest')
        with patch.object(publish.urllib.request, 'urlopen', side_effect=second):
            self.assertTrue(publish.verify_online(files, timeout=.05, verified=verified))
        self.assertEqual(requests, [publish.ORIGIN + 'viewer.html'])
        with patch.object(publish.urllib.request, 'urlopen') as request:
            self.assertTrue(publish.verify_online(files, timeout=0, verified=verified))
        request.assert_not_called()
        files['viewer.html'] = b'next'
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response(b'latest')):
            self.assertFalse(publish.verify_online(files, timeout=.05, verified=verified))
        self.assertNotIn('viewer.html', verified)

    def test_streaming_response_reads_stop_at_deadline_or_excess_bytes(self):
        class Stream:
            status = 200
            def __init__(self, delay=0):
                self.delay = delay
                self.reads = 0
                self.closed = False
            def read1(self, size):
                self.reads += 1
                time.sleep(self.delay)
                return b'x'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.closed = True
        slow = Stream(delay=.01)
        started = time.monotonic()
        with patch.object(publish.urllib.request, 'urlopen', return_value=slow):
            self.assertFalse(publish.verify_online({'stream.md': b'x' * 1000}, timeout=.05))
        self.assertLess(time.monotonic() - started, .2)
        self.assertLess(slow.reads, 10)
        self.assertTrue(slow.closed)
        endless = Stream()
        with patch.object(publish.urllib.request, 'urlopen', return_value=endless):
            self.assertFalse(publish.verify_online({'stream.md': b'x'}, timeout=.05))
        self.assertEqual(endless.reads, 2)
        self.assertTrue(endless.closed)

    def test_publication_state_keeps_actual_partial_hashes_for_retry(self):
        class Response:
            status = 200
            def __init__(self, raw):
                self.raw = raw
            def read(self):
                return self.raw
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        name = self.run.name
        files = {'blogs/kb_site/runs/_sources/frozen.md': b'frozen',
                 'blogs/kb_site/runs/' + name + '.html': b'first'}
        current = dict(files)
        data = {'run': name, 'snaps': [], 'reports': {}}
        urls = []
        def partial(request, timeout):
            urls.append(request.full_url)
            return Response(b'frozen' if request.full_url.endswith('frozen.md') else b'older')
        with patch.object(publish, 'generate', side_effect=lambda *args: (data, current)), \
                patch.object(publish, 'validate_live'), patch.object(publish, 'push_files', return_value='commit1'), \
                patch.object(publish.urllib.request, 'urlopen', side_effect=partial):
            result = publish.publish_once(self.run, self.pool, self.blog, interval=0, push=True, verify_timeout=.05)
        self.assertEqual(result['status'], 'build_pending')
        state = publish.read_json(self.run / 'publication_state.json')
        immutable = 'blogs/kb_site/runs/_sources/frozen.md'
        viewer = 'blogs/kb_site/runs/' + name + '.html'
        self.assertEqual(state['verified_files'], {immutable: {'sha256': publish.sha256(b'frozen'), 'bytes': 6}})
        self.assertEqual(set(state['file_manifest']), set(files))
        self.assertFalse(state['deployment_verified'])
        urls.clear()
        with patch.object(publish, 'generate', side_effect=lambda *args: (data, current)), \
                patch.object(publish, 'validate_live'), patch.object(publish, 'push_files') as push, \
                patch.object(publish.urllib.request, 'urlopen', side_effect=lambda request, timeout: (urls.append(request.full_url) or Response(b'first'))):
            result = publish.publish_once(self.run, self.pool, self.blog, interval=0, push=True, verify_timeout=.05)
        self.assertEqual(result['status'], 'deployed')
        push.assert_not_called()
        self.assertEqual(urls, [publish.ORIGIN + viewer])
        state = publish.read_json(self.run / 'publication_state.json')
        self.assertTrue(state['deployment_verified'])
        current = {**files, viewer: b'second'}
        urls.clear()
        with patch.object(publish, 'generate', side_effect=lambda *args: (data, current)), \
                patch.object(publish, 'validate_live'), patch.object(publish, 'push_files', return_value='commit2'), \
                patch.object(publish.urllib.request, 'urlopen', side_effect=lambda request, timeout: (urls.append(request.full_url) or Response(b'second'))):
            result = publish.publish_once(self.run, self.pool, self.blog, interval=0, push=True, verify_timeout=.05)
        self.assertEqual(result['status'], 'deployed')
        self.assertEqual(urls, [publish.ORIGIN + viewer])
        state = publish.read_json(self.run / 'publication_state.json')
        self.assertEqual(state['git_commit'], 'commit2')
        self.assertEqual(state['file_manifest'][viewer], {'sha256': publish.sha256(b'second'), 'bytes': 6})

    def test_throttled_changed_generation_retries_previous_unverified_publication(self):
        class Response:
            status = 200
            def read(self):
                return b'published'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        viewer = 'blogs/kb_site/runs/' + self.run.name + '.html'
        published_files = {viewer: b'published'}
        publish.deploy_bytes(self.blog, published_files)
        state = {'git_commit': 'prior_commit', 'pushed_at': publish.time.time(),
                 'pushed_digest': publish.artifact_digest(published_files), 'deployment_verified': False,
                 'file_manifest': {viewer: {'sha256': publish.sha256(b'published'), 'bytes': 9}},
                 'verified_files': {}}
        publish.write_json(self.run / 'publication_state.json', state)
        new_files = {viewer: b'unpublished next'}
        data = {'run': self.run.name, 'snaps': [], 'reports': {}}
        with patch.object(publish, 'generate', return_value=(data, new_files)), \
                patch.object(publish, 'validate_live'), patch.object(publish, 'push_files') as push, \
                patch.object(publish, 'committed_files', return_value=published_files) as saved, \
                patch.object(publish.urllib.request, 'urlopen', return_value=Response()) as request:
            result = publish.publish_once(self.run, self.pool, self.blog, push=True, verify_timeout=.05)
        self.assertEqual(result['status'], 'throttled')
        push.assert_not_called()
        saved.assert_called_once_with(self.blog.resolve(), 'prior_commit', state['file_manifest'])
        self.assertEqual(request.call_count, 1)
        after = publish.read_json(self.run / 'publication_state.json')
        self.assertTrue(after['deployment_verified'])
        self.assertEqual(after['git_commit'], 'prior_commit')
        self.assertEqual(after['pushed_digest'], state['pushed_digest'])
        self.assertEqual(after['file_manifest'], state['file_manifest'])
        self.assertEqual(after['verified_files'], state['file_manifest'])

    def test_interval_verification_uses_commit_bytes_without_regenerating(self):
        viewer = 'blogs/kb_site/runs/' + self.run.name + '.html'
        immutable = 'blogs/kb_site/runs/_sources/saved.md'
        files = {viewer: b'published', immutable: b'fixed'}
        manifest = {path: {'sha256': publish.sha256(raw), 'bytes': len(raw)} for path, raw in files.items()}
        state = {'git_commit': 'saved_commit', 'pushed_at': publish.time.time(),
                 'file_manifest': manifest, 'verified_files': {immutable: manifest[immutable]},
                 'deployment_verified': False}
        publish.write_json(self.run / 'publication_state.json', state)
        with patch.object(publish, 'validate_live'), patch.object(publish, 'generate') as generate, \
                patch.object(publish, 'committed_files', return_value={viewer: files[viewer]}) as saved, \
                patch.object(publish, 'verify_online', side_effect=lambda files, timeout, verified: verified.update({viewer: manifest[viewer]})):
            result = publish.publish_once(self.run, self.pool, self.blog, push=True)
        generate.assert_not_called()
        saved.assert_called_once_with(self.blog.resolve(), 'saved_commit', {viewer: manifest[viewer]})
        self.assertEqual(result['verification_pending'], 0)
        after = publish.read_json(self.run / 'publication_state.json')
        self.assertTrue(after['deployment_verified'])
        self.assertEqual(after['verified_files'], manifest)
        self.assertEqual(after['git_commit'], state['git_commit'])

    def test_detached_publisher_attach_and_verified_stop_recovery(self):
        watcher = self.root / 'fake_watcher.py'
        watcher.write_text('import time\nwhile True: time.sleep(.1)\n')
        real_popen = subprocess.Popen
        processes = []
        def fake_start(argv, **kwargs):
            if '--watch' not in argv:
                return real_popen(argv, **kwargs)
            self.assertIn('--watch', argv)
            self.assertIn('--push', argv)
            self.assertTrue(kwargs['start_new_session'])
            process = real_popen([sys.executable, str(watcher)], **kwargs)
            processes.append(process)
            return process
        try:
            with patch.object(publish, 'validate_live'), patch.object(publish.subprocess, 'Popen', side_effect=fake_start):
                first = publish.ensure_publisher(self.run, self.pool, self.blog)
                self.assertEqual(first['status'], 'started')
                second = publish.ensure_publisher(self.run, self.pool, self.blog)
                self.assertEqual(second['status'], 'running')
                self.assertEqual(second['pid'], first['pid'])
                self.assertEqual(len(processes), 1)
                with self.assertRaisesRegex(ValueError, 'different inputs'):
                    publish.ensure_publisher(self.run, self.pool, self.blog, interval=900)
                processes[0].terminate()
                processes[0].wait(timeout=3)
                recovered = publish.ensure_publisher(self.run, self.pool, self.blog)
                self.assertEqual(recovered['status'], 'started')
                self.assertNotEqual(recovered['pid'], first['pid'])
                self.assertEqual(recovered['restart_count'], 1)
                self.assertEqual(len(processes), 2)
                with patch.object(publish, 'process_alive', return_value=None):
                    unknown = publish.ensure_publisher(self.run, self.pool, self.blog)
                self.assertEqual(unknown['status'], 'observation_unavailable')
                self.assertEqual(len(processes), 2)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)

    def test_process_pid_reuse_is_not_survival(self):
        identity = publish.process_identity(os.getpid())
        self.assertTrue(publish.process_alive(identity))
        self.assertFalse(publish.process_alive({**identity, 'started': 'different start time'}))

    def test_watch_releases_export_lock_between_fresh_cycles(self):
        command = [sys.executable, str(SCRIPT), '--run', str(self.run), '--pool', str(self.pool),
                   '--blog', str(self.blog), '--watch', '--interval', '1']
        with (self.root / 'watch.log').open('w') as log:
            watcher = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            page = self.blog / 'blogs/kb_site/runs/synthetic_test_only.json'
            deadline = time.monotonic() + 10
            while not page.exists() and time.monotonic() < deadline:
                self.assertIsNone(watcher.poll())
                time.sleep(.05)
            self.assertTrue(page.exists())
            with (self.run / 'publication_watch.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = False
            while not acquired and time.monotonic() < deadline:
                with (self.run / 'publication.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(.05)
                    else:
                        acquired = True
            self.assertTrue(acquired, 'watcher retained the per-export lock while idle')
            duplicate = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(duplicate.returncode, 0)
            self.assertIn('another_publisher_is_running', duplicate.stdout)
            snapshot = publish.read_json(self.run / 'kb_live.json')
            snapshot['claims'][0]['text'] = 'Revised between saved publication cycles'
            publish.write_json(self.run / 'kb_live.json', snapshot)
            while 'Revised between saved publication cycles' not in page.read_text() and time.monotonic() < deadline:
                self.assertIsNone(watcher.poll())
                time.sleep(.05)
            self.assertIn('Revised between saved publication cycles', page.read_text())
        finally:
            watcher.terminate()
            watcher.wait(timeout=5)

    def test_watch_starts_a_source_loaded_once_process_each_cycle(self):
        cycles = []
        def run_once(command, **kwargs):
            cycles.append(command)
        waits = 0
        def wait_cycle(seconds):
            nonlocal waits
            waits += 1
            if waits == 2:
                raise KeyboardInterrupt
        args = ['--run', str(self.run), '--pool', str(self.pool), '--blog', str(self.blog),
                '--watch', '--push', '--interval', '1800']
        with patch.object(publish.subprocess, 'run', side_effect=run_once), \
                patch.object(publish.time, 'sleep', side_effect=wait_cycle):
            with self.assertRaises(KeyboardInterrupt):
                publish.main(args)
        self.assertEqual(len(cycles), 2)
        for command in cycles:
            self.assertEqual(command[:2], [sys.executable, str(SCRIPT)])
            self.assertIn('--once', command)
            self.assertIn('--push', command)
            self.assertNotIn('--watch', command)

    def test_latest_late_report_visible_without_changing_frozen_kb(self):
        snapshot = publish.read_json(self.run / 'kb_live.json')
        report = snapshot['reports']['R1']
        repo = Path(report['repo'])
        (repo / 'report.md').write_text('# Late science\nNew measurement, unfinished interpretation.')
        subprocess.run(['git', 'add', 'report.md'], cwd=repo, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@local', 'commit', '-m', 'late test-only'],
                       cwd=repo, check=True, capture_output=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        saved = self.run / 'reports/R1' / (commit + '.md')
        saved.write_text((repo / 'report.md').read_text())
        after = json.loads(json.dumps(snapshot))
        after['reports']['R1'].update(repo_commit=commit, path=str(saved), published_at='2026-10-04T21:00:00Z')
        after['claims'][0]['text'] = 'Unapplied speculative change'
        publications = self.run / 'jobs/J1/publications'
        publications.mkdir()
        (publications / 'p000001.json').write_text(json.dumps({'id': 'J1:000001', 'base': snapshot, 'after': after,
              'origin_epoch': 3, 'published_at': '2026-10-04T21:00:00Z'}))
        publish.publish_once(self.run, self.pool, self.blog)
        data = publish.read_json(self.blog / 'blogs/kb_site/runs/synthetic_test_only.json')
        version = data['latest_saved_science']['R1']
        self.assertEqual(version, 'R1@' + commit)
        self.assertIn('New measurement', data['reports'][version]['text'])
        self.assertFalse(data['reports'][version]['applied_to_live'])
        frozen = next(s for s in data['snaps'] if s['label'] == 'frozen_e0004')
        self.assertEqual(frozen['reports']['R1'], 'R1@' + report['repo_commit'])
        self.assertNotIn('Unapplied speculative change', data['texts'])

    def test_repeated_report_refs_load_once_and_retain_publication_metadata(self):
        snapshot = publish.read_json(self.run / 'kb_live.json')
        original = snapshot['reports']['R1']
        repo = Path(original['repo'])
        report_path = Path(original['path']).resolve()
        alias_path = (self.run / 'reports/R2' / report_path.name).resolve()
        alias_path.parent.mkdir()
        alias_path.write_text(report_path.read_text())
        snapshot['reports']['R2'] = {**original, 'id': 'R2', 'path': str(alias_path)}
        (self.run / 'kb_live.json').write_text(json.dumps(snapshot))
        (self.run / 'solver_snapshots/frozen_e0004.json').write_text(json.dumps(snapshot))
        after = json.loads(json.dumps(snapshot))
        after['reports']['R1'].update(title='Revised metadata', claim_ids=['K1001', 'K1002'],
                                       published_at='2026-10-04T21:00:00Z')
        publications = self.run / 'jobs/J1/publications'
        publications.mkdir()
        for sequence in range(1, 5):
            (publications / f'p{sequence:06d}.json').write_text(json.dumps({
                'id': f'J1:{sequence:06d}', 'base': snapshot if sequence == 1 else after,
                'after': after, 'origin_epoch': 3, 'published_at': '2026-10-04T21:00:00Z'}))
        real_read_text = Path.read_text
        reads = []
        def read_text(path, *args, **kwargs):
            if path in (report_path, alias_path):
                reads.append(path)
            return real_read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', new=read_text), \
                patch.object(publish.subprocess, 'check_output', wraps=subprocess.check_output) as git_reads:
            data = publish.make_data(self.run, self.pool)
        self.assertEqual(reads.count(report_path), 1)
        self.assertEqual(reads.count(alias_path), 1)
        self.assertEqual([call.args[0][1] for call in git_reads.call_args_list], ['ls-tree', 'cat-file'])
        self.assertTrue(all(call.kwargs['cwd'] == repo.resolve() for call in git_reads.call_args_list))
        version = 'R1@' + original['repo_commit']
        report = data['reports'][version]
        self.assertEqual(report['publication_id'], 'J1:000001')
        self.assertEqual(report['title'], 'Revised metadata')
        self.assertEqual(report['claim_ids'], ['K1001', 'K1002'])
        self.assertEqual(report['published_at'], '2026-10-04T21:00:00Z')
        self.assertEqual(report['origin_epoch'], 3)
        self.assertEqual(report['job_id'], 'J1')
        self.assertEqual(report['text'], real_read_text(report_path))
        self.assertEqual(report['files']['measure.py']['text'], real_read_text(repo / 'measure.py'))
        self.assertNotIn('secret.json', report['files'])
        self.assertNotIn('PRIVATE_VALUE', json.dumps(data))
        alias = data['reports']['R2@' + original['repo_commit']]
        self.assertIn('/R1/', report['files']['measure.py']['url'])
        self.assertIn('/R2/', alias['files']['measure.py']['url'])
        self.assertEqual(alias['files']['measure.py']['text'], report['files']['measure.py']['text'])
        self.assertEqual(data['latest_saved_science']['R1'], version)
        frozen = next(s for s in data['snaps'] if s['label'] == 'frozen_e0004')
        self.assertEqual(frozen['reports']['R1'], version)

    def test_research_files_batch_read_exact_saved_commit_and_paths(self):
        snapshot = publish.read_json(self.run / 'kb_live.json')
        repo = Path(snapshot['reports']['R1']['repo'])
        fixtures = {'nested/name with spaces\nand newline.md': '# Saved unicode 内容\n',
                    'nested/observations.csv': 'seed,loss\n1,0.25\n',
                    'empty.json': '', 'measure.py': 'print("saved commit")\n'}
        for name, content in fixtures.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        publish.git(repo, 'add', '--', '.')
        publish.git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-m', 'saved batch fixture')
        commit = publish.git(repo, 'rev-parse', 'HEAD')
        (self.run / 'reports/R1' / (commit + '.md')).write_text((repo / 'report.md').read_text())
        snapshot['reports']['R1']['repo_commit'] = commit
        publish.write_json(self.run / 'kb_live.json', snapshot)
        publish.write_json(self.run / 'solver_snapshots/frozen_e0004.json', snapshot)
        (repo / 'measure.py').write_text('newer uncommitted work')
        with patch.object(publish.subprocess, 'check_output', wraps=subprocess.check_output) as reads:
            data = publish.make_data(self.run, self.pool)
        self.assertEqual([call.args[0][1] for call in reads.call_args_list], ['ls-tree', 'cat-file'])
        files = data['reports']['R1@' + commit]['files']
        self.assertEqual({name: item['text'] for name, item in files.items()}, fixtures)
        self.assertNotIn('secret.json', files)

    def test_report_versions_read_unchanged_blobs_and_redact_strings_once(self):
        snapshot = publish.read_json(self.run / 'kb_live.json')
        original = snapshot['reports']['R1']
        repo = Path(original['repo'])
        content = 'api_key=PRIVATE_VALUE repeated scientific measurement\n'
        (repo / 'measure.py').write_text(content)
        publish.git(repo, 'add', '--', 'measure.py')
        publish.git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-m', 'saved unchanged blob fixture')
        commit = publish.git(repo, 'rev-parse', 'HEAD')
        (self.run / 'reports/R1' / (commit + '.md')).write_text((repo / 'report.md').read_text())
        snapshot['reports']['R1']['repo_commit'] = commit
        publish.write_json(self.run / 'kb_live.json', snapshot)
        publish.write_json(self.run / 'solver_snapshots/frozen_e0004.json', snapshot)
        (repo / 'report.md').write_text('# Later report\nSame measurement files.')
        publish.git(repo, 'add', '--', 'report.md')
        publish.git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-m', 'report-only update')
        later_commit = publish.git(repo, 'rev-parse', 'HEAD')
        (self.run / 'reports/R1' / (later_commit + '.md')).write_text((repo / 'report.md').read_text())
        later = json.loads(json.dumps(snapshot))
        later['reports']['R1']['repo_commit'] = later_commit
        publish.write_json(self.run / 'kb_live.json', later)
        with patch.object(publish.subprocess, 'check_output', wraps=subprocess.check_output) as reads:
            data = publish.make_data(self.run, self.pool)
        batches = [call for call in reads.call_args_list if call.args[0][1] == 'cat-file']
        self.assertEqual(len(batches), 1)
        first = data['reports']['R1@' + commit]['files']['measure.py']['text']
        second = data['reports']['R1@' + later_commit]['files']['measure.py']['text']
        self.assertIs(first, second)
        self.assertEqual(first, content.replace('PRIVATE_VALUE', '[redacted]'))

    def test_repeated_large_artifact_is_compressed_once(self):
        snapshot = publish.read_json(self.run / 'kb_live.json')
        report = snapshot['reports']['R1']
        repo = Path(report['repo'])
        (repo / 'large.json').write_text(json.dumps({'measurement': 'x' * 300000}))
        publish.git(repo, 'add', '--', 'large.json')
        publish.git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-m', 'large repeated fixture')
        commit = publish.git(repo, 'rev-parse', 'HEAD')
        (self.run / 'reports/R1' / (commit + '.md')).write_text((repo / 'report.md').read_text())
        report['repo_commit'] = commit
        alias_path = self.run / 'reports/R2' / (commit + '.md')
        alias_path.parent.mkdir()
        alias_path.write_text((repo / 'report.md').read_text())
        snapshot['reports']['R2'] = {**report, 'id': 'R2'}
        publish.write_json(self.run / 'kb_live.json', snapshot)
        publish.write_json(self.run / 'solver_snapshots/frozen_e0004.json', snapshot)
        with patch.object(publish.gzip, 'compress', wraps=publish.gzip.compress) as compress:
            _, files = publish.generate(self.run, self.pool, self.root)
        self.assertEqual(compress.call_count, 1)
        self.assertEqual(len([path for path in files if path.endswith('.gz')]), 1)

    def test_release_and_lab_proof_hashes_are_checked(self):
        manifest = self.pool.parent / 'manifest.json'
        manifest.write_text(json.dumps({'files': {self.pool.name: publish.sha256(self.pool.read_bytes())}}))
        lab = self.root / 'lab_manifest.json'
        lab.write_text(json.dumps({'pool': {'sha256': publish.sha256(self.pool.read_bytes())}, 'files': {}, 'splits': []}))
        proof = {'kind': 'real-experiment', 'runtime_verified': True,
                 'pool_sha256': publish.sha256(self.pool.read_bytes()),
                 'config_sha256': publish.sha256((self.run / 'config.json').read_bytes()),
                 'release_manifest_sha256': publish.sha256(manifest.read_bytes()),
                 'lab_manifest_path': str(lab), 'lab_manifest_sha256': publish.sha256(lab.read_bytes())}
        (self.run / 'publication_provenance.json').write_text(json.dumps(proof))
        publish.validate_live(self.run, self.pool)
        lab.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'lab manifest'):
            publish.validate_live(self.run, self.pool)

    def test_historical_protocol_and_failed_science_are_reported_as_saved(self):
        config = publish.read_json(self.run / 'config.json')
        config.update(render='tool', progressive_disclosure=False, macro_every=5)
        (self.run / 'config.json').write_text(json.dumps(config))
        error = '# Science digest, epoch 4\n\nFailed to authenticate. API Error: 403 user quota is not enough\n'
        (self.run / 'epochs/e0004/science.md').write_text(error)
        data = publish.make_data(self.run, self.pool)
        self.assertIn('可选 KB 工具', data['flow'][0])
        self.assertNotIn('40 分钟', data['flow'][2])
        self.assertIn('每 5 轮', data['flow'][3])
        self.assertEqual(data['epochs'][0]['science'], '')
        self.assertEqual(data['epochs'][0]['scienceError'], error)


if __name__ == '__main__':
    unittest.main()
