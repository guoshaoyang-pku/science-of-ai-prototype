#!/usr/bin/env python3
"""Render science-loop artifacts with the existing luna_main viewer.

Reads stored artifacts; never invents missing claims or snapshot contents.
Accepts raw run directories or normalized luna_main viewer JSON via --data.
"""
from __future__ import annotations
import argparse
import io
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_CONFIG = {"run_name", "provider", "model", "solver_backend", "solve_effort", "discover_effort",
                 "discover_wrong_model", "discover_wrong_backend", "summ_model", "summ_backend",
                 "summ_effort", "batch_size", "epochs", "concurrency", "science", "render",
                 "macro_every", "research_calls", "consolidate_research", "research_effort",
                 "research_max_cands", "lab_workers", "seed", "sources", "async_agents",
                 "agent_timeout", "job_workers", "gate_repeats", "eval_every", "test_repeats",
                 "harness", "progressive_disclosure", "growth_policy", "updated_at"}
PRIVATE_KEY = re.compile(r'api[_-]?key|password|secret|authorization|access[_-]?token|credentials|session[_-]?(?:id|state)|environment', re.I)
PRIVATE_FILE = re.compile(r'(?:^|[/._-])(?:eval[_-]?keys|credentials?|auth|secrets?|tokens?)(?:[._/-]|$)', re.I)


def public_value(value, strings=None):
    if strings is None:
        strings = {}
    if isinstance(value, dict):
        return {k: public_value(v, strings) for k, v in value.items() if not PRIVATE_KEY.search(k)}
    if isinstance(value, list):
        return [public_value(v, strings) for v in value]
    if isinstance(value, str):
        if value in strings:
            return strings[value]
        original = value
        value = re.sub(r'(?i)Bearer\s+[^\s]+', 'Bearer [redacted]', value)
        value = re.sub(r'\bsk-[A-Za-z0-9_-]{12,}', '[redacted]', value)
        value = re.sub(r'(?i)((?:api[_-]?key|password|access[_-]?token|authorization)\s*[=:]\s*)([^\s,;]+)', r'\1[redacted]', value)
        value = re.sub(r'(?i)(["\x27](?:api[_-]?key|password|access[_-]?token|authorization)["\x27]\s*:\s*["\x27])[^"\x27]+', r'\1[redacted]', value)
        strings[original] = value
    return value


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_rows(path):
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith('\n'):
        try:
            json.loads(lines[-1])
        except ValueError:
            lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def trace(raw):
    if isinstance(raw, dict):
        raw = raw.get('trace', raw.get('turns', raw.get('messages', [])))
    out = []
    for i, t in enumerate(raw or []):
        if not isinstance(t, dict):
            continue
        clean = []
        if t.get('name'):
            args = t.get('arguments', {})
            clean.append({'name': t['name'], 'arguments': args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
        for c in t.get('calls', t.get('tool_calls', [])):
            c = c.get('function', c)
            args = c.get('arguments', c.get('input', {}))
            clean.append({'name': c.get('name', ''), 'arguments': args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
        out.append({'turn': t.get('turn', i), 'content': t.get('content', t.get('text')),
                    'reasoning': t.get('reasoning', t.get('reasoning_content')),
                    'calls': clean, 'results': t.get('results', [t['result']] if 'result' in t else []),
                    'usage': t.get('usage') or {}, 'secs': t.get('secs')})
    return out


def make_data(run: Path, pool=None, *, include_test=True):
    config = read_json(run / 'config.json', {})
    config = {k: v for k, v in config.items() if k in PUBLIC_CONFIG}
    data = {'run': config.get('run_name', run.name), 'config': config, 'texts': [],
            'snaps': [], 'commits': read_rows(run / 'commits.jsonl'), 'epochs': [],
            'evals': {}, 'prompts': {}, 'home': '../index.html',
            'test': read_json(run / 'test_report.json', {}) if include_test else {}}
    indices = {}
    reports = {}
    repo_contents = {}
    blob_contents = {}
    latest_science = {}
    frozen_versions = set()

    def report_versions(doc):
        refs = {}
        for identifier, source in doc.get("reports", {}).items():
            version = identifier + "@" + source["repo_commit"]
            if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or not re.fullmatch(r'[0-9a-f]{40,64}', source['repo_commit']):
                raise ValueError('invalid saved report identity')
            refs[identifier] = version
            report = reports.setdefault(version, {})
            report.update({k: source[k] for k in ("id", "title", "topic", "claim_ids",
                                "repo_commit", "origin_epoch", "job_id", "published_at") if k in source})
            if 'text' in report:
                continue
            report_path = (run / 'reports' / identifier / (source['repo_commit'] + '.md')).resolve()
            root = (run / "reports").resolve()
            if root in report_path.parents and report_path.is_file():
                report["text"] = report_path.read_text()
            else:
                raise ValueError('missing report version: ' + version)
            report['url'] = data['run'] + '/reports/' + identifier + '/' + source['repo_commit'] + '.md'
            report['files'] = {}
            repo = Path(source.get('repo', str(run / 'jobs' / source.get('job_id', '') / 'repo'))).resolve()
            if repo.is_relative_to((run / 'jobs').resolve()) and (repo / '.git').exists():
                key = (repo, source['repo_commit'])
                if key not in repo_contents:
                    repo_contents[key] = {}
                    tree = subprocess.check_output(['git', 'ls-tree', '-rz', source['repo_commit']], cwd=repo)
                    objects = []
                    for entry in tree.split(b'\0'):
                        if not entry:
                            continue
                        metadata, raw_name = entry.split(b'\t', 1)
                        _, kind, object_id = metadata.split()
                        if kind != b'blob':
                            continue
                        name = raw_name.decode('utf-8')
                        rel = Path(name)
                        if (rel.is_absolute() or '..' in rel.parts or PRIVATE_KEY.search(name) or PRIVATE_FILE.search(name)
                                or any(p.startswith('.') for p in rel.parts) or rel.suffix not in ('.py', '.json', '.csv', '.tsv', '.md')
                                or name == 'report.md'):
                            continue
                        objects.append((name, object_id))
                    missing = list(dict.fromkeys(object_id for _, object_id in objects if object_id not in blob_contents))
                    if missing:
                        request = b''.join(object_id + b'\n' for object_id in missing)
                        raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=repo, input=request)
                        stream = io.BytesIO(raw)
                        for object_id in missing:
                            header = stream.readline().split()
                            if len(header) != 3 or header[:2] != [object_id, b'blob']:
                                raise ValueError('missing saved research blob: ' + object_id.decode())
                            content = stream.read(int(header[2]))
                            if stream.read(1) != b'\n':
                                raise ValueError('truncated saved research blob: ' + object_id.decode())
                            blob_contents[object_id] = content.decode('utf-8')
                    repo_contents[key] = {name: blob_contents[object_id] for name, object_id in objects}
                for name, content in repo_contents[key].items():
                    url = data['run'] + '/reports/' + identifier + '/' + source['repo_commit'] + '/' + name
                    report['files'][name] = {'text': content, 'url': url}
        return refs

    def text_id(s):
        s = s or ''
        if s not in indices:
            indices[s] = len(data['texts'])
            data['texts'].append(s)
        return indices[s]

    for source, suffix in [('kb', ''), ('kb_rejected', '（候选，被拒）'), ('snapshots', ''), ('solver_snapshots', '')]:
        for path in sorted((run / source).glob('*.json')):
            doc = read_json(path)
            refs = report_versions(doc)
            frozen_versions.update(refs.values())
            claims = doc.get('claims', [])
            ranked = sorted(claims, key=lambda c: (-c.get('credibility', 0.5), c['id']))
            selected = {c['id'] for c in ranked[:40]}
            fresh = sorted([c for c in ranked if c['id'] not in selected and c.get('support_count', 0) + c.get('failure_count', 0) < 2], key=lambda c: (-c.get('created_epoch', 0), c['id']))[:10]
            selected.update(c['id'] for c in fresh)
            tool = config.get('render') == 'tool'
            brief = config.get('progressive_disclosure', False) or config.get('render') == 'brief'
            rows, details = [], {}
            for c in claims:
                s, f = c.get('support_count', 0), c.get('failure_count', 0)
                credit = c.get('credibility', (s + 1) / (s + f + 2))
                rows.append([c['id'], s, f, credit, text_id(c.get('text')), c.get('created_epoch', 0), int(brief or (not tool and c['id'] in selected)), c.get('origin', '')])
                details[c['id']] = {k: v for k, v in c.items() if k in ("created_epoch", "sources",
                    "merged_from", "origin", "mechanism", "regime", "phase", "refines", "report_ids")}
            rendered = '\n'.join(f"[{c['id']}] {c.get('text', '')}" for c in claims if tool or brief or c['id'] in selected)
            epoch = int(path.stem[-4:]) if source == 'solver_snapshots' and path.stem[-4:].isdigit() else doc.get('epoch', 0)
            data['snaps'].append({'label': path.stem + suffix, 'base': path.stem, 'epoch': epoch, 'commitCounter': doc.get('commit_counter'), 'claims': rows, 'render': rendered, 'complete': True, 'candidate': bool(suffix), 'claimDetails': details, "reports": refs,
                                  'solverSelection': doc.get('solver_selection'),
                                  'note': ('解题使用稳定快照 ' + doc['solver_selection']['label'] + '；研究 KB 继续生长。')
                                          if doc.get('solver_selection', {}).get('mode') == 'stable' else ''})
    data['snaps'].sort(key=lambda s: (s['epoch'], s.get('candidate', False)))
    live = read_json(run / "kb_live.json")
    if live:
        refs = report_versions(live)
        frozen_versions.update(refs.values())
        data["snaps"].append({"label": "latest_saved", "epoch": live["epoch"], "commitCounter": live.get('commit_counter'),
            "claims": [[c["id"], c.get("support_count", 0), c.get("failure_count", 0),
                         c.get("credibility", .5), text_id(c.get("text")), c.get("created_epoch", 0),
                         1, c.get("origin", "")] for c in live.get("claims", [])],
            "claimDetails": {c["id"]: {k: v for k, v in c.items() if k in ("created_epoch",
                "sources", "merged_from", "origin", "mechanism", "regime", "phase", "refines",
                "report_ids")} for c in live.get("claims", [])}, "reports": refs, "complete": True,
            "render": "\n".join("[" + c["id"] + "] " + c["text"] for c in live.get("claims", [])),
            "note": "最新已保存版本；已开始的解题仍使用其原冻结快照。"})
    jobs = []
    job_traces = {}
    for path in sorted((run / "jobs").glob("J*/job.json")):
        source = read_json(path)
        jobs.append({k: source[k] for k in ("id", "kind", "origin_epoch", "current_epoch",
                     "member_epoch", "label", "status", "report_id", "checkpoint_commit") if k in source})
        job_traces[source['id']] = trace(read_json(path.parent / 'summarizer_trace.json', []))
        durable_tools = path.parent / 'session/_codex_tools.jsonl'
        if durable_tools.is_file():
            job_traces[source['id']] += trace(read_rows(durable_tools))
        if source.get('kind') == 'research':
            for saved in sorted(path.parent.glob('r*.json')):
                report = read_json(saved)
                if not durable_tools.is_file():
                    job_traces[source['id']] += trace(report.get('tool_log', []))
        for publication in sorted(path.parent.glob("publications/p*.json")):
            saved = read_json(publication)
            refs = report_versions(saved['after'])
            for identifier, version in refs.items():
                source_report = saved['after']['reports'][identifier]
                if source_report != saved.get('base', {}).get('reports', {}).get(identifier):
                    reports[version]['publication_id'] = saved['id']
                    reports[version]['published_at'] = source_report.get('published_at', saved.get('published_at', ''))
    for version, report in reports.items():
        identifier = report['id']
        prior = latest_science.get(identifier)
        if prior is None or (str(report.get('published_at', '')), report.get('publication_id', '')) >= (str(reports[prior].get('published_at', '')), reports[prior].get('publication_id', '')):
            latest_science[identifier] = version
        report['applied_to_live'] = bool(live and live.get('reports', {}).get(identifier, {}).get('repo_commit') == report['repo_commit'])
        report['application_pending'] = version not in frozen_versions
    data["jobs"], data["reports"] = jobs, reports
    data['latest_saved_science'] = latest_science
    data['job_traces'] = job_traces
    qpool = {q['question_id']: q for q in read_rows(Path(pool))} if pool else {}
    metrics = {m['epoch']: m for m in read_rows(run / 'metrics.jsonl')}
    for ep in sorted((run / 'epochs').glob('e[0-9]*')):
        if not re.fullmatch(r'e\d+', ep.name):
            continue
        n = int(ep.name[1:])
        records = read_rows(ep / 'records.jsonl')
        if not records:
            records = [read_json(p) for p in sorted((ep / 'solves').glob('*.json'))]
        if not records and n not in metrics and not any((ep / name).exists() for name in
                ('summarizer_trace.json', 'science.md', 'summarizer_input.md')):
            continue
        full = []
        for record in records:
            qid = record.get('question_id')
            r = {**record, **read_json(ep / 'solves' / f'{qid}.json', {})}
            q = qpool.get(qid, {})
            question = r.get('question', r.get('prompt', ''))
            if not question and q.get('messages'):
                question = '\n\n'.join(f"## {m['role']}\n{m['content']}" for m in q['messages'])
            r['question'] = question or '未取得题目原文。'
            r['discover_trace'] = trace(r.get('discover_trace', []))
            r['solve_usage'] = r.get('solve_usage') or {}
            r['solve_trace'] = trace(r.get('solve_trace', r.get('solver_trace', [])))
            for key, val in [('discover_effort', ''), ('cited', []), ('kb_retrieved', []), ('kb_queries', []), ('solution', ''), ('solver_reasoning', ''), ('comment', ''), ('hypothesis', ''), ('skipped', False)]:
                r.setdefault(key, val)
            full.append(r)
        research = []
        for p in sorted(ep.glob('research*')):
            if p.suffix == '.json':
                research.append({'file': p.name, 'data': read_json(p)})
            elif p.suffix in ('.md', '.txt'):
                research.append({'file': p.name, 'text': p.read_text()})
        sci = ep / 'science.md'
        science = sci.read_text() if sci.exists() else ''
        science_error = ''
        if re.match(r'^\s*(?:#.*\n+)?(?:Failed to authenticate|API Error:|Traceback \(most recent call last\))', science):
            science, science_error = '', science
        data['epochs'].append({'epoch': n, 'records': full, 'metrics': metrics.get(n, {}),
                               'summarizer': trace(read_json(ep / 'summarizer_trace.json', [])),
                               'science': science, 'scienceError': science_error, 'research': research,
                               'summarizer_input': (ep / 'summarizer_input.md').read_text() if (ep / 'summarizer_input.md').exists() else '',
                               "jobs": [job for job in jobs if job["origin_epoch"] == n or job.get("member_epoch") == n]})
        for job in jobs:
            if job.get('kind') == 'summarizer' and job['origin_epoch'] == n:
                data['epochs'][-1]['summarizer'] += job_traces[job['id']]
    for p in sorted((run / 'eval').glob('*/_summary.json')):
        if not include_test and p.parent.name.startswith('test'):
            continue
        data['evals'][p.parent.name] = read_json(p)
    for p in sorted(run.glob('*prompt*.md')):
        data['prompts'][p.name] = p.read_text()
    data['availability'] = '已载入保存的 KB 快照、实验记录与评测结果。'
    publication_note = read_json(run / 'publication_note.json', {})
    for key in ('summary', 'evalNote', 'availability'):
        if isinstance(publication_note.get(key), str):
            data[key] = publication_note[key]
    best = read_json(run / 'best.json', {})
    data['defaultSnap'] = "latest_saved" if live else best.get('label') or next((s['label'] for s in reversed(data['snaps']) if not s.get('candidate')), None)
    data['settingURL'] = data['run'] + '/setting.md'
    brief = config.get('progressive_disclosure', False) or config.get('render') == 'brief'
    window = config.get('agent_timeout', 2400)
    window = f"{window / 60:g} 分钟" if window % 60 == 0 else f"{window:g} 秒"
    data['flow'] = [('① 解题：默认完整简洁 KB，science 按需展开；每题使用冻结版本，答案尚未揭示。' if brief else
                     '① 解题：题目 + 可选 KB 工具；每题使用冻结版本，答案尚未揭示。' if config.get('render') == 'tool' else
                     '① 解题：题目 + 上一轮只读 KB；每题使用冻结版本，答案尚未揭示。'),
                    '② 发现：揭示答案，提出可检查的规律与研究问题。',
                    (f'③ 总结与研究：持久任务，{window} 后标记为下轮延续成员，原任务继续。' if config.get('async_agents') else
                     '③ 总结与研究：整理本轮记录，修订、增加、合并与撤回 claim。'),
                    (f"④ 宏观审查：每 {config['macro_every']} 轮测成长 KB；持续退化只切换 solver 输入，研究和报告保留。"
                     if config.get('growth_policy') == 'research_persistent_solver_fallback_v1' else
                     '④ 下一轮：使用最新已保存并应用的报告版本；每次应用记录所属与应用轮次。' if config.get('async_agents') else
                     f"④ KB 成长：每 {config['macro_every']} 轮集中整理和宏观评测。" if config.get('macro_every') else
                     '④ KB 快照：依据保存的评测记录选择候选或现任版本。')]
    return public_value(data)


def render(data, output):
    if not data.get('snaps'):
        data['snaps'] = [{'label': '未取得 KB 快照', 'epoch': 0, 'claims': [], 'render': '', 'complete': False, 'note': '未取得完整 KB 快照。'}]
    for key, val in [('texts', []), ('commits', []), ('epochs', []), ('evals', {}), ('prompts', {}), ('home', '../index.html'), ('config', {})]:
        data.setdefault(key, val)
    blob = json.dumps(data, ensure_ascii=False).replace('<', '\u003c').replace('\u2028', '\u2028').replace('\u2029', '\u2029')
    head = (HERE / 'run-viewer-head.html').read_text()
    js = (HERE / 'run-viewer.js').read_text().replace('let claimId = "K0134";', 'let claimId = D.snaps.flatMap(s => s.claims)[0]?.[0] || "";')
    js = js.rsplit('render();', 1)[0]
    extra = (HERE / 'sol-viewer.js').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(head + 'const D = ' + blob + ';\n' + js + extra + '\nrender();\n</script>\n</body>\n</html>\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--run', type=Path)
    src.add_argument('--data', type=Path)
    p.add_argument('--pool', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-data', type=Path)
    p.add_argument('--exclude-test', action='store_true')
    a = p.parse_args(argv)
    d = read_json(a.data) if a.data else make_data(a.run, a.pool, include_test=not a.exclude_test)
    if a.save_data:
        a.save_data.parent.mkdir(parents=True, exist_ok=True)
        a.save_data.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    render(d, a.output)
    print(json.dumps({'output': str(a.output), 'run': d['run'], 'snapshots': len(d['snaps']), 'epochs': len(d['epochs']), 'claims_latest': len(d['snaps'][-1]['claims'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
