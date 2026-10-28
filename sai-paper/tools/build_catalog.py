#!/usr/bin/env python3
"""Build a readable SAI literature catalog and audit declared local PDFs.

Reads only literature/{interp,rl,pretrain}.json. Writes catalog.md and audit.json;
never downloads, deletes, modifies, or guesses an unavailable PDF.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ('interp', 'rl', 'pretrain')
ARXIV_ID = r'(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.-]*/\d{7})'
CLASSES = {'knowledge': 'Knowledge', 'k': 'Knowledge', 'law': 'Law', 'l': 'Law', 'science': 'Science', 's': 'Science'}


def text(value):
    if value is None:
        return ''
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def first(item, *keys):
    for key in keys:
        value = item.get(key)
        if value is not None and value != '' and value != []:
            return value
    return ''


def md(value):
    return text(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def safe_url(value):
    value = text(value)
    return value if re.match(r'^https?://[^\s<>]+$', value) else ''


def classify(item):
    raw = text(first(item, 'classification', 'category', 'level', 'label'))
    label = CLASSES.get(raw.lower(), raw)
    # An article can contain several claim types and a provisional qualifier;
    # keep the author's exact annotation rather than inventing a single label.
    return label if re.search(r'\b(?:Knowledge|Law|Science)\b', label, re.I) else ''


def audit_pdf(item):
    raw = text(first(item, 'pdf_path', 'local_pdf', 'pdf'))
    result = {'declared_path': raw or None, 'status': 'not_declared', 'pages': None, 'sha256': None, 'bytes': None, 'first_page_title_match_heuristic': None}
    if not raw:
        return result
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else ROOT / path
    try:
        path = path.resolve(strict=True)
        path.relative_to((ROOT / 'papers').resolve())
    except (OSError, ValueError):
        result['status'] = 'missing_or_outside_papers'
        return result
    if not path.is_file():
        result['status'] = 'not_file'
        return result
    result['path'] = path.relative_to(ROOT).as_posix()
    result['bytes'] = path.stat().st_size
    h = hashlib.sha256()
    with path.open('rb') as f:
        header = f.read(1024)
        h.update(header)
        if not header.startswith(b'%PDF-'):
            result['status'] = 'invalid_magic'
            return result
        while block := f.read(1024 * 1024):
            h.update(block)
    result['sha256'] = h.hexdigest()
    binary = shutil.which('pdfinfo')
    if not binary:
        result['status'] = 'header_only_pdfinfo_unavailable'
        return result
    try:
        process = subprocess.run([binary, str(path)], capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        result['status'] = 'pdfinfo_error'
        result['error'] = str(exc)
        return result
    if process.returncode:
        result['status'] = 'pdfinfo_error'
        result['error'] = process.stderr[:400]
        return result
    match = re.search(r'^Pages:\s*(\d+)\s*$', process.stdout, re.M)
    if not match or int(match.group(1)) < 1:
        result['status'] = 'no_pages'
        return result
    result['pages'] = int(match.group(1))
    result['status'] = 'valid_pdf'
    extractor = shutil.which('pdftotext')
    expected_title = text(first(item, 'title', 'name'))
    if extractor and expected_title:
        try:
            out = subprocess.run([extractor, '-f', '1', '-l', '1', str(path), '-'], capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                stopwords = {'the', 'and', 'for', 'from', 'with', 'are', 'not', 'using', 'through', 'via', 'into', 'towards', 'toward'}
                tokenize = lambda s: set(re.findall(r'[a-z0-9]+', s.lower())) - stopwords
                expected = {w for w in tokenize(expected_title) if len(w) > 2}
                found = tokenize(out.stdout[:16000])
                # Some academic fonts extract words as "H IGH". Also handle
                # short titles, but only accept a complete compact match with
                # sufficient characters; absence remains unknown for short titles.
                compact = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
                compact_title = compact(expected_title)
                same_compact_title = len(compact_title) >= 8 and compact_title in compact(out.stdout[:16000])
                if len(expected) >= 3:
                    result['title_token_coverage'] = round(len(expected & found) / len(expected), 3)
                    result['first_page_title_match_heuristic'] = same_compact_title or result['title_token_coverage'] >= 0.6
                elif same_compact_title:
                    result['first_page_title_match_heuristic'] = True
                arxiv = text(first(item, 'arxiv', 'arxiv_id'))
                if re.fullmatch(ARXIV_ID + r'(?:v\d+)?', arxiv):
                    base = re.sub(r'v\d+$', '', arxiv)
                    result['first_page_arxiv_id_match'] = bool(re.search(r'(?<![\w.-])' + re.escape(base) + r'(?:v\d+)?(?![\w.])', out.stdout[:16000], re.I))
                    observed_ids = re.findall(r'\barxiv\s*:\s*(' + ARXIV_ID + r')(?:v\d+)?', out.stdout[:16000], re.I)
                    result['first_page_explicit_arxiv_ids'] = list(dict.fromkeys(observed_ids))
        except (subprocess.TimeoutExpired, OSError):
            pass
    return result


def audit_html(item):
    """Inspect explicitly declared local HTML; do not guess article mappings."""
    raw = text(first(item, 'html_path', 'local_html'))
    result = {'declared_path': raw or None, 'status': 'not_declared', 'sha256': None, 'bytes': None,
              'kind': text(first(item, 'html_kind')) or 'unspecified'}
    if not raw:
        return result
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else ROOT / path
    try:
        path = path.resolve(strict=True)
        path.relative_to((ROOT / 'papers').resolve())
    except (OSError, ValueError):
        result['status'] = 'missing_or_outside_papers'
        return result
    if not path.is_file():
        result['status'] = 'not_file'
        return result
    payload = path.read_bytes()
    result.update(path=path.relative_to(ROOT).as_posix(), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    result['status'] = 'valid_html' if len(payload) >= 512 and (b'<html' in payload[:8192].lower() or b'<!doctype html' in payload[:8192].lower()) else 'invalid_html'
    return result


def main():
    domain_data = {}
    problems = []
    for domain in DOMAINS:
        path = ROOT / 'literature' / f'{domain}.json'
        if not path.exists():
            problems.append(f'{domain}: index missing')
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(obj, dict) or not isinstance(obj.get('papers'), list) or not isinstance(obj.get('concepts'), list):
                raise ValueError('expected object with papers and concepts arrays')
            for kind in ('papers', 'concepts'):
                if not all(isinstance(x, dict) for x in obj[kind]):
                    raise ValueError(f'{kind} must contain objects')
            domain_data[domain] = obj
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f'{domain}: invalid index: {exc}')

    papers = [(d, p) for d, o in domain_data.items() for p in o['papers']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        audits = list(pool.map(audit_pdf, (p for _, p in papers)))
    html_audits = [audit_html(p) for _, p in papers]
    # Textbooks and other resources remain distinct from paper counts.
    resources = [(d, r) for d, o in domain_data.items() for r in o.get('resources', []) if isinstance(r, dict)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        resource_audits = list(pool.map(audit_pdf, (r for _, r in resources)))
    resource_by_id = {id(r): a for (_, r), a in zip(resources, resource_audits)}
    supplements = [(d, p, s) for d, p in papers for s in p.get('supplemental', []) if isinstance(s, dict)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        supplement_audits = list(pool.map(audit_pdf, (s for _, _, s in supplements)))
    supplements_by_paper = {}
    for (domain, paper, supplement), audit in zip(supplements, supplement_audits):
        supplements_by_paper.setdefault(id(paper), []).append((supplement, audit))
        if audit['status'] != 'valid_pdf':
            problems.append(f"{domain}: supplemental file unavailable or invalid: {text(first(supplement, 'title', 'type'))[:100]}")
    # A first-page image inspection can resolve OCR failures, but only for the
    # exact file whose SHA-256 was reviewed; keep the heuristic result visible.
    manual_path = ROOT / 'literature' / 'manual_pdf_identity.json'
    if manual_path.exists():
        try:
            reviewed = json.loads(manual_path.read_text(encoding='utf-8'))['entries']
            for a in audits + resource_audits:
                matches = [e for e in reviewed if e.get('path') == a.get('path') and e.get('sha256') == a.get('sha256') and e.get('result') == 'matched']
                if len(matches) == 1:
                    a['manual_identity'] = matches[0]
        except (ValueError, KeyError, OSError) as exc:
            problems.append(f'invalid manual PDF identity record: {exc}')
    audit_by_id = {id(p): a for (_, p), a in zip(papers, audits)}
    html_by_id = {id(p): a for (_, p), a in zip(papers, html_audits)}
    classes = Counter()
    review_levels = Counter()
    lines = ['# Science of AI 文献与概念图谱', '',
             '> 自动整理自各方向清单。分类、scope 和审阅程度取自条目；PDF 结构检查不等于内容正确，也不代表完成全文精读。覆盖仍在扩展。', '']
    for domain in DOMAINS:
        if domain not in domain_data:
            lines += [f'## {domain}', '', '尚无有效清单。', '']
            continue
        obj = domain_data[domain]
        lines += [f'## {domain}', '', f"论文 {len(obj['papers'])} 篇；概念 {len(obj['concepts'])} 条。", '',
                  '| 论文 | 层次 | 主张及范围 | 原文依据 | 审阅程度 | 本地原文 |', '|---|---|---|---|---|---|']
        for p in obj['papers']:
            title = text(first(p, 'title', 'name'))
            source = safe_url(first(p, 'url', 'source_url', 'source'))
            label = classify(p)
            classes[label or 'unknown'] += 1
            if not title or not source or not label:
                problems.append(f'{domain}: paper missing title/source/classification: {title[:100]}')
            claim = md(first(p, 'claim', 'summary', 'description'))
            scope = md(first(p, 'scope', 'limitations', 'boundary'))
            review = md(first(p, 'review_level', 'reading_level', 'verification')) or '未标明'
            evidence = md(first(p, 'evidence', 'evidence_location')) or '未标明'
            review_levels[review] += 1
            if not claim or not scope or review == '未标明' or not first(p, 'evidence', 'evidence_location'):
                problems.append(f'{domain}: paper missing claim/scope/evidence/review level: {title[:100]}')
            a = audit_by_id[id(p)]
            html = html_by_id[id(p)]
            if a['status'] == 'valid_pdf':
                pdf = f"[PDF](../{a['path']}) ({a['pages']} 页)"
            elif html['status'] == 'valid_html':
                html_label = '许可 HTML 转录' if html['kind'] == 'licensed_transcription' else '网页'
                pdf = f"[{html_label}](../{html['path']})；无已核验 PDF"
            else:
                pdf = f"未核验：{md(a['status'])}"
            for supplement, extra in supplements_by_paper.get(id(p), []):
                if extra['status'] == 'valid_pdf':
                    kind = md(first(supplement, 'type', 'title')) or '补充材料'
                    pdf += f"; [补充材料：{kind}](../{extra['path']})"
            heading = f'[{md(title)}]({source})' if source else md(title) or '未注明标题'
            detail = claim.rstrip('。；; ') + (('；范围：' if claim else '范围：') + scope if scope else '')
            lines.append(f'| {heading} | {md(label) or "待核"} | {detail or "待补充"} | {evidence} | {review} | {pdf} |')
        lines += ['', '### 概念', '', '| 概念 | 层次 | 定义、范围与来源 |', '|---|---|---|']
        for c in obj['concepts']:
            name = text(first(c, 'name', 'title', 'concept'))
            label = classify(c)
            desc = md(first(c, 'definition', 'description', 'claim', 'summary'))
            scope = md(first(c, 'scope', 'limitations', 'boundary'))
            rawsource = first(c, 'url', 'source_url', 'source', 'sources')
            if isinstance(rawsource, list):
                rawsource = next((x for x in rawsource if isinstance(x, str) and safe_url(x)), '')
            src = safe_url(rawsource)
            evidence = f' [来源]({src})' if src else ''
            if not name or not label:
                problems.append(f'{domain}: concept missing name/classification: {name[:100]}')
            if not src:
                problems.append(f'{domain}: concept missing direct source URL: {name[:100]}')
            if not desc or not scope:
                problems.append(f'{domain}: concept missing definition/scope: {name[:100]}')
            lines.append(f'| {md(name) or "待补"} | {md(label) or "待核"} | {desc or "待补充"}{("；范围：" + scope) if scope else ""}{evidence} |')
        lines.append('')
        domain_resources = [r for d, r in resources if d == domain]
        if domain_resources:
            lines += ['### 参考资源（不计入论文数）', '', '| 名称 | 来源与范围 | 本地资料 |', '|---|---|---|']
            for resource in domain_resources:
                title = md(first(resource, 'title', 'name'))
                source = safe_url(first(resource, 'url', 'source_url', 'source'))
                heading = f'[{title}]({source})' if source else title
                scope = md(first(resource, 'scope', 'limitations', 'boundary'))
                audit = resource_by_id[id(resource)]
                local = f"[PDF](../{audit['path']}) ({audit['pages']} 页)" if audit['status'] == 'valid_pdf' else f"未核验：{md(audit['status'])}"
                lines.append(f'| {heading} | {scope} | {local} |')
            lines.append('')
    status_counts = Counter(a['status'] for a in audits)
    html_status_counts = Counter(a['status'] for a in html_audits)
    category_mentions = Counter()
    for label, count in classes.items():
        for category in ('Knowledge', 'Law', 'Science'):
            if re.search(r'\b' + category + r'\b', label, re.I):
                category_mentions[category] += count
    for (domain, paper), a in zip(papers, audits):
        if a['status'] == 'valid_pdf':
            title = text(first(paper, 'title', 'name'))[:100]
            declared_hash = text(first(paper, 'pdf_sha256', 'sha256')).strip().lower()
            if declared_hash and declared_hash != a['sha256']:
                problems.append(f'{domain}: declared PDF SHA-256 mismatch: {title}')
            for names, actual, label in ((('pdf_bytes', 'size_bytes'), a['bytes'], 'bytes'), (('pdf_pages',), a['pages'], 'pages')):
                declared = first(paper, *names)
                if declared not in (None, '') and str(declared) != str(actual):
                    problems.append(f'{domain}: declared PDF {label} mismatch: {title}')
        if a['status'] != 'valid_pdf' and (a['declared_path'] or 'arxiv.org/' in text(first(paper, 'url', 'source_url', 'source'))):
            problems.append(f"{domain}: PDF unavailable or invalid ({a['status']}): {text(first(paper, 'title', 'name'))[:100]}")
        if a.get('first_page_title_match_heuristic') is False and not a.get('manual_identity'):
            problems.append(f"{domain}: PDF first-page title may not match: {text(first(paper, 'title', 'name'))[:100]}")
        if a.get('first_page_explicit_arxiv_ids') and not a.get('first_page_arxiv_id_match'):
            problems.append(f"{domain}: expected arXiv ID absent while other IDs appear on page 1; inspect: {text(first(paper, 'title', 'name'))[:100]}")
    for (domain, paper), html in zip(papers, html_audits):
        title = text(first(paper, 'title', 'name'))[:100]
        if html['declared_path'] and html['status'] != 'valid_html':
            problems.append(f"{domain}: declared HTML unavailable or invalid ({html['status']}): {title}")
        if html['status'] == 'valid_html':
            # A generic sha256/size_bytes refers to the HTML only when no
            # separate PDF is declared; otherwise use explicit HTML fields.
            no_pdf = not first(paper, 'pdf_path', 'local_pdf', 'pdf')
            declared_hash = text(first(paper, 'html_sha256') or (paper.get('sha256') if no_pdf else '')).strip().lower()
            declared_size = first(paper, 'html_bytes') or (paper.get('size_bytes') if no_pdf else '')
            if declared_hash and declared_hash != html['sha256']:
                problems.append(f'{domain}: declared HTML SHA-256 mismatch: {title}')
            if declared_size not in (None, '') and str(declared_size) != str(html['bytes']):
                problems.append(f'{domain}: declared HTML bytes mismatch: {title}')
    for (domain, resource), audit in zip(resources, resource_audits):
        title = text(first(resource, 'title', 'name'))[:100]
        if audit['declared_path'] and audit['status'] != 'valid_pdf':
            problems.append(f"{domain}: resource PDF unavailable or invalid ({audit['status']}): {title}")
        if audit['status'] == 'valid_pdf':
            expected = text(first(resource, 'pdf_sha256', 'sha256')).strip().lower()
            if expected and expected != audit['sha256']:
                problems.append(f'{domain}: resource PDF SHA-256 mismatch: {title}')
            for names, actual, label in ((('pdf_bytes', 'size_bytes'), audit['bytes'], 'bytes'), (('pdf_pages',), audit['pages'], 'pages')):
                declared = first(resource, *names)
                if declared not in (None, '') and str(declared) != str(actual):
                    problems.append(f'{domain}: resource PDF {label} mismatch: {title}')
            if audit.get('first_page_title_match_heuristic') is False and not audit.get('manual_identity'):
                problems.append(f'{domain}: resource PDF first-page title may not match: {title}')
    duplicate_hashes = {}
    for (domain, paper), audit in zip(papers, audits):
        if audit.get('sha256'):
            duplicate_hashes.setdefault(audit['sha256'], []).append({'domain': domain, 'title': text(first(paper, 'title', 'name')), 'path': audit.get('path')})
    duplicate_hashes = {k: v for k, v in duplicate_hashes.items() if len(v) > 1}
    # A source can be downloaded in different formats or versions. Hash-only
    # detection misses such duplicates; preserve possible title/identifier
    # collisions for manual review rather than automatically deleting records.
    title_groups = {}
    arxiv_groups = {}
    for domain, paper in papers:
        title = text(first(paper, 'title', 'name'))
        entry = {'domain': domain, 'title': title}
        normalized = re.sub(r'[^a-z0-9]', '', title.lower())
        if normalized:
            title_groups.setdefault(normalized, []).append(entry)
        arxiv = text(first(paper, 'arxiv', 'arxiv_id'))
        match = re.fullmatch(r'(' + ARXIV_ID + r')(?:v\d+)?', arxiv)
        if match:
            arxiv_groups.setdefault(match.group(1).lower(), []).append(entry)
    duplicate_titles = {key: value for key, value in title_groups.items() if len(value) > 1}
    duplicate_arxiv = {key: value for key, value in arxiv_groups.items() if len(value) > 1}
    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'scope_note': 'Representative, expanding corpus; not a census or independent reproduction.',
              'domains_present': list(domain_data),
              'papers': len(papers), 'concepts': sum(len(v['concepts']) for v in domain_data.values()),
              'paper_classifications': dict(classes), 'category_mentions_nonexclusive': dict(category_mentions), 'paper_review_levels': dict(review_levels), 'pdf_statuses': dict(status_counts), 'html_statuses': dict(html_status_counts),
              'problems': problems, 'duplicate_sha256': duplicate_hashes, 'duplicate_normalized_title': duplicate_titles, 'duplicate_arxiv': duplicate_arxiv,
              'paper_pdf_audit': [{'domain': d, 'title': text(first(p, 'title', 'name')), **a, 'html': html} for (d, p), a, html in zip(papers, audits, html_audits)],
              'supplemental_pdf_audit': [{'domain': d, 'paper': text(first(p, 'title', 'name')), 'supplement_type': text(first(s, 'type', 'title')), **a} for (d, p, s), a in zip(supplements, supplement_audits)],
              'resource_pdf_audit': [{'domain': d, 'title': text(first(r, 'title', 'name')), **a} for (d, r), a in zip(resources, resource_audits)]}
    lines[3:3] = [f"论文 {report['papers']} 篇，概念 {report['concepts']} 条；有效 PDF {status_counts['valid_pdf']} 个，本地网页 {html_status_counts['valid_html']} 个（具体材料类型见各条记录）。", '']
    if problems:
        lines += ['## 待处理的问题', ''] + [f'- {md(p)}' for p in problems] + ['']
    # A second view answers the user's primary request directly. Multi-label
    # papers/concepts appear under each label, so category counts are not disjoint.
    level_lines = ['# 按 Knowledge、Law、Science 浏览', '',
                   '> 同一论文可有多个主张，复合标签会在多个层次重复显示。分类和阅读深度来自各方向索引；不是对论文价值的排名，也不是已经穷尽全部文献。', '']
    for category in ('Knowledge', 'Law', 'Science'):
        pattern = re.compile(r'\b' + category + r'\b', re.I)
        group = [(d, p) for d, p in papers if pattern.search(classify(p))]
        concepts = [(d, c) for d, obj in domain_data.items() for c in obj['concepts'] if pattern.search(classify(c))]
        level_lines += [f'## {category}', '', f'论文条目 {len(group)}；概念条目 {len(concepts)}（多标签计入每个相关组）。', '',
                        '| 方向 | 论文 | 标签 | 主张及范围 | 审阅程度 | 本地资料 |',
                        '|---|---|---|---|---|---|']
        for domain, p in group:
            title = text(first(p, 'title', 'name'))
            source = safe_url(first(p, 'url', 'source_url', 'source'))
            heading = f'[{md(title)}]({source})' if source else md(title)
            claim = md(first(p, 'claim', 'summary', 'description')).rstrip('。；; ')
            scope = md(first(p, 'scope', 'limitations', 'boundary'))
            review = md(first(p, 'review_level', 'reading_level', 'verification')) or '未标明'
            a = audit_by_id[id(p)]
            h = html_by_id[id(p)]
            local = f"[PDF](../{a['path']})" if a['status'] == 'valid_pdf' else (f"[网页](../{h['path']})" if h['status'] == 'valid_html' else '未取得')
            detail = claim + (('；范围：' if claim else '范围：') + scope if scope else '')
            level_lines.append(f'| {md(domain)} | {heading} | {md(classify(p))} | {detail} | {review} | {local} |')
        level_lines += ['', '### 概念', '', '| 方向 | 概念 | 标签 | 定义与范围 | 原始来源 | 审阅程度 |', '|---|---|---|---|---|---|']
        for domain, c in concepts:
            name = md(first(c, 'name', 'title', 'concept'))
            desc = md(first(c, 'definition', 'description', 'claim', 'summary'))
            scope = md(first(c, 'scope', 'limitations', 'boundary'))
            rawsource = first(c, 'url', 'source_url', 'source', 'sources')
            if isinstance(rawsource, list):
                rawsource = next((x for x in rawsource if isinstance(x, str) and safe_url(x)), '')
            source = safe_url(rawsource)
            source_link = f'[来源]({source})' if source else '待核'
            review = md(first(c, 'review_level', 'reading_level', 'verification')) or '未标明'
            level_lines.append(f'| {md(domain)} | {name} | {md(classify(c))} | {desc.rstrip("。；; ")}；范围：{scope} | {source_link} | {review} |')
        level_lines.append('')
    # Claim annotations are not the paper's primary label. Surface them
    # separately so the same work can make a theoretical and empirical claim.
    annotations = []
    for domain, paper in papers:
        for annotation in paper.get('claim_annotations', []):
            if not isinstance(annotation, dict):
                problems.append(f"{domain}: invalid claim annotation: {text(first(paper, 'title', 'name'))[:100]}")
                continue
            annotations.append((domain, paper, annotation))
            if not all((classify(annotation), first(annotation, 'claim'), first(annotation, 'scope'),
                        first(annotation, 'evidence'), first(annotation, 'review_level'))):
                problems.append(f"{domain}: incomplete claim annotation: {text(first(paper, 'title', 'name'))[:100]}")
    # Some RL records retain secondary claims separately; do not lose these
    # when rendering the claim view. Their audit level is inherited explicitly,
    # not presented as a separate proof review.
    for domain, paper in papers:
        secondary = paper.get('secondary_claims', [])
        if not isinstance(secondary, list):
            problems.append(f"{domain}: secondary_claims must be a list: {text(first(paper, 'title', 'name'))[:100]}")
            continue
        for original in secondary:
            if not isinstance(original, dict):
                problems.append(f"{domain}: invalid secondary claim: {text(first(paper, 'title', 'name'))[:100]}")
                continue
            annotation = dict(original)
            if not first(annotation, 'review_level'):
                parent_review = text(first(paper, 'review_level', 'reading_level', 'verification'))
                annotation['review_level'] = ('沿用论文等级：' + parent_review) if parent_review else '未标明'
            if not all((classify(annotation), first(annotation, 'claim'), first(annotation, 'scope'),
                        first(annotation, 'evidence'), first(annotation, 'review_level'))):
                problems.append(f"{domain}: incomplete secondary claim: {text(first(paper, 'title', 'name'))[:100]}")
            annotations.append((domain, paper, annotation))
    overlay_path = ROOT / 'literature' / 'anchor_claims.json'
    if overlay_path.exists():
        try:
            overlay = json.loads(overlay_path.read_text(encoding='utf-8'))['annotations']
            if not isinstance(overlay, list):
                raise ValueError('annotations must be an array')
            for annotation in overlay:
                if not isinstance(annotation, dict):
                    problems.append('anchor_claims: annotation must be an object')
                    continue
                identifier = text(annotation.get('arxiv'))
                matches = [(d, p) for d, p in papers if text(first(p, 'arxiv', 'arxiv_id')) == identifier]
                if not identifier or len(matches) != 1:
                    problems.append(f'anchor_claims: arXiv ID must match exactly one indexed paper: {identifier}')
                    continue
                domain, paper = matches[0]
                if not all((classify(annotation), first(annotation, 'claim'), first(annotation, 'scope'),
                            first(annotation, 'evidence'), first(annotation, 'review_level'))):
                    problems.append(f'anchor_claims: incomplete annotation: {identifier}')
                annotations.append((domain, paper, annotation))
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f'invalid anchor claims: {exc}')
    claim_lines = ['# 按具体主张分类', '',
                   '> 本页仅展示已拆分的主张，并不意味着没有展示的论文只包含一个主张。每个层次均限于所列条件；形式结果不自动证明真实系统的完整边界。', '',
                   f'已拆分 {len(annotations)} 条主张；按主张层次分组，论文主标签仍见按层次总表。', '']
    for category in ('Knowledge', 'Law', 'Science'):
        pattern = re.compile(r'\b' + category + r'\b', re.I)
        selected = [(domain, paper, a) for domain, paper, a in annotations if pattern.search(classify(a))]
        claim_lines += [f'## {category}', '', f'{len(selected)} 条主张（复合标签可重复）。', '',
                        '| 方向 | 论文 | 层次 | 主张 | 范围 | 原文位置 | 审阅程度 |',
                        '|---|---|---|---|---|---|---|']
        for domain, paper, annotation in selected:
            title = md(first(paper, 'title', 'name'))
            source = safe_url(first(paper, 'url', 'source_url', 'source'))
            heading = f'[{title}]({source})' if source else title
            claim_lines.append(f"| {md(domain)} | {heading} | {md(classify(annotation))} | {md(first(annotation, 'claim'))} | {md(first(annotation, 'scope'))} | {md(first(annotation, 'evidence'))} | {md(first(annotation, 'review_level'))} |")
        claim_lines.append('')
    (ROOT / 'literature' / 'by_claim.md').write_text('\n'.join(claim_lines) + '\n', encoding='utf-8')
    report['claim_annotations'] = len(annotations)
    (ROOT / 'literature' / 'by_level.md').write_text('\n'.join(level_lines) + '\n', encoding='utf-8')
    (ROOT / 'literature' / 'catalog.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'literature' / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('domains_present','papers','concepts','paper_classifications','paper_review_levels','pdf_statuses','html_statuses','problems')}, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
