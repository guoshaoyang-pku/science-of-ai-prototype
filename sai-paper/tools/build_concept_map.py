#!/usr/bin/env python3
"""Join concepts to explicit primary-source identifiers, preserving source versions.

Writes concept_sources.md, concept_source_map.json and concept_coverage_gaps.md.
A URL/identifier join does not establish semantic coverage or claim validity.
No downloads and no changes to source indexes or PDFs.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ('interp', 'rl', 'pretrain')
ARXIV_ID = r'(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.-]*/\d{7})'


def md(value):
    return str(value or '').replace('|', r'\|').replace('\n', ' ').replace('\r', ' ')


def first(item, *keys):
    for key in keys:
        value = item.get(key)
        if value not in (None, '', []):
            return value
    return ''


def arxiv_key(raw):
    if not isinstance(raw, str):
        return ''
    match = re.fullmatch(r'(' + ARXIV_ID + r')(?:v\d+)?', raw.strip())
    return 'arxiv:' + match.group(1).lower() if match else ''


def url_key(raw):
    if not isinstance(raw, str) or not re.fullmatch(r'https?://[^\s<>]+', raw):
        return ''
    parsed = urlsplit(raw)
    host = parsed.netloc.lower().removeprefix('www.')
    path = parsed.path.rstrip('/')
    if host in ('arxiv.org', 'export.arxiv.org'):
        match = re.fullmatch(r'/(?:abs|pdf|html)/(' + ARXIV_ID + r')(?:v\d+)?(?:\.pdf)?', path)
        if match:
            return 'arxiv:' + match.group(1).lower()
    return 'url:' + host + path + ('?' + parsed.query if parsed.query else '')


def source_list(item):
    result = []
    for name in ('primary_source', 'primary_sources', 'url', 'source_url', 'source', 'sources', 'pdf_source', 'pdf_source_url', 'pdf_url'):
        raw = item.get(name)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if url_key(value) and value not in result:
                result.append(value)
    return result


def source_record(domain, paper, kind, audit_map, supplement_map):
    title = first(paper, 'title', 'name')
    status = audit_map.get((domain, title), {})
    pdf = status.get('path') if status.get('status') == 'valid_pdf' else None
    html = status.get('html', {})
    html_path = html.get('path') if html.get('status') == 'valid_html' else None
    supplements = []
    for supplement in paper.get('supplemental', []):
        if not isinstance(supplement, dict):
            continue
        checked = supplement_map.get((domain, title, supplement.get('pdf_path')), {})
        supplements.append({
            'type': first(supplement, 'type', 'title'),
            'title': first(supplement, 'title', 'type'),
            'url': first(supplement, 'url', 'pdf_source', 'pdf_source_url'),
            'version_note': first(supplement, 'version_note', 'source_version'),
            'review_level': first(supplement, 'review_level', 'reading_level'),
            'audit_status': checked.get('status', 'not_in_audit'),
            'local_path': checked.get('path') if checked.get('status') == 'valid_pdf' else None,
        })
    return {
        'domain': domain, 'title': title, 'kind': kind,
        'url': first(paper, 'url', 'source_url'),
        'source_version': first(paper, 'source_version', 'version_note'),
        'classification': first(paper, 'classification', 'category'),
        'scope': paper.get('scope', ''),
        'review_level': first(paper, 'review_level', 'reading_level'),
        'main_pdf_status': status.get('status', 'not_in_audit'),
        'verified_main_pdf': pdf, 'verified_html': html_path,
        'html_kind': html.get('kind', 'unspecified'),
        'verified_native_html': html_path if html.get('kind') == 'native_article' else None,
        'supplements': supplements,
    }


def render_record(record):
    path = record['verified_main_pdf'] or record['verified_html']
    if path:
        if record['verified_main_pdf']:
            label = '本地登记版本'
        elif record['html_kind'] == 'licensed_transcription':
            label = '本地许可 HTML 转录'
        else:
            label = '本地网页'
        local = f' [{label}](../{path})'
    else:
        local = '（主文登记文件未核验可用）'
    rendered = f"{md(record['title'])} [{md(record['domain'])}，{md(record['kind'])}]{local}"
    labels = {
        'primary_paper_anonymous_review_draft_not_final': '匿名审稿稿，非最终版',
        'author_conference_poster_not_paper': '作者海报，非论文全文',
    }
    for supplement in record['supplements']:
        if supplement['local_path']:
            label = labels.get(supplement['type'], supplement['title'])
            rendered += f"；补充：[{md(label)}](../{supplement['local_path']})"
    return rendered


def main():
    docs = {d: json.loads((ROOT / 'literature' / (d + '.json')).read_text()) for d in DOMAINS}
    audit = json.loads((ROOT / 'literature' / 'audit.json').read_text())
    audit_map = {(x['domain'], x['title']): x for x in audit['paper_pdf_audit']}
    papers = [(d, p) for d, doc in docs.items() for p in doc['papers']]
    if set(audit_map) != {(d, first(p, 'title', 'name')) for d, p in papers}:
        raise ValueError('Paper inventory differs from audit.json; run build_catalog.py first.')
    audit_map.update({(x['domain'], x['title']): x for x in audit.get('resource_pdf_audit', [])})
    supplement_map = {
        (x['domain'], x['paper'], x.get('declared_path')): x
        for x in audit.get('supplemental_pdf_audit', [])
    }
    resources = [(d, r) for d, doc in docs.items() for r in doc.get('resources', []) if isinstance(r, dict)]
    entries = [(d, p, '论文') for d, p in papers] + [(d, r, '教材／资源') for d, r in resources]
    by_key = defaultdict(list)
    for domain, item, kind in entries:
        keys = {url_key(s) for s in source_list(item)}
        keys.add(arxiv_key(first(item, 'arxiv', 'arxiv_id')))
        for key in keys - {''}:
            by_key[key].append((domain, item, kind))
    records = []
    linked_papers = set()
    for domain, doc in docs.items():
        for concept in doc['concepts']:
            sources = source_list(concept)
            matches = []
            seen = set()
            for source in sources:
                for pdomain, paper, kind in by_key.get(url_key(source), []):
                    identity = (pdomain, first(paper, 'title', 'name'), kind)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if kind == '论文':
                        linked_papers.add(identity[:2])
                    matches.append(source_record(pdomain, paper, kind, audit_map, supplement_map))
            records.append({
                'domain': domain, 'name': first(concept, 'name', 'title', 'concept'),
                'classification': first(concept, 'classification', 'category', 'level', 'label'),
                'review_level': first(concept, 'review_level', 'reading_level', 'verification'),
                'definition': concept.get('definition', ''), 'claim': concept.get('claim', ''),
                'scope': concept.get('scope', ''), 'evidence': concept.get('evidence', ''),
                'primary_evidence': concept.get('primary_evidence', ''),
                'source_annotations': concept.get('source_annotations', []),
                'sources': sources, 'matches': matches,
            })
    gaps = [
        source_record(d, p, '论文', audit_map, supplement_map)
        for d, p in papers if (d, first(p, 'title', 'name')) not in linked_papers
    ]
    mapped = sum(bool(r['matches']) for r in records)
    metadata = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'audit_generated_utc': audit.get('generated_utc'),
        'concepts': len(records), 'mapped': mapped, 'unmatched': len(records) - mapped,
        'paper_records': len(papers), 'papers_with_explicit_concept_source_link': len(linked_papers),
        'papers_without_explicit_concept_source_link': len(gaps),
        'unlinked_by_domain': dict(Counter(r['domain'] for r in gaps)),
    }
    lines = [
        '# 概念与来源论文对照', '',
        '> 按明确来源 URL 或 arXiv ID 关联，不使用语义相似度；匹配不代表论文全面定义或验证该概念。概念审阅程度沿用源记录。', '',
        f'已自动关联 {mapped} 条概念；{len(records) - mapped} 条未匹配。', '',
        f"另有 {len(gaps)} 篇论文没有任何概念显式引用其已登记来源，见[反向核查队列](concept_coverage_gaps.md)。这不等于它们没有被已有概念覆盖。", '',
        f"文件可用性来自 {audit.get('generated_utc', '未知时间')} 的总审计。登记主文、许可 HTML 转录、匿名审稿稿、作者海报分别显示；转录及补充材料不冒充出版 PDF 或最终版。网页的资产可能不完整。", '',
    ]
    for domain in DOMAINS:
        lines += [f'## {domain}', '', '| 概念 | 层次 | 来源 | 自动匹配的目录条目 | 概念审阅程度 |', '|---|---|---|---|---|']
        for record in records:
            if record['domain'] != domain:
                continue
            sources = '<br>'.join(f'[来源 {i}]({s})' for i, s in enumerate(record['sources'], 1)) or '未标明'
            matches = '<br>'.join(render_record(r) for r in record['matches']) or '未自动匹配；需人工核查'
            lines.append(f"| {md(record['name'])} | {md(record['classification'])} | {sources} | {matches} | {md(record['review_level']) or '未标明'} |")
        lines.append('')
    gap_lines = [
        '# 已收论文与概念来源的反向核查队列', '',
        f"当前 {len(papers)} 篇论文中，{len(linked_papers)} 篇被至少一个概念通过明确来源标识关联，{len(gaps)} 篇尚未关联。", '',
        '这是书目关联的待核队列，不是语义缺漏判决或领域覆盖率。一个已有概念可以覆盖多篇论文；应先核查同义概念，再决定补来源、补定义或建立独立概念。无需为每篇论文机械新建概念。', '',
        '同时保留论文标签、scope 与审阅深度，避免把没有概念链接误读成没有分类或没有读过论文。', '',
    ]
    for domain in DOMAINS:
        gap_lines += [f'## {domain}', '', '| 已收论文与文件 | 现有分类 | 现有 scope | 论文审阅程度 |', '|---|---|---|---|']
        for record in gaps:
            if record['domain'] == domain:
                gap_lines.append(f"| {render_record(record)} | {md(record['classification'])} | {md(record['scope'])} | {md(record['review_level'])} |")
        gap_lines.append('')
    gap_lines += [f"源文件审计时间：{audit.get('generated_utc')}。生成时间：{metadata['generated_utc']}。", '']
    (ROOT / 'literature' / 'concept_sources.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'literature' / 'concept_coverage_gaps.md').write_text('\n'.join(gap_lines) + '\n', encoding='utf-8')
    (ROOT / 'literature' / 'concept_source_map.json').write_text(
        json.dumps({'metadata': metadata, 'scope_note': 'Identifier links, not semantic coverage or validation.',
                    'records': records, 'papers_without_explicit_concept_source_link': gaps}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
