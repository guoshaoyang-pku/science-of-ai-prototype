#!/usr/bin/env python3
"""Render this run into the existing D2 page; Git push stays a separate step."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

SOURCE = Path(__file__).resolve().parent
SPEC = json.loads((SOURCE / 'preregistration.json').read_text())
REPO = SOURCE.parents[1]
RUN = Path(os.environ['AIQ_KB_DATA_ROOT']) / 'runs' / SPEC['run_name'] if 'AIQ_KB_DATA_ROOT' in os.environ else REPO / 'evidence' / SPEC['run_name']
BLOG = Path(os.environ.get('AIQ_KB_BLOG_ROOT', 'BLOG_CHECKOUT'))
DEST = BLOG / 'blogs/kb_site/v2'
LEGACY = REPO / 'science_program_v2'


def main():
    if not os.environ.get('AIQ_KB_BLOG_ROOT'):
        raise RuntimeError('Set AIQ_KB_BLOG_ROOT to the intended blog checkout')
    spec = importlib.util.spec_from_file_location('legacy_renderer', LEGACY / 'reports/publish_to_blog.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    markdown = (RUN / 'report.md').read_text()
    body = renderer.md_to_html(markdown)
    DEST.mkdir(parents=True, exist_ok=True)
    used_figs = re.findall(r'<img[^>]*src="([^"]+)"', body)
    for relative in used_figs:
        source = RUN / relative
        if not source.is_file():
            raise RuntimeError(f'Missing figure: {source}')
        for suffix in ('png', 'pdf', 'svg'):
            figure = source.with_suffix('.' + suffix)
            if figure.is_file():
                target = DEST / 'figs' / ('d2_20261009_' + figure.name)
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(figure, target)
                if suffix == 'svg':
                    target.write_text(chr(10).join(line.rstrip() for line in target.read_text().splitlines()) + chr(10))
        body = body.replace(f'src="{relative}"', f'src="figs/d2_20261009_{source.name}"')
    evidence = {
        'studies/n_sigma_tasks/preregistration.json': 'preregistration.json',
        'studies/n_sigma_tasks/summary.json': 'summary.json',
        'studies/n_sigma_tasks/supplement_summary.json': 'supplement_summary.json',
        'studies/n_sigma_tasks/verification.json': 'verification.json',
        'studies/legacy_curve_analysis/summary.json': 'legacy_summary.json',
    }
    evidence_dir = DEST / 'd2_evidence_20261009'
    evidence_dir.mkdir(exist_ok=True)
    for relative, name in evidence.items():
        value = json.loads((RUN / relative).read_text())
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, str):
                return obj.replace(str(RUN), 'AIQ_KB_DATA/runs/' + SPEC['run_name']).replace(str(LEGACY), 'science_program_v2').replace(SPEC['data_root_default'], 'AIQ_KB_DATA')
            return obj
        (evidence_dir / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + chr(10))
        encoded = f'（<code>{relative}</code>）'
        body = body.replace(encoded, f'（<a href="d2_evidence_20261009/{name}">存档</a>）')
    body += '<p><a href="d2_evidence_20261009/report.md">下载 Markdown</a> · <a href="d2_evidence_20261009/sources.zip">重建源码</a></p>'
    shutil.copy2(RUN / 'report.md', evidence_dir / 'report.md')
    import zipfile
    with zipfile.ZipFile(evidence_dir / 'sources.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.iterdir()):
            if path.suffix in ('.py', '.json'):
                archive.write(path, arcname=path.name)
        archive.write(RUN / 'studies/legacy_curve_analysis/analysis.py', arcname='legacy_analysis.py')
    body = re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>', body, flags=re.S)
    html = renderer.page('D2：风险何时回升，以及这个规律能否跨任务', body)
    html = html.replace('</head>', '<style>img{max-width:100%;height:auto} .table-wrap{overflow-x:auto}table{min-width:620px}body{overflow-wrap:break-word}mjx-container[display="true"]{max-width:100%;overflow-x:auto;overflow-y:hidden}</style></head>')
    pages = ['D2_overfitting_u_curve.html', 'D2_overfitting_u_curve_integrated.html']
    for name in pages:
        (DEST / name).write_text(html)
    index = DEST / 'index.html'
    text = index.read_text()
    item = re.compile(r'(<li><a class="t" href="D2_overfitting_u_curve.html">).*?(</li>)', re.S)
    replacement = '\\1D2：风险回升与跨任务边界</a> · <a href="D2_overfitting_u_curve_integrated.html">整合版</a><span class="s">旧范围 U≈0.28·n/σ²；扩域与跨任务统一指数失败。新增 train/val、信号噪声与机制图。</span>\\2'
    updated, count = item.subn(replacement, text)
    if count != 1:
        raise RuntimeError('D2 index entry not uniquely found')
    index.write_text(updated)
    manifest = {'report_sha256': hashlib.sha256((RUN / 'report.md').read_bytes()).hexdigest(), 'html_sha256': hashlib.sha256(html.encode()).hexdigest(), 'pages': pages, 'figures': used_figs, 'canonical_run': str(RUN)}
    (RUN / 'publication.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
