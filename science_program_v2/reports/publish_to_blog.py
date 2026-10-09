#!/usr/bin/env python3
"""Publish v2 direction reports to the kb_site blog for human review.

Converts each directions/Dx/report.md (+ morning/lines reports) to styled
HTML under <blog>/blogs/kb_site/v2/, builds an index page, commits and
pushes. Rerunnable: run again after the line parks for a final sync.

Usage: python reports/publish_to_blog.py [--no-push]
"""
import re
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import markdown

PROGRAM_DIR = Path(__file__).resolve().parent.parent
BLOG = Path(os.environ.get("AIQ_KB_BLOG_ROOT", "BLOG_CHECKOUT"))
DEST = BLOG / "blogs/kb_site/v2"
SITE = "https://guoshaoyang-pku.github.io/blogs/kb_site/v2"

CSS = """
:root { --dark:#141413; --light:#faf9f5; --mid:#b0aea5; --soft:#e8e6dc;
        --muted:#5e5d59; --orange:#d97757; --blue:#6a9bcc; --green:#788c5d;
        --head:"Poppins","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
        --body:"Lora","Noto Serif CJK SC","Songti SC",Georgia,serif;
        --mono:"SF Mono",Menlo,Consolas,monospace; }
* { box-sizing:border-box; }
body { margin:0; background:var(--light); color:var(--dark); font:17px/1.8 var(--body); }
main { max-width:780px; margin:0 auto; padding:56px 24px 96px; }
h1,h2,h3 { font-family:var(--head); font-weight:600; line-height:1.35; }
h1 { font-size:28px; margin:0 0 14px; }
h2 { font-size:21px; margin:44px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--soft); }
a { color:inherit; text-decoration:underline; text-decoration-color:var(--orange); text-underline-offset:3px; }
code { font-family:var(--mono); font-size:.85em; background:var(--soft); padding:1px 5px; border-radius:4px; }
pre { background:#0d1117; color:#d7dde4; border-radius:8px; padding:12px 14px; overflow-x:auto; font-size:13px; }
pre code { background:none; padding:0; color:inherit; }
table { width:100%; border-collapse:collapse; font:14px/1.6 var(--head); margin:14px 0; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--soft); vertical-align:top; }
th { font-weight:500; color:var(--muted); }
blockquote { border-left:3px solid var(--orange); margin:16px 0; padding:2px 0 2px 14px; color:var(--muted); }
.nav { font:14px var(--head); margin-bottom:28px; color:var(--muted); }
.stamp { font:13px var(--head); color:var(--mid); margin-top:60px; }
ul.idx { list-style:none; padding:0; }
ul.idx li { margin:0 0 18px; }
ul.idx .t { font:600 16px var(--head); }
ul.idx .s { display:block; font-size:14.5px; color:var(--muted); margin-top:2px; }
"""

TITLE_RE = re.compile(r"^#\s+(.*)$", re.M)


def neutralize_links(md_text):
    # relative links cannot resolve on the web: keep text, show path as code;
    # leave image syntax ![alt](path) untouched
    return re.sub(r"(?<!\!)\[([^\]]+)\]\((?!https?:)([^)]+)\)",
                  r"\1（<code>\2</code>）", md_text)


def first_conclusion(md_text, limit=160):
    m = re.search(r"^##\s*结论\s*$(.*?)(?=^##\s|\Z)", md_text, re.M | re.S)
    if not m:
        return ""
    body = m.group(1)
    for line in body.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if len(line) >= 12 and not line.startswith(("#", "|", "```", ">")):
            line = re.sub(r"[*_`]", "", line)
            return line[:limit] + ("…" if len(line) > limit else "")
    return ""


def page(title, body_html, nav='<a href="index.html">← 报告索引</a>'):
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><link rel="stylesheet" href="style.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)'],['$','$']],displayMath:[['\\\\[','\\\\]'],['$$','$$']]}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body><main><div class="nav">{nav}</div>
{body_html}
<div class="stamp">同步于 {datetime.now().strftime('%Y-%m-%d %H:%M')}（北京时间）· 源文件在研究仓库内随轮次更新，本页为快照</div>
</main></body></html>"""


def protect_math(text):
    """Extract LaTeX math spans before markdown so _ * < are not mangled."""
    store = []

    def repl(m):
        store.append(m.group(0).replace("<", "&lt;"))
        return f"@@MATH{len(store) - 1}@@"

    def sub_seg(seg):
        seg = re.sub(r"\$\$[\s\S]+?\$\$", repl, seg)
        seg = re.sub(r"\\\[[\s\S]+?\\\]", repl, seg)
        seg = re.sub(r"\\\(.+?\\\)", repl, seg)
        seg = re.sub(r"\$[^$\n]+?\$", repl, seg)
        # keep underscores inside link/image targets raw so markdown still parses them
        seg = re.sub(r"\]\([^)\s]+\)",
                     lambda m: m.group(0).replace("_", "\x01"), seg)
        # identifiers like U_rise / t_* must not become markdown italics
        seg = seg.replace("_", "\\_")
        seg = seg.replace("\x01", "_")
        seg = re.sub(r"(?<!\*)\*(?!\*)", "\\*", seg)
        return seg

    out = [sub_seg(p) if i % 2 == 0 else p
           for i, p in enumerate(re.split(r"(```.*?```|`[^`\n]+`)", text, flags=re.S))]
    return "".join(out), store


def md_to_html(md_text):
    protected, store = protect_math(neutralize_links(md_text))
    html = markdown.markdown(protected,
                             extensions=["tables", "fenced_code", "sane_lists"])
    for i, s in enumerate(store):
        html = html.replace(f"@@MATH{i}@@", s)
    return html


def git(*args):
    r = subprocess.run(["git", "-C", str(BLOG), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def main():
    raise RuntimeError('Historical publisher would overwrite the newer D2 page. Use the D2 publisher after explicit configuration; this copy is a read-only renderer.')
    push = "--no-push" not in sys.argv
    git("pull", "--rebase", "origin", "main")  # before generating: tracked-file edits block rebase
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "style.css").write_text(CSS)

    directions = sorted((PROGRAM_DIR / "directions").iterdir(),
                        key=lambda p: int(re.match(r"D(\d+)", p.name).group(1))
                        if re.match(r"D(\d+)", p.name) else 999)
    index_items = []
    for d in directions:
        report = d / "report.md"
        if not report.exists():
            continue
        text = report.read_text()
        title = (TITLE_RE.search(text).group(1).strip()
                 if TITLE_RE.search(text) else d.name)
        html = md_to_html(text)
        for src in re.findall(r'<img[^>]*src="([^"]+)"', html):
            if src.startswith("http"):
                continue
            f = d / src
            if f.exists():
                (DEST / "figs").mkdir(exist_ok=True)
                shutil.copy(f, DEST / "figs" / f.name)
                html = html.replace(f'src="{src}"', f'src="figs/{f.name}"')
        (DEST / f"{d.name}.html").write_text(page(title, html))
        integ = d / "INTEGRATED.md"
        has_integ = integ.exists()
        if has_integ:
            ihtml = md_to_html(integ.read_text())
            for src in re.findall(r'<img[^>]*src="([^"]+)"', ihtml):
                if src.startswith("http"):
                    continue
                f = d / src
                if f.exists():
                    (DEST / "figs").mkdir(exist_ok=True)
                    shutil.copy(f, DEST / "figs" / f.name)
                    ihtml = ihtml.replace(f'src="{src}"', f'src="figs/{f.name}"')
            (DEST / f"{d.name}_integrated.html").write_text(
                page(title + "（整合版）", ihtml))
        index_items.append((d.name, title, first_conclusion(text), has_integ))

    for name, src in [("morning_2026-10-07", PROGRAM_DIR / "reports/MORNING_REPORT_2026-10-07.md"),
                      ("lines_overview", PROGRAM_DIR / "reports/LINES_REPORT_2026-10-06.md")]:
        if src.exists():
            text = src.read_text()
            title = (TITLE_RE.search(text).group(1).strip()
                     if TITLE_RE.search(text) else name)
            (DEST / f"{name}.html").write_text(page(title, md_to_html(text)))

    idx = ['<h1>science_program_v2 · 方向报告（供 review）</h1>',
           '<p style="font-size:15px">中心化自动研究线（Sol 6.1 Ultra，2026-10-06 启动）。'
           '每份报告结构：结论 → Formulation → 成立程度 → support 材料后置。'
           '研究仍在进行，本页为快照；<a href="morning_2026-10-07.html">晨报</a> · '
           '<a href="lines_overview.html">新旧实验线总览</a> · '
           '<a href="../index.html#runs">旧线 runs</a></p>', '<ul class="idx">']
    for name, title, concl, has_integ in index_items:
        extra = (f' · <a href="{name}_integrated.html">整合版（读者面向）</a>'
                 if has_integ else "")
        idx.append(f'<li><a class="t" href="{name}.html">{title}</a>{extra}'
                   f'<span class="s">{concl}</span></li>')
    idx.append("</ul>")
    (DEST / "index.html").write_text(page(
        "science_program_v2 报告索引", "\n".join(idx),
        nav='<a href="../index.html">← kb_site 首页</a>'))

    print(f"generated {len(index_items)} direction pages + index + 2 reports")
    subprocess.run(["git", "-C", str(BLOG), "add", "blogs/kb_site/v2"], check=True)
    if subprocess.run(["git", "-C", str(BLOG), "diff", "--cached", "--quiet"]).returncode == 0:
        print("no changes to commit")
        return
    git("commit", "-m", "blog(kb_site): sync v2 direction reports snapshot")
    if push:
        git("push", "origin", "main")
        print("pushed;", "Pages rebuild ~1 min; verify:", SITE + "/index.html")


if __name__ == "__main__":
    main()
