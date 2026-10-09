#!/usr/bin/env python3
"""Audit a report panel (HTML or Markdown) before handing it to a human.

Checks, in reading order:
  1. Extra elements  - subtitles, KPI cards, section numbers, status/PID/SHA/snapshot lines,
                       nav/download buttons, color emoji. Minimalism: delete what the reader does not need.
  2. Jargon          - project-specific terms, symbols, codenames and run IDs, listed at their FIRST
                       appearance, flagged when no plain-language explanation follows nearby.
  3. First screen    - everything before the first <h2>: title + one-sentence conclusion + settings
                       row. Nothing else. Jargon here is the worst offense.

The script only flags candidates. The agent must read each flag and fix or dismiss it.

Usage:
  python panel_audit.py report.html
  python panel_audit.py report.html --allow SGD Adam loss token   # terms the reader already knows
"""
import argparse
import re
import sys
from html.parser import HTMLParser

SKIP = {"script", "style", "noscript", "head", "title"}
BLOCK = {"h1", "h2", "h3", "h4", "p", "li", "td", "th", "figcaption", "summary", "div", "section",
         "header", "footer", "nav", "button", "a", "span", "text", "label", "option", "dt", "dd", "pre"}

EXTRA_TEXT = [
    (r"PID\s*\d+", "process ID: operational noise, not a finding"),
    (r"SHA-?256|sha256", "checksum: belongs in a log, not the panel"),
    (r"快照|静态快照|snapshot|生成时间|生成于|generated (at|on)|built \d", "snapshot/generation stamp: keep one footer line at most"),
    (r"^\s*\d{1,2}\s*[·、]\s*\S|^\s*\d{1,2}\.\s+\S", "numbered section prefix: drop the number"),
    (r"跳到正文|skip to content", "skip link: unneeded on a short page"),
    (r"查看\s*Markdown|下载\s*Markdown|打印|download|export", "action buttons: does the reader need them?"),
    (r"本页是|本页面|this page (is|shows)", "page talking about itself: delete"),
    (r"阅读范围|reading scope|说明[:：]", "scope/disclaimer banner: fold into the conclusion or delete"),
    (r"运行中|存活|alive|running|已归档|归档", "status label: keep only if the reader acts on it"),
]
EXTRA_CLASS = re.compile(r"kpi|stat-?card|metric-?card|big-?num|hero|subtitle|tagline|eyebrow|kicker|badge-lg|scorecard", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F000-\U0001F2FF]")

TERM = re.compile(
    r"[A-Za-z]+_[A-Za-z0-9_]+"                 # snake_case codenames: soa_v1, luna_main
    r"|[A-Za-z]+[-_]?v\d+(?:\.\d+)*"           # versioned names: v2, soa_v4
    r"|\b[A-Z]\d{1,4}\b"                       # round/run IDs: R55, D3, E8
    r"|\b[a-z]+\d+[a-z0-9]*\b"                 # e8, macro15, r0
    r"|\b[A-Z]{2,}[a-z]*\b"                    # acronyms: RGI, KB, noKB
    r"|\b[a-z]+[A-Z][A-Za-z]*\b"               # camelCase: noKB
    r"|[A-Za-zα-ωΑ-Ω][₀-₉]+"                   # subscripted symbols: Q₂
    r"|[α-ωΑ-Ω]"                               # bare Greek symbols
    r"|(?<![A-Za-z0-9.])[A-Z](?![A-Za-z0-9])"  # single-letter variables: D, A, M (not the B in 0.8B)
)
DEFINE_NEAR = re.compile(r"^.{0,3}[（(：:=]|^.{0,6}(是|指|即|表示|代表|意思是|也就是|，即|——)")


class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.blocks = []          # (tag, text, h2_count_at_start)
        self.buf = []
        self.h2 = 0
        self.extra_class = []

    def flush(self, tag="p"):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        if t:
            self.blocks.append((tag, t, self.h2))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
            return
        cls = " ".join(v or "" for k, v in attrs if k in ("class", "id"))
        if cls and EXTRA_CLASS.search(cls):
            self.extra_class.append(f"<{tag} {cls[:60]}>")
        if tag in BLOCK:
            self.flush()
        if tag == "h2":
            self.h2 += 1

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag in BLOCK:
            self.flush(tag)

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def load_blocks(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    if path.endswith((".md", ".markdown")):
        blocks, h2 = [], 0
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("```"):
                continue
            if s.startswith("## "):
                h2 += 1
            blocks.append(("md", s.lstrip("#").strip(), h2))
        return blocks, []
    p = Text()
    p.feed(raw)
    p.flush()
    return p.blocks, p.extra_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--allow", nargs="*", default=[], help="terms the reader already knows")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args()
    allow = {t.lower() for t in a.allow}

    blocks, extra_class = load_blocks(a.path)
    first_screen = [b for b in blocks if b[2] < 1]
    issues = 0

    print(f"== 1. 多余元素（{a.path}）")
    for c in extra_class:
        print(f"  [class] {c}  -> KPI 卡 / 副标题类元素：确认读者需要，否则删除")
        issues += 1
    seen = set()
    for tag, t, _ in blocks:
        for pat, why in EXTRA_TEXT:
            if re.search(pat, t, re.I | re.M) and (pat, t[:30]) not in seen:
                seen.add((pat, t[:30]))
                print(f"  [{tag}] {t[:70]!r}  -> {why}")
                issues += 1
    n_emoji = sum(len(EMOJI.findall(t)) for _, t, _ in blocks)
    if n_emoji and not a.path.endswith(".md"):
        print(f"  [emoji] 页面里有 {n_emoji} 个彩色 emoji -> HTML 里换成单色线性图标")
        issues += 1

    print("\n== 2. 首屏（第一个 h2 之前）")
    print(f"  {len(first_screen)} 个文本块（目标：标题 + 一句话结论 + settings 行，约 ≤ 8 块）")
    if len(first_screen) > 8:
        issues += 1

    print("\n== 3. 术语首次出现（只列附近没有解释的）")
    first = {}
    for i, (tag, t, h2) in enumerate(blocks):
        for m in TERM.finditer(t):
            term = m.group(0)
            if term.lower() in allow or term in first:
                continue
            if t[m.start() - 1:m.start()] in ("/", ".") and m.start() or t[m.end():m.end() + 1] in ("/", "."):
                continue  # part of a path or file name
            after = t[m.end():m.end() + 24]
            defined = bool(DEFINE_NEAR.search(after))
            first[term] = (i, h2, defined, t[max(0, m.start() - 16):m.end() + 24])
    shown = 0
    for term, (i, h2, defined, ctx) in sorted(first.items(), key=lambda kv: kv[1][0]):
        if defined:
            continue
        where = "首屏" if h2 < 1 else f"第 {h2} 节"
        print(f"  {term:<14} {where:<6} …{ctx}…")
        shown += 1
        issues += 1
        if shown >= a.max:
            print(f"  …（还有更多，用 --max 调大）")
            break

    print(f"\n共 {issues} 处待确认。读者已知的通用术语用 --allow 排除后重跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
