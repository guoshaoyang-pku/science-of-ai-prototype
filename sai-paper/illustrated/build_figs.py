#!/usr/bin/env python3
"""Draw the figures of SAI_illustrated.md as static SVG.

Every figure carries its own light surface, so it reads the same on light and dark pages, and uses CJK
system fonts. Marks that text must not cross have class "nt"; boxes that must contain their text have
class "card". A browser layout check relies on both classes.
"""
import itertools
import math
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
W = 880
FONT = "PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, sans-serif"

C = {
    "surface": "#FAF9F5", "frame": "#E6E3DA", "card": "#F3F1EA", "white": "#FFFFFF",
    "ink": "#141413", "ink2": "#3D3D3A", "muted": "#5E5D59", "faint": "#6F6D66",
    "edge": "#DAD6CB", "grid": "#E8E6DC", "cube": "#DFDBD0", "axisln": "#CFCBC0",
    "clay": "#D97757", "clayd": "#A8492C", "clayf": "#EFC1A8", "clayw": "#F7E3D8",
    "red": "#A3291F", "olive": "#9DB57E", "olived": "#4A6328", "olivek": "#2F4A16",
    "blue": "#2E6CAB", "blued": "#245A93", "bluef": "#DCE8F4",
    "amber": "#8A6410", "gray": "#E4E1D8", "grayd": "#B9B5AA",
}

SUBSUP = re.compile(r"([_^])\{([^}]*)\}")


def n(v):
    return f"{round(v, 2):g}"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plain(s):
    return SUBSUP.sub(lambda m: m.group(1) + m.group(2), s)


def rich(s, size):
    """Text with _{sub} and ^{sup} markup as tspans."""
    out, pos, shift = [], 0, 0.0
    for m in SUBSUP.finditer(s):
        if m.start() > pos:
            seg = esc(s[pos:m.start()])
            out.append(f'<tspan dy="{n(-shift)}">{seg}</tspan>' if shift else seg)
            shift = 0.0
        d = round(size * (0.3 if m.group(1) == "_" else -0.4), 1)
        out.append(f'<tspan dy="{n(d - shift)}" font-size="{n(size * 0.72)}">{esc(m.group(2))}</tspan>')
        shift = d
        pos = m.end()
    if pos < len(s):
        seg = esc(s[pos:])
        out.append(f'<tspan dy="{n(-shift)}">{seg}</tspan>' if shift else seg)
    return "".join(out)


def _cw(ch):
    if ch == " ":
        return 0.32
    if ord(ch) >= 0x2E80 or ch in "，。：；、（）“”‘’！？《》【】—…":
        return 1.0
    if ch in "ilj.,:;'|!·()[]":
        return 0.32
    if ch in "mwMW":
        return 0.86
    if ch.isupper() or ch.isdigit():
        return 0.64
    return 0.58


def tw(s, size):
    """Approximate rendered width of a label, for chips and label backgrounds."""
    total, pos = 0.0, 0
    for m in SUBSUP.finditer(s):
        total += sum(map(_cw, s[pos:m.start()])) * size + sum(map(_cw, m.group(2))) * size * 0.72
        pos = m.end()
    return total + sum(map(_cw, s[pos:])) * size


class Fig:
    def __init__(self, name, h, title, subtitle=None):
        self.name, self.h, self.title = name, h, title
        self.p = name.split("_")[0]
        self.body, self.extra = [], []
        self.body.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="14" '
                         f'fill="{C["surface"]}" stroke="{C["frame"]}"/>')
        self.t(32, 42, title, 19, weight=700)
        if subtitle:
            self.t(32, 66, subtitle, 13, C["muted"])

    def url(self, k):
        return f"url(#{self.p}-{k})"

    # primitives -----------------------------------------------------------------------------
    def t(self, x, y, s, size=13, fill=None, weight=None, anchor=None, extra=""):
        a = f'x="{n(x)}" y="{n(y)}" font-size="{n(size)}" fill="{fill or C["ink"]}"'
        if weight:
            a += f' font-weight="{weight}"'
        if anchor:
            a += f' text-anchor="{anchor}"'
        self.body.append(f"<text {a}{extra}>{rich(s, size)}</text>")

    def tbg(self, x, y, s, size=12, fill=None, weight=None, anchor=None, pad=4):
        """Text on a surface-colored plate, for labels that sit on hatching."""
        w = tw(s, size)
        x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        self.r(x0 - pad, y - size * 0.98, w + 2 * pad, size * 1.32, C["surface"], rx=3, opacity=0.94)
        self.t(x, y, s, size, fill, weight, anchor)

    def r(self, x, y, w, h, fill="none", stroke=None, sw=1, rx=0, cls=None, dash=None, opacity=None, extra=""):
        a = f'x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" fill="{fill}"'
        if rx:
            a += f' rx="{n(rx)}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{n(sw)}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        if cls:
            a += f' class="{cls}"'
        self.body.append(f"<rect {a}{extra}/>")

    def l(self, x1, y1, x2, y2, stroke, sw=1.5, dash=None, arrow=None, cls="nt"):
        a = f'x1="{n(x1)}" y1="{n(y1)}" x2="{n(x2)}" y2="{n(y2)}" stroke="{stroke}" stroke-width="{n(sw)}" stroke-linecap="round"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if arrow:
            a += f' marker-end="url(#{self.p}-ar-{arrow})"'
        if cls:
            a += f' class="{cls}"'
        self.body.append(f"<line {a}/>")

    def path(self, d, stroke="none", sw=1.5, fill="none", dash=None, arrow=None, cls="nt", extra=""):
        a = f'd="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{n(sw)}" stroke-linecap="round" stroke-linejoin="round"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if arrow:
            a += f' marker-end="url(#{self.p}-ar-{arrow})"'
        if cls:
            a += f' class="{cls}"'
        self.body.append(f"<path {a}{extra}/>")

    def c(self, cx, cy, r, fill, stroke=None, sw=2, cls="nt", opacity=None):
        a = f'cx="{n(cx)}" cy="{n(cy)}" r="{n(r)}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{n(sw)}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        if cls:
            a += f' class="{cls}"'
        self.body.append(f"<circle {a}/>")

    def dot(self, cx, cy, r, fill):
        """A data mark with a 2px surface ring."""
        self.c(cx, cy, r, fill, C["surface"], 2)

    def ring(self, cx, cy, r=7, color=None):
        self.c(cx, cy, r, C["surface"], color or C["clay"], 2.4)
        self.c(cx, cy, r * 0.4, color or C["clay"])

    def cross(self, cx, cy, s, stroke, sw=2.2):
        self.path(f"M{n(cx - s)} {n(cy - s)}L{n(cx + s)} {n(cy + s)}M{n(cx + s)} {n(cy - s)}L{n(cx - s)} {n(cy + s)}", stroke, sw)

    def check(self, cx, cy, s, stroke, sw=2.2):
        self.path(f"M{n(cx - s)} {n(cy)}L{n(cx - s * 0.3)} {n(cy + s * 0.7)}L{n(cx + s)} {n(cy - s * 0.8)}", stroke, sw)

    def diamond(self, cx, cy, s, fill):
        self.path(f"M{n(cx)} {n(cy - s)}L{n(cx + s)} {n(cy)}L{n(cx)} {n(cy + s)}L{n(cx - s)} {n(cy)}Z",
                  C["surface"], 1.5, fill)

    def pill(self, x, y, w, h, s, size=13.5, fill=None, color=None):
        self.r(x, y, w, h, fill or C["ink2"], rx=h / 2, cls="card")
        self.t(x + w / 2, y + h / 2 + size * 0.36, s, size, color or C["white"], 700, "middle")

    def chip(self, x, y, s, size=12.5, h=26, fill=None, stroke=None, color=None):
        w = tw(s, size) + 22
        self.r(x, y, w, h, fill or C["card"], stroke or C["edge"], 1, h / 2, cls="card")
        self.t(x + w / 2, y + h / 2 + size * 0.36, s, size, color, None, "middle")
        return w

    def swatch(self, x, y, kind, label, size=12.5):
        """A legend entry; returns the x after it."""
        if kind == "ok":
            self.r(x, y - 11, 16, 14, C["clayf"], rx=2)
        elif kind == "bad":
            self.r(x, y - 11, 16, 14, C["red"], rx=2)
            self.cross(x + 8, y - 4, 3.6, C["white"], 1.8)
        elif kind == "no":
            self.r(x, y - 11, 16, 14, self.url("hatch"), C["edge"], 1, 2)
        elif kind == "run":
            self.r(x, y - 11, 16, 14, C["ink2"], rx=2)
        elif kind == "dm":
            self.r(x, y - 11, 16, 14, C["clayw"], C["clay"], 1.8, 3)
        elif kind == "support":
            self.r(x, y - 11, 22, 16, C["olive"], rx=2)
            self.check(x + 11, y - 3, 5, C["olivek"], 1.8)
        elif kind == "fail":
            self.r(x, y - 11, 22, 16, C["red"], rx=2)
            self.cross(x + 11, y - 3, 4, C["white"], 1.8)
        elif kind == "weak":
            self.r(x, y - 11, 22, 16, self.url("hatchA"), rx=2)
            self.c(x + 11, y - 3, 4.5, "none", C["amber"], 1.8, cls=None)
        elif kind == "untested":
            self.r(x, y - 11, 22, 16, C["gray"], rx=2)
            self.t(x + 11, y + 1, "?", 12, C["muted"], 700, "middle")
        elif kind == "pt-ok":
            self.c(x + 7, y - 4, 5, C["olived"], cls=None)
        elif kind == "pt-fail":
            self.cross(x + 7, y - 4, 4.5, C["red"], 2.4)
        elif kind == "pt-weak":
            self.c(x + 7, y - 4, 5, "none", C["amber"], 2, cls=None)
        elif kind == "pt-premise":
            self.diamond(x + 7, y - 4, 6.5, C["blue"])
        wid = 22 if kind in ("support", "fail", "weak", "untested") else 16
        self.t(x + wid + 6, y, label, size, C["ink2"])
        return x + wid + 6 + tw(label, size)

    # output ---------------------------------------------------------------------------------
    def save(self):
        p = self.p
        defs = []
        for k, col in [("ink", C["ink2"]), ("muted", C["muted"]), ("clay", C["clay"]), ("blue", C["blue"]),
                       ("red", C["red"])]:
            defs.append(f'<marker id="{p}-ar-{k}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="10" '
                        f'markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">'
                        f'<path d="M0,0.6 L10,5 L0,9.4 z" fill="{col}"/></marker>')
        defs.append(f'<pattern id="{p}-hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
                    f'patternTransform="rotate(45)"><rect width="7" height="7" fill="#EFECE4"/>'
                    f'<line x1="0" y1="0" x2="0" y2="7" stroke="#D6D2C7" stroke-width="2.2"/></pattern>')
        defs.append(f'<pattern id="{p}-hatchA" width="6" height="6" patternUnits="userSpaceOnUse" '
                    f'patternTransform="rotate(45)"><rect width="6" height="6" fill="#FBF1D9"/>'
                    f'<line x1="0" y1="0" x2="0" y2="6" stroke="#E2C070" stroke-width="2"/></pattern>')
        label = esc(plain(self.title))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="{W}" height="{self.h}" '
               f'font-family="{FONT}" role="img" aria-label="{label}"><title>{label}</title>'
               f'<defs>{"".join(defs + self.extra)}</defs>{"".join(self.body)}</svg>\n')
        (OUT / f"{self.name}.svg").write_text(svg, encoding="utf-8")


def outline(cells, gx, gy):
    """Boundary of a set of grid cells (c, r), drawn on the gap lines."""
    segs = []
    for c, r in cells:
        if (c, r - 1) not in cells:
            segs.append((gx(c), gy(r), gx(c + 1), gy(r)))
        if (c, r + 1) not in cells:
            segs.append((gx(c), gy(r + 1), gx(c + 1), gy(r + 1)))
        if (c - 1, r) not in cells:
            segs.append((gx(c), gy(r), gx(c), gy(r + 1)))
        if (c + 1, r) not in cells:
            segs.append((gx(c + 1), gy(r), gx(c + 1), gy(r + 1)))
    return "".join(f"M{n(a)} {n(b)}L{n(c)} {n(d)}" for a, b, c, d in segs)


# ---------------------------------------------------------------------------------------------
def core():
    f = Fig("fig01_core", 460, "SAI 的核心：衡量科学家的直觉",
            "直觉是不把实验跑完就判断结果的能力。SAI 定量衡量它，并研究它怎样变成写明成立范围的知识")
    f.r(30, 92, 210, 340, C["white"], C["clay"], 1.6, 12, cls="card")
    f.t(48, 122, "科学家的直觉", 16, weight=700)
    f.t(48, 142, "人，或者模型", 12, C["faint"])
    f.t(48, 170, "不把实验跑完，", 13)
    f.t(48, 189, "就判断结果会怎样", 13)
    for x, s in [(48, "配置 A"), (150, "配置 B")]:
        f.r(x, 210, 72, 40, C["card"], C["edge"], 1, 8, cls="card")
        f.t(x + 36, 235, s, 12.5, anchor="middle")
    f.t(135, 276, "判断：A 的最终损失更低", 12.5, C["ink2"], 700, "middle")
    for i, (k, v) in enumerate([("代价", "近 0，不用跑"), ("误差", "多大？"), ("边界", "在哪里不灵？")]):
        y = 318 + 26 * i
        f.t(48, y, k, 12.5, C["faint"], 700)
        f.t(84, y, v, 12.5, C["ink2"])
    f.t(48, 404, "后两个问号，就是 SAI 要回答的", 12, C["clayd"], 700)
    f.l(244, 173, 273, 173, C["clay"], 2, arrow="clay")
    f.l(244, 349, 273, 349, C["ink2"], 2, arrow="ink")

    f.r(278, 92, 572, 162, C["card"], rx=12, cls="card")
    f.pill(296, 106, 150, 24, "定量判据（目前唯一）", 12.5, C["clay"])
    f.t(458, 124, "衡量：这份直觉值多少训练算力", 15, weight=700)

    def sx(b):
        return 300 + b / 12 * 196

    def sy(acc):
        return 226 - (acc - 50) / 35 * 76

    f.l(300, 226, 506, 226, C["ink2"], 1.2, arrow="ink")
    f.l(300, 226, 300, 144, C["ink2"], 1.2, arrow="ink")
    f.path("M" + "L".join(f"{n(sx(b / 10))} {n(sy(50 + 33 * (1 - math.exp(-b / 50))))}" for b in range(121)),
           C["clay"], 2.2)
    b_k = -5 * math.log(1 - 27 / 33)
    f.l(300, sy(77), sx(b_k), sy(77), C["ink2"], 1.3, dash="4 3")
    f.l(sx(b_k), sy(77), sx(b_k), 226, C["ink2"], 1.3, dash="4 3")
    f.dot(sx(b_k), sy(77), 4.5, C["ink2"])
    f.t(sx(b_k), 244, "ρ^{*}", 13.5, weight=700, anchor="middle")
    f.t(306, 162, "直觉的选对率", 11.5, C["ink2"])
    f.t(372, 214, "基线试探", 11.5, C["ink2"])
    for i, s in enumerate(["1. 在一份合同下，测直觉的选对率（误差 e）", "2. 看基线试探要花多少预算才能追平",
                           "3. 这个预算就是等价算力 ρ^{*}"]):
        f.t(528, 160 + 24 * i, s, 12.5, C["ink2"])
    f.t(528, 236, "panel 实测：最强的闭卷模型 ρ^{*} ≈ 8.5%", 12, C["faint"])

    f.r(278, 266, 572, 166, C["white"], C["ink2"], 1.6, 12, cls="card")
    f.pill(296, 280, 112, 24, "更多：研究对象", 12.5)
    f.t(420, 298, "知识怎样产生，在什么范围内成立", 15, weight=700)
    for i, (lv, how) in enumerate([("知识", "观察到"), ("定律", "测出来"), ("科学", "推出来")]):
        x = 296 + 190 * i
        f.r(x, 318, 160, 40, C["card"], C["edge"], 1, 10, cls="card")
        f.t(x + 16, 343, lv, 14, weight=700)
        f.t(x + 58, 343, how, 12.5, C["muted"])
        if i < 2:
            f.l(x + 164, 338, x + 186, 338, C["muted"], 1.5, arrow="muted")
    f.t(296, 384, "成立范围：声明域 D_{M} 内作答，界外声明不适用，记作 u", 13, C["ink2"])
    f.t(296, 408, "知道边界在哪，界外就能回退到直接训练；不知道，就会在未知的地方出错", 12.5, C["muted"])
    f.save()


def overview():
    f = Fig("fig02_overview", 640, "SAI 全景：一条主张怎样被描述、评价和记录",
            "直觉写成主张以后，坐标说它好在哪里，层次说它的边界是怎么知道的；单位是主张，不是论文")
    xs, bw = [30, 238, 446, 654], 196
    f.t(30, 100, "一条主张（基本单位）", 13, C["muted"], 700)
    boxes = [("问题族 q", ["对每个条件 ω 要预测什么", "例：两个配置谁的", "最终验证损失更低"]),
             ("合同 (Ω, μ, ℓ, ε)", ["在哪些条件、按什么权重、", "怎样算错、容忍多少；", "真值是把设定跑完的结果"]),
             ("答案规则 M", ["对每个条件给出答案，", "或声明“不适用”；", "写明版本"]),
             ("声明域 D_{M} ⊆ Ω", ["M 愿意作答的条件集合，", "以及划出这个集合的依据"])]
    for x, (head, body) in zip(xs, boxes):
        f.r(x, 110, bw, 112, C["white"], C["edge"], 1, 10, cls="card")
        f.t(x + 16, 138, head, 15, weight=700)
        for i, s in enumerate(body):
            f.t(x + 16, 162 + 19 * i, s, 12.5, C["muted"])
    for x in (232, 648):
        f.l(x, 226, x, 251, C["muted"], 1.5, arrow="muted")

    f.r(30, 256, 404, 200, C["card"], rx=12, cls="card")
    f.t(48, 284, "三个坐标（0–1，越小越好）", 15, weight=700)
    for i, (k, s) in enumerate([("e", "误差：声明域里答错的比例"), ("c", "代价：相对“直接把训练跑完”（= 1）"),
                                ("u", "不适用：M 不作答的条件比例")]):
        f.t(48, 316 + 26 * i, k, 15, C["clayd"], 700)
        f.t(72, 316 + 26 * i, s, 13)
    f.l(48, 386, 416, 386, C["edge"], 1, cls=None)
    f.c(56, 406, 5.5, C["ink2"], cls=None)
    f.t(70, 410, "直接训练 ≈ (种子噪声, 1, 0)：处处可用，但最贵", 12.5, C["muted"])
    f.c(56, 432, 5.5, C["surface"], C["clay"], 2.2, cls=None)
    f.t(70, 436, "原点 (0, 0, 0)：理想方向，不是已证结论", 12.5, C["muted"])

    f.r(446, 256, 404, 200, C["card"], rx=12, cls="card")
    f.t(464, 284, "三层：边界是怎么知道的", 15, weight=700)
    for i, (lv, how, ex, desc) in enumerate([
            ("知识", "观察到", "例：GRPO、GSPO", "只在测过的条件上看到结果，对新条件不作预言"),
            ("定律", "测出来", "例：scaling law", "沿条件轴系统扩展，由观测到的失败定界"),
            ("科学", "推出来", "例：Hyperball", "从可独立检验的前提推出关系和边界")]):
        y = 318 + 48 * i
        f.pill(464, y - 17, 48, 24, lv, 13)
        f.t(522, y, how, 14, weight=700)
        f.t(522 + tw(how, 14) + 10, y, ex, 12, C["faint"])
        f.t(522, y + 20, desc, 12.5, C["muted"])

    f.t(440, 482, "坐标相对一份合同，层次看依据的类型：两者都记，不互相推出", 12.5, C["faint"], anchor="middle")
    f.t(30, 512, "另外分开记录", 13, C["muted"], 700)
    f.t(654, 512, "派生概念", 13, C["muted"], 700)
    rowc = [("Scope", ["适用范围多大：", "合同下的 u 及其上界"]),
            ("证据协议", ["坐标多可信：", "预注册、盲测、新鲜题、", "程序裁决、复现、证明核查"]),
            ("阅读层级", ["我们读到多深：", "元数据 → 摘要 →", "指定章节 → 全文"]),
            ("SAI 系统", ["自动产出主张的系统；", "看前沿推进、生产代价", "和划界诚实度"])]
    for i, (x, (head, body)) in enumerate(zip(xs, rowc)):
        f.r(x, 522, bw, 100, C["white"], C["clay"] if i == 3 else C["edge"], 1.6 if i == 3 else 1, 10, cls="card")
        f.t(x + 16, 548, head, 15, weight=700)
        for j, s in enumerate(body):
            f.t(x + 16, 570 + 19 * j, s, 12.5, C["muted"])
    f.save()


def claim():
    f = Fig("fig03_claim", 480, "一条主张：在条件空间 Ω 上划出声明域 D_{M}",
            "Ω 的每个格子是一次完整的训练设定（任务 × 架构 × 超参 × 数据 × 预算），不是一个数据样本")
    cols, rows, cell, pitch, x0, y0 = 12, 8, 26, 30, 78, 104
    maxc = [5, 7, 8, 9, 9, 8, 6, 4]
    inside = {(c, r) for r in range(rows) for c in range(cols) if c <= maxc[r]}
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * pitch, y0 + r * pitch
            f.r(x, y, cell, cell, C["clayw"] if (c, r) in inside else f.url("hatch"), rx=3)
            w = math.exp(-((c - 4) ** 2 / (2 * 3.2 ** 2) + (r - 3.5) ** 2 / (2 * 2.4 ** 2)))
            f.c(x + cell / 2, y + cell / 2, 1.6 + 6.2 * w, C["ink2"], cls=None, opacity=0.55)
    f.path(outline(inside, lambda k: x0 + k * pitch - 2, lambda k: y0 + k * pitch - 2), C["clay"], 2.6)
    f.t(78, 366, "条件轴 1（示意：模型规模）→", 12.5, C["muted"])
    f.t(56, 222, "条件轴 2（示意：学习率）→", 12.5, C["muted"], anchor="middle", extra=' transform="rotate(-90 56 222)"')
    f.swatch(78, 399, "dm", "声明域 D_{M}：M 作答")
    f.swatch(250, 399, "no", "D_{M} 之外：声明不适用")
    for cx, rad in [(84, 2.4), (98, 4.6), (117, 7.4)]:
        f.c(cx, 422, rad, C["ink2"], cls=None, opacity=0.55)
    f.t(132, 427, "μ：圆点越大，这类设定的权重越高", 12.5, C["ink2"])

    cards = [(100, 76, "问题族 q", ["对每个条件 ω 要预测什么。", "例：A、B 两个配置，谁的最终验证损失更低。"]),
             (186, 96, "合同 (Ω, μ, ℓ, ε)", ["Ω 是条件空间，μ 是 Ω 上的权重；", "ℓ 和 ε 规定怎样算错、容忍多少；",
                                            "真值是把该设定实际跑完的结果。"]),
             (292, 76, "答案规则 M", ["对 D_{M} 里的每个 ω 给出答案，", "D_{M} 之外声明“不适用”；写明版本。"]),
             (378, 76, "声明域 D_{M} ⊆ Ω", ["M 愿意作答的条件集合，", "以及给出这个集合的依据。"])]
    for y, h, head, body in cards:
        f.r(470, y, 380, h, C["white"], C["edge"], 1, 10, cls="card")
        f.t(486, y + 24, head, 14.5, weight=700)
        for i, s in enumerate(body):
            f.t(486, y + 46 + 18 * i, s, 12.5, C["muted"])
    f.save()


def coords():
    f = Fig("fig04_coords", 472, "三个坐标怎么算：一个 100 格的例子",
            "设 100 个条件权重相同（μ 均匀），每格是一次完整的训练设定")
    x0, y0, cell, pitch = 60, 100, 28, 32
    excluded = {0: range(5, 10), 1: range(6, 10), 2: range(6, 10), 3: range(7, 10), 4: range(8, 10),
                5: range(9, 10), 6: range(9, 10)}
    wrong = {(1, 0), (4, 1), (2, 2), (5, 3), (6, 5), (3, 7), (8, 8), (5, 9)}
    inside = {(c, r) for r in range(10) for c in range(10) if c not in excluded.get(r, ())}
    assert len(inside) == 80 and wrong <= inside
    for r in range(10):
        for c in range(10):
            x, y = x0 + c * pitch, y0 + r * pitch
            if (c, r) not in inside:
                f.r(x, y, cell, cell, f.url("hatch"), rx=3)
            elif (c, r) in wrong:
                f.r(x, y, cell, cell, C["red"], rx=3)
                f.cross(x + cell / 2, y + cell / 2, 6, C["white"], 2.4)
            else:
                f.r(x, y, cell, cell, C["clayf"], rx=3)
    f.path(outline(inside, lambda k: x0 + k * pitch - 2, lambda k: y0 + k * pitch - 2), C["clay"], 2.6)
    nxt = f.swatch(60, 447, "ok", "作答且在 ε 内：72 格")
    nxt = f.swatch(nxt + 18, 447, "bad", "作答但超出 ε：8 格")
    f.swatch(nxt + 18, 447, "no", "声明不作答：20 格")

    f.r(404, 100, 446, 190, C["white"], C["edge"], 1, 10, cls="card")
    for i, (k, a, b) in enumerate([("e", "= 声明域里答错的格 ÷ 声明域的格", "= 8 ÷ 80 = 0.10"),
                                   ("c", "= 声明域里每次作答的平均代价", "= 0.05（直接训练 = 1）"),
                                   ("u", "= 不作答的格 ÷ 全部的格", "= 20 ÷ 100 = 0.20")]):
        y = 132 + 60 * i
        f.t(424, y, k, 18, C["clayd"], 700)
        f.t(452, y, a, 13.5)
        f.t(452, y + 22, b, 13.5, weight=700)
    f.r(404, 302, 446, 118, C["white"], C["edge"], 1, 10, cls="card")
    f.r(404, 302, 5, 118, C["clay"], rx=2)
    f.t(424, 328, "回退：不作答的 20 格改为直接训练", 14, weight=700)
    f.t(424, 354, "e′ = (1 − u) · e = 0.8 × 0.10 = 0.08", 13.5)
    f.t(424, 378, "c′ = (1 − u) · c + u = 0.8 × 0.05 + 0.2 = 0.24", 13.5)
    f.t(424, 402, "u′ = 0", 13.5)
    f.save()


def axes():
    f = Fig("fig05_axes", 560, "三轴空间：科学的方向，是用更少的试探得到同样的答案")

    def pr(e, c, u):
        return 300 + 420 * c - 190 * u, 350 - 250 * e + 140 * u

    corners = {k: pr(*k) for k in itertools.product((0, 1), repeat=3)}
    for a, b in itertools.combinations(sorted(corners), 2):
        if sum(x != y for x, y in zip(a, b)) == 1 and a != (0, 0, 0):
            f.l(*corners[a], *corners[b], C["cube"], 1)
    for tip in [(1.12, 0, 0), (0, 1.12, 0), (0, 0, 1.12)]:
        f.l(300, 350, *pr(*tip), C["ink2"], 1.5, arrow="ink")
    f.t(288, 80, "误差 e", 14.5, weight=700, anchor="end")
    f.t(288, 98, "错得少（0–1）", 12, C["muted"], anchor="end")
    f.t(782, 354, "代价 c", 14.5, weight=700)
    f.t(782, 372, "直接训练 = 1", 12, C["muted"])
    f.t(104, 516, "不适用 u（0–1）", 14.5, weight=700)
    f.t(104, 534, "管得宽：越小越宽", 12, C["muted"])

    f.path("M708 337 Q510 296 318 341", C["clay"], 3, arrow="clay")
    f.t(513, 296, "科学：用更少的试探，得到同样的答案", 14, C["clayd"], 700, "middle")
    dx, dy = pr(0.02, 1, 0)
    f.dot(dx, dy, 7, C["ink2"])
    f.t(692, 392, "直接训练", 14, weight=700)
    f.t(692, 410, "处处可用，误差只来自种子噪声", 12, C["muted"])
    f.t(692, 428, "但最贵：c = 1", 12, C["muted"])
    f.ring(300, 350, 7.5)
    f.t(286, 326, "原点 (0, 0, 0)：万物理论", 14, C["clayd"], 700, "end")
    f.t(286, 344, "理想方向，不是已证结论", 12, C["muted"], anchor="end")
    ux, uy = pr(0.3, 0.04, 0)
    f.dot(ux, uy, 6.5, C["blue"])
    f.t(296, 270, "U：代价近 0、误差大于 0", 13, weight=700, anchor="end")
    f.t(296, 288, "便宜但不精确的候选规律", 12, C["muted"], anchor="end")
    kx, ky = pr(0.02, 0.05, 0.95)
    f.dot(kx, ky, 6, C["olived"])
    f.l(kx + 8, ky - 4, 171, 466, C["faint"], 1, cls=None)
    f.t(176, 462, "只测过少数设定的知识", 13, weight=700)
    f.t(176, 479, "u 接近 1：未测的设定记为未知", 12, C["muted"])

    f.r(636, 446, 224, 82, C["card"], C["edge"], 1, 10, cls="card")
    f.t(650, 470, "三个坐标都相对同一份合同", 12, C["ink2"], 700)
    f.t(650, 490, "(Ω, μ, ℓ, ε)。换了 μ，u 会变；", 12, C["muted"])
    f.t(650, 510, "比较两条主张须用同一份合同。", 12, C["muted"])
    f.save()


def fallback():
    f = Fig("fig06_fallback", 568, "知道边界的价值：同样在 30% 的条件上不灵，结果完全不同",
            "设每次作答的代价 c = 0.1。每个小方阵都是同样的 100 个条件")
    bad = set(random.Random(11).sample(range(100), 30))

    def grid(x0, kind):
        for r in range(10):
            for c in range(10):
                x, y, k = x0 + c * 17, 128 + r * 17, kind(c, r)
                f.r(x, y, 15, 15, {"ok": C["clayf"], "bad": C["red"], "no": f.url("hatch"), "run": C["ink2"]}[k], rx=2)
                if k == "bad":
                    f.cross(x + 7.5, y + 7.5, 3.6, C["white"], 1.7)

    grid(56, lambda c, r: "bad" if r * 10 + c in bad else "ok")
    grid(342, lambda c, r: "no" if c >= 7 else "ok")
    grid(628, lambda c, r: "run" if c >= 7 else "ok")
    f.r(341, 127, 7 * 17, 170, "none", C["clay"], 2.2, 3)
    f.l(300, 100, 300, 350, C["grid"], 1.2, cls=None)
    f.l(522, 212, 614, 212, C["ink2"], 1.6, arrow="ink")
    f.t(568, 202, "回退", 13, C["muted"], 700, "middle")
    for x, head, coord, note in [(56, "A：不知道边界", "(e, c, u) = (0.30, 0.1, 0)", "照常作答，错在哪里不知道"),
                                 (342, "B：知道边界", "(e, c, u) = (0, 0.1, 0.30)", "知道哪里不灵，就在那里拒答"),
                                 (628, "B 回退后", "(e, c, u) = (0, 0.37, 0)", "拒答处直接训练：0.7 × 0.1 + 0.3")]:
        f.t(x, 114, head, 14.5, weight=700)
        f.t(x, 322, coord, 13.5, weight=700)
        f.t(x, 342, note, 12.5, C["muted"])
    nxt = f.swatch(56, 376, "ok", "作答，答对")
    nxt = f.swatch(nxt + 22, 376, "bad", "作答，答错")
    nxt = f.swatch(nxt + 22, 376, "no", "拒答")
    f.swatch(nxt + 22, 376, "run", "直接训练")

    x0, x1, yb, yt, emax = 100, 820, 512, 404, 0.35

    def sx(c):
        return x0 + c * (x1 - x0)

    def sy(e):
        return yb - e / emax * (yb - yt)

    for e in (0.1, 0.2, 0.3):
        f.l(x0, sy(e), x1, sy(e), C["grid"], 1, cls=None)
        f.t(92, sy(e) + 4, f"{e:.1f}", 12, C["muted"], anchor="end")
    f.l(x0, yb, x1 + 10, yb, C["ink2"], 1.2)
    f.l(x0, yb, x0, yt - 4, C["ink2"], 1.2)
    for c in (0, 0.25, 0.5, 0.75, 1):
        f.t(sx(c), 530, f"{c:g}", 12, C["muted"], anchor="middle")
    f.t(108, 398, "误差 e", 12.5, C["muted"], 700)
    f.t(820, 552, "代价 c（直接训练 = 1）", 12.5, C["muted"], anchor="end")
    f.ring(sx(0), sy(0), 6)
    f.dot(sx(0.1), sy(0.3), 6, C["clay"])
    f.t(sx(0.1) + 12, 410, "A：便宜，但有 30% 答错", 12.5)
    f.dot(sx(0.37), sy(0), 6, C["clay"])
    f.t(sx(0.37) + 12, 500, "B 回退后：e = 0，c = 0.37", 12.5)
    f.dot(sx(1), sy(0.02), 6, C["ink2"])
    f.t(808, 498, "直接训练", 12.5, weight=700, anchor="end")
    f.t(808, 470, "A 要消除误差，只能全部重跑：回到这里", 12, C["muted"], anchor="end")
    f.save()


def rho():
    f = Fig("fig07_rho", 430, "衡量直觉：等价算力 ρ^{*} 把误差折算成代价",
            "示意图。先在一份合同下测直觉（例如不训练、闭卷判断的模型）的选对率，再看基线试探要花多少预算才能追平")
    x0, x1, yb, yt = 110, 600, 360, 110

    def sx(b):
        return x0 + b / 12 * (x1 - x0)

    def sy(acc):
        return yb - (acc - 50) / 35 * (yb - yt)

    f.l(x0, yb, x1 + 12, yb, C["ink2"], 1.4, arrow="ink")
    f.l(x0, yb, x0, yt - 10, C["ink2"], 1.4, arrow="ink")
    f.t(120, 104, "选对率", 12.5, C["muted"], 700)
    f.t(600, 404, "试探预算（占每个候选完整训练的比例）", 12.5, C["muted"], anchor="end")
    f.t(x0, 380, "0", 12, C["muted"], anchor="middle")
    curve = [(sx(b / 10), sy(50 + 33 * (1 - math.exp(-b / 50)))) for b in range(0, 121)]
    f.path("M" + "L".join(f"{n(x)} {n(y)}" for x, y in curve), C["clay"], 2.4)
    acc_k = 77
    b_k = -5 * math.log(1 - (acc_k - 50) / 33)
    f.l(x0, sy(acc_k), sx(b_k), sy(acc_k), C["ink2"], 1.5, dash="5 4")
    f.l(sx(b_k), sy(acc_k), sx(b_k), yb, C["ink2"], 1.5, dash="5 4")
    f.dot(sx(b_k), sy(acc_k), 5.5, C["ink2"])
    f.t(sx(b_k), 382, "ρ^{*}", 15, weight=700, anchor="middle")
    f.t(122, 158, "直觉的选对率（不训练，闭卷判断）", 12.5)
    f.t(250, 270, "基线试探：预算越多，选得越准", 12.5, C["ink2"])

    f.r(640, 108, 216, 194, C["card"], C["edge"], 1, 10, cls="card")
    f.t(656, 134, "怎么读", 14, weight=700)
    for i, s in enumerate(["直觉越准，水平线越高，", "追平它要花的预算 ρ^{*} 越大。", "ρ^{*} 是目前唯一定量的判据：",
                           "它把误差换算成代价，", "是三轴的一个切片。", "panel 的合同取", "ArchitectureIQ。"]):
        f.t(656, 160 + 20 * i, s, 12.5, C["muted"])
    f.save()


def upper_bound():
    f = Fig("fig08_upper_bound", 436, "全部成功，也只给出上界",
            "n 个独立条件全部成功时，失败率的单侧 95% 上限是 1 − 0.05^{1/n}，大约 3/n")

    def xn(v):
        return 100 + math.log10(v) * 460 / 3

    def yp(p):
        return 372 - (math.log10(p) + 3) * 92

    for v in (1, 10, 100, 1000):
        f.l(xn(v), 96, xn(v), 372, C["grid"], 1, cls=None)
        f.t(xn(v), 392, str(v), 12, C["muted"], anchor="middle")
    for p, s in [(0.001, "0.1%"), (0.01, "1%"), (0.1, "10%"), (1, "100%")]:
        f.l(100, yp(p), 560, yp(p), C["grid"], 1, cls=None)
        f.t(92, yp(p) + 4, s, 12, C["muted"], anchor="end")
    f.l(100, 372, 560, 372, C["ink2"], 1.2)
    f.l(100, 96, 100, 372, C["ink2"], 1.2)
    f.t(330, 420, "全部成功的独立条件数 n（对数刻度）", 12.5, C["muted"], anchor="middle")
    f.t(40, 234, "失败率的 95% 上限（对数刻度）", 12.5, C["muted"], anchor="middle", extra=' transform="rotate(-90 40 234)"')
    pts = [(xn(10 ** (3 * i / 120)), yp(1 - 0.05 ** (1 / 10 ** (3 * i / 120)))) for i in range(121)]
    f.path("M" + "L".join(f"{n(x)} {n(y)}" for x, y in pts), C["clay"], 2.2)
    for v, s in [(29, "n = 29：上限 9.8%"), (100, "n = 100：上限 2.95%"), (299, "n = 299：上限 1.0%")]:
        p = 1 - 0.05 ** (1 / v)
        f.dot(xn(v), yp(p), 4.5, C["clay"])
        f.t(xn(v) + 10, yp(p) - 8, s, 12.5, C["ink2"])

    f.t(618, 108, "报告 u 时，两部分分开列", 14, weight=700)
    segs = [(0.55, C["clayf"], "声明域 D_{M}：55%", "已测并作答"), (0.15, C["grayd"], "已证不适用：15%", None),
            (0.30, f.url("hatch"), "未测：30%", "不算成立，也计入 u")]
    y = 126
    for share, fill, head, sub in segs:
        h = 240 * share - 2
        f.r(624, y, 36, h, fill, rx=3)
        mid = y + h / 2
        if sub:
            f.t(672, mid - 3, head, 13, weight=700)
            f.t(672, mid + 15, sub, 12, C["muted"])
        else:
            f.t(672, mid + 5, head, 13, weight=700)
        y += 240 * share
    f.l(666, 126 + 240 * 0.55, 666, 366, C["clay"], 1.6)
    f.t(618, 392, "u = 0.15 + 0.30 = 0.45", 13.5, weight=700)
    f.t(618, 412, "坐标取保守值：两部分都计入 u", 12, C["muted"])
    f.save()


def levels():
    f = Fig("fig09_levels", 512, "三层：区别不在范围大小，而在边界是怎么知道的")
    xs, pw, top, ph = [30, 309, 588], 262, 120, 200
    heads = [("知识", "观察到", "Knowledge"), ("定律", "测出来", "Law"), ("科学", "推出来", "Science")]
    for x, (lv, how, en) in zip(xs, heads):
        f.pill(x, 82, 50, 26, lv)
        f.t(x + 60, 101, how, 15, weight=700)
        f.t(x + 60 + tw(how, 15) + 8, 101, en, 12, C["faint"])
        f.r(x, top, pw, ph, f.url("hatch"), C["edge"], 1, 10)

    # knowledge: a few observed points and their near neighbours
    for px, py in [(98, 196), (130, 234), (180, 176)]:
        f.c(px, py, 19, C["clayw"], C["clay"], 2)
        f.dot(px, py, 5, C["olived"])
    f.tbg(280, 306, "其余条件：不作预言", 12, C["muted"], anchor="end")

    # law: systematic extension along two axes, bounded where a failure was observed
    x = xs[1]
    cx, cy = x + 80, 240
    f.l(x + 12, cy, x + 246, cy, C["axisln"], 1, cls=None)
    f.l(cx, top + 8, cx, top + ph - 8, C["axisln"], 1, cls=None)
    f.r(x + 24, 156, 134, 144, C["clayw"], C["clay"], 2, 14, opacity=None)
    f.r(x + 162, 196, 88, 88, "none", C["clay"], 1.6, 12, dash="5 4")
    for px in (36, 58, 80, 102, 124, 146):
        f.dot(x + px, cy, 4.8, C["olived"])
    for py in (288, 264, 216, 192, 168):
        f.dot(cx, py, 4.8, C["olived"])
    for px, py in [(58, 192), (124, 192), (58, 288), (124, 288)]:
        f.dot(x + px, py, 4.8, C["olived"])
    f.cross(cx, 138, 5, C["red"], 2.4)
    f.tbg(cx + 12, 143, "失败处定界", 12, C["muted"])
    f.c(x + 172, cy, 5.5, C["surface"], C["amber"], 2)
    f.t(x + 218, 228, "?", 18, C["faint"], 700, "middle")

    # science: premises checked, boundary derived where they fail
    x = xs[2]
    f.extra.append(f'<clipPath id="{f.p}-clipS"><rect x="{x}" y="{top}" width="{pw}" height="{ph}" rx="10"/></clipPath>')
    curve = f"M{x} 150 C{x + 90} 140 {x + 170} 210 {x + pw} 232"
    f.path(curve + f" L{x + pw} {top + ph} L{x} {top + ph} Z", fill=C["clayw"], cls=None,
           extra=f' clip-path="{f.url("clipS")}"')
    f.path(curve, C["blue"], 2.6, extra=f' clip-path="{f.url("clipS")}"')
    for px, py in [(40, 250), (110, 276), (170, 262), (214, 252)]:
        f.diamond(x + px, py, 6.5, C["blue"])
    f.tbg(x + 250, 190, "前提失效处", 12, C["blued"], 700, "end")
    f.tbg(x + 250, 142, "界外：未知，不是已证失效", 12, C["muted"], anchor="end")
    f.t(x + 16, 306, "前提已核验处：没测过结果，也能作答", 12, C["ink"])

    rows = [["在测过的条件上观察到结果", "测到哪里算哪里", "不作预言", "GRPO、GSPO"],
            ["沿声明的条件轴系统地改变条件", "由观测到的失败定位", "已测外延内作答；外推是新主张", "scaling law"],
            ["显式、可独立检验的前提", "前提失效的地方", "前提已核验处可以作答", "Hyperball"]]
    for x, vals in zip(xs, rows):
        for i, (lab, val) in enumerate(zip(["依据", "边界", "新条件", "锚点"], vals)):
            f.t(x, 344 + 24 * i, lab, 12, C["faint"], 700)
            f.t(x + 46, 344 + 24 * i, val, 12.5, C["ink2"])
    nxt = f.swatch(30, 458, "pt-ok", "测过且成立")
    for kind, lab in [("pt-fail", "测过且失败"), ("pt-weak", "测量不足"), ("pt-premise", "前提检验"),
                      ("dm", "声明域 D_{M}"), ("no", "不作答")]:
        nxt = f.swatch(nxt + 22, 458, kind, lab)
    f.t(30, 490, "层次不是价值排名：一条覆盖广、便宜又准的定律，可能比一条窄域的科学推导更有用。", 12.5, C["muted"])
    f.save()


def law_axes():
    f = Fig("fig10_law_axes", 360, "定律怎样扩展：沿每条条件轴记录支持、失败和未测",
            "示意，不是真实数据。每行是一条条件轴，每格是这条轴上的一段取值，越往右离最初测过的设定越远")
    names = ["模型规模", "数据量", "训练步数", "架构族", "优化器", "数据分布"]
    status = ["SSSSSSIU", "SSSSSIUU", "SSSSSSSI", "SSFSUUUU", "SSSIUUUU", "SSSFIUUU"]
    x0, y0, cw, ch, px, py = 164, 96, 50, 30, 53, 33
    for r, (name, row) in enumerate(zip(names, status)):
        y = y0 + r * py
        f.t(150, y + 20, name, 13, anchor="end")
        for c, s in enumerate(row):
            x = x0 + c * px
            if s == "S":
                f.r(x, y, cw, ch, C["olive"], rx=3)
                f.check(x + cw / 2, y + ch / 2, 6, C["olivek"], 2.2)
            elif s == "F":
                f.r(x, y, cw, ch, C["red"], rx=3)
                f.cross(x + cw / 2, y + ch / 2, 5.5, C["white"], 2.4)
            elif s == "I":
                f.r(x, y, cw, ch, f.url("hatchA"), rx=3)
                f.c(x + cw / 2, y + ch / 2, 6, "none", C["amber"], 2)
            else:
                f.r(x, y, cw, ch, C["gray"], rx=3)
                f.t(x + cw / 2, y + ch / 2 + 5, "?", 13, C["muted"], 700, "middle")
        for k, grp in itertools.groupby(enumerate(row), key=lambda t: t[1]):
            idx = [i for i, _ in grp]
            if k == "S":
                f.r(x0 + idx[0] * px - 3, y - 3, (idx[-1] - idx[0] + 1) * px - 3 + 6, ch + 6, "none", C["clay"], 2.2, 6)
    f.r(x0 + 6 * px - 3, y0 - 3, 2 * px - 3 + 6, ch + 6, "none", C["clay"], 1.8, 6, dash="5 4")

    notes = [("dm", "橙色框：声明域，以已测外延为界", "逐轴取已测且支持的那一段"),
             ("fail", "已测失败：把必要的限制写进定律，", "这不算研究失败"),
             ("weak", "测量不足、未测：记为未知，", "不算成立；预算用完不等于到头"),
             ("ext", "虚线框：越界外推（如更大规模），", "是一条新的、证据更弱的待检主张")]
    for i, (kind, a, b) in enumerate(notes):
        y = 110 + 52 * i
        if kind == "dm":
            f.r(606, y - 12, 18, 14, "none", C["clay"], 2.2, 3)
        elif kind == "fail":
            f.r(606, y - 12, 18, 14, C["red"], rx=2)
            f.cross(615, y - 5, 3.6, C["white"], 1.8)
        elif kind == "weak":
            f.r(606, y - 12, 18, 14, f.url("hatchA"), rx=2)
            f.c(615, y - 5, 3.8, "none", C["amber"], 1.6, cls=None)
        else:
            f.r(606, y - 12, 18, 14, "none", C["clay"], 1.8, 3, dash="4 3")
        f.t(632, y, a, 12.5, C["ink2"])
        f.t(632, y + 18, b, 12.5, C["muted"])
    nxt = f.swatch(164, 334, "support", "已测支持")
    for kind, lab in [("fail", "已测失败"), ("weak", "测量不足"), ("untested", "未测")]:
        nxt = f.swatch(nxt + 22, 334, kind, lab)
    f.save()


def science():
    f = Fig("fig11_science", 516, "科学：从可独立检验的前提，推出关系和它的边界",
            "要分开问两件事：推导在假设内对不对（数学问题）；前提在真实系统里成不成立（映射问题，要用实验检验）")
    f.t(30, 104, "“第一性原理”的可操作含义：前提满足三条", 13.5, weight=700)
    for i, (head, sub) in enumerate([("(a) 显式写出", "每条前提都写下来，而不是藏在拟合里"),
                                     ("(b) 能独立于结论检验", "不能把拟合出的关系换个说法当作前提"),
                                     ("(c) 结论由前提推出", "边界就是前提失效的地方")]):
        y = 116 + 60 * i
        f.r(30, y, 390, 52, C["white"], C["edge"], 1, 10, cls="card")
        f.t(46, y + 22, head, 14, weight=700)
        f.t(46, y + 42, sub, 12.5, C["muted"])
    f.l(225, 292, 225, 325, C["blue"], 2, arrow="blue")
    f.t(240, 314, "推导：在假设内对不对（数学问题）", 12.5, C["blued"], 700)
    f.r(30, 330, 390, 52, C["bluef"], C["blue"], 1.4, 10, cls="card")
    f.t(46, 352, "关系 R，以及它的边界", 14, weight=700)
    f.t(46, 372, "边界 = 前提失效的地方", 12.5, C["muted"])
    f.r(30, 392, 390, 52, C["white"], C["edge"], 1, 10, cls="card")
    f.t(46, 414, "映射问题：前提在真实系统中成立吗？", 13.5, weight=700)
    f.t(46, 434, "要像定律一样，用实验逐条检验", 12.5, C["muted"])
    f.r(30, 454, 390, 40, C["white"], C["red"], 1.2, 10, cls="card", dash="5 4")
    f.cross(50, 474, 5, C["red"], 2.2)
    f.t(64, 479, "循环论证：把拟合出的关系换个说法当前提，违反 (b)", 12.5, C["red"])

    f.r(450, 96, 400, 300, f.url("hatch"), C["edge"], 1, 10)
    f.extra.append(f'<clipPath id="{f.p}-clipR"><rect x="450" y="96" width="400" height="300" rx="10"/></clipPath>')
    true_region = "M490 250 A150 112 0 1 0 790 250 A150 112 0 1 0 490 250 Z"
    derived = "M514 262 A98 72 0 1 0 710 262 A98 72 0 1 0 514 262 Z"
    f.path(true_region, fill=C["surface"], cls=None)
    f.path(true_region, C["ink2"], 1.6, dash="6 5")
    f.path(derived, fill=C["clayw"], cls=None)
    f.path(derived, C["blue"], 2.6)
    f.tbg(640, 128, "真实成立的范围（未知，示意）", 12, C["muted"], anchor="middle")
    f.t(612, 226, "推出的范围（前提已核验）", 12, C["blued"], 700, "middle")
    for k, (px, py) in enumerate([(590, 270), (742, 220), (812, 352)], 1):
        f.c(px, py, 10, C["ink2"], C["surface"], 2, cls=None)
        f.t(px, py + 4.5, str(k), 12, C["white"], 700, "middle")
    for k, s in enumerate(["前提已核验：没测过结果，也能作答", "其实成立，但推导不保证：记为未知", "不成立"], 1):
        y = 422 + 24 * (k - 1)
        f.c(464, y - 4, 9, C["ink2"], cls=None)
        f.t(464, y, str(k), 11, C["white"], 700, "middle")
        f.t(482, y, s, 12.5, C["ink2"])
    f.t(456, 496, "推导只给充分条件时，推出的边界是内界：界外是未知，不是已证失效", 12, C["muted"])
    f.save()


def promote():
    f = Fig("fig12_promote", 470, "晋升、降级，以及哪些东西不构成层次依据")
    for x, lv, sub in [(40, "知识", "边界：观察到"), (360, "定律", "边界：系统扩展测出"), (680, "科学", "边界：从前提推出")]:
        f.r(x, 96, 160, 64, C["white"], C["edge"], 1, 12, cls="card")
        f.t(x + 80, 124, lv, 18, weight=700, anchor="middle")
        f.t(x + 80, 146, sub, 12.5, C["muted"], anchor="middle")
    for x0, x1, a, b in [(206, 354, "沿条件轴扩展外延，", "记录支持、失败和未测"), (526, 674, "找到可独立检验的前提，", "推出关系与边界")]:
        f.l(x0, 128, x1, 128, C["clay"], 2.4, arrow="clay")
        f.t((x0 + x1) / 2, 104, a, 12, C["clayd"], 700, "middle")
        f.t((x0 + x1) / 2, 119, b, 12, C["clayd"], anchor="middle")

    f.t(40, 206, "出现反例时", 13, C["muted"], 700)
    f.r(40, 222, 170, 56, C["white"], C["red"], 1.4, 10, cls="card")
    f.t(125, 245, "声明域内出现反例", 13, weight=700, anchor="middle")
    f.t(125, 265, "证伪记录保留", 12, C["muted"], anchor="middle")
    f.l(214, 250, 250, 250, C["ink2"], 1.5, arrow="ink")
    f.r(254, 222, 180, 56, C["white"], C["edge"], 1, 10, cls="card")
    f.t(344, 245, "先收窄声明域 D_{M}", 13, weight=700, anchor="middle")
    f.t(344, 265, "误差 e 随之降低", 12, C["muted"], anchor="middle")
    f.l(438, 250, 470, 250, C["ink2"], 1.5, arrow="ink")
    f.path("M474 250L556 204L638 250L556 296Z", C["ink2"], 1.4, C["white"], cls=None)
    f.t(556, 246, "收窄后还满足", 13, weight=700, anchor="middle")
    f.t(556, 264, "本层的依据吗？", 13, weight=700, anchor="middle")
    f.l(638, 250, 690, 216, C["ink2"], 1.5, arrow="ink")
    f.l(638, 250, 690, 288, C["ink2"], 1.5, arrow="ink")
    f.t(654, 226, "是", 12, C["muted"], 700)
    f.t(654, 286, "否", 12, C["muted"], 700)
    f.r(694, 190, 160, 48, C["white"], C["edge"], 1, 10, cls="card")
    f.t(774, 211, "满足：留在本层", 13, weight=700, anchor="middle")
    f.t(774, 229, "范围更小", 12, C["muted"], anchor="middle")
    f.r(694, 266, 160, 48, C["white"], C["edge"], 1, 10, cls="card")
    f.t(774, 287, "不满足：降一层", 13, weight=700, anchor="middle")
    f.t(774, 305, "证伪记录保留", 12, C["muted"], anchor="middle")

    f.l(40, 346, 840, 346, C["grid"], 1, cls=None)
    f.t(40, 376, "不构成层次依据，也不自动升层（它们记入证据协议）：", 13, weight=700)
    x = 40
    for s in ["形式定义", "程序可执行", "一次验证", "高分", "预注册", "盲测", "机器可读"]:
        x += f.chip(x, 390, s) + 10
    f.t(40, 444, "它们说明坐标有多可信，不说明边界是怎么知道的。", 12.5, C["muted"])
    f.save()


def records():
    f = Fig("fig13_records", 488, "分开记录的四件事：各记各的，不互相推出",
            "四个例子的取值是示意。层次看依据的类型，Scope 看坐标，证据协议看坐标多可信，阅读层级看核查得多深")
    cols = [(220, "层次", "边界是怎么知道的", ["知识 · 定律 · 科学"]),
            (380, "Scope", "适用范围多大", ["合同下的 u 及其上界"]),
            (540, "证据协议", "坐标有多可信", ["预注册、盲测、复现、", "程序裁决、证明核查…"]),
            (700, "阅读层级", "我们读到多深", ["元数据 → 摘要 →", "指定章节 → 全文"])]
    for x, head, q, vals in cols:
        f.t(x, 106, head, 14.5, weight=700)
        f.t(x, 126, q, 12.5, C["muted"])
        for i, s in enumerate(vals):
            f.t(x, 146 + 16 * i, s, 12, C["faint"])
    f.l(30, 176, 850, 176, C["edge"], 1, cls=None)
    examples = [("窄域推导", "只在一小片条件上推出", "科学", 0.85, "窄：u ≈ 0.85", ["推导已核查"], 4, "全文"),
                ("广域经验关系", "覆盖很多条件的拟合", "定律", 0.15, "宽：u ≈ 0.15", ["独立复现"], 3, "指定章节"),
                ("预注册过的观察", "只在测过的设定上", "知识", 0.95, "窄：u ≈ 0.95", ["预注册", "盲测"], 4, "全文"),
                ("未经实验检验的推导", "前提还没在真实系统检验", "科学", None, "映射未检验：坐标待测",
                 ["推导已核查", "前提未检验"], 2, "摘要")]
    for i, (name, desc, lv, u, ulab, tags, depth, dlab) in enumerate(examples):
        y = 186 + 72 * i
        if i:
            f.l(30, y - 5, 850, y - 5, C["grid"], 1, cls=None)
        f.t(30, y + 26, name, 13.5, weight=700)
        f.t(30, y + 46, desc, 12, C["muted"])
        f.pill(220, y + 12, 52, 26, lv)
        if u is None:
            f.t(220, y + 58, "映射问题悬而未决", 11.5, C["faint"])
            f.r(380, y + 18, 120, 12, "none", C["grayd"], 1.2, 3, dash="4 3")
        else:
            f.r(380, y + 18, 120, 12, f.url("hatch"), C["edge"], 1, 3)
            f.r(380, y + 18, 120 * (1 - u), 12, C["clayf"], rx=3)
        f.t(380, y + 50, ulab, 12.5, C["ink2"])
        tx, ty = 540, y + 14
        for tag in tags:
            if tx > 540 and tx + tw(tag, 12) + 22 > 690:
                tx, ty = 540, ty + 30
            tx += f.chip(tx, ty, tag, 12, 24) + 8
        for k in range(4):
            f.c(706 + 16 * k, y + 26, 5, C["ink2"] if k < depth else C["surface"], C["ink2"], 1.4)
        f.t(772, y + 31, dlab, 12.5, C["ink2"])
    f.save()


def system():
    f = Fig("fig14_system", 440, "SAI 系统：自动产出主张的系统，评价看三件事",
            "里程碑 M0–M6 是对系统的另一种评分，不是主张的层次")
    f.r(30, 104, 170, 64, C["ink2"], rx=12, cls="card")
    f.t(115, 132, "SAI 系统", 16, C["white"], 700, "middle")
    f.t(115, 154, "自动产出主张", 12.5, C["white"], anchor="middle")
    f.l(115, 172, 115, 203, C["muted"], 1.5, arrow="muted")
    for dx in (12, 6):
        f.r(40 + dx, 208 + dx, 150, 44, C["white"], C["edge"], 1, 8)
    f.r(40, 208, 150, 44, C["white"], C["edge"], 1, 8, cls="card")
    f.t(54, 227, "主张 + 声明域", 12.5, weight=700)
    f.t(54, 244, "坐标 (e, c, u)", 12, C["muted"])
    f.l(206, 236, 268, 236, C["muted"], 1.5, arrow="muted")

    x0, x1, yb, yt = 300, 590, 380, 110

    def sx(c):
        return x0 + c * (x1 - x0)

    def sy(e):
        return yb - e / 0.5 * (yb - yt)

    def stairs(pts):
        d = f"M{n(sx(pts[0][0]))} {n(sy(0.5))}"
        for (c, e), nxt in zip(pts, pts[1:] + [None]):
            d += f"V{n(sy(e))}"
            if nxt:
                d += f"H{n(sx(nxt[0]))}"
        return d

    old = [(0.08, 0.46), (0.25, 0.25), (0.5, 0.12), (0.75, 0.05), (1, 0.012)]
    new = [(0.02, 0.30), (0.08, 0.14), (0.2, 0.06), (0.42, 0.02), (1, 0.012)]
    poly =[(sx(new[0][0]), sy(0.5))]
    for (c, e), nxt in zip(new, new[1:] + [None]):
        poly.append((sx(c), sy(e)))
        if nxt:
            poly.append((sx(nxt[0]), sy(e)))
    back = []
    for (c, e), nxt in zip(old, old[1:] + [None]):
        back.append((sx(c), sy(e)))
        if nxt:
            back.append((sx(nxt[0]), sy(e)))
    back.insert(0, (sx(old[0][0]), sy(0.5)))
    shape = poly + back[::-1]
    f.path("M" + "L".join(f"{n(a)} {n(b)}" for a, b in shape) + "Z", fill=C["clay"], cls=None, extra=' fill-opacity="0.12"')
    f.path(stairs(old), C["grayd"], 2)
    f.path(stairs(new), C["clay"], 2.4)
    f.l(x0, yb, x1 + 14, yb, C["ink2"], 1.3, arrow="ink")
    f.l(x0, yb, x0, yt - 12, C["ink2"], 1.3, arrow="ink")
    f.t(x0 + 10, 104, "误差 e", 12.5, C["muted"], 700)
    f.t((x0 + x1) / 2, 400, "代价 c", 12.5, C["muted"], 700, "middle")
    f.ring(x0, yb, 6)
    f.dot(sx(1), sy(0.012), 6, C["ink2"])
    f.t(600, 400, "直接训练", 12.5, weight=700, anchor="end")
    f.t(sx(0.5) + 8, sy(0.25) - 6, "旧前沿", 12.5, C["muted"], 700)
    f.t(314, 338, "新前沿", 12.5, C["clayd"], 700)
    f.t(380, 340, "前沿推进", 12.5, C["clayd"], 700)
    f.t(x0, 424, "u 已按“回退到直接训练”折算进 e 和 c", 12, C["muted"])

    cards = [(96, 92, "前沿推进", ["同一份合同下，产出的主张把", "(e, c, u) 前沿向原点推了多少"]),
             (198, 112, "生产代价", ["产出主张花的计算，和使用代价", "c 分开记：花 10^{6} GPU 小时", "拟合出的定律，用起来可以很便宜"]),
             (320, 92, "划界诚实度", ["声明域里实际出错的比例，", "即越界声明（overclaim）率"])]
    for y, h, head, body in cards:
        f.r(620, y, 236, h, C["white"], C["edge"], 1, 10, cls="card")
        f.t(636, y + 26, head, 14, weight=700)
        for i, s in enumerate(body):
            f.t(636, y + 50 + 19 * i, s, 12.5, C["muted"])
    f.save()


def vision():
    f = Fig("fig15_vision", 490, "前景：从“凭直觉挑、跑了才知道”，到“查了、算了就知道”",
            "主张库越大、边界越清楚，落在声明域内、不必重跑的问题就越多；原点是方向，不是已经到达的终点")
    f.t(30, 104, "今天：凭直觉挑，跑了才知道", 14, weight=700)
    f.r(50, 120, 180, 40, C["white"], C["edge"], 1, 10, cls="card")
    f.t(140, 145, "新问题", 13.5, weight=700, anchor="middle")
    f.l(140, 162, 140, 179, C["muted"], 1.5, arrow="muted")
    f.r(50, 184, 180, 48, C["white"], C["edge"], 1, 10, cls="card")
    f.t(140, 204, "凭直觉挑候选", 13.5, weight=700, anchor="middle")
    f.t(140, 222, "准不准、哪里不灵，说不清", 12, C["muted"], anchor="middle")
    f.l(140, 234, 140, 251, C["muted"], 1.5, arrow="muted")
    f.r(50, 256, 180, 48, C["ink2"], rx=10, cls="card")
    f.t(140, 276, "把训练跑完", 13.5, C["white"], 700, "middle")
    f.t(140, 294, "c = 1", 12.5, C["white"], anchor="middle")
    f.l(140, 306, 140, 323, C["muted"], 1.5, arrow="muted")
    f.r(50, 328, 180, 40, C["white"], C["edge"], 1, 10, cls="card")
    f.t(140, 353, "得到这一个结果", 13.5, weight=700, anchor="middle")
    f.path("M50 348 C20 348 20 140 46 140", C["muted"], 1.4, dash="5 4", arrow="muted")
    f.t(140, 390, "下一个问题，再挑、再跑", 12.5, C["muted"], anchor="middle")
    f.l(270, 92, 270, 380, C["grid"], 1.2, cls=None)

    f.t(290, 104, "SAI：先查主张，再决定跑不跑", 14, weight=700)
    f.r(300, 122, 130, 44, C["white"], C["edge"], 1, 10, cls="card")
    f.t(365, 149, "新问题", 13.5, weight=700, anchor="middle")
    f.l(434, 144, 466, 145, C["ink2"], 1.5, arrow="ink")
    f.path("M470 146L566 102L662 146L566 190Z", C["ink2"], 1.4, C["white"], cls=None)
    f.t(566, 142, "落在某条主张的", 13, weight=700, anchor="middle")
    f.t(566, 160, "声明域 D_{M} 内？", 13, weight=700, anchor="middle")
    f.l(662, 146, 700, 146, C["ink2"], 1.5, arrow="ink")
    f.t(680, 138, "是", 12, C["muted"], 700, "middle")
    f.r(704, 120, 150, 52, C["clayw"], C["clay"], 1.5, 10, cls="card")
    f.t(779, 142, "直接作答", 13.5, weight=700, anchor="middle")
    f.t(779, 161, "c 很小，误差 e 已知", 12, C["muted"], anchor="middle")
    f.l(566, 190, 566, 225, C["ink2"], 1.5, arrow="ink")
    f.t(578, 212, "否", 12, C["muted"], 700)
    f.r(476, 230, 180, 52, C["ink2"], rx=10, cls="card")
    f.t(566, 252, "回退：把训练跑完", 13.5, C["white"], 700, "middle")
    f.t(566, 271, "c = 1，得到真值", 12, C["white"], anchor="middle")
    f.l(566, 284, 566, 315, C["ink2"], 1.5, arrow="ink")
    f.r(436, 320, 260, 60, C["white"], C["edge"], 1, 10, cls="card")
    f.t(566, 343, "新结果回流", 13.5, weight=700, anchor="middle")
    f.t(566, 363, "扩展外延、找到前提：知识 → 定律 → 科学", 12, C["muted"], anchor="middle")
    f.r(300, 230, 140, 52, C["card"], C["clay"], 1.5, 10, cls="card")
    f.t(370, 252, "主张库", 13.5, weight=700, anchor="middle")
    f.t(370, 271, "主张 + 声明域 + 坐标", 12, C["muted"], anchor="middle")
    f.l(420, 228, 514, 170, C["clay"], 1.5, dash="5 4", arrow="clay")
    f.t(452, 186, "查询", 12, C["clayd"], 700, "end")
    f.path("M436 350H370V287", C["clay"], 1.5, arrow="clay")
    f.t(378, 330, "更新", 12, C["clayd"], 700)

    f.l(30, 400, 850, 400, C["grid"], 1, cls=None)
    f.t(30, 426, "长期（示意）：同样多的问题里，不必重跑就能作答的比例上升", 13, weight=700)
    for i, (lab, share) in enumerate([("今天", 0.08), ("方向", 0.8)]):
        y = 442 + 22 * i
        f.t(30, y + 10, lab, 12.5, C["muted"])
        f.r(70, y, 520 * share - 1, 12, C["clayf"], rx=3)
        f.r(70 + 520 * share + 1, y, 520 * (1 - share) - 1, 12, C["ink2"], rx=3)
    f.swatch(620, 452, "ok", "落在声明域内：直接作答")
    f.swatch(620, 474, "run", "回退：把训练跑完")
    f.save()


if __name__ == "__main__":
    for fig in (core, overview, claim, coords, axes, fallback, rho, upper_bound, levels, law_axes, science, promote,
                records, system, vision):
        fig()
    print("wrote", len(list(OUT.glob("fig*.svg"))), "figures to", OUT)
