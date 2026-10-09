"""Shared figure style for report-to-human panels.

Palette: Anthropic brand-guidelines (github.com/anthropics/skills, skills/brand-guidelines).
Chart rules: Orchestra AI-Research-SKILLs academic-plotting
(github.com/Orchestra-Research/AI-Research-SKILLs, 20-ml-paper-writing/academic-plotting).

Usage:
    from panel_style import apply, C, SERIES, save
    apply()
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(x, y, color=C["orange"])      # the main result is always orange
    save(fig, "docs/reports/figs/reward") # writes reward.svg + reward.png (2x)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {
    "dark": "#141413",    # text, axes labels
    "light": "#faf9f5",   # page background
    "mid": "#b0aea5",     # baselines, reference lines, secondary series
    "soft": "#e8e6dc",    # grid
    "muted": "#5e5d59",   # tick labels
    "orange": "#d97757",  # the main result
    "blue": "#6a9bcc",    # comparison 1
    "green": "#788c5d",   # comparison 2
}
# Main result first. More than 5 series: switch to Okabe-Ito (colorblind-safe) or split the figure.
SERIES = [C["orange"], C["blue"], C["green"], C["dark"], C["mid"]]
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

FIG_WIDE = (7.2, 3.2)    # one figure across the 760px panel column
FIG_HALF = (3.6, 2.8)    # two figures side by side

CJK_SANS = ["PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "Hiragino Sans GB",
            "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]


def _cjk_first():
    """Put the first installed CJK sans font first: matplotlib's per-glyph fallback is unreliable."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    cjk = [n for n in CJK_SANS if n in have]
    return cjk[:1] + ["Poppins", "Arial"] + cjk[1:]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _cjk_first(),
        "axes.unicode_minus": False,
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "axes.titlesize": 10, "axes.titleweight": "normal",
        "text.color": C["dark"], "axes.labelcolor": C["dark"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.edgecolor": C["mid"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["soft"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 1.8, "lines.markersize": 4,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
        "svg.fonttype": "none",
    })


def save(fig, stem):
    """Write <stem>.svg (sharp in the panel) and <stem>.png (2x, for markdown/chat previews)."""
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)
