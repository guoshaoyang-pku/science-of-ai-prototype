#!/usr/bin/env python3
"""Coordinator zero-training integration analysis for D2 (2026-10-08).

Post-hoc (development) on saved curves only. Tests the mechanism claim
t* = f(sigma^2/n; spectrum): (i) exact risk identity R = B + s^2*N per cell;
(ii) n*N(t) concentration across n (the 1/n factor of the noise rate);
(iii) signal_bias concentration across n; (iv) folding scatter old(sample
normalization) vs population normalization, per seed.
Also emits the two integration figures (collapse + rising branch).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "LOCAL_HOME/.verdent/skills/report_to_human/scripts")
import matplotlib.pyplot as plt
from panel_style import apply, save, C

apply()
D = Path(__file__).resolve().parent.parent.parent  # directions/D2_.../
ST = D / "studies"
ETA = 0.3
OUT = Path(__file__).resolve().parent
FIG = D / "figs"
FIG.mkdir(exist_ok=True)

cells = []
for study, norm in [("r003_noise_sample_scaling", "sample"),
                    ("r019_population_label_normalization", "population"),
                    ("r051_noise_halving", "population"),
                    ("r083_low_noise_u_threshold", "population"),
                    ("r095_low_noise_seed413", "population"),
                    ("r112_seed411_low_noise", "population")]:
    for f in sorted(glob.glob(str(ST / study / "results" / "*.npz"))):
        name = Path(f).stem
        parts = name.split("_")
        n = int(parts[0][1:])
        seed = int(parts[1][4:])
        var = float(parts[2][3:])
        d = np.load(f)
        cells.append(dict(study=study, norm=norm, n=n, seed=seed, var=var,
                          path=f, d=d))

# (i) identity R = B + s^2 N
ident_err = max(float(np.abs(c["d"]["expected_risk"]
                             - (c["d"]["signal_bias"] + c["var"] * c["d"]["variance_unit"])).max())
                for c in cells)

# t* and U_rise
for c in cells:
    r = c["d"]["expected_risk"]
    c["tstar"] = int(np.argmin(r))
    c["U"] = ETA * c["tstar"]
    c["Lmin"] = float(r[c["tstar"]])

# (ii)/(iii) concentration of n*N(t) and B(t) across n, per seed, sample norm
tgrid = np.arange(64, 4097)
conc = {}
for seed in [411, 412, 413, 414]:
    byn = {}
    for c in cells:
        if c["norm"] == "sample" and c["seed"] == seed and c["var"] == 1.0:
            byn[c["n"]] = c
    if len(byn) < 2:
        continue
    ns = sorted(byn)
    Nref = byn[ns[0]]["d"]["variance_unit"][tgrid] * ns[0]
    Bref = byn[ns[0]]["d"]["signal_bias"][tgrid]
    devN, devB = [], []
    for n in ns[1:]:
        Nn = byn[n]["d"]["variance_unit"][tgrid] * n
        Bn = byn[n]["d"]["signal_bias"][tgrid]
        devN.append(float(np.abs(Nn - Nref).max() / max(Nref.max(), 1e-12)))
        devB.append(float(np.abs(Bn - Bref).max() / max(Bref.max(), 1e-12)))
    conc[seed] = dict(ns=ns, devN=devN, devB=devB)

# (iv) folding scatter: equal n/sigma^2=128 pairs n32/n64, per seed, both norms
pairs = {}
for norm in ["sample", "population"]:
    rows = {}
    for c in cells:
        if c["norm"] == norm and abs(c["n"] / c["var"] - 128) < 1e-9:
            rows[(c["seed"], c["n"])] = c
    ratios = {}
    for seed in [411, 412, 413, 414]:
        if (seed, 32) in rows and (seed, 64) in rows:
            ratios[seed] = rows[(seed, 64)]["tstar"] / rows[(seed, 32)]["tstar"]
    pairs[norm] = ratios


def mean_abs_log2(rs):
    return float(np.mean([abs(np.log2(v)) for v in rs.values()]))


def mean_abs_log2_ex(rs):
    return float(np.mean([abs(np.log2(v)) for k, v in rs.items() if k != 414]))


scatter = {
    "sample_all4": mean_abs_log2(pairs["sample"]),
    "population_all4": mean_abs_log2(pairs["population"]),
    "sample_ex414": mean_abs_log2_ex(pairs["sample"]),
    "population_ex414": mean_abs_log2_ex(pairs["population"]),
    "ratios": pairs,
}

# global power-law fit on sample-norm r003 cells
r003 = [c for c in cells if c["study"] == "r003_noise_sample_scaling"]
x = np.log(np.array([c["n"] / c["var"] for c in r003]))
y = np.log(np.array([c["U"] for c in r003]))
b, a = np.polyfit(x, y, 1)
pred = np.exp(a + b * x)
err = np.exp(np.abs(np.log(np.array([c["U"] for c in r003]) / pred)))
fit = dict(a=float(a), b=float(b), C=float(np.exp(a)),
           within2=int((err <= 2).sum()), ncells=len(r003),
           max_factor=float(err.max()))

# ---- figures ----
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for c in r003:
    ax.loglog(c["n"] / c["var"], c["U"], "o", ms=4.5,
              color={32: C["orange"], 64: C["blue"], 128: C["green"]}[c["n"]],
              mec="white", mew=.6)
for c in cells:
    if c["norm"] == "population":
        ax.loglog(c["n"] / c["var"], c["U"], "x", ms=5, color=C["mid"])
g = np.logspace(np.log(8), np.log(512), 50)
ax.loglog(g, np.exp(a) * g ** b, "--", color=C["dark"], lw=1.4,
          label=f"全局拟合 U={np.exp(a):.3f}·(n/σ²)^{b:.3f}")
ax.set_xlabel("n/σ²（样本数 ÷ 标签噪声方差）")
ax.set_ylabel("U_rise = η·t*（回升起点训练量）")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("回升起点随 n/σ² 的折叠：圆点=样本归一化（按 n 着色），×=population 归一化")
save(fig, str(FIG / "collapse_n_over_sigma2"))

fig, ax = plt.subplots(figsize=(7.2, 4.0))
xs_all = np.logspace(-1, 2.2, 200)
ax.loglog(xs_all, xs_all ** 1.5455654305138984, "--", color=C["dark"], lw=1.4,
          label="x^1.5456（短段律）")
for c in cells:
    r = c["d"]["expected_risk"]
    t = np.arange(len(r))
    tr = t[c["tstar"]:]
    lr = r[c["tstar"]:]
    Cc = r[min(2 * c["tstar"], len(r) - 1)] - c["Lmin"]
    if Cc <= 0:
        continue
    xx = (ETA * tr - c["U"]) / c["U"]
    yy = (lr - c["Lmin"]) / Cc
    m = xx > 0.02
    ax.loglog(xx[m], yy[m], "-", lw=.7, alpha=.45,
              color=C["blue"] if c["norm"] == "sample" else C["orange"])
ax.axvspan(0.1, 1.0, color=C["green"], alpha=.10)
ax.set_xlabel("x = (U−U_rise)/U_rise")
ax.set_ylabel("(L_test−L_min)/C")
ax.set_xlim(0.05, 200)
ax.set_ylim(1e-3, 1e4)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("上升支折叠：绿带 x∈[0.1,1] 内贴幂律，x>1 全部发散（78/78 超两倍）")
save(fig, str(FIG / "rising_branch_collapse"))

summary = dict(identity_max_error=ident_err, concentration=conc,
               scatter=scatter, global_fit=fit,
               n_cells=len(cells),
               label="coordinator zero-training post-hoc (development), 2026-10-08")
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "concentration"},
                 ensure_ascii=False, indent=1)[:1600])
print("concentration:", json.dumps(conc)[:600])
