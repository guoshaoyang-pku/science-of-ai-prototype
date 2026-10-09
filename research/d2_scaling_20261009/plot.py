#!/usr/bin/env python3
"""Plot saved D2 measurements without recomputing experiment data."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


SOURCE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
GRAY, INK = '#777777', '#242424'


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'legend.fontsize': 9, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .14, 'axes.axisbelow': True,
        'lines.linewidth': 1.8, 'savefig.dpi': 180, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def save(fig, out, name):
    paths = []
    for suffix in ('svg', 'png', 'pdf'):
        path = out / f'{name}.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        paths.append(str(path))
    plt.close(fig)
    return paths


def expanded_scaling(summary, spec, out):
    task = next(t for t in summary['tasks'] if (t['protocol'], t['task']) == ('d4_homogeneous', 'interaction'))
    rows = [r for r in summary['rows'] if (r['protocol'], r['task']) == ('d4_homogeneous', 'interaction')]
    coefficient = task['models']['linear']['c']
    fig, (ax, hm) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.05, 1]})
    groups = [(lambda n: n <= 64, BLUE, 'n = 16–64'),
              (lambda n: 64 < n <= 128, GREEN, 'n = 96–128'),
              (lambda n: n > 128, ORANGE, 'n = 256–1024')]
    for accepts, color, label in groups:
        selected = [r for r in rows if r['interior'] and accepts(r['n'])]
        ax.scatter([r['n'] / r['sigma2'] for r in selected], [r['U_rise'] for r in selected],
                   s=20, alpha=.65, color=color, linewidths=0, label=label)
    calibration = [r for r in rows if r['n'] in spec['calibration']['n'] and r['sigma2'] in spec['calibration']['sigma2'] and r['interior']]
    ax.scatter([r['n'] / r['sigma2'] for r in calibration], [r['U_rise'] for r in calibration],
               s=39, facecolors='none', edgecolors=INK, linewidths=.65, label='Calibration')
    xx = np.geomspace(2, max(r['n'] / r['sigma2'] for r in rows), 200)
    ax.fill_between(xx, coefficient * xx / 2, coefficient * xx * 2, color=GRAY, alpha=.1)
    ax.plot(xx, coefficient * xx, '--', color=INK, label=rf'$U_*={coefficient:.2f}\,n/\sigma^2$')
    ax.set(xscale='log', yscale='log', xlabel=r'$n/\sigma^2$', ylabel=r'Rise time $U_*=\eta t_*$')
    ax.set_title('Original interaction task: expanded grid', loc='left')
    ax.legend(loc='upper left', fontsize=8.5)
    expanded = task['models']['linear']['expanded']
    ax.text(.97, .04, f"Expanded: {expanded['within2']}/{expanded['total']} within factor 2\nBoundary minima count as failures",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    ns, variances = spec['n'], spec['sigma2']
    matrix = np.full((len(ns), len(variances)), np.nan)
    boundary = np.zeros_like(matrix, dtype=int)
    for i, n in enumerate(ns):
        for j, variance in enumerate(variances):
            selected = [r for r in rows if r['n'] == n and r['sigma2'] == variance]
            interior = [r for r in selected if r['interior']]
            boundary[i, j] = len(selected) - len(interior)
            if interior:
                matrix[i, j] = np.median([np.log2(r['U_rise'] / (coefficient * n / variance)) for r in interior])
    cmap = matplotlib.colormaps['RdBu'].copy()
    cmap.set_bad('#eeeeee')
    im = hm.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), aspect='auto')
    for i in range(len(ns)):
        for j in range(len(variances)):
            if boundary[i, j]:
                hm.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='///', edgecolor='#888888', linewidth=0))
                label = f'B:{boundary[i,j]}/3'
            else:
                label = f'{matrix[i,j]:+.1f}'
            hm.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 2 else INK)
    hm.grid(False)
    hm.set_xticks(range(len(variances)), ['1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8'])
    hm.set_yticks(range(len(ns)), ns)
    hm.set(xlabel=r'Noise variance $\sigma^2$', ylabel='Training samples n')
    hm.set_title('Deviation from the straight-line prediction', loc='left')
    cb = fig.colorbar(im, ax=hm, fraction=.047, pad=.025, ticks=[-3, -2, -1, 0, 1, 2, 3], extend='both')
    cb.set_label(r'Median $\log_2[U_* /(c n/\sigma^2)]$')
    fig.text(.53, .012, 'Three seeds per tile; B marks boundary minima. Colors use interior medians only.', fontsize=8.5)
    fig.subplots_adjust(wspace=.3, bottom=.16)
    return save(fig, out, 'expanded_scaling')


def cross_task_laws(summary, out):
    tasks = summary['tasks']
    labels = [f"{t['protocol'].split('_')[0][1:]}D · {t['task']}" + (' · bias' if t['protocol'].endswith('_biased') else '') for t in tasks]
    y = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 6.8), sharey=True,
                             gridspec_kw={'width_ratios': [1.1, 1.1, 2.2]})
    for ax in axes:
        ax.set_ylim(len(tasks) - .55, -.6)
        ax.set_yticks(y, labels)
        for split in [5.5, 7.5, 10.5]:
            ax.axhline(split, color='#cccccc', linewidth=.8)
        ax.grid(axis='y', visible=False)
    coeff = np.array([t['models']['linear']['c'] for t in tasks])
    axes[0].scatter(coeff, y, color=INK, s=30)
    for value, pos in zip(coeff, y):
        axes[0].annotate(f'{value:.2f}', (value, pos), xytext=(6, 4), textcoords='offset points', fontsize=9)
    axes[0].set(xscale='log', xlim=(.018, 5), xlabel=r'$c_k$ in $U_*=c_k n/\sigma^2$')
    axes[0].set_title('Task coefficient', loc='left')
    power = np.array([t['models']['power']['b'] for t in tasks])
    axes[1].scatter(power, y, color=BLUE, s=30)
    axes[1].axvline(1, linestyle='--', color=GRAY)
    for value, pos in zip(power, y):
        axes[1].annotate(f'{value:.2f}', (value, pos), xytext=(6, 4), textcoords='offset points', fontsize=9)
    axes[1].set(xlim=(.15, 1.45), xlabel=r'$b_k$ in $U_*=c_k(n/\sigma^2)^{b_k}$')
    axes[1].set_title('Fitted exponent', loc='left')
    for name, color, offset, marker, label in [
        ('linear', ORANGE, -.18, 'o', 'Exponent fixed at 1'),
        ('power', BLUE, 0, 's', 'Free power'),
        ('log', GREEN, .18, '^', 'Logarithmic')]:
        fractions = [100 * t['models'][name]['expanded']['pass_rate'] for t in tasks]
        axes[2].scatter(fractions, y + offset, color=color, s=29, marker=marker, label=label)
        if name == 'linear':
            for t, fraction, pos in zip(tasks, fractions, y):
                result = t['models'][name]['expanded']
                axes[2].annotate(f"{result['within2']}/{result['total']}", (fraction, pos + offset),
                                 xytext=(5, -1), textcoords='offset points', fontsize=8, color=color)
    axes[2].axvline(80, linestyle='--', color=GRAY)
    axes[2].text(80.8, -.35, '80% gate', fontsize=8, va='center', color=GRAY)
    axes[2].set(xlim=(-2, 109), xlabel='Expanded cells within factor 2 (%)')
    axes[2].set_xticks([0, 20, 40, 60, 80, 100])
    axes[2].set_title('Prediction beyond calibration', loc='left')
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.12), fontsize=8.5, ncol=3)
    fig.text(.03, .018, 'Calibration: n = 32/64/128, σ² = 0.25/0.5/1/2/4. Expanded: 75 conditions × 3 seeds per task. Boundary minima count as failures.', fontsize=9)
    fig.subplots_adjust(left=.19, right=.99, bottom=.17, top=.93, wspace=.18)
    return save(fig, out, 'cross_task_laws')


def train_val_shapes(summary, study, out):
    selected = next(r for r in summary['rows'] if (r['protocol'], r['task'], r['n'], r['sigma2'], r['seed']) == ('d4_homogeneous', 'interaction', 64, .5, 6101))
    with np.load(study / selected['curve']) as saved:
        a = {k: saved[k] for k in saved.files}
    u, star = a['U'], selected['U_rise']
    x = u / star
    inflection = selected['curvature_inflection_over_rise']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax = axes[0, 0]
    for key, realized, color, label in [('expected_noisy_train', 'realized_noisy_train', GREEN, 'Train (noisy labels)'),
                                        ('expected_clean_val', 'realized_clean_val', BLUE, 'Validation (clean labels)'),
                                        ('expected_noisy_val', 'realized_noisy_val', ORANGE, 'Validation (noisy labels)')]:
        ax.plot(u, a[key], color=color, label=label)
        ax.plot(u, a[realized], color=color, linestyle='--', linewidth=.8, alpha=.45)
    ax.axvline(star, color=INK, linestyle='--', linewidth=1)
    ax.set(xscale='symlog', xlabel=r'Training time $U=\eta t$', ylabel='MSE', xlim=(0, u[-1]))
    ax.set_title('A   Train keeps falling while validation rises', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.text(.025, .045, f"At rise: train = {selected['expected_train_fraction_at_rise']:.0%} of initial loss\nSolid: noise expectation; thin dashed: one realization",
            transform=ax.transAxes, fontsize=8.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    ax = axes[0, 1]
    shown = x <= 6
    for key, scale, color, label in [('signal_bias', 1, BLUE, r'Signal residual $B$'),
                                     ('variance_unit', selected['sigma2'], ORANGE, r'Noise cost $\sigma^2 N$'),
                                     ('expected_clean_val', 1, INK, r'Clean validation $R$')]:
        ax.plot(x[shown], scale * a[key][shown], color=color, label=label)
    ax.axvline(1, color=INK, linestyle='--', linewidth=1)
    ax.axvline(inflection, color=GRAY, linestyle=':', linewidth=1.5)
    ax.set(xlabel=r'Relative time $U/U_*$', ylabel='MSE', ylim=(0, .47), xlim=(0, 6))
    ax.set_title('B   Noise has accumulated before the minimum', loc='left')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.text(.26, .5, f"Noise cost at half-time = {selected['noise_at_half_over_at_rise']:.0%} of cost at rise\nMinimum: 1; first later curvature inflection: {inflection:.1f}",
            transform=ax.transAxes, fontsize=8.5,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
    ax = axes[1, 0]
    shown = (x >= .25) & (x <= 4)
    k = np.flatnonzero(a['t'] == selected['t_star'])[0]
    cost_at_rise = a['noise_slope_U'][k]
    ax.plot(x[shown], -a['signal_slope_U'][shown] / cost_at_rise, color=BLUE, label=r'Signal gain $-B\prime$')
    ax.plot(x[shown], a['noise_slope_U'][shown] / cost_at_rise, color=ORANGE, label=r'Noise cost rate $\sigma^2N\prime$')
    ax.axvline(1, color=INK, linestyle='--', linewidth=1)
    ax.axvline(inflection, color=GRAY, linestyle=':', linewidth=1.5)
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.set(xlim=(.25, 4), ylim=(-.3, 7), xlabel=r'Relative time $U/U_*$', ylabel='Rate / noise cost rate at minimum')
    ax.set_title('C   Rise begins when the two rates balance', loc='left')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.annotate('Signal gain = noise cost rate', xy=(1, 1), xytext=(1.3, 3.3),
                fontsize=9, arrowprops={'arrowstyle': '->', 'color': GRAY})
    ax = axes[1, 1]
    fractions = np.sort([r['expected_train_fraction_at_rise'] for r in summary['rows'] if r['interior']])
    cdf = np.arange(1, len(fractions) + 1) / len(fractions)
    ax.plot(fractions, cdf * 100, color=GREEN)
    result = summary['registered_predictions']['P3']
    ax.axvline(.1, color=GRAY, linestyle='--', linewidth=1)
    ax.scatter([.1], [result['rate'] * 100], s=40, color=ORANGE, zorder=4)
    ax.annotate(f"{result['passed']}/{result['total']} = {result['rate']:.0%}\nTrain ≤10% of initial loss",
                (.1, result['rate'] * 100), xytext=(.25, 14), fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': GRAY})
    ax.set(xlim=(0, 1), ylim=(0, 102), xlabel='Train loss at rise / initial train loss', ylabel='Interior curves at or below this value (%)')
    ax.set_title('D   Very low train loss is not necessary', loc='left')
    fig.text(.055, .014, 'A–C: original 4D interaction, n = 64, σ² = 0.5, seed 6101, frozen ReLU features. D: all 3137 interior curves across 12 task recipes.', fontsize=9)
    fig.subplots_adjust(wspace=.25, hspace=.4, bottom=.11, top=.95)
    return save(fig, out, 'train_val_shapes'), selected


def logic_schematic(out):
    fig = plt.figure(figsize=(12.3, 5.7))
    ax = fig.add_axes([.07, .25, .46, .66])
    x = np.linspace(0, 4.8, 1000)
    decay = 2 ** (-x)
    b, noise = decay ** 2, (1 - decay) ** 2
    ax.plot(x, b, color=BLUE, label=r'Signal residual $B$')
    ax.plot(x, noise, color=ORANGE, label=r'Noise cost $\sigma^2N$')
    ax.plot(x, b + noise, color=INK, linewidth=2.5, label=r'Validation risk $R=B+\sigma^2N$')
    ax.axvline(1, color=INK, linewidth=1, linestyle='--')
    ax.axvline(2, color=GRAY, linewidth=1.3, linestyle=':')
    ax.scatter([1], [.5], color=INK, s=38, zorder=5)
    ax.scatter([2], [.625], facecolors='white', edgecolors=GRAY, s=40, zorder=5)
    ax.annotate('Minimum: slope = 0', (1, .5), xytext=(.25, .29),
                fontsize=9, arrowprops={'arrowstyle': '->', 'color': GRAY})
    ax.annotate('Later inflection:\ncurvature changes sign', (2, .625), xytext=(2.35, .42),
                fontsize=9, arrowprops={'arrowstyle': '->', 'color': GRAY})
    ax.set(xlim=(0, 4.8), ylim=(0, 1.06), xlabel=r'Relative training time $U/U_*$', ylabel='Schematic loss')
    ax.set_title('A one-mode example of the mechanism', loc='left')
    ax.legend(loc='upper right', fontsize=8.5)
    diagram = fig.add_axes([.58, .16, .39, .8])
    diagram.set(xlim=(0, 1), ylim=(0, 1))
    diagram.axis('off')
    diagram.text(.5, .99, 'At fixed data and features', ha='center', va='top', fontsize=11, weight='bold')
    for xpos, label, color in [(.03, 'Target changes\nsignal residual $B_k(U)$', BLUE),
                               (.52, 'Label noise scales\nnoise cost $\\sigma^2N_n(U)$', ORANGE)]:
        patch = FancyBboxPatch((xpos, .73), .45, .15, boxstyle='round,pad=.012',
                               facecolor='white', edgecolor=color, linewidth=1.2)
        diagram.add_patch(patch)
        diagram.text(xpos + .225, .805, label, ha='center', va='center', fontsize=9.5, color=color)
        diagram.annotate('', xy=(.5, .63), xytext=(xpos + .225, .72),
                         arrowprops={'arrowstyle': '->', 'color': GRAY, 'linewidth': 1})
    diagram.text(.5, .59, r'$R_{k,n}(U)=B_{k,n}(U)+\sigma^2N_n(U)$', ha='center', fontsize=12)
    stages = [(.44, 'Before the minimum', r'$-B\prime > \sigma^2N\prime$  : risk falls'),
              (.31, 'At the minimum', r'$-B\prime = \sigma^2N\prime$  : rates balance'),
              (.18, 'After the minimum', r'$-B\prime < \sigma^2N\prime$  : risk rises')]
    for ypos, heading, formula in stages:
        diagram.text(.04, ypos, heading, weight='bold', fontsize=9.5)
        diagram.text(.04, ypos - .055, formula, fontsize=10)
    diagram.text(.04, .025, 'A shared noise shape does not imply\na universal rise-time exponent.', fontsize=9.5, color=GRAY)
    fig.text(.07, .085, 'Noise fitting begins before the rise. The minimum is a rate balance; it need not coincide with train loss approaching zero.', fontsize=10)
    fig.text(.07, .035, r'Schematic only: $B=e^{-4\lambda U}$, noise $=(1-e^{-2\lambda U})^2$, $U_*=(\log 2)/(2\lambda)$. Real curves can have several inflections.', fontsize=8.5, color=GRAY)
    return save(fig, out, 'd2_logic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path)
    args = parser.parse_args()
    spec = json.loads((SOURCE / 'preregistration.json').read_text())
    study = args.study or SOURCE.parents[1] / 'evidence' / spec['run_name'] / 'studies' / spec['study']
    summary_path = study / 'summary.json'
    summary = json.loads(summary_path.read_text())
    out = study.parents[1] / 'figs'
    out.mkdir(parents=True, exist_ok=True)
    style()
    plots = {
        'expanded_scaling': expanded_scaling(summary, spec, out),
        'cross_task_laws': cross_task_laws(summary, out),
    }
    plots['train_val_shapes'], selected = train_val_shapes(summary, study, out)
    plots['d2_logic'] = logic_schematic(out)
    manifest = {
        'summary': str(summary_path),
        'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'representative_curve': selected,
        'schematic': 'Analytic one-mode example, not measured data; equations stated in figure.',
        'plots': plots,
    }
    (out / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'figures': str(out), 'plots': list(plots)}))


if __name__ == '__main__':
    main()
