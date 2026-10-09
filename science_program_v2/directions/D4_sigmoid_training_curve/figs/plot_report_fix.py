"""只读保存结果生成报告点图；不训练、不拟合、不创建 cell。"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FIGS = Path(__file__).resolve().parent
D4 = FIGS.parent
D12 = D4.parent / 'D12_cascade_sigmoid'


def main():
    sources = sorted((D4 / 'studies').rglob('*'))
    sources += sorted((D12 / 'studies' / 'r028_gap_cascade').rglob('*'))
    sources = [p for p in sources if p.is_file()]
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in sources}
    with tempfile.TemporaryDirectory(dir=FIGS) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        sys.path.insert(0, 'LOCAL_HOME/.verdent/skills/report_to_human/scripts')
        import panel_style
        panel_style.apply()
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from scipy.special import expit

        summaries = {}
        rows = []
        for study in ['r002_two_mode_logistic', 'r018_slow_energy_logistic',
                      'r038_maxabs_objective', 'r050_alpha025_maxabs',
                      'r082_alpha050_maxabs', 'r094_alpha075_maxabs']:
            summary = json.loads((D4 / 'studies' / study / 'summary.json').read_text())
            summaries[study] = summary
            for cell in summary['cells']:
                if 'best_errors' in cell:
                    rows.append(('least_squares', cell['best_errors'], cell['adequate']))
                else:
                    rows.append(('minimax', cell['errors']['minimax'], cell['adequate']))
        boundary = json.loads((D12 / 'studies/r028_gap_cascade/summary.json').read_text())
        for cell in boundary['cells']:
            rows.append(('least_squares', cell['single'], cell['single']['adequate']))

        colors = {'least_squares': panel_style.C['blue'],
                  'minimax': panel_style.C['orange']}
        names = {'least_squares': '最小二乘', 'minimax': 'min-maxabs'}
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for protocol in colors:
            for passed, marker in [(True, 'o'), (False, 'X')]:
                group = [e for p, e, ok in rows if p == protocol and ok == passed]
                ax.scatter([e['rmse'] for e in group], [e['maxabs'] for e in group],
                           color=colors[protocol], marker=marker, s=44, alpha=0.7)
        ax.set(xscale='log', yscale='log', xlim=(0.009, 0.085), ylim=(0.027, 0.21),
               xlabel=r'均方根误差 $E_2$（对数轴）', ylabel=r'最差点误差 $E_\infty$（对数轴）')
        ax.axvline(0.03, color=panel_style.C['muted'], linestyle='--', linewidth=1)
        ax.axhline(0.05, color=panel_style.C['muted'], linestyle='--', linewidth=1)
        ax.text(0.0308, 0.185, r'$E_2 = 0.03$', fontsize=9)
        ax.text(0.079, 0.0515, r'$E_\infty = 0.05$', ha='right', fontsize=9)
        handles = [Line2D([], [], color=colors[p], marker='o', linestyle='', label=names[p])
                   for p in colors]
        handles += [Line2D([], [], color=panel_style.C['dark'], marker=m,
                           linestyle='', label=n) for m, n in [('o', '通过'), ('X', '失败')]]
        ax.legend(handles=handles, loc='upper left', ncol=2)
        fig.subplots_adjust(bottom=0.25)
        fig.text(0.12, 0.035,
                 '图 1｜每个保存 cell 一个点；54 个最小二乘、15 个 min-maxabs 点，坐标重复重叠。\n'
                 '等占比谱比：1/3/10/12/15/20/25/30/100；谱比 100 的 α：0–1（九个离散点）。\n'
                 'min-maxabs：谱比 100，α=0/0.01/0.25/0.5/0.75；误差以真实动态范围 1 归一化。',
                 fontsize=8.5)
        panel_style.save(fig, FIGS / 'decision_map')

        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
        metric_checks = []
        for ax, study, cell_id in zip(axes, ['r038_maxabs_objective', 'r050_alpha025_maxabs'],
                                    ['alpha_0000_seed_0', 'alpha_0250_seed_0']):
            cell = next(c for c in summaries[study]['cells'] if c['cell_id'] == cell_id)
            folder = D4 / 'studies' / study / 'results' / cell_id
            metadata = json.loads((folder / 'metadata.json').read_text())
            assert hashlib.sha256((folder / 'arrays.npz').read_bytes()).hexdigest() == metadata['arrays_sha256']
            with np.load(folder / 'arrays.npz') as arrays:
                grid, observed = arrays['grid'], arrays['observed']
                assert grid.tolist() == summaries[study]['grid']
                for protocol in colors:
                    log_half, beta = cell['parameters_log_half_beta'][protocol]
                    prediction = np.ones_like(observed)
                    positive = grid > 0
                    prediction[positive] = expit(beta * (log_half - np.log(grid[positive])))
                    assert np.max(np.abs(prediction - arrays[protocol + '_prediction'])) < 1e-12
                    error = np.abs(prediction - observed)
                    metrics = {'rmse': float(np.sqrt(np.mean(error**2))), 'maxabs': float(error.max())}
                    assert all(abs(metrics[m] - cell['errors'][protocol][m]) < 1e-12 for m in metrics)
                    metric_checks.append({'study': study, 'cell': cell_id, 'protocol': protocol, **metrics})
                    ax.scatter(grid[positive], error[positive], s=12, color=colors[protocol],
                               alpha=0.85, label=names[protocol])
                ax.set(xscale='log', xlabel='更新步数 t（对数轴）',
                       title=f'谱比 100，α={cell["alpha"]}，坐标 seed=0')
                ax.axhline(0.05, color=panel_style.C['muted'], linestyle='--', linewidth=1)
                ax.set_ylim(-0.003, 0.19)
                ax.text(1.2, 0.18, f'最小二乘峰值：第 1 步\n{cell["errors"]["least_squares"]["maxabs"]:.8f}',
                        fontsize=8.5, va='top')
        axes[0].set_ylabel('逐点绝对误差 |拟合 − 真实|')
        axes[1].legend(loc='upper right')
        fig.subplots_adjust(bottom=0.27, wspace=0.1)
        fig.text(0.1, 0.035,
                 '图 2｜只用原 150 点网格；t=0 两协议误差均为 0，因对数轴省略该点。\n'
                 '最小二乘误差集中于最早快降段；min-maxabs 压低峰值、抬高其他位置误差。\n'
                 '“均匀”指多个位置接近同一误差上限，并非所有点误差相等；虚线为 0.05。',
                 fontsize=8.5)
        panel_style.save(fig, FIGS / 'pointwise_errors')
    assert all((hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) == before[p]
               for p in sources)
    print(json.dumps({'decision_points': len(rows), 'source_files_unchanged': len(sources),
                      'pointwise_metric_checks': metric_checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
