"""注册系数散布点图：显示全部 seed、聚合极差与减半阈值。"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'executed/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import STUDY, contract, saved


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--commit',required=True); args=ap.parse_args()
    cfg=contract(args.commit); saved(cfg,args.commit)
    s=json.loads((STUDY/'summary.json').read_text())
    fig, axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    colors={'sample':'#32658c','population':'#ae532f'}
    for j,norm in enumerate(['sample','population']):
        ax=axes[j]
        for seed in cfg['seeds']:
            for n,marker in [(32,'o'),(64,'s')]:
                rows=[r for r in s['cells'] if r['normalization']==norm and r['seed']==seed and r['n']==n]
                ax.scatter([r['x'] for r in rows],[r['k_point'] for r in rows],marker=marker,
                           facecolors='none' if n==32 else colors[norm],edgecolors=colors[norm],s=30,
                           label=f'n={n}' if seed==411 else None)
        for n,style in [(32,'--'),(64,':')]:
            ax.axhline(s['coefficients'][norm][str(n)],color=colors[norm],ls=style,lw=1.5)
        ax.set_title(norm+' labels')
        ax.set_xlabel('x = n / sigma²'); ax.set_ylabel('k = U_rise / x')
        ax.set_xticks([128,256]); ax.legend(frameon=False)
    ax=axes[2]
    for i,norm in enumerate(['sample','population']):
        vals=[v[norm]['range'] for v in s['per_seed']]
        ax.scatter([i+(.04*(j-1.5)) for j in range(4)],vals,color=colors[norm],s=25)
        ax.scatter([i],[s['absolute_spreads'][norm]],marker='D',color='black',s=45,
                   label='pooled range' if i==0 else None)
    ax.axhline(.5*s['absolute_spreads']['sample'],color='#444444',ls='--',label='half sample range')
    ax.set_xticks([0,1],['sample','population']); ax.set_ylabel('|c64 - c32|')
    ax.set_title('P1 '+s['P1']['status']); ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed exponent 1, x = 128 and 256; all four seeds retained')
    for ext in ['svg','png']:
        out=STUDY.parents[1]/'figs'/('r113_population_coefficient.'+ext)
        assert not out.exists()
        fig.savefig(out,dpi=180)
    plt.close(fig)


if __name__=='__main__': main()
