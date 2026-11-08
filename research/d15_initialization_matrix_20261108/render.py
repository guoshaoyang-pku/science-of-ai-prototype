"""Render saved D15 metrics as an interactive factor matrix and SVGs; no training."""
from pathlib import Path
import hashlib
import json
import math
import os
import itertools
import tempfile
from html import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from bs4 import BeautifulSoup
import design

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
DOCS=ROOT/'docs'/'模糊方向'
DATA=Path(os.environ.get('AIQ_KB_DATA_ROOT',str(ROOT.parent/'AIQ_KB_DATA'))).resolve()
RUN=DATA/'runs/d15_initialization_matrix_20261108'
LABELS={'uniform':'均匀基线','front':'前大后小','back':'前小后大','alternating':'偶数层大',
 'reverse_alternating':'奇数层大','hourglass':'沙漏形','middle_heavy':'中间大','step':'前半大',
 'reverse_step':'后半大','single_bottleneck':'单层近零','sparse_gates':'稀疏近零','geometric':'等比递减',
 'periodic':'三层周期','identity_random':'单位+随机','head_zero':'末层精确零','first_zero':'首层精确零',
 'middle_zero':'中层精确零','two_zero':'首末双零','zas_embed':'ZAS 矩形嵌入','head_near_zero':'末层×0.001',
 'linear':'线性','relu_teacher':'ReLU教师','interaction':'交互 x₁x₂',
 'plain':'串联','residual_hidden':'中层残差','layernorm_hidden':'隐藏层LN'}


def select(cells,**criteria):
    return [c for c in cells if all(c.get(k)==v for k,v in criteria.items())]


def number(v):
    if v is None:return '—'
    if v==0:return '0'
    return f'{v:.2f}' if .01<=abs(v)<100 else f'{v:.1e}'


def annotation(c):
    if not c:return '未测'
    if c['median_unavailable']:text='基线缺失'
    elif c['median_failed']:text='∞'
    elif c.get('baseline_nearzero_n',0):text=number(c['absolutevalmedian'])+'†'
    else:text=number(c['ratio_median'])+'×'
    return text+'\n'+f"{c['hit01_n']}/{c['seed_n']} 达标"+(f" 失{c['failed_n']}" if c['failed_n'] else '')


def strategy_cards():
    """Definition diagrams from frozen design rules, never training results."""
    parts=['<section id="initialization-strategies" aria-labelledby="strategies-title">',
      '<h2 id="strategies-title">初始化策略图解</h2>',
      '<p class="strategy-note">每图用 6 层展示从输入到输出的结构。高度按各图自身最大值归一化，数字是实际尺度乘子；空心圆表示精确零。I 与 I+G 表示矩阵构造，不表示随机权重的尺度。</p>',
      '<div class="strategy-grid">']
    descriptions={
      'uniform':'各层保持基准尺度。','front':'从输入到输出逐渐缩小。','back':'从输入到输出逐渐放大。',
      'alternating':'偶数层较大，奇数层较小。','reverse_alternating':'奇数层较大，偶数层较小。',
      'hourglass':'两端大，中间小。','middle_heavy':'两端小，中间大。','step':'前半层大，后半层小。',
      'reverse_step':'前半层小，后半层大。','single_bottleneck':'仅中间一层缩小，再整体对数居中。',
      'sparse_gates':'每三层的第二层缩小，再整体对数居中。','geometric':'相邻层尺度比为 0.5。',
      'periodic':'每三层重复一次尺度变化。','identity_random':'每层为单位嵌入与随机矩阵的混合，范数匹配基线。',
      'head_zero':'前层随机，末层精确置零。','first_zero':'首层精确置零，其余为随机基线。',
      'middle_zero':'中间一层精确置零，其余为随机基线。','two_zero':'首层、末层同时精确置零。',
      'zas_embed':'前层为单位嵌入或单位矩阵，末层精确置零。','head_near_zero':'只将末层随机权重乘以 0.001。'}
    for profile in design.PROFILES:
        config=dict(depth=6,profile=profile)
        scales=design.scales(config).copy()
        if profile in ('head_zero','zas_embed','two_zero'):scales[-1]=0
        if profile in ('first_zero','two_zero'):scales[0]=0
        if profile=='middle_zero':scales[3]=0
        if profile=='head_near_zero':scales[-1]*=.001
        symbolic=profile in ('identity_random','zas_embed')
        label=LABELS[profile]
        parts.append('<figure class="strategy-card" data-profile="'+profile+'" data-scales="'+escape(json.dumps(scales.tolist()),quote=True)+'">')
        parts.append('<figcaption>'+escape(label)+'</figcaption>')
        parts.append('<svg viewBox="0 0 276 145" role="img" aria-label="'+escape(label)+'：六层初始化结构"><title>'+escape(label+'；'+descriptions[profile])+'</title>')
        parts.append('<line x1="12" y1="104" x2="264" y2="104" stroke="#bbc7ce"/>')
        for i,value in enumerate(scales):
            x=18+42*i;height=68*value/max(scales)
            text=('I+G' if profile=='identity_random' else ('I' if i<5 else '0')) if symbolic else f'{value:.3g}'
            color='#487e9c' if not symbolic else '#548b78'
            if value==0:
                parts.append(f'<circle cx="{x+12}" cy="101" r="4" fill="white" stroke="#bd704d" stroke-width="2"/>')
            else:
                parts.append(f'<rect x="{x}" y="{104-height:.3f}" width="24" height="{height:.3f}" rx="2" fill="{color}"><title>第{i+1}层：{escape(text)}</title></rect>')
            parts.append(f'<text x="{x+12}" y="{max(16,96-height):.3f}" text-anchor="middle" font-size="11" fill="#243b4b">{escape(text)}</text>')
            parts.append(f'<text x="{x+12}" y="121" text-anchor="middle" font-size="11" fill="#64717e">{i+1}</text>')
        parts.append('<text x="12" y="140" font-size="10" fill="#64717e">输入 → 输出（层编号）</text></svg>')
        parts.append('<p>'+escape(descriptions[profile])+'</p></figure>')
    parts.append('</div></section>')
    return '\n'.join(parts)


def heatmap(cells,rows,columns,getter,title,subtitle,name):
    fig,ax=plt.subplots(figsize=(max(8,len(columns)*1.05),max(5,len(rows)*.52+2.4)))
    values=np.full((len(rows),len(columns)),np.nan)
    for i,r in enumerate(rows):
        for j,c in enumerate(columns):
            item=getter(r,c)
            if item and item.get('baseline_nearzero_n',0) and not item['median_failed']:values[i,j]=0
            elif item and item['ratio_median'] is not None:values[i,j]=np.log2(max(item['ratio_median'],1e-12))
            elif item and item['median_failed']:values[i,j]=4
    cmap=plt.get_cmap('PuOr_r').copy();cmap.set_bad('#e1e4e7')
    im=ax.pcolormesh(np.arange(len(columns)+1)-.5,np.arange(len(rows)+1)-.5,
        np.ma.masked_invalid(values),cmap=cmap,norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2),
        shading='flat',edgecolors='white',linewidth=1,rasterized=False)
    ax.set_ylim(len(rows)-.5,-.5);ax.set_xlim(-.5,len(columns)-.5)
    for i,r in enumerate(rows):
        for j,c in enumerate(columns):
            v=values[i,j]
            ax.text(j,i,annotation(getter(r,c)),ha='center',va='center',fontsize=9,
                    color='white' if np.isfinite(v) and abs(v)>1.1 else '#17202b')
    ax.set_xticks(range(len(columns)),[c[1] for c in columns],fontsize=10)
    ax.set_yticks(range(len(rows)),[r[1] for r in rows],fontsize=10)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_title(title+'\n'+subtitle,fontsize=12,pad=16)
    cb=fig.colorbar(im,ax=ax,pad=.02,shrink=.7,extend='both');cb.set_ticks([-2,-1,0,1,2],labels=['≤0.25×','0.5×','1×','2×','≥4×'])
    if cb.solids is not None:cb.solids.set_rasterized(False)
    cb.set_label('验证MSE / 同条件均匀基线（颜色为log₂，文本不截断）',fontsize=10)
    fig.tight_layout();target=DOCS/name;fig.savefig(target,bbox_inches='tight',metadata={'Date':None})
    fig.savefig(Path(tempfile.gettempdir())/name.replace('.svg','_preview.png'),dpi=130,bbox_inches='tight')
    plt.close(fig)
    return name


def main():
    summary=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    raw=json.loads((SOURCE/'cells.json').read_text(encoding='utf-8'))
    cells=raw['cells'] if isinstance(raw,dict) else raw
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'svg.fonttype':'none','axes.unicode_minus':False,'svg.hashsalt':'d15-initialization-matrix'})
    figures=[]
    for task in design.TASKS:
        pool=select(cells,round='r001',task=task,rate=0)
        rows=[(p,LABELS[p]) for p in design.PROFILES]
        cols=[((a,o),f'{a}\n{o.upper()}') for a,o in itertools.product(('linear','relu','tanh'),('sgd','adam'))]
        def lookup(r,c):return next((x for x in pool if x['profile']==r[0] and (x['activation'],x['optimizer'])==c[0]),None)
        figures.append(heatmap(pool,rows,cols,lookup,'初始化结构的表现依赖任务、激活与优化器：'+LABELS[task],
          '串联 · depth=4 · width=16 · 512步 · 4 seeds · SGD .01 / Adam .001\n每格：配对验证MSE倍数中位数；训练MSE≤0.1种子数/4；失=非有限路径\n†基线≤1e−10时显示绝对MSE，颜色中性',f'D15_matrix_{task}.svg'))
    pool=select(cells,round='r002',task='relu_teacher',optimizer='adam',rate=0,condition=1)
    rows=[(p,LABELS[p]) for p in ('uniform','front','back','alternating','head_zero','zas_embed')]
    cols=[((a,d,w),f'{LABELS[a]}\nD{d} W{w}') for a,d,w in itertools.product(('plain','residual_hidden','layernorm_hidden'),(2,4,6),(8,16))]
    def lookup2(r,c):return next((x for x in pool if x['profile']==r[0] and (x['architecture'],x['depth'],x['width'])==c[0]),None)
    figures.append(heatmap(pool,rows,cols,lookup2,'架构、深度与宽度改变初始化响应',
       'ReLU教师 · ReLU学生 · Adam .001 · 输入条件数1 · 512步 · 3 seeds\n每格：配对验证MSE倍数；达训练MSE≤0.1数/3；depth2残差模型为线性\n†基线≤1e−10时显示绝对MSE，颜色中性', 'D15_matrix_geometry.svg'))
    pool=select(cells,round='r003',activation='relu',optimizer='adam',rate=0)
    rows=[((t,a),f'{LABELS[t]}\nα={a:g}') for t,a in itertools.product(design.TASKS,(-1.,0.,1.))]
    cols=[((s,b),f's₀={s:g}\nβ={b:g}') for s,b in itertools.product((.3,1.,3.),(-.7,0.,.7))]
    def lookup3(r,c):return next((x for x in pool if (x['task'],x['alpha'])==r[0] and (x['global_scale'],x['beta'])==c[0]),None)
    figures.append(heatmap(pool,rows,cols,lookup3,'趋势、交替与整体尺度的联合响应',
       '串联ReLU · D4 W16 · Adam .001 · 512步 · 3 seeds\n基线 α=β=0、s₀=1；倍数为验证MSE相对值，达标为训练MSE≤0.1\n†基线≤1e−10时显示绝对MSE，颜色中性','D15_matrix_alpha_beta.svg'))
    page=DOCS/'D15_asymmetric_initialization.html'
    original=(SOURCE/'page_template.html').read_text(encoding='utf-8')
    keys=['id','round','task','activation','optimizer','rate','profile','architecture','depth','width','condition','lr',
      'alpha','beta','global_scale','seed_n','seeds','case_ids','ratio_median','median_failed','median_unavailable','ratio_by_seed',
      'ratio_failed_by_seed','ratio_unavailable_by_seed','failed_n','baseline_failed_n','hit01_n','hit001_n','initial_allzero_n',
      'model_unreachable_n','absolutevalmedian','absolutetrainmedian','improved_n','first_hit_0_1_by_seed',
      'baseline_absolutevalmedian','baseline_nearzero_n','both_nearzero_n','finalval_by_seed','baselineval_by_seed',
      'val_median_failed','ratio_median_finite','failure_codes','baseline_missing_n',
      'expected_seed_n','partial_cell','first_hit_0_01_by_seed']
    compact=[{k:c[k] for k in keys if k in c} for c in cells]
    matrix=BeautifulSoup((SOURCE/'matrix_fragment.html').read_text(encoding='utf-8'),'html.parser')
    counts=summary['counts'];tot=sum(summary['rounds'][r]['loaded_n'] for r in ('r001','r002','r003'))
    matrix.select_one('#path-count').string=f'{tot:,}'
    matrix.select_one('#cell-count').string=f'{len(cells):,}'
    matrix.select_one('#nonfinite-count').string=str(sum(c['failed_n'] for c in cells))
    matrix.select_one('#zero-count').string=str(sum(c['initial_allzero_n'] for c in cells))
    matrix.select_one('#front-verdict').string=f"前大后小并非普遍更好：仅 {summary['r001_front_vs_back']['passed_context_n']}/{summary['r001_front_vs_back']['expected_context_n']} 个上下文通过预设判据"
    gallery=matrix.select_one('#matrix-figures')
    for name in figures:
        details=matrix.new_tag('details');su=matrix.new_tag('summary');su.string='原尺寸矩阵：'+name.replace('D15_matrix_','').replace('.svg','');details.append(su)
        a=matrix.new_tag('a',href=name,target='_blank',rel='noopener');img=matrix.new_tag('img',src=name,alt='D15矩阵，数值和条件见图内，点击可查看原尺寸',loading='lazy');a.append(img);details.append(a);gallery.append(details)
    html=str(matrix)
    data='<script id="d15-matrix-data" type="application/json">'+json.dumps(compact,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')+'</script>'
    css=(SOURCE/'matrix.css').read_text(encoding='utf-8');js=(SOURCE/'matrix.js').read_text(encoding='utf-8')
    css+='\n'+(SOURCE/'strategy_cards.css').read_text(encoding='utf-8')
    original=original.replace('</head>','<style>'+css+'</style></head>',1)
    original=original.replace('</main>',strategy_cards()+'\n'+html+'\n</main>',1)
    original=original.replace('<a href="../RESEARCH_DIRECTIONS.md">研究方向地图</a>',
        '<a href="../RESEARCH_DIRECTIONS.md">研究方向地图</a> · <a href="#initialization-strategies">策略图解</a> · <a href="#d15-results">实验矩阵</a>',1)
    original=original.replace('</body>',data+'<script>'+js+'</script></body>',1)
    page.write_bytes(original.encode('utf-8'))
    manifest={'figures':{n:hashlib.sha256((DOCS/n).read_bytes()).hexdigest() for n in figures},
       'page_sha256':hashlib.sha256(page.read_bytes()).hexdigest(),'paths':tot,'cells':len(cells)}
    (SOURCE/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
