"""Three paired scientific figures, generated only from saved r002 records."""
import hashlib,json
import sys
import numpy as np
import matplotlib.pyplot as plt
from run_r002 import SOURCE,RUN,REPO
sys.path.insert(0,str(REPO/'research'/'d1_scaling_exploration_20261107'))
from plot_r001 import configure_style,clean_axes,save_figure

OUT=REPO/'docs'/'模糊方向'
LABELS={'d':'参数位移 d','V':'函数位移 V','P':'函数投影 P'}


def main():
    configure_style();preview=RUN/'figures';preview.mkdir(exist_ok=True)
    allrows=[json.loads(l) for l in (RUN/'trajectory_scores.jsonl').open(encoding='utf-8')]
    rows=[r for r in allrows if r['split']=='confirmation' and r['cohort']=='core' and r['balance']!=1 and r['method'] in ['Adam','RMSprop','Adagrad']]
    selected=[r for r in rows if r['progress']>=.1 and r['common_dV_coverage']>=.8]
    fig,ax=plt.subplots(figsize=(10.8,6.6),layout='constrained')
    for activation,color,marker in [('identity','#0072B2','o'),('relu','#D55E00','s')]:
        for protocol,filled in [('parameter_radius',False),('function_radius',True)]:
            sub=[r for r in selected if r['activation']==activation and r['eta_protocol']==protocol]
            x=[100*r['common_dV_mae']['d'] for r in sub];y=[100*r['common_dV_mae']['V'] for r in sub]
            ax.scatter(x,y,s=27,alpha=.6,marker=marker,edgecolors=color,facecolors=color if filled else 'none',
                       label=('线性' if activation=='identity' else 'ReLU')+' · '+('匹配参数半径' if not filled else '匹配函数半径'))
    maximum=max(max(100*r['common_dV_mae'][clock] for r in selected) for clock in ['d','V'])*1.04
    ax.plot([0,maximum],[0,maximum],color='#64748b',linestyle='--',linewidth=1.2,label='两种坐标误差相同')
    ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='参数位移坐标：共同测点 MAE / 初始损失（%）',ylabel='函数位移坐标：共同测点 MAE / 初始损失（%）')
    ax.legend(loc='upper left');clean_axes(ax,'both')
    first=save_figure(fig,'D15_layer_scale_paired_errors',OUT,preview,'同路径共同测点评价，点在对角线下代表函数位移误差更小。P2合格591/768，固定窗口且覆盖>=80%、进度>=10%。')
    groups=[('层间缩放 · 线性',rows,'identity'),('层间缩放 · ReLU',rows,'relu')]
    depth=[r for r in allrows if r['split']=='confirmation' and r['cohort']=='depth_holdout' and r['method'] in ['Adam','RMSprop','Adagrad']]
    groups += [('深度留出 · 线性',depth,'identity'),('深度留出 · ReLU',depth,'relu')]
    matrix=[];yl=[];counts=[]
    for label,base,activation in groups:
        base=[r for r in base if r['activation']==activation]
        sub=[r for r in base if all(r['scores'][clock]['complete'] for clock in ['d','V','P'])]
        matrix.append([100*float(np.median([r['scores'][clock]['mae'] for r in sub])) for clock in ['d','V','P']])
        yl.append(label+f'（{len(sub)}/{len(base)}）');counts.append({'label':label,'common_complete':len(sub),'total':len(base)})
    matrix=np.array(matrix)
    fig,ax=plt.subplots(figsize=(10.8,6.6),layout='constrained')
    mesh=ax.pcolormesh(np.arange(4),np.arange(5),matrix,cmap='Blues',vmin=0,vmax=np.max(matrix),edgecolors='white',linewidth=.8)
    ax.set(xticks=np.arange(3)+.5,xticklabels=[LABELS[k] for k in ['d','V','P']],yticks=np.arange(4)+.5,yticklabels=yl)
    ax.invert_yaxis();ax.tick_params(length=0,pad=10)
    for i in range(4):
        for j in range(3):ax.text(j+.5,i+.5,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=16,color='white' if matrix[i,j]>.6*np.max(matrix) else '#243442')
    colorbar=fig.colorbar(mesh,ax=ax,pad=.025);colorbar.set_label('同一完整路径子集上的中位 MAE（%）')
    colorbar.solids.set_rasterized(False)
    second=save_figure(fig,'D15_layer_scale_transfer_heatmap',OUT,preview,'每行只比较d/V/P都完整参考覆盖的相同路径，括号为共同完整路径/全部路径；不同方法的缺失不算成功。')
    examples=json.loads((RUN/'illustrative_curves.json').read_text())
    fig,ax=plt.subplots(figsize=(10.8,6.6),layout='constrained')
    for c,color,marker in [(.1,'#0072B2','o'),(1.,'#009E73','s'),(10.,'#D55E00','^')]:
        for depth,style in [(2,'-'),(4,'--')]:
            row=next(e for e in examples if e['case']['balance']==c and e['case']['depth']==depth)
            ax.plot(row['V'],row['loss_ratio'],color=color,linestyle=style,marker=marker,markevery=64,markersize=4,label=f'c={c:g} · {depth} 个权重层')
    low=min(min(e['loss_ratio']) for e in examples);high=max(max(e['loss_ratio']) for e in examples)
    ax.set(xlim=(0,1.025),ylim=(max(0.,low-.04),max(1.03,high+.03)),xlabel='初始函数位移 V',ylabel='训练损失 / 初始损失')
    clean_axes(ax);ax.legend(loc='best',ncol=2)
    third=save_figure(fig,'D15_layer_scale_function_matched_paths',OUT,preview,'预先固定ReLU teacher/input covariance condition10/ReLU student/width32/seed8201/Adam/function radius1；全部256步，相同f0和相同预计V终点仍不保证真实损失重合。')
    manifest={'figures':[{'svg':str(svg.relative_to(REPO)),'png':str(png),'sha256':hashlib.sha256(svg.read_bytes()).hexdigest()} for svg,png in [first,second,third]],
              'paired_scatter_paths':len(selected),'paired_scatter_V_wins':sum(r['common_dV_mae']['d']-r['common_dV_mae']['V']>1e-10 for r in selected),
              'heatmap_groups':counts,'heatmap_values_percent':matrix.tolist(),'curves_preassigned':True}
    (SOURCE/'r002_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='figures'},indent=2))


if __name__=='__main__':main()
