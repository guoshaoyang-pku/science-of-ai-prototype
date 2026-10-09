import json
from pathlib import Path
import matplotlib.pyplot as plt
summary=json.loads((Path(__file__).resolve().parents[1]/'summary.json').read_text())
new=summary['new']['endpoint_minus_minimum']; old=summary['baseline']['endpoint_minus_minimum']
fig,ax=plt.subplots(figsize=(4.6,3.4))
xs=['σ²=.125\nseed411','σ²=.0625\nseed411']; ys=[old,new]
colors=['#6b7280','#2563eb']
ax.scatter(xs,ys,c=colors,s=70,zorder=3)
ax.axhline(.05,color='#dc2626',linestyle='--',linewidth=1.2,label='阈值 .05')
for x,y in zip(xs,ys): ax.text(x,y+0.008,f'{y:.6f}',ha='center',va='bottom',fontsize=9)
ax.set_ylabel('终点减最低风险 Δₜ')
ax.set_ylim(0,0.21); ax.set_title('n32 / seed411：降噪后的 U 型幅度')
ax.legend(frameon=False,fontsize=8); fig.tight_layout()
for ext in ('png','svg'): fig.savefig(Path('directions/D2_overfitting_u_curve/figs')/f'r112_seed411_delta.{ext}',dpi=180)
