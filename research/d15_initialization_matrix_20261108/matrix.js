(()=>{'use strict';
const source=document.getElementById('d15-matrix-data');if(!source)return;
const raw=JSON.parse(source.textContent),cells=Array.isArray(raw)?raw:raw.cells;
const $=id=>document.getElementById(id),root=$('d15-results');if(!root)return;
const labels={uniform:'均匀基线',front:'前大后小',back:'前小后大',alternating:'偶数层大',reverse_alternating:'奇数层大',hourglass:'沙漏形',middle_heavy:'中间大',step:'前半大',reverse_step:'后半大',single_bottleneck:'单层近零',sparse_gates:'稀疏近零',geometric:'等比递减',periodic:'三层周期',identity_random:'单位＋随机',head_zero:'末层精确零',first_zero:'首层精确零',middle_zero:'中层精确零',two_zero:'首末双零',zas_embed:'ZAS 矩形嵌入',head_near_zero:'末层 ×.001',linear:'线性',relu_teacher:'ReLU 教师',interaction:'交互 x₁x₂',plain:'串联',residual_hidden:'中层残差',layernorm_hidden:'隐藏层 LN',relu:'ReLU',tanh:'tanh',sgd:'SGD',adam:'Adam'};
const profiles=['uniform','front','back','alternating','reverse_alternating','hourglass','middle_heavy','step','reverse_step','single_bottleneck','sparse_gates','geometric','periodic','identity_random','head_zero','first_zero','middle_zero','two_zero','zas_embed','head_near_zero'];
const tasks=['linear','relu_teacher','interaction'],opts=['sgd','adam'],architectures=['plain','residual_hidden','layernorm_hidden'];
const controls=['round','task','activation','optimizer','rate','condition','metric'];
let selectedButton=null;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function finite(v){return typeof v==='number'&&Number.isFinite(v);}
function fmt(v,digits=3){if(!finite(v))return '—';if(v===0)return '0';if(Math.abs(v)<.001||Math.abs(v)>=10000)return v.toExponential(2);return Number(v.toPrecision(digits)).toString();}
function ratio(c){return c.median_unavailable?'N/A':c.median_failed?'∞':finite(c.ratio_median)?fmt(c.ratio_median)+'×':'N/A';}
function ratioColor(v){if(!finite(v))return '#e9edf0';const a=[246,246,242],t=Math.max(-2,Math.min(2,Math.log2(Math.max(v,1e-300)))),b=t<0?[176,207,232]:[238,193,157],w=Math.abs(t)/2;return 'rgb('+a.map((x,i)=>Math.round(x+(b[i]-x)*w)).join(',')+')';}
function simpleColor(fraction,bad=false){const a=[247,248,248],b=bad?[238,193,157]:[176,207,232],w=Math.max(0,Math.min(1,fraction));return 'rgb('+a.map((x,i)=>Math.round(x+(b[i]-x)*w)).join(',')+')';}
function read(){return Object.fromEntries(controls.map(k=>[k,$('matrix-'+k).value]));}
function filterState(){const s=read(),r=s.round,act=$('matrix-activation'),condition=$('matrix-condition');act.disabled=r==='r002';condition.disabled=r!=='r002';
if(r==='r002'){act.value='relu';s.activation='relu';}else{condition.value='1';s.condition='1';}
if(r!=='r001'){if(s.task==='all'){s.task='relu_teacher';$('matrix-task').value=s.task;}if(s.optimizer==='all'){s.optimizer='adam';$('matrix-optimizer').value=s.optimizer;}}
return s;}
function rowsColumns(s){if(s.round==='r001'){const ts=s.task==='all'?tasks:[s.task],os=s.optimizer==='all'?opts:[s.optimizer];return {rows:profiles.map(p=>({label:labels[p],match:{profile:p}})),columns:ts.flatMap(t=>os.map(o=>({label:labels[t]+'\n'+labels[o],match:{task:t,optimizer:o}})))};}
if(s.round==='r002')return {rows:['uniform','front','back','alternating','head_zero','zas_embed'].map(p=>({label:labels[p],match:{profile:p}})),columns:architectures.flatMap(a=>[2,4,6].flatMap(d=>[8,16].map(w=>({label:labels[a]+'\n深度 '+d+' · 宽 '+w,match:{architecture:a,depth:d,width:w}}))))};
return {rows:[-1,0,1].map(a=>({label:'α = '+a,match:{alpha:a}})),columns:[.3,1,3].flatMap(g=>[-.7,0,.7].map(b=>({label:'s₀ = '+g+'\nβ = '+b,match:{global_scale:g,beta:b}})))};}
function poolFor(s){return cells.filter(c=>c.round===s.round&&c.activation===s.activation&&c.rate===Number(s.rate)&&(s.round==='r001'||(c.task===s.task&&c.optimizer===s.optimizer))&&(s.round!=='r002'||c.condition===Number(s.condition)));}
function matches(c,obj){return Object.entries(obj).every(([k,v])=>c[k]===v);}
function cellText(c,metric){if(!c)return {text:'未测',color:'#e9edf0',state:'cell-unavailable'};const near=(c.baseline_nearzero_n||0)>0;
if(metric==='hit')return {text:c.hit01_n+'/'+c.seed_n,color:simpleColor(c.hit01_n/c.seed_n)};
if(metric==='zero')return {text:c.initial_allzero_n+'/'+c.seed_n,color:simpleColor(c.initial_allzero_n/c.seed_n,true)};
if(metric==='failure')return {text:c.failed_n+'/'+c.seed_n,color:simpleColor(c.failed_n/c.seed_n,true)};
if(metric==='absoluteval'||(metric==='comparison'&&near&&!c.median_unavailable)){return {text:(c.val_median_failed?'∞':fmt(c.absolutevalmedian))+(near?'†':''),color:'#f4f5f3',state:c.val_median_failed?'cell-failure':''};}
return {text:ratio(c)+(near?'†':''),color:ratioColor(c.ratio_median),state:c.median_unavailable?'cell-unavailable':c.median_failed?'cell-failure':''};}
function detail(c){if(!c)return;const panel=$('matrix-cell-detail'),container=$('matrix-detail-content');container.replaceChildren();$('matrix-detail-title').textContent='— '+(labels[c.profile]||'α='+c.alpha+' β='+c.beta+' s₀='+c.global_scale)+' · '+labels[c.task]+' · '+labels[c.optimizer];
container.appendChild(el('p','',c.round+' · '+labels[c.architecture]+' · 激活 '+labels[c.activation]+' · 深度 '+c.depth+' · 宽度 '+c.width+' · 学习率 '+c.lr+' · 输入条件数 '+c.condition));
container.appendChild(el('p','', '主比值中位数 '+ratio(c)+'；绝对验证 MSE '+(c.val_median_failed?'∞':fmt(c.absolutevalmedian,6))+'；基线绝对 MSE '+fmt(c.baseline_absolutevalmedian,6)+'；达训练 MSE≤.1：'+c.hit01_n+'/'+c.seed_n+'；非有限 '+c.failed_n+'/'+c.seed_n+'；初始全零梯度 '+c.initial_allzero_n+'/'+c.seed_n));
if((c.baseline_nearzero_n||0)>0)container.appendChild(el('p','matrix-help','† '+c.baseline_nearzero_n+'/'+c.seed_n+' 个基线 seed 的验证 MSE≤10⁻¹⁰，其中 '+(c.both_nearzero_n||0)+' 对两边均接近零。原始比值保留，不能把数值残差的大倍数当作实质差异。'));
if(c.model_unreachable_n)container.appendChild(el('p','matrix-help','不可达：'+c.model_unreachable_n+'/'+c.seed_n+' 条线性串联路径的最小二乘训练地板大于 .1。'));
if(c.architecture==='residual_hidden'&&c.depth===2)container.appendChild(el('p','matrix-help','此深度没有内部残差层，实际是两层线性模型。'));
const scroll=el('div','matrix-small-scroll'),table=el('table'),head=el('thead'),hr=el('tr');['seed／路径','验证 MSE','基线 MSE','原始比值','首次训练≤.1','首次训练≤.01'].forEach(t=>hr.appendChild(el('th','',t)));head.appendChild(hr);table.appendChild(head);const body=el('tbody');
c.seeds.forEach((seed,i)=>{const tr=el('tr'),td=el('td');td.appendChild(el('div','',String(seed)));td.appendChild(el('div','case-id',(c.case_ids||[])[i]||''));tr.appendChild(td);const val=(c.finalval_by_seed||[])[i],base=(c.baselineval_by_seed||[])[i],rv=(c.ratio_by_seed||[])[i],rr=(c.ratio_unavailable_by_seed||[])[i]?'N/A':(c.ratio_failed_by_seed||[])[i]?'∞':finite(rv)?fmt(rv,7)+'×':'∞';
[finite(val)?fmt(val,7):'非有限／未提供',finite(base)?fmt(base,7):'非有限／缺失',rr,(c.first_hit_0_1_by_seed||[])[i],(c.first_hit_0_01_by_seed||[])[i]].forEach((x,j)=>tr.appendChild(el('td','',(j>=3&&x===-1)?'未达到':x===undefined?'—':String(x))));body.appendChild(tr);});table.appendChild(body);scroll.appendChild(table);container.appendChild(scroll);panel.open=true;}
function render(){const s=filterState(),pool=poolFor(s),{rows,columns}=rowsColumns(s),table=$('matrix-table'),thead=table.querySelector('thead'),tbody=table.querySelector('tbody');thead.replaceChildren();tbody.replaceChildren();selectedButton=null;
const hr=el('tr');hr.appendChild(el('th','matrix-rowhead',s.round==='r003'?'深度趋势':'初始化结构'));columns.forEach(c=>hr.appendChild(el('th','',c.label)));thead.appendChild(hr);
let found=0;rows.forEach(r=>{const tr=el('tr'),rh=el('th','matrix-rowhead',r.label);rh.scope='row';tr.appendChild(rh);columns.forEach(col=>{const candidates=pool.filter(c=>matches(c,r.match)&&matches(c,col.match)),c=candidates[0],td=el('td'),display=cellText(c,s.metric),button=el('button','matrix-cell '+(display.state||''));button.type='button';button.style.backgroundColor=display.color;
if(candidates.length>1)throw new Error('Duplicate matrix context: '+r.label+' '+col.label);
button.appendChild(el('span','cell-value',display.text));button.appendChild(el('span','cell-secondary',c?'达标 '+c.hit01_n+'/'+c.seed_n:''));let flags=[];if(c){found++;if(c.failed_n)flags.push('非有限 '+c.failed_n);if(c.initial_allzero_n)flags.push('零梯度 '+c.initial_allzero_n);if(c.model_unreachable_n)flags.push('不可达 '+c.model_unreachable_n);if(c.partial_cell)flags.push('未齐种子');button.dataset.cellId=c.id;button.title=r.label+' / '+col.label.replace('\n',' ')+'\n原始比值 '+ratio(c)+'\n绝对验证 MSE '+fmt(c.absolutevalmedian,7)+'\n基线验证 MSE '+fmt(c.baseline_absolutevalmedian,7)+'\n达标 '+c.hit01_n+'/'+c.seed_n+'；非有限 '+c.failed_n+'；零梯度 '+c.initial_allzero_n;button.setAttribute('aria-label',button.title.replace(/\n/g,'，'));button.addEventListener('click',()=>{if(selectedButton){selectedButton.classList.remove('cell-selected');selectedButton.setAttribute('aria-pressed','false');}selectedButton=button;button.classList.add('cell-selected');button.setAttribute('aria-pressed','true');detail(c);});}else{button.disabled=true;button.title='此组合尚无保存结果';}button.appendChild(el('span','cell-flags',flags.join(' · ')));td.appendChild(button);tr.appendChild(td);});tbody.appendChild(tr);});
const common=s.round==='r001'?'串联 · 深度4 · 宽度16 · 输入条件数1 · 每格4 seeds':s.round==='r002'?labels[s.task]+' · ReLU · '+labels[s.optimizer]+' · 输入条件数'+s.condition+' · 每格3 seeds':labels[s.task]+' · '+labels[s.activation]+' · '+labels[s.optimizer]+' · 串联 D4 W16 · 每格3 seeds';
$('matrix-context').textContent=common+' · 512 步';$('matrix-caption').textContent=found+'/'+(rows.length*columns.length)+' 格有保存结果 · 点击数值查看全部 seed 与路径编号';
const legend=$('matrix-legend');legend.hidden=!['comparison','ratio'].includes(s.metric);$('matrix-cell-detail').open=false;}
controls.forEach(k=>$('matrix-'+k).addEventListener('change',render));render();
})();
