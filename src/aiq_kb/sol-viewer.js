// Keep the six luna_main views; show the sol run's actual read protocol.
const num = (v, d = 3) => typeof v === 'number' ? v.toFixed(d) : '—';
const isTool = D.config.render === 'tool';
const table = (heads, rows) => '<table><tr>' + heads.map(x => '<th>' + x + '</th>').join('') + '</tr>' + rows + '</table>';
const card = (title, body, cls = '') => '<div class="card ' + cls + '"><h3>' + title + '</h3>' + body + '</div>';
const td = x => '<td>' + x + '</td>';
const anchor = (text, action) => '<a href="#" onclick="' + action + ';return false">' + text + '</a>';
const commitOperations = [];
const jobIdMaps = {};
for (const c of D.commits) {
  const wrapped = c.op === 'async_apply';
  const raw = wrapped ? c.payload?.operation || c : c;
  const maps = wrapped ? Object.assign(jobIdMaps[c.job_id] ||= {}, c.payload?.id_remap || {}) : {};
  const operations = raw.op === 'proposal' ? raw.payload?.operations || [] : [raw];
  for (const op of operations) {
    const ids = [...(op.claim_ids || []), ...(op.new_id ? [op.new_id] : [])];
    commitOperations.push({...op, commit: c.commit, source_commit: op.commit, epoch: c.epoch,
      origin_epoch: c.origin_epoch ?? op.origin_epoch ?? op.epoch ?? c.epoch,
      applied_epoch: c.applied_epoch ?? c.epoch,
      claim_ids: (op.claim_ids || []).map(id => maps[id] || id), new_id: maps[op.new_id] || op.new_id,
      wrapped, proposal: raw.op === 'proposal', claim_status: wrapped ? c.payload?.claim_status : 'applied',
      conflicts: wrapped ? c.payload?.conflicts || [] : [],
      id_remap: Object.fromEntries(ids.filter(id => maps[id] && maps[id] !== id).map(id => [id, maps[id]])),
      reason: op.reason || raw.reason || c.reason || ''});
  }
}
const operationClass = op => op === 'add' ? 'add' : op === 'delete' ? 'del' : ['merge', 'revise', 'rollback'].includes(op) ? 'chg' : '';
state.snap = D.defaultSnap && snapBy[D.defaultSnap] ? D.defaultSnap : D.snaps.filter(s => !s.candidate).at(-1)?.label || D.snaps[0].label;
state.epoch = D.epochs[0]?.epoch || 1;
state.tab = D.defaultTab || 'kb';
if (location.hash.startsWith('#kb=')) {
  const label = decodeURIComponent(location.hash.slice(4));
  if (snapBy[label]) { state.tab = 'kb'; state.snap = label; }
}

readViewHash();

function nav() {
  const tabs = [['overview','概览'],['epoch','每个 epoch：解题 / 发现 / 总结'],['kb','KB 快照'],['claim','单条 claim 演化'],['commits','commit 日志'],['prompts','提示词与配置']];
  $('nav').innerHTML = tabs.map(([k,v]) => '<button class="' + (state.tab===k?'on':'') + '" onclick="go(\''+k+'\')">'+v+'</button>').join('');
  $('title').textContent = 'KB 科学循环 · '+D.run;
  if(D.home) $('home').href=D.home; else $('home').style.display='none';
  $('meta').textContent = (D.config.model||'模型未取得')+' · effort='+(D.config.solve_effort||'—')+' · '+D.epochs.length+' 轮已有记录';
}

function overview() {
  $('side').innerHTML='';
  const rows=D.epochs.map(e=>{
    const m=e.metrics||{}, g=m.gate||m.review;
    const A=snapBy[lab(e.epoch-1)],B=snapBy[lab(e.epoch)];
    const counts={};
    for(const c of snapshotOperations(lab(e.epoch-1),lab(e.epoch),e.epoch,commitOperations)) {
      if(c.claim_status!=='deferred')counts[c.op]=(counts[c.op]||0)+1;
    }
    const ops=Object.entries(counts).map(([k,v])=>k+':'+v).join(' ');
    const review=g?(g.accept===true?'接受 · 候选 − 现任':g.accept===false?'拒绝 · 候选 − 现任':'宏观审查 · KB − 不带 KB')+' '+num((g.delta??g.val_vs_nokb)*100,2)+' ±'+num((g.se??g.val_se)*100,2)+' pt':'—';
    return '<tr>'+td(anchor('e'+e.epoch,'state.epoch='+e.epoch+';go(\'epoch\')'))+td(m.n??(e.records.length||'—'))+td(num(m.mean_score))+td((A?.claims.length??m.kb_before??'—')+' → '+(B?.claims.length??'未完成'))+td(esc(ops)+(g?.accept===false?' · 含被拒候选':''))+td(review)+'</tr>';
  }).join('');
  const ev=Object.entries(D.evals).sort().map(([label,e])=>'<tr>'+td(esc(label))+td(e.n??e.n_q??'—')+td(e.repeats??'—')+td(num(e.mean))+td(e.delta!=null?num(e.delta*100,2)+' ±'+num(e.delta_se*100,2)+' pt':'—')+td(esc(e.by_source?JSON.stringify(e.by_source):e.note||''))+'</tr>').join('');
  const flow=D.flow||[isTool?'① 解题：题目 + 可选 KB 工具；每题使用冻结快照，答案尚未揭示。':'① 解题：题目 + 上一轮只读 KB，答案尚未揭示。','② 发现：揭示答案，分析失败，可查历史、回测与受控实验。','③ 总结：修订、增加、合并、撤回 claim；每次改动记录理由。',D.config.macro_every?'④ KB 成长：每 '+D.config.macro_every+' 轮集中整理和宏观评测。':'④ KB 快照：在验证集比较候选与现任；整轮接受或回滚。'];
  const pairs=(D.comparisons||[]).map(e=>'<tr>'+td(esc(e.label))+td(e.n??'—')+td(e.repeats??'—')+td(num(e.mean))+td(e.delta==null?'—':num(e.delta*100,2)+' ±'+num(e.se*100,2)+' pt')+td(esc(e.note||''))+'</tr>').join('');
  const control=D.training_control;
  const pairTable=pairs?table(['配对比较','题数','重复','均分','得分差','备注'],pairs):'';
  const training=control?'<p>训练流：'+control.n+' 题，KB − 不带KB '+num(control.delta*100,2)+' ±'+num(control.se*100,2)+' pt。'+esc(control.note||'')+'</p>':'';
  $('view').innerHTML=card(esc(D.run),'<p>'+esc(D.availability||'')+'</p>'+(D.summary?'<p>'+esc(D.summary)+'</p>':''))+card('流程','<div class="flow">'+flow.map(x=>'<div>'+esc(x)+'</div>').join('')+'</div>')+card('每个 epoch',table(['轮次','题数','训练题均分','KB 条数','结构操作','评审结果'],rows||'<tr><td colspan=6>未取得逐轮记录。</td></tr>'))+card('评测',pairTable+training+table(['快照 / 对照','题数','重复','均分','相对不带 KB','分题源 / 备注'],ev||'<tr><td colspan=6>尚无已完成评测。</td></tr>')+'<p class="note">得分包含排序题部分分；± 为题级配对 1 SE。</p>'+(D.evalNote?'<p class="note">'+esc(D.evalNote)+'</p>':''));
}

function detail(S,id) {
  const raw=S.claimDetails?.[id]||{};
  const meta=Object.fromEntries(Object.entries(raw).filter(([k,v])=>!['text_excerpt','text_complete','text','id','support_count','failure_count','credibility'].includes(k)&&v!=null&&v!==''&&(!Array.isArray(v)||v.length)&&(!(typeof v==='object'&&!Array.isArray(v))||Object.keys(v).length)));
  const evidence=meta&&Object.keys(meta).length?'<details><summary>条件、机制与来源</summary><pre>'+linkC(JSON.stringify(meta,null,2))+'</pre></details>':'';
  return evidence+(raw.report_ids||[]).map(rid=>reportView(S.reports?.[rid])).join('');
}

function reportView(version) {
  const R=D.reports?.[version];
  if(!R)return '';
  const code=Object.entries(R.files||{}).map(([name,f])=>'<a href="'+esc(encodeURI(f.url))+'">'+esc(name)+(f.compression==='gzip'?' (gzip)':'')+'</a>').join(' · ');
  return '<details><summary>'+esc(R.id+' · '+R.title)+'</summary><p class="note">发起轮 e'+R.origin_epoch+' · '+esc(R.job_id||'')+' · repo commit '+esc(R.repo_commit)+(R.application_pending?' · 已保存报告，KB 操作待应用':'')+'</p>'+(code?'<p>'+code+'</p>':'')+'<pre class="tall">'+linkC(R.text||'未取得报告原文。')+'</pre></details>';
}

function jobView(jobs) {
  return jobs?.length?table(['任务','角色','发起轮','本轮成员','状态'],jobs.map(j=>'<tr>'+td(esc(j.id))+td(esc(j.kind))+td(j.origin_epoch)+td(esc(j.label||'e'+j.member_epoch))+td(esc(j.status))+'</tr>').join('')):'';
}

function kbView() {
  $('side').innerHTML=D.snaps.map(s=>'<div class="item '+(s.label===state.snap?'on':'')+'" onclick="state.snap=\''+s.label+'\';render()">'+esc(s.label)+' <small>'+(s.totalClaims??s.claims.length)+' 条'+(s.complete===false?' · 摘录':'')+'</small></div>').join('');
  const S=snapBy[state.snap];
  if(!S){$('view').innerHTML=card('未取得此快照','');return;}
  let h='<div class="sub"><button onclick="state.sub=\'table\';render()">全部条目</button><button onclick="state.sub=\'render\';render()">'+(isTool?'可检索的 KB 原文':'解题者看到的 KB 原文')+'</button></div>';
  h+=card(esc(S.label)+' · '+(S.totalClaims??S.claims.length)+' 条'+(S.complete===false?'（摘录）':''),'<p>'+esc(S.note||(S.complete===false?'这里只保存了输出摘录，尚未取得完整快照。':''))+'</p>'+(isTool?'<p class="note">本实验通过工具读取条目；完整 KB 没有常驻解题 prompt。</p>':''),'kbc');
  if(state.sub==='render') h+=card('KB 原文','<pre class="tall">'+linkC(S.render||'未取得原文。')+'</pre>','solver');
  else if(S.claims.length) h+=card('条目内容','<input type="search" placeholder="搜索：LayerNorm / width / Adam" oninput="filt(this.value)">'+table(['编号','引用反馈','胜 / 负','来源','条目正文'],S.claims.map(c=>'<tr data-t="'+esc(T(c[4]).toLowerCase())+'">'+td(linkC(c[0]))+td(num(c[3],2))+td((c[1]??'—')+' / '+(c[2]??'—'))+td(c[5]==null?'—':c[5]?'第 '+c[5]+' 轮':'种子')+td(esc(T(c[4]))+(S.excerptIds?.includes(c[0])?'<div class="note">保存的输出在此截断；未取得完整条目。</div>':'')+detail(S,c[0]))+'</tr>').join('')).replace('<table>','<table id="kbt">')+'<p class="note">引用反馈是解题使用后的表现记账，不是规律真实的概率。</p>','kbc');
  if(S.changeReasons)h+=card('该轮修改理由（保存原文）','<pre class="tall">'+linkC(S.changeReasons)+'</pre>','summ');
  if(state.snap===D.defaultSnap&&Object.keys(D.latest_saved_science||{}).length)h+='<details><summary>最新已保存的 science</summary><p class="note">持续研究的报告版本；KB 修改仍按应用日志登记。历史解题保留各自冻结版本。</p>'+Object.values(D.latest_saved_science).map(reportView).join('')+'</details>';
  $('view').innerHTML=h;
}

function epochView() {
  const E=D.epochs.find(e=>e.epoch===state.epoch)||D.epochs[0];
  if(!E){$('side').innerHTML='';$('view').innerHTML=card('逐轮记录','未取得逐轮记录。');return;}
  const subs=[['records','解题 + 发现（逐题）'],['summ','总结者（逐轮）'],['diff','KB 变化'],['sci','科学总结']];
  if(!subs.some(([k])=>k===state.sub))state.sub='records';
  $('side').innerHTML=D.epochs.map(e=>'<div class="item '+(e.epoch===E.epoch?'on':'')+'" onclick="state.epoch='+e.epoch+';state.q=null;render()">第 '+e.epoch+' 轮 <small>均分 '+num(e.metrics?.mean_score,2)+'</small></div>').join('')+(state.sub==='records'?E.records.map((r,i)=>'<div class="item '+(state.q===i?'on':'')+'" onclick="state.q='+i+';render()">'+esc(r.question_id)+' '+scoreTag(r.score)+'<br><small>'+esc(r.source)+' · '+esc(r.family)+'</small></div>').join(''):'');
  let h='<div class="sub">'+subs.map(([k,v])=>'<button class="'+(state.sub===k?'on':'')+'" onclick="state.sub=\''+k+'\';render()">'+v+'</button>').join('')+'</div>';
  if(state.sub==='records')h+=E.records.length?state.q==null?recordsTable(E):recordView(E,E.records[state.q]):card('逐题记录',esc(E.recordNote||'未取得本轮逐题解题与发现轨迹。'));
  if(state.sub==='summ')h+=summView(E)+(E.summarizerNote?'<p class="note">'+esc(E.summarizerNote)+'</p>':'')+(E.research||[]).map(r=>card(esc(r.file),'<pre>'+esc(r.text||JSON.stringify(r.data,null,2))+'</pre>')).join('')+(E.jobs||[]).filter(j=>j.kind==='research').map(j=>card(esc(j.id)+' · 研究工具日志',turns(D.job_traces?.[j.id]||[]))).join('');
  if(state.sub==='diff') {
    const candidate=D.snaps.find(s=>s.candidate&&s.epoch===E.epoch);
    if(candidate)h+=card('本轮候选被拒','以下展示上一轮 KB → 本轮候选的修改。候选未被采纳；本轮最终保留 '+esc(lab(E.epoch))+'，共 '+snapBy[lab(E.epoch)].claims.length+' 条。')+diffView(lab(E.epoch-1),candidate.label,E.epoch);
    else h+=diffView(lab(E.epoch-1),lab(E.epoch),E.epoch);
  }
  if(state.sub==='sci')h+=card('科学总结 · 第 '+E.epoch+' 轮','<pre class="tall">'+linkC(E.science||(E.scienceError?'本轮科学总结未保存成功。':'未取得科学总结原文。'))+'</pre>'+(E.scienceError?'<details><summary>运行错误原文</summary><pre>'+esc(E.scienceError)+'</pre></details>':'')+jobView(E.jobs)+(Object.keys(D.reports||{}).filter(v=>D.reports[v].origin_epoch===E.epoch).map(reportView).join('')),'summ');
  $('view').innerHTML=h;
}

function recordView(E,r) {
  const S=snapBy[r.solver_snapshot||lab(E.epoch-1)], ids=[...new Set([...(r.kb_retrieved||[]),...(r.cited||[])])];
  const brief=(D.config.progressive_disclosure||D.config.render==='brief')?'完整简洁 KB；science 按需展开':isTool?'可选 KB 工具检索':'只读 KB 注入';
  const reportVersions=Object.entries(r.report_versions||{}).map(([id,commit])=>reportView(id+'@'+commit)).join('');
  const rows=ids.map(id=>{const c=S?.claims.find(z=>z[0]===id);return '<tr>'+td(linkC(id))+td(c?esc(T(c[4]))+detail(S,id):'未取得解题时该条目的原文。')+'</tr>';}).join('');
  return card(esc(r.question_id)+' · '+esc(r.source)+' · '+esc(r.family),scoreTag(r.score)+' 预测 <b>'+esc(r.prediction)+'</b> · 答案 <b>'+esc(r.answer)+'</b><details><summary>题目全文</summary><pre>'+esc(r.question)+'</pre></details>')+
    card('① 解题 agent（effort='+(D.config.solve_effort||'low')+'）','<p>'+brief+' · 冻结快照 '+esc(r.solver_snapshot||S?.label||'未取得')+'</p>'+(r.kb_queries?.length?'<pre>'+esc(JSON.stringify(r.kb_queries,null,2))+'</pre>':'')+(r.solver_reasoning?'<details><summary>返回的推理摘要</summary><pre>'+esc(r.solver_reasoning)+'</pre></details>':'')+'<p><b>完整输出</b></p><pre>'+linkC(r.solution||'未取得完整输出。')+'</pre>'+(r.solve_trace?.length?turns(r.solve_trace):'')+table(['检索 / 引用 ID','解题时内容'],rows||'<tr><td colspan=2>没有检索或引用记录。</td></tr>')+(reportVersions?'<details><summary>解题时的 science 版本</summary>'+reportVersions+'</details>':''),'solver')+
    card('② 发现 agent（effort='+esc(r.discover_effort)+'）',turns(r.discover_trace||[])+'<p><b>评论</b></p><pre>'+linkC(r.comment||'—')+'</pre><p><b>猜想</b></p><pre>'+linkC(r.hypothesis||'—')+'</pre>','disc');
}

function diffView(a,b,epoch) {
  return kbDiffView(a,b,epoch,commitOperations);
}

function claimRows(id) {
  let last=null;
  return D.snaps.map(S=>{
    const c=S.claims.find(z=>z[0]===id);
    if(!c)return '<tr>'+td(esc(S.label))+'<td colspan=3>'+(S.complete===false?'摘录中未出现，不能判断是否存在。':'不存在')+'</td></tr>';
    const content=c[4]!==last?esc(T(c[4]))+detail(S,id):'<span class="note">文本同上</span>';last=c[4];
    return '<tr>'+td(esc(S.label))+td((c[1]??'—')+' / '+(c[2]??'—'))+td(num(c[3],2))+td(content+(S.excerptIds?.includes(id)?'<p class="note">此条原文被截断。</p>':''))+'</tr>';
  }).join('');
}
function claimView() {
  const ids=[...new Set(D.snaps.flatMap(s=>s.claims.map(c=>c[0])))].sort();
  $('side').innerHTML='<div style="padding:6px"><input type="search" placeholder="K1006" onchange="goClaim(this.value.trim().toUpperCase())"></div>'+ids.map(id=>'<div class="item '+(id===claimId?'on':'')+'" onclick="goClaim(\''+id+'\')">'+id+'</div>').join('');
  const cs=commitOperations.filter(c=>(c.claim_ids||[]).includes(claimId)||c.new_id===claimId);
  $('view').innerHTML=card(esc(claimId)+' 在各快照中的状态',table(['快照','胜 / 负','引用反馈','内容'],claimRows(claimId)),'kbc')+card('相关操作',commitTable(cs),'summ');
}
function openClaim(id,ev) {
  const cs=commitOperations.filter(c=>c.op!=='credit'&&((c.claim_ids||[]).includes(id)||c.new_id===id));
  $('popc').innerHTML='<h3>'+esc(id)+' '+anchor('查看演化 →','closePop();goClaim(\''+id+'\')')+'<button id="popx" onclick="closePop()">关闭 ×</button></h3>'+table(['快照','胜 / 负','引用反馈','内容'],claimRows(id))+'<h3>相关操作</h3>'+commitTable(cs);
  $('pop').style.display='block';$('popc').style.left=Math.max(12,(innerWidth-$('popc').offsetWidth)/2)+'px';$('popc').style.top='6vh';$('popc').scrollTop=0;
}

function summView(E) {
  const ops=commitOperations.filter(c=>c.origin_epoch===E.epoch&&!['credit','seed','report'].includes(c.op));
  return card('③ 总结者 · 第 '+E.epoch+' 轮','<p>保存的 KB 操作理由</p>'+commitTable(ops)+(E.summarizer.length?turns(E.summarizer):'<p class="note">未取得逐步工具轨迹；不能据此判断调用次数。</p>'),'summ');
}

function commitTable(cs) {
  if(!cs.length)return '<p class="note">无</p>';
  return table(['commit','所属轮 / 应用轮','操作','条目','来源题','内容 / 理由'],cs.map(c=>{
    const p=c.payload||{}, deferred=c.claim_status==='deferred';
    const remap=Object.entries(c.id_remap||{}).map(([a,b])=>esc(a)+' → '+linkC(b)).join(' · ');
    const body=(p.text?'<b>新内容：</b> '+linkC(p.text)+'<br>':'')+(p.old_text?'<span class="note">旧内容：</span> '+esc(p.old_text)+'<br>':'')+(c.op==='credit'?esc(JSON.stringify(p))+'<br>':'')+(c.reason?'<b>理由：</b> '+linkC(c.reason):'')+(remap?'<p class="note">任务内 ID → 应用 ID：'+remap+'</p>':'')+(c.op==='report'?'<p class="note">science '+esc(c.report_id)+' · research repo commit '+esc(c.repo_commit)+'（报告版本）</p>':'')+(c.conflicts?.length?'<pre>冲突：'+esc(JSON.stringify(c.conflicts))+'</pre>':'');
    const label=c.op+(c.wrapped?deferred?' · 异步提案':' · 异步应用':'')+(c.proposal?' · 原子提案':'')+(deferred?' · 待应用':'');
    const commit=esc(c.commit)+(c.source_commit!==c.commit?'<br><small>任务内 '+esc(c.source_commit)+'</small>':'');
    return '<tr class="'+(deferred?'':operationClass(c.op))+'">'+td(commit)+td('<span style="white-space:nowrap">e'+c.origin_epoch+' / e'+c.applied_epoch+'</span>')+td(esc(label))+td(linkC((c.claim_ids||[]).join(' '))+(c.new_id?' → '+linkC(c.new_id):''))+td(esc((c.source_question_ids||[]).join(' ')))+td(body)+'</tr>';
  }).join(''));
}

function commitView() {
  const ops=[...new Set(commitOperations.map(c=>c.op))], selected=state.op||'(结构操作)';
  const filtered=o=>o.startsWith('(')?commitOperations.filter(c=>!['credit','seed','report'].includes(c.op)):commitOperations.filter(c=>c.op===o);
  $('side').innerHTML=['(结构操作)',...ops].map(o=>`<div class="item ${selected===o?'on':''}" onclick="state.op='${o}';render()">${esc(o)} <small>${filtered(o).length}</small></div>`).join('');
  const cs=filtered(selected);
  $('view').innerHTML=card('commit 日志 · '+esc(selected)+'（'+cs.length+' 条）',commitTable(cs));
}

function promptView() {
  $('side').innerHTML='';
  $('view').innerHTML=(D.settingURL?'<p><a href="'+esc(D.settingURL)+'">当前 setting</a></p>':'')+(D.sourceArchive?'<p><a href="'+esc(D.sourceArchive.zipURL)+'">源码 '+esc(D.sourceArchive.commit.slice(0,7))+'</a> · <a href="'+esc(D.sourceArchive.manifestURL)+'">文件与哈希</a></p>':'')+Object.entries(D.prompts).map(([k,v])=>card(esc(k),'<pre class="tall">'+esc(v)+'</pre>')).join('')+card('运行配置','<pre>'+esc(JSON.stringify(D.config,null,1))+'</pre>');
}
