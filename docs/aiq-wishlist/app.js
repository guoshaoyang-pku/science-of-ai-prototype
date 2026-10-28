(() => {
  'use strict';
  const aspects = window.AIQ_ASPECTS;
  const directions = new Map(window.AIQ_DIRECTIONS.map(direction => [direction.id, direction]));
  const $ = id => document.getElementById(id);
  const key = 'aiq-wishlist-v1';
  const allowed = new Set(aspects.map(a => a.id));
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let motionEnabled = true, resizeAnimation = null;
  const motionAllowed = () => motionEnabled && !reducedMotion.matches;
  function syncMotion() {
    const on = motionAllowed();
    document.body.classList.toggle('motion-paused', !on);
    $('motion-toggle').setAttribute('aria-pressed', String(on));
    $('motion-toggle').setAttribute('aria-label', on ? '关闭装饰动效' : '开启动效');
    $('motion-toggle').disabled = reducedMotion.matches;
    $('motion-toggle').title = reducedMotion.matches ? '遵循系统减少动态效果设置' : '';
    $('motion-label').textContent = on ? '动效开' : '动效关';
    if (!on) resizeAnimation?.cancel();
  }
  $('motion-toggle').addEventListener('click', () => { motionEnabled = !motionEnabled; syncMotion(); });
  reducedMotion.addEventListener('change', syncMotion);
  syncMotion();
  let state = { wishes: [], drafts: {} }, selected = null, step = 1, editing = null, storageOK = true, toastTimer;
  const clean = (value, max) => typeof value === 'string' ? value.slice(0, max) : '';
  try {
    const saved = JSON.parse(localStorage.getItem(key));
    if (saved && typeof saved === 'object') {
      state.wishes = Array.isArray(saved.wishes) ? saved.wishes.filter(w => w && allowed.has(w.aspect) && typeof w.id === 'string' && typeof w.wish === 'string').map(w => ({id: clean(w.id, 100), aspect: w.aspect, wish: clean(w.wish, 180), why: clean(w.why, 300), date: clean(w.date, 40)})) : [];
      for (const id of allowed) if (saved.drafts?.[id]) state.drafts[id] = {wish: clean(saved.drafts[id].wish, 180), why: clean(saved.drafts[id].why, 300)};
    }
  } catch { storageOK = false; }
  function notify(message) { $('status').textContent = message; $('status').classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('status').classList.remove('visible'), 4500); }
  function persist() {
    try { localStorage.setItem(key, JSON.stringify(state)); storageOK = true; return true; }
    catch { storageOK = false; notify('浏览器无法保存，请在关闭页面前导出愿望。'); document.querySelectorAll('.storage-hint').forEach(el => el.textContent = '当前仅暂存，请及时导出'); return false; }
  }
  const titleFor = id => aspects.find(a => a.id === id)?.title || '未分类';
  const researchOverview = {title:'13 个方向与当前结论', sourceHref:'../模糊方向/index.html', literatureHref:'../模糊方向/背景与文献.html'};
  let reading = null;
  function loadReading(kind) {
    const literature = kind === 'literature';
    const href = literature ? reading.literatureHref : reading.sourceHref;
    $('reader-report').setAttribute('aria-pressed', String(!literature));
    $('reader-literature').setAttribute('aria-pressed', String(literature));
    $('research-frame').title = reading.title + (literature ? ' · 文献对照' : ' · 研究报告');
    $('research-frame').src = href;
    $('reader-full-page').href = href;
  }
  function openReading(direction, kind = 'report') {
    reading = direction;
    $('reader-title').textContent = (direction.id ? direction.id + ' · ' : '') + direction.title;
    $('reader-report').textContent = direction.id ? '研究报告' : '方向总览';
    $('reader-literature').textContent = direction.id ? '文献对照' : '背景与文献';
    loadReading(kind);
    document.body.classList.add('reading-open');
    $('research-dialog').showModal();
  }
  function interceptReading(link, direction, kind = 'report') {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault(); openReading(direction, kind);
    });
  }
  $('reader-report').addEventListener('click', () => loadReading('report'));
  $('reader-literature').addEventListener('click', () => loadReading('literature'));
  $('close-reader').addEventListener('click', () => $('research-dialog').close());
  $('research-dialog').addEventListener('close', () => {
    document.body.classList.remove('reading-open');
    $('research-frame').removeAttribute('src');
    reading = null;
  });
  $('research-frame').addEventListener('load', () => {
    if (!$('research-dialog').open) return;
    // Keep the reader controls aligned when following links inside a report.
    // file:// hosts may isolate frames; the explicit report buttons still work.
    try {
      const frame = $('research-frame').contentWindow;
      const current = new URL(frame.location.href);
      if (!['http:', 'https:', 'file:'].includes(current.protocol)) return;
      current.hash = '';
      const match = [...directions.values(), researchOverview].find(d =>
        [d.sourceHref, d.literatureHref].some(href => new URL(href, document.baseURI).href === current.href));
      if (match) {
        reading = match;
        const literature = new URL(match.literatureHref, document.baseURI).href === current.href;
        $('reader-title').textContent = (match.id ? match.id + ' · ' : '') + match.title;
        $('reader-report').textContent = match.id ? '研究报告' : '方向总览';
        $('reader-literature').textContent = match.id ? '文献对照' : '背景与文献';
        $('reader-report').setAttribute('aria-pressed', String(!literature));
        $('reader-literature').setAttribute('aria-pressed', String(literature));
      } else {
        $('reader-title').textContent = frame.document.title || '补充材料';
        $('reader-report').setAttribute('aria-pressed', 'false');
        $('reader-literature').setAttribute('aria-pressed', 'false');
      }
      $('research-frame').title = frame.document.title || '研究材料';
      $('reader-full-page').href = frame.location.href;
    } catch { /* Cross-origin navigation retains the explicit report controls. */ }
  });
  interceptReading($('open-research-overview'), researchOverview);
  function showStep(next, focus = true) {
    const workspace = $('workspace');
    const oldHeight = workspace.getBoundingClientRect().height;
    resizeAnimation?.cancel();
    step = next;
    workspace.style.setProperty('--step-index', next === 'done' ? 2 : next - 1);
    for (const id of ['1', '2', '3', 'done']) $('step-' + id).hidden = String(next) !== id;
    document.querySelectorAll('[data-step]').forEach(button => {
      const n = Number(button.dataset.step);
      button.disabled = n > 1 && !selected || n === 1 && Boolean(editing);
      if (n === next || next === 'done' && n === 3) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    });
    $('use-suggestion').firstChild.textContent = editing ? '继续编辑 ' : '沿着这个想法写 ';
    $('own-idea').hidden = Boolean(editing);
    $('change-aspect').hidden = Boolean(editing);
    if (focus) {
      const nextHeight = workspace.getBoundingClientRect().height;
      if (motionAllowed() && Math.abs(nextHeight - oldHeight) > 3) resizeAnimation = workspace.animate([{height: oldHeight+'px'}, {height:nextHeight+'px'}], {duration:450, easing:'cubic-bezier(.22,1,.36,1)'});
      const heading = $('step-' + next).querySelector('h2'); heading.focus({preventScroll: true});
      const top = workspace.getBoundingClientRect().top;
      if (top < 0 || top > 95) workspace.scrollIntoView({behavior:motionAllowed() ? 'smooth' : 'auto',block:'start'});
    }
  }
  function displayAspect(aspect) {
    selected = aspect;
    $('aspect-label').textContent = 'ASPECT / ' + aspect.title;
    $('inspiration-title').textContent = aspect.question;
    $('aspect-explanation').textContent = aspect.explanation;
    $('suggested-wish').textContent = aspect.wish;
    $('current-evidence').textContent = aspect.current;
    $('next-step').textContent = aspect.next;
    $('source-link').href = aspect.sourceHref;
    $('examples-label').textContent = '看看已有研究例子 · ' + aspect.exampleIds.length;
    $('examples-note').textContent = aspect.exampleNote || '已有问题，非已验证结论；归类为建议，可跨方向。';
    $('examples-list').replaceChildren();
    for (const id of aspect.exampleIds) {
      const direction = directions.get(id);
      const item = element('li', '', 'research-example');
      const link = element('a', '', 'example-source');
      link.href = direction.sourceHref;
      link.target = '_blank'; link.rel = 'noopener';
      link.setAttribute('aria-label', direction.id + ' ' + direction.title + '：阅读研究报告');
      link.append(element('span', direction.id, 'example-id'), element('span', direction.title), element('span', '↗'));
      interceptReading(link, direction);
      item.append(link, element('p', direction.question, 'example-question'), element('p', direction.boundary, 'example-boundary'));
      const references = element('div', '', 'example-reading-links');
      const report = element('a', '读研究报告', 'text-button');
      report.href = direction.sourceHref; report.target = '_blank'; report.rel = 'noopener';
      interceptReading(report, direction);
      const literature = element('a', '文献对照', 'text-button');
      literature.href = direction.literatureHref; literature.target = '_blank'; literature.rel = 'noopener';
      interceptReading(literature, direction, 'literature');
      references.append(report, literature); item.append(references);
      $('examples-list').append(item);
    }
    $('research-examples').open = false;
    // Only bundled, authored SVG markup is inserted here; wish text always uses textContent.
    $('detail-glyph').innerHTML = window.AIQ_GLYPHS?.[aspect.id] || '';
    $('evidence').open = false;
  }
  function populateForm(value) {
    $('writing-aspect').textContent = '我的愿望 / ' + selected.title;
    $('wish-input').value = value?.wish || '';
    $('why-input').value = value?.why || '';
    $('wish-input').placeholder = selected.placeholder;
    $('char-count').textContent = $('wish-input').value.length;
    $('optional-fields').open = Boolean(value?.why);
    $('save-wish').firstChild.textContent = editing ? '保存修改 ' : '保存愿望 ';
    $('cancel-edit').hidden = !editing;
    $('wish-input').setCustomValidity('');
  }
  function startWriting(useSuggestion) {
    if (editing) { showStep(3); $('wish-input').focus({preventScroll:true}); return; }
    const draft = state.drafts[selected.id];
    editing = null;
    populateForm(draft?.wish || draft?.why ? draft : {wish: useSuggestion ? selected.wish : '', why: ''});
    if (draft?.wish && useSuggestion) notify('已恢复这个方向的草稿，可继续修改。');
    showStep(3);
    $('wish-input').focus({preventScroll:true});
    saveDraft();
  }
  function saveDraft() {
    $('char-count').textContent = $('wish-input').value.length;
    $('wish-input').setCustomValidity('');
    if (selected && !editing) { state.drafts[selected.id] = {wish:$('wish-input').value, why:$('why-input').value}; persist(); }
  }
  function count() {
    const counter = $('wish-count'), previous = counter.textContent;
    counter.textContent = state.wishes.length;
    if (previous !== counter.textContent && motionAllowed()) { counter.classList.remove('count-pop'); void counter.offsetWidth; counter.classList.add('count-pop'); }
    $('export-wishes').disabled = state.wishes.length === 0;
  }
  aspects.forEach((aspect, index) => {
    const button = document.createElement('button'); button.className = 'aspect-card'; button.type = 'button';
    button.style.setProperty('--card-index', index);
    button.style.setProperty('--card-accent', '#2855e8');
    const glyph = document.createElement('span'); glyph.className = 'card-glyph'; glyph.setAttribute('aria-hidden','true');
    glyph.innerHTML = window.AIQ_GLYPHS?.[aspect.id] || ''; button.append(glyph);
    for (const [tag, cls, text] of [['span','number',String(index+1).padStart(2,'0')],['strong','',aspect.title],['small','',aspect.question],['span','arrow','↗']]) {
      const element = document.createElement(tag); element.className = cls; element.textContent = text; if (cls==='arrow'||cls==='number') element.setAttribute('aria-hidden','true'); button.append(element);
    }
    button.addEventListener('click', () => { editing = null; displayAspect(aspect); showStep(2); });
    button.addEventListener('pointermove', event => {
      if (!motionAllowed() || event.pointerType === 'touch') return;
      const rect = button.getBoundingClientRect();
      button.style.setProperty('--pointer-x', (event.clientX-rect.left)+'px');
      button.style.setProperty('--pointer-y', (event.clientY-rect.top)+'px');
    });
    $('aspect-grid').append(button);
  });
  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => {
    const next = Number(button.dataset.step);
    if (next === 3 && step !== 3) { if (editing) showStep(3); else startWriting(false); }
    else showStep(next);
  }));
  $('change-aspect').addEventListener('click', () => showStep(1));
  $('use-suggestion').addEventListener('click', () => startWriting(true));
  $('own-idea').addEventListener('click', () => startWriting(false));
  $('back-to-idea').addEventListener('click', () => showStep(2));
  $('wish-input').addEventListener('input', saveDraft);
  $('why-input').addEventListener('input', saveDraft);
  $('wish-form').addEventListener('submit', event => {
    event.preventDefault();
    const wish = $('wish-input').value.trim(), why = $('why-input').value.trim();
    if (!wish) { $('wish-input').setCustomValidity('写下一个具体愿望吧。'); $('wish-input').reportValidity(); return; }
    const prior = editing && state.wishes.find(w => w.id === editing);
    const item = {id: prior?.id || (globalThis.crypto?.randomUUID?.() || Date.now()+'-'+Math.random().toString(36).slice(2)), aspect:selected.id, wish, why, date:prior?.date || new Date().toISOString()};
    if (prior) state.wishes[state.wishes.indexOf(prior)] = item; else { state.wishes.unshift(item); delete state.drafts[selected.id]; }
    editing = null;
    const saved = persist(); count();
    $('done-title').textContent = saved ? '愿望记下了。' : '愿望已暂存。';
    $('saved-preview').textContent = wish;
    showStep('done');
  });
  $('cancel-edit').addEventListener('click', () => { editing = null; showStep(1); });
  $('another-wish').addEventListener('click', () => { selected = null; showStep(1); });
  function element(tag, text, cls) { const node = document.createElement(tag); if (text) node.textContent = text; if (cls) node.className = cls; return node; }
  function renderSaved() {
    $('saved-list').replaceChildren(); count();
    if (!state.wishes.length) { $('saved-list').append(element('p','这里还没有愿望。选一个方向，写下第一句。','empty-state')); return; }
    for (const wish of state.wishes) {
      const card = element('article', '', 'saved-item');
      const meta = element('div','','saved-meta'); meta.append(element('span',titleFor(wish.aspect)+' · 待讨论'));
      const date = new Date(wish.date); if (!Number.isNaN(date.valueOf())) meta.append(element('span', date.toLocaleDateString('zh-CN')));
      card.append(meta, element('h3',wish.wish)); if (wish.why) card.append(element('p',wish.why));
      const actions = element('div','','saved-item-actions'), edit = element('button','编辑','text-button'), remove = element('button','删除','text-button delete-button');
      edit.addEventListener('click', () => { $('wishes-dialog').close(); editing = wish.id; displayAspect(aspects.find(a=>a.id===wish.aspect)); populateForm(wish); showStep(3); $('wish-input').focus({preventScroll:true}); });
      remove.addEventListener('click', () => {
        if (remove.dataset.confirm !== 'yes') { remove.dataset.confirm='yes'; remove.textContent='确认删除'; const cancel=element('button','取消','text-button'); cancel.addEventListener('click',()=>{remove.dataset.confirm='';remove.textContent='删除';cancel.remove();}); actions.append(cancel); return; }
        state.wishes = state.wishes.filter(w=>w.id!==wish.id);
        if (editing === wish.id) { editing = null; showStep(1, false); }
        persist(); renderSaved(); notify('已删除这条愿望。');
      });
      actions.append(edit,remove); card.append(actions); $('saved-list').append(card);
    }
  }
  function openSaved() { renderSaved(); $('export-fallback').hidden = true; $('wishes-dialog').showModal(); }
  $('open-wishes').addEventListener('click', openSaved); $('view-saved').addEventListener('click', openSaved);
  $('close-dialog').addEventListener('click', () => $('wishes-dialog').close());
  $('wishes-dialog').addEventListener('click', event => { if(event.target===$('wishes-dialog')) { const rect=event.target.getBoundingClientRect(); if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom) event.target.close(); }});
  $('export-wishes').addEventListener('click', () => {
    const date=new Date().toLocaleDateString('sv-SE');
    const quote=text=>text.split('\n').map(line=>'> '+line).join('\n');
    const text=['# AIQ 许愿池','',`导出日期：${date}`,'','以下为用户保存的待讨论愿望，尚未立项。','',...state.wishes.flatMap((w,i)=>[`## ${i+1} ${titleFor(w.aspect)}`,'',quote(w.wish),'',...(w.why?['为什么或具体例子：','',quote(w.why),'']:[]),'状态：待讨论',''])].join('\n');
    $('export-text').value = text; $('export-fallback').hidden = false;
    const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'})); const a=document.createElement('a'); a.href=url; a.download=`AIQ-wishlist-${date}.md`; document.body.append(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),30000);
  });
  $('copy-export').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('export-text').value); $('copy-export').textContent='已复制 ✓'; }
    catch { $('export-text').focus(); $('export-text').select(); $('copy-export').textContent='已选中，请复制'; }
  });
  count(); showStep(1,false);
  if (!storageOK) notify('浏览器存储不可用或记录无法读取，保存后请导出备份。');
})();
