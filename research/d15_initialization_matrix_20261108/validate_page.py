"""Check every matrix cell's reachability, preserved explorer, local links, SVGs."""
from pathlib import Path
import hashlib
import itertools
import json
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import design
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]
PAGE=REPO/'docs'/'模糊方向'/'D15_asymmetric_initialization.html'


def main():
    soup=BeautifulSoup(PAGE.read_text(encoding='utf-8'),'html.parser')
    embedded=json.loads(soup.select_one('#d15-matrix-data').string)
    cells=json.loads((SOURCE/'cells.json').read_text(encoding='utf-8'))
    expected={c['id']:c for c in cells}
    assert {c['id'] for c in embedded}==set(expected)
    for e in embedded:
        assert all(expected[e['id']][k]==v for k,v in e.items())
    assert soup.h1.string=='怎样解释零非对称初始化'
    assert str(soup).index('id="profile-chart"') < str(soup).index('id="initialization-strategies"') < str(soup).index('id="d15-results"')
    cards=soup.select('.strategy-card')
    assert len(cards)==len(design.PROFILES)==20
    for card in cards:
        profile=card['data-profile'];scales=np.array(json.loads(card['data-scales']))
        c=dict(depth=6,width=16,seed=9301,profile=profile)
        ws=design.initial_weights(c)
        baseline=design.initial_weights(dict(c,profile='uniform'))
        if profile=='zas_embed':
            for w in ws[:-1]:assert np.array_equal(w,np.eye(*w.shape))
            assert not np.any(ws[-1])
        elif profile=='identity_random':
            assert np.allclose([np.linalg.norm(w) for w in ws],[np.linalg.norm(w) for w in baseline])
        else:
            for w,b,s in zip(ws,baseline,scales):assert np.allclose(w,b*s,rtol=1e-12,atol=1e-14)
        assert card.find('svg') and card.find('figcaption')
    link_n=0
    for a in soup.find_all(['a','img']):
        h=a.get('href') or a.get('src')
        if h and not h.startswith(('http','#','data:')):
            assert (PAGE.parent/h.split('#')[0]).exists(),h
            link_n+=1
    svgs=list(PAGE.parent.glob('D15_matrix_*.svg'))
    for svg in svgs:
        tree=ET.parse(svg)
        assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
    seen=set();rendered=0;errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':1400,'height':1080})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(PAGE.as_uri(),wait_until='networkidle')
        assert page.locator('#matrix-table tbody tr').count()==20
        assert page.locator('#matrix-table tbody button').count()==120
        def check():
            nonlocal rendered
            buttons=page.locator('#matrix-table tbody button').evaluate_all('(els)=>els.map(e=>({id:e.dataset.cellId,text:e.innerText,disabled:e.disabled}))')
            for b in buttons:
                assert not b['disabled'],b
                c=expected[b['id']]
                assert f"达标 {c['hit01_n']}/{c['seed_n']}" in b['text'],b
                if c['failed_n']:assert f"非有限 {c['failed_n']}" in b['text']
                if c['initial_allzero_n']:assert f"零梯度 {c['initial_allzero_n']}" in b['text']
                seen.add(b['id']);rendered+=1
        for act,rate in itertools.product(('linear','relu','tanh'),('0','1')):
            page.select_option('#matrix-activation',act);page.select_option('#matrix-rate',rate);check()
        page.select_option('#matrix-round','r002')
        for task,opt,rate,cond in itertools.product(('linear','relu_teacher','interaction'),('sgd','adam'),('0','1'),('1','10')):
            for k,v in [('task',task),('optimizer',opt),('rate',rate),('condition',cond)]:page.select_option('#matrix-'+k,v)
            assert page.locator('#matrix-table tbody button').count()==108;check()
        page.select_option('#matrix-round','r003')
        for task,act,opt,rate in itertools.product(('linear','relu_teacher','interaction'),('linear','relu','tanh'),('sgd','adam'),('0','1')):
            for k,v in [('task',task),('activation',act),('optimizer',opt),('rate',rate)]:page.select_option('#matrix-'+k,v)
            assert page.locator('#matrix-table tbody button').count()==27;check()
        assert seen==set(expected),(len(seen),len(expected))
        for metric in ('ratio','absoluteval','hit','zero','failure','comparison'):
            page.select_option('#matrix-metric',metric)
            page.locator('#matrix-table tbody button').first.click()
            assert page.locator('#matrix-cell-detail').get_attribute('open') is not None
            assert page.locator('#matrix-detail-content tbody tr').count()==3
        page.locator('#uniform').click()
        assert page.locator('#profile-chart rect').evaluate_all('(els)=>els.every(e=>Number(e.dataset.scale)===1)')
        page.locator('#example').click();assert page.locator('#profile-chart rect').count()==12
        page.wait_for_function('window.MathJax && MathJax.typesetPromise')
        page.evaluate('MathJax.typesetPromise()')
        assert page.locator('mjx-merror').count()==0
        assert page.locator('mjx-container').count()>20
        page.select_option('#matrix-round','r001');page.select_option('#matrix-task','all');page.select_option('#matrix-optimizer','all');page.select_option('#matrix-activation','relu');page.select_option('#matrix-rate','0')
        page.evaluate('window.scrollTo(0,0)')
        screenshot=Path(tempfile.gettempdir())/'d15-matrix-final-desktop.png';page.screenshot(path=str(screenshot))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        mobile=Path(tempfile.gettempdir())/'d15-matrix-final-mobile.png';page.screenshot(path=str(mobile))
        assert not errors,errors
        browser.close()
    report={'status':'passed','all_cells_reachable':len(seen),'rendered_checks':rendered,'strategy_diagrams_checked':len(cards),'page_order':'introduction/explorer/formulas, strategy diagrams, experiment results','local_links_checked':link_n,'svg_xml_checked':len(svgs),'js_errors':errors,'mathjax':'rendered; no error nodes','explorer':'uniform reset and example restored','mobile_width':390,'page_sha256':hashlib.sha256(PAGE.read_bytes()).hexdigest(),'screenshots':[str(screenshot),str(mobile)]}
    (SOURCE/'page_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
