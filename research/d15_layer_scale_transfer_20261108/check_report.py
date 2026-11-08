"""Local browser checks of saved report; no training or publishing."""
import functools
import http.server
import json
import threading
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'd1_scaling_exploration_20261107'))
from run_r001 import REPO
from run_r002 import RUN
SOURCE = __import__('pathlib').Path(__file__).resolve().parent


def main():
    docs = REPO / 'docs' / '模糊方向'
    raster = sum(len(ET.parse(p).getroot().findall('.//{http://www.w3.org/2000/svg}image'))
                 for p in list(docs.glob('D1_scaling_r001_*.svg')) + list(docs.glob('D15_layer_scale_*.svg')))
    assert raster == 0
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO))
    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        result = {'embedded_raster_nodes': raster, 'viewports': []}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for width, height in [(1440, 1000), (390, 844)]:
                page = browser.new_page(viewport={'width': width, 'height': height})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/docs/%E6%A8%A1%E7%B3%8A%E6%96%B9%E5%90%91/D15_layer_scale_transfer.html', wait_until='networkidle')
                page.wait_for_function('window.MathJax && MathJax.typesetPromise', timeout=30000)
                page.evaluate('async () => { await MathJax.startup.promise; }')
                page.evaluate('document.querySelectorAll("details").forEach(e=>e.open=true)')
                images = page.locator('figure img')
                assert images.count() == 3
                for item in images.all():
                    item.scroll_into_view_if_needed()
                    item.evaluate('(el) => el.decode()')
                doc_width = page.evaluate('document.documentElement.scrollWidth')
                math_errors = page.locator('mjx-merror,[data-mjx-error]').count()
                assert doc_width == width, (width, doc_width)
                assert not errors and math_errors == 0, (errors, math_errors)
                assert page.locator('mjx-container').count() > 0
                page.evaluate('document.querySelectorAll("details").forEach(e=>e.open=false);window.scrollTo(0,0)')
                screenshot = RUN / 'figures' / f'report_{width}.png'
                page.screenshot(path=str(screenshot))
                result['viewports'].append({'width': width, 'height': height, 'document_width': doc_width,
                                            'figures_loaded': images.count(), 'mathjax_error_nodes': math_errors,
                                            'mathjax_nodes': page.locator('mjx-container').count(), 'page_errors': errors})
                page.close()
            browser.close()
        server.shutdown()
    (SOURCE / 'r002_page_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
