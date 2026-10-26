import tempfile
from pathlib import Path

from aiq_kb.build_run_pages import render


def test_saved_model_output_cannot_close_viewer_script():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "viewer.html"
        render({"texts": ["</script><img src=x onerror=alert(1)>\u2028\u2029"]}, path)
        html = path.read_text(encoding="utf-8")
        assert "</script><img" not in html
        assert r"\u003c/script>" in html
        assert r"\u2028\u2029" in html
