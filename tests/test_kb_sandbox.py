import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiq_kb import kb_sandbox


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.visible = self.root / "visible"
        self.visible.mkdir()
        self.allowed = self.visible / "history.json"
        self.allowed.write_text('{"measurements": [1, 2, 3]}')
        self.private = self.root / "private_input.txt"
        self.private.write_text("outside fixture must not be read")

    def tearDown(self):
        self.temp.cleanup()

    def run_code(self, code, roots=None):
        return kb_sandbox.run_python(code, self.work, [self.visible] if roots is None else roots)

    def test_read_visible_and_write_research_code(self):
        output = self.run_code(f'values = json.load(open({str(self.allowed)!r}))["measurements"]\n'
                               'open("experiment.py", "w").write("print(6)\\n")\nprint(sum(values))')
        self.assertEqual(output.strip(), "6")
        self.assertEqual((self.work / "experiment.py").read_text(), "print(6)\n")

    def test_external_file_reads_writes_and_symlink_escape_denied(self):
        for statement in (f'open({str(self.private)!r}).read()',
                          f'open({str(self.private)!r}, "w").write("bad")',
                          f'os.listdir({str(self.root)!r})',
                          f'os.symlink({str(self.private)!r}, "escape")\nopen("escape").read()'):
            with self.subTest(statement=statement):
                output = self.run_code(statement)
                self.assertIn("PermissionError", output)
        self.assertEqual(self.private.read_text(), "outside fixture must not be read")

    def test_read_only_file_root_does_not_allow_siblings(self):
        sibling = self.visible / "sibling.txt"
        sibling.write_text("separate input")
        self.assertIn("measurements", self.run_code(f'print(open({str(self.allowed)!r}).read())', [self.allowed]))
        self.assertIn("PermissionError", self.run_code(f'print(open({str(sibling)!r}).read())', [self.allowed]))
        self.assertIn("PermissionError", self.run_code(f'open({str(self.allowed)!r},"w").write("bad")'))

    def test_network_subprocess_and_exec_denied(self):
        for code in ('import socket\nsocket.socket()',
                     'import subprocess\nsubprocess.run(["/usr/bin/true"])',
                     'os.system("/usr/bin/true")',
                     'os.execl(sys.executable, sys.executable, "-c", "print(1)")'):
            with self.subTest(code=code):
                self.assertIn("PermissionError", self.run_code(code))

    def test_git_metadata_and_scrubbed_credentials(self):
        git = self.work / ".git"
        git.mkdir()
        output = self.run_code('open(".git/config","w").write("malicious config")')
        self.assertIn("PermissionError", output)
        with patch.dict(kb_sandbox.os.environ, {"CCTQ_API_KEY": "test-only-secret"}):
            self.assertEqual(self.run_code('print("CCTQ_API_KEY" in os.environ)').strip(), "False")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "active Python has no NumPy")
    def test_numpy_analysis_works(self):
        self.assertEqual(self.run_code('print(np.arange(10).sum())').strip(), "45")

    def test_audit_only_fallback_is_explicit_and_still_restricts_routine_access(self):
        with patch.object(kb_sandbox, "SANDBOX_EXEC", None):
            self.assertEqual(self.run_code('print(2 + 3)').strip(), "5")
            self.assertIn("PermissionError", self.run_code(f'open({str(self.private)!r}).read()'))
        self.assertIn("not an OS boundary", kb_sandbox.__doc__)

    @unittest.skipUnless(sys.platform == "darwin" and kb_sandbox.SANDBOX_EXEC, "requires macOS OS sandbox")
    def test_os_layer_blocks_external_io_network_and_spawn_without_audit(self):
        # Remove the Python hook to verify the independent OS restrictions.
        with patch.object(kb_sandbox, "PREAMBLE", ""):
            for code in (f'print(open({str(self.private)!r}).read())',
                         f'open({str(self.private)!r}, "w").write("bad")',
                         'import socket\nsocket.socket().connect(("127.0.0.1", 9))',
                         'import subprocess\nsubprocess.run(["/usr/bin/true"], check=True)'):
                with self.subTest(code=code):
                    output = self.run_code(code)
                    self.assertTrue("PermissionError" in output or "Operation not permitted" in output
                                    or "sandbox failed" in output, output)
        self.assertEqual(self.private.read_text(), "outside fixture must not be read")


if __name__ == "__main__":
    unittest.main()
