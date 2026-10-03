import os
import tempfile
import unittest
from pathlib import Path

import agent.tools as tools


class ToolsPathCompatibilityTests(unittest.TestCase):
    def test_write_file_uses_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools.PROJECT_ROOT = Path(tmpdir)
            relative_path = "nested\\demo.txt" if os.name == "nt" else "nested/demo.txt"

            result = tools.write_file(relative_path, "hello")

            self.assertTrue(result.startswith("已写入"))
            self.assertEqual((Path(tmpdir) / "nested" / "demo.txt").read_text(encoding="utf-8"), "hello")

    def test_read_file_handles_posix_like_project_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools.PROJECT_ROOT = Path(tmpdir)
            target = Path(tmpdir) / "src" / "main.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("print('ok')", encoding="utf-8")

            content = tools.read_file("/src/main.py")

            self.assertEqual(content, "print('ok')")


if __name__ == "__main__":
    unittest.main()
