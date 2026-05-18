"""CLI 测试。"""

from cvlab.cli.main import main


class TestCLI:
    def test_help(self):
        assert main([]) == 0

    def test_init(self, tmp_path):
        import os
        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            assert main(["init"]) == 0
            assert (tmp_path / ".cvlab").exists()
        finally:
            os.chdir(orig_dir)

    def test_list_empty(self):
        import tempfile
        import os
        orig_dir = os.getcwd()
        try:
            tmp = tempfile.mkdtemp()
            os.chdir(tmp)
            assert main(["list"]) == 0
        finally:
            os.chdir(orig_dir)

    def test_unknown_experiment(self):
        assert main(["show", "nonexistent"]) == 1
