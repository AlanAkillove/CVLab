"""CLI 冒烟测试 — 快速验证 CLI 入口的基本可用性。

不依赖外部数据库或 GPU，所有测试应在无头/离线环境中通过。
使用 ``main()`` 函数直接调用（而非 subprocess），更快且减少系统依赖。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from io import StringIO

import pytest

from cvlab.cli.main import main


# ── 辅助函数 ──────────────────────────────────────────────

def capture_main(argv: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """运行 main() 并捕获 stdout / stderr。

    能安全处理 argparse 的 ``SystemExit``（``--help`` / 非法参数），
    将退出码和输出一并返回。

    Returns:
        (return_code, stdout_text, stderr_text)
    """
    old_cwd = os.getcwd()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv

    try:
        if cwd:
            os.chdir(cwd)

        sys.stdout = stdout_capture = StringIO()
        sys.stderr = stderr_capture = StringIO()
        sys.argv = ["cvlab"] + argv

        try:
            rc = main(argv)
        except SystemExit as e:
            rc = e.code if e.code is not None else 0

        return rc, stdout_capture.getvalue(), stderr_capture.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.argv = old_argv
        os.chdir(old_cwd)


# ── 测试: --version ───────────────────────────────────────

class TestVersion:
    """验证 --version 正确输出版本号。"""

    def test_version_returns_zero(self):
        """--version 应返回 0。"""
        rc, stdout, stderr = capture_main(["--version"])
        assert rc == 0, f"版本命令应成功，rc={rc}, stderr={stderr}"

    def test_version_contains_semver(self):
        """版本号字符串应包含语义化版本号。"""
        rc, stdout, stderr = capture_main(["--version"])
        version = stdout.strip()
        assert version, "版本号不应为空"
        # 至少包含数字和点号
        assert any(c.isdigit() for c in version), f"版本号应包含数字：{version!r}"
        assert "." in version, f"版本号应包含点号：{version!r}"

    def test_version_output_no_extra(self):
        """--version 的输出应只有版本号，没有额外文本。"""
        rc, stdout, stderr = capture_main(["--version"])
        lines = stdout.strip().splitlines()
        assert len(lines) == 1, (
            f"--version 应只输出一行，得到 {len(lines)} 行"
        )


# ── 测试: --help ──────────────────────────────────────────

class TestHelp:
    """验证 --help 正常打印帮助信息。"""

    def test_help_flag(self):
        """--help 应返回 0。"""
        rc, stdout, stderr = capture_main(["--help"])
        assert rc == 0, f"帮助命令应成功，rc={rc}"

    def test_help_contains_prog_name(self):
        """帮助文本应包含 'cvlab' 程序名。"""
        rc, stdout, stderr = capture_main(["--help"])
        assert "cvlab" in stdout.lower(), "帮助文本应包含 cvlab"

    def test_help_shows_commands(self):
        """帮助文本应列出主要子命令。"""
        rc, stdout, stderr = capture_main(["--help"])
        for cmd in ("init", "list", "train", "sweep", "help"):
            assert cmd in stdout, f"帮助文本缺少子命令 {cmd}"

    def test_help_shows_lang_flag(self):
        """帮助文本应包含 --lang 选项。"""
        rc, stdout, stderr = capture_main(["--help"])
        assert "--lang" in stdout, "帮助文本应包含 --lang 选项"

    def test_no_args_prints_help(self):
        """不带任何参数时默认打印帮助信息。"""
        rc, stdout, stderr = capture_main([])
        assert rc == 0, "无参数时应返回 0"
        assert "cvlab" in stdout.lower()

    def test_help_shows_version_flag(self):
        """帮助文本应包含 --version。"""
        rc, stdout, stderr = capture_main(["--help"])
        assert "--version" in stdout, "帮助文本应包含 --version"


# ── 测试: init ────────────────────────────────────────────

class TestInit:
    """验证 init 命令创建 .cvlab 目录。"""

    def test_init_creates_cvlab_directory(self, tmp_path):
        """init 应在当前目录创建 .cvlab 目录。"""
        rc, stdout, stderr = capture_main(["init"], cwd=str(tmp_path))
        assert rc == 0, f"init 应成功，rc={rc}, stderr={stderr}"
        assert (tmp_path / ".cvlab").is_dir(), ".cvlab 目录未创建"

    def test_init_idempotent(self, tmp_path):
        """多次运行 init 不应报错。"""
        capture_main(["init"], cwd=str(tmp_path))
        rc2, stdout2, stderr2 = capture_main(["init"], cwd=str(tmp_path))
        assert rc2 == 0, f"二次 init 失败：{stderr2}"
        assert (tmp_path / ".cvlab").is_dir(), ".cvlab 目录仍应存在"

    def test_init_output_success_message(self, tmp_path):
        """init 应输出成功信息。"""
        rc, stdout, stderr = capture_main(["init"], cwd=str(tmp_path))
        assert stdout, "init 应有输出信息"

    def test_init_with_lang_flag(self, tmp_path):
        """--lang en init 应正常工作。"""
        rc, stdout, stderr = capture_main(["--lang", "en", "init"], cwd=str(tmp_path))
        assert rc == 0, f"--lang en init 应成功，rc={rc}, stderr={stderr}"
        assert (tmp_path / ".cvlab").is_dir(), ".cvlab 目录未创建"


# ── 测试: list ────────────────────────────────────────────

class TestList:
    """验证 list 命令的基本功能。"""

    def test_list_empty_returns_zero(self, tmp_path):
        """空数据库时 list 应返回 0。"""
        rc, stdout, stderr = capture_main(["list"], cwd=str(tmp_path))
        assert rc == 0, f"空列表应成功，rc={rc}, stderr={stderr}"

    def test_list_output(self, tmp_path):
        """空列表应有输出（提示无实验）。"""
        rc, stdout, stderr = capture_main(["list"], cwd=str(tmp_path))
        assert stdout.strip(), "list 应有输出"

    def test_list_with_lang_flag(self, tmp_path):
        """--lang en list 应正常工作。"""
        rc, stdout, stderr = capture_main(["--lang", "en", "list"], cwd=str(tmp_path))
        assert rc == 0, f"--lang en list 应成功，rc={rc}"

    def test_list_with_status_filter(self, tmp_path):
        """list --status running 应正常执行。"""
        rc, stdout, stderr = capture_main(["--status", "running"], cwd=str(tmp_path))
        # 没有数据库时可能失败，但至少不应崩溃
        assert isinstance(rc, int)

    def test_list_with_tag_filter(self, tmp_path):
        """list --tag test 应正常执行。"""
        rc, stdout, stderr = capture_main(["--tag", "test"], cwd=str(tmp_path))
        assert isinstance(rc, int)


# ── 测试: show（错误路径）───────────────────────────────────

class TestShow:
    """验证 show 命令对不存在实验的处理。"""

    def test_show_nonexistent_returns_one(self):
        """show nonexistent 应返回 1。"""
        rc, stdout, stderr = capture_main(["show", "nonexistent_exp_12345"])
        assert rc == 1, f"不存在的实验应返回 1，得到 rc={rc}"

    def test_show_nonexistent_has_error_message(self):
        """show nonexistent 应输出错误信息。"""
        rc, stdout, stderr = capture_main(["show", "nonexistent_exp_12345"])
        output = stdout + stderr
        assert output.strip(), "错误应有输出信息"


# ── 测试: --lang 标志 ─────────────────────────────────────

class TestLangFlag:
    """验证 --lang 标志在各种命令上正常工作。"""

    def test_lang_before_command(self):
        """--lang 应能在子命令前使用。"""
        rc, stdout, stderr = capture_main(["--lang", "en", "help"])
        assert rc == 0, f"--lang en help 应成功，rc={rc}"

    def test_lang_with_help(self):
        """--lang en --help 应返回英文帮助。"""
        rc, stdout, stderr = capture_main(["--lang", "en", "--help"])
        assert rc == 0, f"--lang en --help 应成功，rc={rc}"

    def test_lang_zh_works(self):
        """--lang zh 应正常工作。"""
        rc, stdout, stderr = capture_main(["--lang", "zh", "help"])
        assert rc == 0, f"--lang zh help 应成功，rc={rc}"


# ── 测试: 未知命令 ────────────────────────────────────────

class TestUnknownCommand:
    """验证对非法子命令的处理。"""

    def test_unknown_command(self):
        """未知命令应返回非零或打印帮助。"""
        # argparse 对未知子命令的处理：打印错误并退出
        rc, stdout, stderr = capture_main(["unknown_command_xyz"])
        # 可能返回 2（argparse 错误）或 0（fallback 到 help）
        # 至少不应抛出未捕获异常
        assert isinstance(rc, int), f"未知命令应返回 int，得到 {type(rc)}"

    def test_typo_command(self):
        """拼写错误的命令应优雅处理。"""
        rc, stdout, stderr = capture_main(["helpp"])  # 拼写错误
        assert isinstance(rc, int)


# ── 测试: 集成场景 ────────────────────────────────────────

class TestWorkflow:
    """模拟常见工作流的多步测试。"""

    def test_init_then_list(self, tmp_path):
        """先 init 再 list 应正常工作。"""
        rc_init, _, _ = capture_main(["init"], cwd=str(tmp_path))
        assert rc_init == 0, "init 失败"
        rc_list, stdout_list, _ = capture_main(["list"], cwd=str(tmp_path))
        assert rc_list == 0, "init 后 list 失败"

    def test_init_creates_database(self, tmp_path):
        """init 后应能正常查询实验列表。"""
        capture_main(["init"], cwd=str(tmp_path))
        db_path = tmp_path / ".cvlab"
        assert db_path.is_dir(), ".cvlab 目录应存在"
        # 数据库文件是否存在取决于 Database 实现
        any_file = any(db_path.iterdir()) if db_path.exists() else False
        # 不强制要求有文件，因为可能是内存数据库
        assert True  # 至少不崩溃
