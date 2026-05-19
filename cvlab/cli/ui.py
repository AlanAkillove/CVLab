"""cvlab ui - 启动 Streamlit Web 界面。

自动解析 cvlab/ui/app.py 的安装路径，用户无需手动寻找。
支持 --port、--host 等 streamlit 常用参数。

用法:
    cvlab ui                    # 默认端口 8501
    cvlab ui --port 8502        # 自定义端口
    cvlab ui --host 0.0.0.0     # 允许外部访问
    cvlab ui --lang en          # 英文界面
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cvlab.cli.console import error, info
from cvlab.i18n import _


def cmd_ui(args: argparse.Namespace) -> int:
    # 自动解析 ui/app.py 的安装路径（兼容开发环境和 site-packages）
    app_path = _resolve_app_path()
    if not app_path:
        error(_("UI 入口文件未找到"))
        return 1

    info(_("启动 CVLab Web UI: {}").format(str(app_path)))
    info(_("浏览器打开: http://{}:{}").format(
        args.host if args.host != "0.0.0.0" else "localhost",
        args.port,
    ))

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]

    if args.lang:
        # 通过环境变量传递语言设置
        env = dict(__import__("os").environ)
        env["CVLAB_LANG"] = args.lang
    else:
        env = None

    try:
        proc = subprocess.run(cmd, env=env)
        return proc.returncode
    except FileNotFoundError:
        error(_("streamlit 未安装。请执行: pip install streamlit"))
        return 1
    except KeyboardInterrupt:
        return 0


def _resolve_app_path() -> Path | None:
    """Resolve the path to cvlab/ui/app.py.

    Tries multiple strategies:
    1. Development: relative to this file (../../ui/app.py)
    2. Installed package: via importlib.resources
    3. Direct import resolution
    """
    # Strategy 1: relative to this file (development / editable install)
    dev_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    if dev_path.exists():
        return dev_path

    # Strategy 2: use importlib to find the package
    try:
        import importlib.resources as resources

        if hasattr(resources, "files"):
            # Python 3.9+
            app = resources.files("cvlab") / "ui" / "app.py"
            if app.is_file():
                return app
    except (ImportError, ModuleNotFoundError):
        pass

    # Strategy 3: import and inspect
    try:
        import cvlab.ui.app
        app_file = Path(cvlab.ui.app.__file__)
        if app_file.exists():
            return app_file
    except (ImportError, AttributeError):
        pass

    return None


def add_subparser(sub) -> None:
    p = sub.add_parser("ui", help=_("启动 Streamlit Web 界面"))
    p.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help=_("端口号（默认 8501）"),
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help=_("监听地址（默认 127.0.0.1，使用 0.0.0.0 允许外部访问）"),
    )
    p.add_argument(
        "--lang",
        choices=["zh", "en"],
        default=None,
        help=_("界面语言（默认：自动检测）"),
    )
    p.set_defaults(func=cmd_ui)
