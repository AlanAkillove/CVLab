"""CLI 输出工具 - 基于 Rich 的终端输出管理。"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console as _RichConsole
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

console = _RichConsole()


def header(text: str) -> None:
    """打印章节标题。"""
    console.rule(f"[bold]{text}[/bold]", style="blue")


def result(label: str, value: str, ok: bool = True) -> None:
    """打印键值对结果。"""
    mark = "[green][OK][/green]" if ok else "[red][FAIL][/red]"
    console.print(f"  {mark}  [bold]{label}:[/bold] {value}")


def table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    """打印格式化表格（ID 列完整显示，不截断不折叠）。"""
    t = Table(
        title=title, box=box.ROUNDED, header_style="bold cyan",
        padding=(0, 1), safe_box=True,
    )
    for i, col in enumerate(columns):
        if i == 0:
            t.add_column(col, no_wrap=True)
        else:
            t.add_column(col, no_wrap=True)
    for row in rows:
        t.add_row(*(str(r) for r in row))
    console.print(t)


def json_block(data: Any, title: str = "") -> None:
    """语法高亮显示 JSON。"""
    import json as _json
    text = _json.dumps(data, indent=2, default=str)
    syntax = Syntax(text, "json", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title=title, border_style="dim"))


def progress() -> Progress:
    """创建一个进度条。"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def success(text: str) -> None:
    console.print(f"  [green][OK][/green] {text}")


def error(text: str) -> None:
    console.print(f"  [red][FAIL][/red] {text}")


def warning(text: str) -> None:
    console.print(f"  [yellow][WARN][/yellow] {text}")


def info(text: str) -> None:
    console.print(f"  [dim]{text}[/dim]")


def panel(text: str, title: str = "", style: str = "blue") -> None:
    """打印面板。"""
    console.print(Panel(text, title=title, border_style=style))
