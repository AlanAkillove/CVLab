"""CVLab 国际化（i18n）支持。

用法:
    from cvlab.i18n import _, set_language, current_language

    set_language("zh")    # 切换到中文
    print(_("Hello"))     # 自动翻译

语言自动检测优先级:
    1. CVLAB_LANG 环境变量
    2. --lang CLI 参数（由调用方注入）
    3. 系统 locale
    4. 默认中文
"""

from __future__ import annotations

import locale
import os
import json
import threading
from pathlib import Path
from typing import Any

# 存储当前语言（线程安全）
_current_lang = threading.local()
_current_lang.value = "zh"  # 默认中文

# 翻译缓存
_translations: dict[str, dict[str, str]] = {}
_loaded: set[str] = set()


def _detect_system_language() -> str:
    """检测系统语言。"""
    try:
        lang, _ = locale.getdefaultlocale()
        if lang:
            lang = lang.lower()
            if lang.startswith("zh"):
                return "zh"
            if lang.startswith("en"):
                return "en"
    except Exception:
        pass
    return "zh"


def get_available_languages() -> list[dict[str, str]]:
    """返回可用语言列表。

    Returns:
        [{"code": "zh", "name": "中文", "name_en": "Chinese"},
         {"code": "en", "name": "English", "name_en": "English"}]
    """
    return [
        {"code": "zh", "name": "中文", "name_en": "Chinese"},
        {"code": "en", "name": "English", "name_en": "English"},
    ]


def current_language() -> str:
    """获取当前语言代码。"""
    return getattr(_current_lang, "value", "zh")


def set_language(lang: str) -> None:
    """设置当前语言。

    Args:
        lang: 语言代码，如 "zh" 或 "en"。
    """
    valid = {l["code"] for l in get_available_languages()}
    if lang not in valid:
        lang = "zh"
    _current_lang.value = lang


def _load_translations(lang: str) -> dict[str, str]:
    """加载指定语言的翻译文件。"""
    if lang in _loaded:
        return _translations.get(lang, {})

    path = Path(__file__).parent / "translations" / f"{lang}.json"
    if not path.exists():
        _loaded.add(lang)
        _translations[lang] = {}
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _translations[lang] = data
        _loaded.add(lang)
        return data
    except (json.JSONDecodeError, IOError) as e:
        _loaded.add(lang)
        _translations[lang] = {}
        return {}


def _(text: str, *args: Any, **kwargs: Any) -> str:
    """翻译字符串。

    如果当前语言是中文，直接返回原文（中文是基础语言）。
    如果是其他语言，查找翻译表。

    Args:
        text: 原始文本（中文）。
        *args: 格式化参数（位置）。
        **kwargs: 格式化参数（命名）。

    Returns:
        翻译后的文本。
    """
    lang = current_language()
    if lang == "zh":
        translated = text
    else:
        translations = _load_translations(lang)
        translated = translations.get(text, text)

    if args or kwargs:
        try:
            translated = translated.format(*args, **kwargs)
        except (IndexError, KeyError):
            pass

    return translated


def _n(text_single: str, text_plural: str, count: int, *args: Any, **kwargs: Any) -> str:
    """复数翻译（备用接口，当前英语无需区分复数）。"""
    text = text_single if count == 1 else text_plural
    return _(text, count, *args, **kwargs)


def init() -> None:
    """初始化 i18n 系统（自动检测语言）。"""
    env_lang = os.environ.get("CVLAB_LANG", "")
    if env_lang:
        set_language(env_lang)
        return

    detected = _detect_system_language()
    set_language(detected)


def language_selector_html() -> str:
    """生成语言选择器的 HTML（供 Streamlit UI 使用）。"""
    current = current_language()
    options = get_available_languages()
    items = []
    for lang in options:
        code = lang["code"]
        name = lang["name"]
        selected = "selected" if code == current else ""
        items.append(f'<option value="{code}" {selected}>{name}</option>')
    return f"""
    <select id="cvlab-lang-selector" onchange="changeLanguage(this.value)"
            style="font-size:0.75rem;padding:0.25rem 0.5rem;
                   border:1px solid var(--border);background:var(--surface-card);
                   font-family:var(--font);cursor:pointer;">
        {chr(10).join(items)}
    </select>
    <script>
    function changeLanguage(lang) {{
        const params = new URLSearchParams(window.location.search);
        params.set('lang', lang);
        window.location.search = params.toString();
    }}
    </script>
    """


def cli_lang_option() -> dict[str, Any]:
    """返回 argparse 的语言选项（供 CLI 子解析器复用）。"""
    return {
        "flags": ["--lang"],
        "kwargs": {
            "choices": ["zh", "en"],
            "default": None,
            "help": "显示语言 (zh/en)",
        },
    }


def init_from_args(lang: str | None = None) -> None:
    """从 CLI 参数初始化语言。

    Args:
        lang: 语言代码，None 则自动检测。
    """
    if lang:
        set_language(lang)
    else:
        init()
