"""CVLab i18n 模块单元测试。

测试策略：
- 隔离测试每个公开 API，不依赖外部环境
- 语言切换类测试注意重置全局状态，避免污染后续 case
- 格式字符串测试覆盖位置参数、命名参数、参数缺失等边缘情况
"""

from __future__ import annotations

import os
import threading

import pytest

from cvlab.i18n import (
    _,
    _n,
    cli_lang_option,
    current_language,
    get_available_languages,
    init,
    init_from_args,
    language_selector_html,
    set_language,
)


# ── 辅助 fixture ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_i18n_state():
    """每个测试前后重置 i18n 状态到默认中文。

    因为 ``set_language`` 修改全局线程本地状态，不重置的话
    测试之间的执行顺序会影响结果。
    """
    set_language("zh")
    yield
    set_language("zh")


# ── 基本翻译功能 ──────────────────────────────────────────

class TestBasicTranslation:
    """测试 ``_()`` 函数在不同语言下的基本翻译行为。"""

    def test_zh_returns_original(self):
        """中文模式下 _() 应直接返回原文，不做翻译表查找。"""
        set_language("zh")
        result = _("实验列表")
        assert result == "实验列表", f"中文应返回原文，得到 {result!r}"

    def test_en_translates_known_key(self):
        """英文模式下 _() 应返回 en.json 中的翻译值。"""
        set_language("en")
        result = _("实验列表")
        assert result == "Experiment List", f"英文翻译错误，得到 {result!r}"

    def test_en_translates_another_key(self):
        """验证多个已知 key 的英文翻译。"""
        set_language("en")
        assert _("暂无实验") == "No experiments"
        assert _("名称") == "Name"
        assert _("状态") == "Status"
        assert _("创建时间") == "Created At"

    def test_unknown_key_returns_original_in_en(self):
        """英文模式下不存在的 key 应回退到原文。"""
        set_language("en")
        result = _("ThisKeyDoesNotExist_abc123")
        assert result == "ThisKeyDoesNotExist_abc123", (
            f"不存在的 key 应回退原文，得到 {result!r}"
        )

    def test_empty_string(self):
        """空字符串应保持不变。"""
        for lang in ("zh", "en"):
            set_language(lang)
            assert _("") == "", f"语言 {lang} 下空字符串被改变"
            set_language("zh")

    def test_string_with_special_chars(self):
        """包含特殊字符的字符串应原样返回。"""
        set_language("en")
        result = _("Hello! @#$% ^&*()")
        assert result == "Hello! @#$% ^&*()", f"特殊字符被改变：{result!r}"


# ── set_language / current_language ───────────────────────

class TestLanguageSwitching:
    """测试语言切换和当前语言查询。"""

    def test_set_and_get_language(self):
        """set_language 后 current_language 应返回对应值。"""
        assert current_language() == "zh", "默认应为中文"

        set_language("en")
        assert current_language() == "en", "切换到英文后查询结果错误"

        set_language("zh")
        assert current_language() == "zh", "切回中文后查询结果错误"

    def test_invalid_language_falls_back_to_zh(self):
        """设置不支持的语言代码应静默回退到中文。"""
        set_language("fr")
        assert current_language() == "zh", "非法语言 fr 应回退到 zh"

        set_language("jp")
        assert current_language() == "zh", "非法语言 jp 应回退到 zh"

        set_language("")
        assert current_language() == "zh", "空字符串应回退到 zh"

    def test_case_sensitivity(self):
        """语言代码应区分大小写，小写 'en' 有效，大写 'EN' 无效。"""
        set_language("EN")
        assert current_language() == "zh", "大写 EN 应视为非法，回退到 zh"

        set_language("En")
        assert current_language() == "zh", "混合大小写 En 应视为非法"

    def test_multiple_switches(self):
        """频繁切换语言应始终正确。"""
        for lang in ("zh", "en", "zh", "en", "zh"):
            set_language(lang)
            expected = lang if lang in ("zh", "en") else "zh"
            assert current_language() == expected, (
                f"切换至 {lang} 后期望 {expected}，得到 {current_language()}"
            )

    def test_get_available_languages(self):
        """get_available_languages 应返回完整的语言列表。"""
        langs = get_available_languages()
        codes = {l["code"] for l in langs}
        assert "zh" in codes, "缺少中文"
        assert "en" in codes, "缺少英文"
        # 每个语言条目应有 code/name/name_en
        for lang in langs:
            assert "code" in lang, f"条目缺少 code: {lang}"
            assert "name" in lang, f"条目缺少 name: {lang}"
            assert "name_en" in lang, f"条目缺少 name_en: {lang}"


# ── 格式字符串 ────────────────────────────────────────────

class TestFormatStrings:
    """测试带参数的格式化翻译。"""

    def test_positional_args(self):
        """带 {} 占位符的文本应被 .format() 正确替换。

        _() 内部使用 ``str.format(*args, **kwargs)``，
        占位符语法是 ``{}`` 而非 ``%s``。
        """
        # 中文模式：直接对原文做 .format()
        set_language("zh")
        result = _("实验 {} 已完成", "exp_001")
        assert result == "实验 exp_001 已完成", (
            f"位置参数替换错误：{result!r}"
        )

        # 英文模式：key "实验 {} 已完成" 在 en.json 中不存在
        # （翻译表使用 %s 风格），因此回退到原文，再执行 .format()
        set_language("en")
        result_en = _("实验 {} 已完成", "exp_001")
        assert "exp_001" in result_en, (
            f"英文模式位置参数未替换：{result_en!r}"
        )

    def test_percent_s_placeholder_passes_through(self):
        """翻译表中 ``%s`` 是字面文本，.format() 不会替换它。

        .. note::
           翻译文件 en.json 中的 ``%s`` 是字面 ``%s``，
           不会被 ``.format()`` 识别为占位符。如果要格式化，
           请使用 ``{}`` 语法。
        """
        set_language("zh")
        result = _("实验 %s 不存在", "exp_001")
        assert "%s" in result, "%s 应作为字面文本保留"
        assert result == "实验 %s 不存在"

    def test_multiple_positional_args(self):
        """多个 {} 占位符应被按位置替换。"""
        text = "{0} + {1} = {2}"
        result = _(text, "1", "1", "2")
        assert result == "1 + 1 = 2", f"多参数替换错误：{result!r}"

    def test_kwargs_format(self):
        """命名参数应正确替换 {name} 占位符。"""
        set_language("zh")
        # _() 内部使用 .format(*args, **kwargs)
        text = "Hello {name}, your score is {score}"
        result = _(text, name="Alice", score=95)
        assert result == "Hello Alice, your score is 95", (
            f"命名参数替换错误：{result!r}"
        )

    def test_args_and_kwargs_together(self):
        """位置参数和命名参数同时使用。"""
        text = "{greeting}, {name}! You are #{rank}"
        result = _(text, greeting="Hi", name="Bob", rank=1)
        assert result == "Hi, Bob! You are #1", f"混合参数替换错误：{result!r}"

    def test_missing_format_key(self):
        """格式占位符缺少参数时应保持原样，不崩溃。"""
        text = "Hello {name}, today is {day}"
        # 只提供 name，不提供 day
        result = _(text, name="Alice")
        # _() 内部捕获 KeyError，所以返回未完全替换的文本
        assert "{day}" in result or result == text, (
            f"缺少格式参数时应保持原样，得到 {result!r}"
        )

    def test_extra_args_ignored(self):
        """多余的参数应被 .format() 忽略（如果占位符是 {} 自动编号）。"""
        # 注意：str.format() 在混合自动编号和手动编号时会抛 ValueError，
        # 但这里用了纯粹手动编号 {name}，所以多余的 kwargs 会被忽略
        text = "Only {name}"
        result = _(text, name="Alice", extra="ignored")
        assert result == "Only Alice", f"多余参数导致问题：{result!r}"

    def test_no_args_no_change(self):
        """无参数时格式字符串应保持原样。"""
        set_language("en")
        result = _("实验列表")
        assert result == "Experiment List", "无参数 key 被意外修改"


# ── 语言自动检测 ──────────────────────────────────────────

class TestLanguageDetection:
    """测试 ``init()`` 和 ``init_from_args()`` 的自动检测逻辑。"""

    def test_init_from_args_sets_language(self):
        """init_from_args("en") 应显式设置语言为英文。"""
        init_from_args("en")
        assert current_language() == "en", "init_from_args(en) 后语言不是 en"

    def test_init_from_args_none_triggers_auto(self):
        """init_from_args(None) 应走自动检测路径（默认中文）。"""
        # 确保没有 CVLAB_LANG
        old_env = os.environ.pop("CVLAB_LANG", None)
        try:
            init_from_args(None)
            # 在没有 CVLAB_LANG、系统 locale 非 zh 时可能检测为其它，
            # 但我们的 _detect_system_language 在无法识别时返回 "zh"
            lang = current_language()
            assert lang in ("zh", "en"), f"自动检测返回了意外语言 {lang}"
        finally:
            if old_env is not None:
                os.environ["CVLAB_LANG"] = old_env

    def test_init_respects_cvlab_lang(self):
        """init() 应优先读取 CVLAB_LANG 环境变量。"""
        old_env = os.environ.get("CVLAB_LANG")
        os.environ["CVLAB_LANG"] = "en"
        try:
            init()
            assert current_language() == "en", (
                "CVLAB_LANG=en 后 init() 未设为英文"
            )
        finally:
            if old_env is not None:
                os.environ["CVLAB_LANG"] = old_env
            else:
                del os.environ["CVLAB_LANG"]

    def test_cvlab_lang_empty_string(self):
        """CVLAB_LANG 为空字符串时应走自动检测。"""
        old_env = os.environ.get("CVLAB_LANG")
        os.environ["CVLAB_LANG"] = ""
        try:
            init()
            # 空字符串视为未设置，走自动检测
            lang = current_language()
            assert lang in ("zh", "en"), f"空 CVLAB_LANG 后语言异常：{lang}"
        finally:
            if old_env is not None:
                os.environ["CVLAB_LANG"] = old_env
            else:
                del os.environ["CVLAB_LANG"]

    def test_cli_lang_takes_priority_over_env(self):
        """模拟 CLI --lang 参数（通过 init_from_args）应覆盖 CVLAB_LANG。"""
        old_env = os.environ.get("CVLAB_LANG")
        os.environ["CVLAB_LANG"] = "zh"
        try:
            # 用户显式传入 --lang en
            init_from_args("en")
            assert current_language() == "en", (
                "CLI 显式 --lang en 应覆盖 CVLAB_LANG=zh"
            )
        finally:
            if old_env is not None:
                os.environ["CVLAB_LANG"] = old_env
            else:
                del os.environ["CVLAB_LANG"]

    def test_cli_lang_option_structure(self):
        """cli_lang_option() 应返回 argparse 兼容的选项字典。"""
        opt = cli_lang_option()
        assert isinstance(opt, dict), "cli_lang_option 应返回 dict"
        assert "flags" in opt, "缺少 flags"
        assert "--lang" in opt["flags"], "flags 中应包含 --lang"
        assert "kwargs" in opt, "缺少 kwargs"
        assert opt["kwargs"].get("choices") == ["zh", "en"], (
            "choices 应为 [zh, en]"
        )


# ── 复数翻译 ──────────────────────────────────────────────

class TestPluralTranslation:
    """测试 ``_n()`` 复数翻译函数。"""

    def test_singular(self):
        """count == 1 时应返回单数形式。"""
        result = _n("实验", "实验s", 1)
        # 中文下不区分单复数，所以返回第一个参数
        assert result == "实验", f"单数形式错误：{result!r}"

    def test_plural(self):
        """count != 1 时应返回复数形式。"""
        result = _n("实验", "实验s", 2)
        assert result == "实验s", f"复数形式错误：{result!r}"

    def test_zero(self):
        """count == 0 也应返回复数形式（符合英文习惯）。"""
        result = _n("实验", "实验s", 0)
        assert result == "实验s", f"零值复数形式错误：{result!r}"

    def test_plural_negative(self):
        """负数也应返回复数形式。"""
        result = _n("实验", "实验s", -5)
        assert result == "实验s", f"负数复数形式错误：{result!r}"


# ── 线程安全 ──────────────────────────────────────────────

class TestThreadSafety:
    """验证语言切换是线程安全的（threading.local）。"""

    def test_thread_inherits_default_not_main(self):
        """新线程应使用默认语言 zh，不受主线程切换影响。"""
        set_language("en")
        assert current_language() == "en", "主线程应为 en"

        langs_in_thread: list[str] = []

        def check_lang():
            langs_in_thread.append(current_language())

        t = threading.Thread(target=check_lang)
        t.start()
        t.join()

        assert langs_in_thread[0] == "zh", (
            f"新线程应使用默认 zh，得到 {langs_in_thread[0]!r}"
        )

    def test_multiple_threads_independent(self):
        """多线程各自设置语言互不干扰。"""
        results: list[str] = []

        def set_and_get(lang: str):
            set_language(lang)
            results.append(current_language())

        threads = [
            threading.Thread(target=set_and_get, args=("zh",)),
            threading.Thread(target=set_and_get, args=("en",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert "zh" in results, "zh 线程结果丢失"
        assert "en" in results, "en 线程结果丢失"
        # 主线程不受影响
        assert current_language() == "zh", "主线程语言被污染"


# ── 边缘情况 ──────────────────────────────────────────────

class TestEdgeCases:
    """测试各种边界和错误路径。"""

    def test_language_selector_html_contains_zh(self):
        """language_selector_html() 应生成包含中文选项的 HTML。"""
        set_language("zh")
        html = language_selector_html()
        assert "zh" in html, "HTML 中缺少 zh 选项"
        assert "中文" in html, "HTML 中缺少中文显示名"
        assert "select" in html, "HTML 不是有效的 select 元素"
        assert "selected" in html, "当前语言应标记为 selected"

    def test_language_selector_html_en_selected(self):
        """英文模式下 selector 中 en 应为 selected。"""
        set_language("en")
        html = language_selector_html()
        assert 'value="en" selected' in html, (
            "英文模式下 en 应标记为 selected"
        )

    def test_nested_percent_in_text(self):
        """原文中包含 % 字符时不应被误认为格式占位符。"""
        # 使用 _() 的 format，但原文中的 % 不是 str.format 语法，没问题
        text = "Accuracy: 95%"
        result = _(text)
        assert result == "Accuracy: 95%", f"百分号被修改：{result!r}"

    def test_braces_in_text(self):
        """原文中包含花括号时 str.format 可能报错，但 _() 应在遇到错误时返回原文。"""
        text = "dict = {key: value}"
        # str.format 会尝试解析 {key}，但 key 不存在于 kwargs，
        # _() 内部会捕获 KeyError，返回未完全替换的文本
        result = _(text)
        # 至少不崩溃
        assert result is not None

    def test_multiple_calls_different_languages(self):
        """交替使用中英文 _() 应各自返回正确结果。"""
        set_language("zh")
        zh_result = _("实验列表")
        set_language("en")
        en_result = _("实验列表")
        set_language("zh")
        zh_result2 = _("实验列表")

        assert zh_result == "实验列表", f"中文结果错误：{zh_result!r}"
        assert en_result == "Experiment List", f"英文结果错误：{en_result!r}"
        assert zh_result2 == "实验列表", f"第二次中文结果错误：{zh_result2!r}"
