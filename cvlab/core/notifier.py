"""训练事件通知系统 — Webhook 推送。

支持将训练完成、失败、OOM 等事件推送到 Webhook URL。
当前支持：Slack / 飞书 / 钉钉 / 通用 Webhook。

用法:
    from cvlab.core.notifier import WebhookNotifier

    notifier = WebhookNotifier("https://hooks.example.com/xxx")
    notifier.notify_complete(exp_id="exp_001", val_acc=0.85)
    notifier.notify_failed(exp_id="exp_001", reason="OOM")
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Webhook 通知器。

    Args:
        webhook_url: Webhook URL。
        format: 消息格式，可选 auto / slack / feishu / dingtalk。
             auto 会根据 URL 自动检测。
    """

    def __init__(self, webhook_url: str, format: str = "auto"):
        self.webhook_url = webhook_url
        self.format = self._detect_format(webhook_url) if format == "auto" else format

    def _detect_format(self, url: str) -> str:
        """根据 URL 自动检测 Webhook 类型。"""
        url_lower = url.lower()
        if "hooks.slack.com" in url_lower:
            return "slack"
        elif "feishu" in url_lower or "larksuite" in url_lower:
            return "feishu"
        elif "dingtalk" in url_lower or "oapi.dingtalk.com" in url_lower:
            return "dingtalk"
        return "generic"

    def notify_complete(self, exp_id: str, val_acc: float | None = None,
                        val_loss: float | None = None, epochs: int = 0,
                        duration: str = "", extra: dict[str, Any] | None = None) -> bool:
        """发送训练完成通知。"""
        title = "✅ CVLab 训练完成"
        fields = [
            ("实验 ID", exp_id),
            ("Epochs", str(epochs)),
        ]
        if val_acc is not None:
            fields.append(("验证准确率", f"{val_acc:.2f}%"))
        if val_loss is not None:
            fields.append(("验证 Loss", f"{val_loss:.4f}"))
        if duration:
            fields.append(("耗时", duration))
        if extra:
            for k, v in extra.items():
                fields.append((k, str(v)))
        return self._send(title, fields, "good")

    def notify_failed(self, exp_id: str, reason: str,
                      error_msg: str = "", extra: dict[str, Any] | None = None) -> bool:
        """发送训练失败通知。"""
        title = "❌ CVLab 训练失败"
        fields = [
            ("实验 ID", exp_id),
            ("原因", reason),
        ]
        if error_msg:
            fields.append(("错误信息", error_msg[:500]))
        if extra:
            for k, v in extra.items():
                fields.append((k, str(v)))
        return self._send(title, fields, "danger")

    def notify_oom(self, exp_id: str, attempt: int, batch_size: int) -> bool:
        """发送 OOM 通知。"""
        title = "⚠️ CVLab OOM 恢复"
        fields = [
            ("实验 ID", exp_id),
            ("重试次数", f"第 {attempt} 次"),
            ("Batch Size", str(batch_size)),
        ]
        return self._send(title, fields, "warning")

    def _send(self, title: str, fields: list[tuple[str, str]], color: str) -> bool:
        """发送消息到 Webhook。"""
        try:
            import urllib.request
            payload = self._build_payload(title, fields, color)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
                logger.info("Webhook sent: %s -> %s", title, body[:100])
                return True
        except Exception as e:
            logger.warning("Webhook failed: %s", e)
            return False

    def _build_payload(self, title: str, fields: list[tuple[str, str]],
                       color: str) -> dict[str, Any]:
        """根据 format 构建消息 payload。"""
        if self.format == "slack":
            return self._build_slack(title, fields, color)
        elif self.format == "feishu":
            return self._build_feishu(title, fields)
        elif self.format == "dingtalk":
            return self._build_dingtalk(title, fields)
        else:
            return self._build_generic(title, fields, color)

    def _build_slack(self, title: str, fields: list[tuple[str, str]],
                     color: str) -> dict[str, Any]:
        """Slack 格式。"""
        return {
            "attachments": [{
                "color": color,
                "title": title,
                "fields": [{"title": k, "value": v, "short": True} for k, v in fields],
                "footer": "CVLab",
            }],
        }

    def _build_feishu(self, title: str, fields: list[tuple[str, str]]) -> dict[str, Any]:
        """飞书格式。"""
        content_lines = [f"**{title}**"]
        for k, v in fields:
            content_lines.append(f"{k}: {v}")
        return {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [{
                    "tag": "markdown",
                    "content": "\n".join(content_lines),
                }],
            },
        }

    def _build_dingtalk(self, title: str, fields: list[tuple[str, str]]) -> dict[str, Any]:
        """钉钉格式。"""
        content = f"# {title}\n\n"
        for k, v in fields:
            content += f"**{k}**: {v}\n"
        return {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": content},
        }

    def _build_generic(self, title: str, fields: list[tuple[str, str]],
                       color: str) -> dict[str, Any]:
        """通用 JSON 格式。"""
        return {
            "title": title,
            "color": color,
            "fields": {k: v for k, v in fields},
            "source": "CVLab",
        }
