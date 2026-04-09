"""
Bot notifications via Telegram.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.core.config import get_settings

log = logging.getLogger(__name__)


class TelegramNotifier:
    """Async sender for Telegram bot messages."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        cfg = get_settings()
        self.token = token or cfg.telegram_bot_token
        self.chat_id = chat_id or cfg.telegram_chat_id

        self.base_url: Optional[str] = None
        if self.token and self.chat_id:
            self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_message(self, text: str) -> bool:
        """Sends an HTML formatted message to the configured chat."""
        if not self.base_url or not self.chat_id:
            log.debug("Telegram not configured. Discarded message: %s", text)
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                response.raise_for_status()
                return True
        except Exception as exc:
            log.error("Error envoyando mensaje a Telegram: %s", exc)
            return False

# Global instance
notifier = TelegramNotifier()
