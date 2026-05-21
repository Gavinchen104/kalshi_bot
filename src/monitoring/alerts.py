from __future__ import annotations

import time
from typing import Any

import httpx

from src.config import MonitoringConfig
from src.monitoring.logging import get_logger


logger = get_logger("monitoring.alerts")


class AlertManager:
    """Cooldown-aware alert sender.

    Alerts always hit structured logs. If BOT_ALERT_WEBHOOK_URL is configured,
    they are also posted out-of-band as JSON.
    """

    def __init__(self, config: MonitoringConfig, webhook_url: str = "") -> None:
        self.config = config
        self.webhook_url = webhook_url.strip()
        self._last_sent: dict[str, float] = {}

    def should_send(self, key: str, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else now
        last = self._last_sent.get(key)
        if last is not None and ts - last < self.config.alert_cooldown_seconds:
            return False
        self._last_sent[key] = ts
        return True

    async def emit(self, key: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if not self.should_send(key):
            return
        body = {"key": key, "message": message, "payload": payload or {}}
        logger.warning("alert", **body)
        if not self.webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self.webhook_url, json=body)
        except Exception as exc:
            logger.warning("alert_delivery_failed", key=key, error=str(exc))
