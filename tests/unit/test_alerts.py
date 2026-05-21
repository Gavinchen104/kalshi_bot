from __future__ import annotations

from src.config import MonitoringConfig
from src.monitoring.alerts import AlertManager


def test_alert_cooldown_suppresses_repeated_key():
    alerts = AlertManager(MonitoringConfig(alert_cooldown_seconds=60))

    assert alerts.should_send("stalled_feed", now=100.0)
    assert not alerts.should_send("stalled_feed", now=120.0)
    assert alerts.should_send("stalled_feed", now=161.0)


def test_alert_cooldown_is_per_key():
    alerts = AlertManager(MonitoringConfig(alert_cooldown_seconds=60))

    assert alerts.should_send("stalled_feed", now=100.0)
    assert alerts.should_send("calibration_drift", now=120.0)
