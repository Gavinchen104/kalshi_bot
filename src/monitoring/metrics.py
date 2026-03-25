class Metrics:
    """Metrics stub. Replace with Prometheus/OpenTelemetry later."""

    def incr(self, name: str, value: int = 1) -> None:
        _ = (name, value)

    def gauge(self, name: str, value: float) -> None:
        _ = (name, value)

