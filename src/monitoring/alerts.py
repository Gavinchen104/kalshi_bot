from src.monitoring.logging import get_logger


logger = get_logger("alerts")


def send_alert(message: str) -> None:
    logger.warning("alert", message=message)

