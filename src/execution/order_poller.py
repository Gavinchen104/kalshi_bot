from __future__ import annotations

from datetime import datetime, timezone

from src.api.kalshi_client import KalshiClient
from src.monitoring.logging import get_logger
from src.portfolio.pnl import PnLTracker
from src.risk.exposure import ExposureBook
from src.storage.repository import Repository


logger = get_logger("order_poller")


class OrderPoller:
    """
    Polls Kalshi REST API for live order fill confirmations.
    Reconciles confirmed fills into the PnL ledger and exposure book.
    Only active in live (non-paper) trading mode.

    Kalshi order lifecycle:
      live_sent → resting → filled / canceled / expired
    On fill: books PnL and updates position exposure.
    """

    def __init__(
        self,
        client: KalshiClient,
        repository: Repository,
        pnl_tracker: PnLTracker,
        exposure: ExposureBook,
        poll_interval_seconds: int = 30,
    ) -> None:
        self.client = client
        self.repository = repository
        self.pnl_tracker = pnl_tracker
        self.exposure = exposure
        self.poll_interval_seconds = poll_interval_seconds
        self._last_poll_at: datetime | None = None

    async def maybe_poll(self) -> None:
        """Call from the main loop; throttled by poll_interval_seconds."""
        now = datetime.now(tz=timezone.utc)
        if self._last_poll_at is not None:
            if (now - self._last_poll_at).total_seconds() < self.poll_interval_seconds:
                return
        self._last_poll_at = now

        pending = self.repository.pending_live_orders()
        if not pending:
            return

        for order_row in pending:
            order_id = order_row.get("order_id")
            if not order_id:
                continue
            try:
                result = await self.client.get_order(order_id)
                kalshi_order = result.get("order", {})
                status = str(kalshi_order.get("status", "")).lower()

                if status in ("filled", "executed"):
                    fill_price = self._extract_fill_price(kalshi_order, order_row)
                    qty = int(order_row["quantity"])
                    side = str(order_row["side"])
                    market_id = str(order_row["market_id"])

                    self.pnl_tracker.on_fill(
                        market_id=market_id,
                        side=side,
                        quantity=qty,
                        fill_price_cents=fill_price,
                        fee_cents=0,
                    )
                    signed_qty = qty if side == "yes" else -qty
                    self.exposure.apply_fill(market_id, signed_qty)
                    self.repository.update_order_status(
                        db_id=int(order_row["id"]),
                        new_status="live_filled",
                        fill_price_cents=fill_price,
                        response=result,
                    )
                    self.repository.save_position_snapshot(
                        positions=self.pnl_tracker.position_snapshot(),
                        realized_cents=self.pnl_tracker.realized_cents,
                        unrealized_cents=self.pnl_tracker.unrealized_cents,
                    )
                    self.repository.log_event(
                        "live_fill_reconciled",
                        market_id=market_id,
                        payload={
                            "order_id": order_id,
                            "fill_price_cents": fill_price,
                            "quantity": qty,
                            "side": side,
                            "realized_cents": self.pnl_tracker.realized_cents,
                        },
                    )
                    logger.info(
                        "live_fill_reconciled",
                        market_id=market_id,
                        order_id=order_id,
                        fill_price_cents=fill_price,
                        side=side,
                        qty=qty,
                    )

                elif status in ("canceled", "cancelled", "expired"):
                    self.repository.update_order_status(
                        db_id=int(order_row["id"]),
                        new_status=f"live_{status}",
                        response=result,
                    )
                    logger.info("live_order_terminal", order_id=order_id, status=status)

            except Exception as exc:
                logger.warning("order_poll_failed", order_id=order_id, error=str(exc))

    @staticmethod
    def _extract_fill_price(kalshi_order: dict, order_row: dict) -> int:
        """Extract fill price in cents from Kalshi order response."""
        raw = kalshi_order.get("avg_price") or kalshi_order.get("price")
        if raw is not None:
            try:
                val = float(raw)
                # Kalshi prices can be 0-1 (fractional) or 0-100 (cents)
                return int(val * 100) if val <= 1.0 else int(val)
            except (TypeError, ValueError):
                pass
        return int(order_row.get("price_cents", 50))
