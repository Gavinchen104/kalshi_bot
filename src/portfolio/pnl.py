from __future__ import annotations

from src.portfolio.ledger import PositionLedger


class PnLTracker:
    """
    Wraps PositionLedger for truthful realized + mark-to-market unrealized PnL.
    Mark prices are updated from live market data each tick.
    Fee and slippage costs are applied at fill time by the execution layer.
    """

    def __init__(self) -> None:
        self.ledger = PositionLedger()
        self._mark_prices: dict[str, int] = {}

    def on_fill(
        self,
        market_id: str,
        side: str,
        quantity: int,
        fill_price_cents: int,
        fee_cents: int = 0,
    ) -> None:
        self.ledger.on_fill(market_id, side, quantity, fill_price_cents, fee_cents=fee_cents)

    def settle_position(self, market_id: str, settlement_cents: int) -> int:
        """Settle an open position at contract expiry. Returns realized PnL (cents)."""
        return self.ledger.settle_position(market_id, settlement_cents)

    def restore_from_snapshot(self, snapshot: list[dict]) -> None:
        """Restore positions persisted from a previous session."""
        self.ledger.restore_from_snapshot(snapshot)

    def update_mark(self, market_id: str, mark_price_cents: int) -> None:
        self._mark_prices[market_id] = mark_price_cents

    @property
    def realized_cents(self) -> int:
        return self.ledger.total_realized_cents

    @property
    def unrealized_cents(self) -> int:
        return self.ledger.total_unrealized_cents(self._mark_prices)

    @property
    def total_cents(self) -> int:
        return self.realized_cents + self.unrealized_cents

    def position_snapshot(self) -> list[dict]:
        return self.ledger.snapshot()