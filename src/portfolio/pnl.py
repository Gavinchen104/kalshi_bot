from __future__ import annotations

from src.portfolio.ledger import PositionLedger


class PnLTracker:
    def __init__(self) -> None:
        self.ledger = PositionLedger()
        self._marks: dict[str, int] = {}

    def on_fill(self, market_id: str, side: str, quantity: int, fill_price_cents: int, fee_cents: int = 0) -> None:
        self.ledger.on_fill(market_id, side, quantity, fill_price_cents, fee_cents)

    def update_mark(self, market_id: str, mark_cents: int) -> None:
        self._marks[market_id] = mark_cents

    @property
    def realized_cents(self) -> int:
        return sum(p.realized_pnl_cents for p in self.ledger.positions.values())

    @property
    def unrealized_cents(self) -> int:
        total = 0
        for pos in self.ledger.positions.values():
            if pos.is_flat:
                continue
            mark = self._marks.get(pos.market_id)
            if mark is None:
                continue
            if pos.net_quantity > 0:
                total += int(round((mark - pos.avg_entry_cents) * pos.net_quantity))
            else:
                total += int(round((pos.avg_entry_cents - mark) * abs(pos.net_quantity)))
        return total

    @property
    def total_cents(self) -> int:
        return self.realized_cents + self.unrealized_cents

    def position_snapshot(self) -> list[dict]:
        return [
            {
                "market_id": p.market_id,
                "net_quantity": p.net_quantity,
                "avg_entry_cents": p.avg_entry_cents,
                "realized_pnl_cents": p.realized_pnl_cents,
            }
            for p in self.ledger.positions.values()
        ]
