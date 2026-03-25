from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    market_id: str
    net_quantity: int = 0
    avg_entry_cents: float = 0.0
    realized_pnl_cents: int = 0

    @property
    def is_flat(self) -> bool:
        return self.net_quantity == 0


class PositionLedger:
    """
    Per-market FIFO position tracker for Kalshi binary contracts.

    YES/NO framing (both settle at 0 or 100 cents):
      YES buy  at p  →  +qty at cost p
      NO  buy  at p  →  equivalent short-YES at (100 − p)

    Realized PnL is booked when positions are reduced or closed.
    Fees are passed in per-fill and deducted from realized PnL.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def on_fill(
        self,
        market_id: str,
        side: str,
        quantity: int,
        fill_price_cents: int,
        fee_cents: int = 0,
    ) -> int:
        """
        Record a fill. Returns net realized PnL change (cents, net of fees).
        """
        if market_id not in self._positions:
            self._positions[market_id] = Position(market_id=market_id)

        pos = self._positions[market_id]

        if side == "yes":
            signed_qty = quantity
            effective_price = float(fill_price_cents)
        else:
            signed_qty = -quantity
            effective_price = float(100 - fill_price_cents)

        realized = 0

        if pos.net_quantity == 0:
            pos.avg_entry_cents = effective_price
            pos.net_quantity = signed_qty

        elif (pos.net_quantity > 0 and signed_qty > 0) or (pos.net_quantity < 0 and signed_qty < 0):
            # Adding to same-side position — update weighted avg entry
            old_qty = abs(pos.net_quantity)
            add_qty = abs(signed_qty)
            pos.avg_entry_cents = (
                old_qty * pos.avg_entry_cents + add_qty * effective_price
            ) / (old_qty + add_qty)
            pos.net_quantity += signed_qty

        else:
            # Reducing or flipping position
            closing_qty = min(abs(signed_qty), abs(pos.net_quantity))
            if pos.net_quantity > 0:
                realized = int((effective_price - pos.avg_entry_cents) * closing_qty)
            else:
                realized = int((pos.avg_entry_cents - effective_price) * closing_qty)

            pos.net_quantity += signed_qty
            remaining = abs(signed_qty) - closing_qty
            if remaining > 0:
                direction = 1 if signed_qty > 0 else -1
                pos.net_quantity = direction * remaining
                pos.avg_entry_cents = effective_price

        net = realized - fee_cents
        pos.realized_pnl_cents += net
        return net

    def mark_to_market(self, market_id: str, mark_price_cents: int) -> int:
        pos = self._positions.get(market_id)
        if pos is None or pos.net_quantity == 0:
            return 0
        if pos.net_quantity > 0:
            return int((mark_price_cents - pos.avg_entry_cents) * pos.net_quantity)
        else:
            return int((pos.avg_entry_cents - mark_price_cents) * abs(pos.net_quantity))

    def total_unrealized_cents(self, mark_prices: dict[str, int]) -> int:
        return sum(
            self.mark_to_market(market_id, price)
            for market_id, price in mark_prices.items()
        )

    @property
    def total_realized_cents(self) -> int:
        return sum(p.realized_pnl_cents for p in self._positions.values())

    def position_for(self, market_id: str) -> Position | None:
        return self._positions.get(market_id)

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def settle_position(self, market_id: str, settlement_cents: int) -> int:
        """
        Settle an open position at contract expiry.
        YES contracts pay out settlement_cents (100 = YES wins, 0 = NO wins).
        Returns realized PnL booked (cents).
        """
        pos = self._positions.get(market_id)
        if pos is None or pos.is_flat:
            return 0
        if pos.net_quantity > 0:
            realized = int((settlement_cents - pos.avg_entry_cents) * pos.net_quantity)
        else:
            realized = int((pos.avg_entry_cents - settlement_cents) * abs(pos.net_quantity))
        pos.realized_pnl_cents += realized
        pos.net_quantity = 0
        return realized

    def restore_from_snapshot(self, snapshot: list[dict]) -> None:
        """Restore positions from a persisted snapshot on restart."""
        for row in snapshot:
            market_id = str(row.get("market_id", ""))
            if not market_id:
                continue
            self._positions[market_id] = Position(
                market_id=market_id,
                net_quantity=int(row.get("net_quantity", 0)),
                avg_entry_cents=float(row.get("avg_entry_cents", 0.0)),
                realized_pnl_cents=int(row.get("realized_pnl_cents", 0)),
            )

    def snapshot(self) -> list[dict]:
        return [
            {
                "market_id": p.market_id,
                "net_quantity": p.net_quantity,
                "avg_entry_cents": round(p.avg_entry_cents, 2),
                "realized_pnl_cents": p.realized_pnl_cents,
            }
            for p in self._positions.values()
            if not p.is_flat or p.realized_pnl_cents != 0
        ]
