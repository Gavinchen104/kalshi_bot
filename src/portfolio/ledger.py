"""Position ledger: tracks net quantity + average entry per market, books realized PnL on offsets."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    market_id: str
    net_quantity: int = 0   # positive = long YES; negative = short YES (== long NO)
    avg_entry_cents: float = 0.0
    realized_pnl_cents: int = 0

    @property
    def is_flat(self) -> bool:
        return self.net_quantity == 0


@dataclass
class PositionLedger:
    positions: dict[str, Position] = field(default_factory=dict)

    def position_for(self, market_id: str) -> Position | None:
        return self.positions.get(market_id)

    def on_fill(
        self,
        market_id: str,
        side: str,
        quantity: int,
        fill_price_cents: int,
        fee_cents: int = 0,
    ) -> None:
        pos = self.positions.setdefault(market_id, Position(market_id=market_id))
        signed = quantity if side == "yes" else -quantity
        new_qty = pos.net_quantity + signed
        if pos.net_quantity == 0 or (pos.net_quantity > 0) == (signed > 0):
            # Open or add to existing direction: update weighted average entry.
            total_cost = pos.avg_entry_cents * abs(pos.net_quantity) + fill_price_cents * abs(signed)
            pos.avg_entry_cents = total_cost / max(1, abs(new_qty)) if new_qty != 0 else 0.0
        else:
            # Offsetting trade: realize PnL on the offset portion.
            offset_qty = min(abs(pos.net_quantity), abs(signed))
            if pos.net_quantity > 0:
                pnl_per = fill_price_cents - pos.avg_entry_cents
            else:
                pnl_per = pos.avg_entry_cents - fill_price_cents
            pos.realized_pnl_cents += int(round(pnl_per * offset_qty))
            if abs(signed) > abs(pos.net_quantity):
                # Flipped direction: residual opens new position at fill price.
                pos.avg_entry_cents = float(fill_price_cents)
            elif new_qty == 0:
                pos.avg_entry_cents = 0.0
        pos.net_quantity = new_qty
        pos.realized_pnl_cents -= int(fee_cents)
