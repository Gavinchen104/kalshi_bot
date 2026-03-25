from __future__ import annotations

from src.monitoring.logging import get_logger
from src.portfolio.ledger import Position
from src.portfolio.pnl import PnLTracker
from src.types import MarketState, ProposedOrder


logger = get_logger("exit_manager")


class ExitManager:
    """
    Monitors open positions and generates close orders when exit conditions are met.

    Exit conditions (percentage of entry price):
      1. Take-profit: unrealized gain >= take_profit_pct of avg entry
      2. Stop-loss:   unrealized loss >= stop_loss_pct of avg entry

    A close is the reverse trade:
      - Long YES (net_quantity > 0) → close by buying NO (at bid side)
      - Short YES (net_quantity < 0) → close by buying YES (at ask side)
    """

    def __init__(
        self,
        take_profit_pct: float = 0.25,
        stop_loss_pct: float = 0.15,
    ) -> None:
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

    def check_exits(
        self,
        state: MarketState,
        pnl_tracker: PnLTracker,
    ) -> list[ProposedOrder]:
        pos = pnl_tracker.ledger.position_for(state.market_id)
        if pos is None or pos.is_flat:
            return []

        mark = self._mark_price(state)
        if mark is None:
            return []

        entry = pos.avg_entry_cents
        if entry <= 0:
            return []

        pnl_per_contract = self._unrealized_per_contract(pos, mark)
        tp_threshold = entry * self.take_profit_pct
        sl_threshold = entry * self.stop_loss_pct

        orders: list[ProposedOrder] = []

        if pnl_per_contract >= tp_threshold:
            order = self._close_order(pos, state, "take_profit")
            if order:
                logger.info(
                    "exit_signal",
                    reason="take_profit",
                    market_id=state.market_id,
                    pnl_per_contract=round(pnl_per_contract, 1),
                    threshold=round(tp_threshold, 1),
                    entry_avg=round(entry, 1),
                    mark=mark,
                )
                orders.append(order)

        elif pnl_per_contract <= -sl_threshold:
            order = self._close_order(pos, state, "stop_loss")
            if order:
                logger.info(
                    "exit_signal",
                    reason="stop_loss",
                    market_id=state.market_id,
                    pnl_per_contract=round(pnl_per_contract, 1),
                    threshold=round(-sl_threshold, 1),
                    entry_avg=round(entry, 1),
                    mark=mark,
                )
                orders.append(order)

        return orders

    def _unrealized_per_contract(self, pos: Position, mark_cents: int) -> float:
        if pos.net_quantity > 0:
            return mark_cents - pos.avg_entry_cents
        else:
            return pos.avg_entry_cents - mark_cents

    def _close_order(
        self, pos: Position, state: MarketState, reason: str
    ) -> ProposedOrder | None:
        qty = abs(pos.net_quantity)
        if qty == 0:
            return None

        if pos.net_quantity > 0:
            if state.bid_cents is None:
                return None
            close_side = "no"
            price_cents = 100 - state.bid_cents
        else:
            if state.ask_cents is None:
                return None
            close_side = "yes"
            price_cents = state.ask_cents

        price_cents = max(1, min(99, price_cents))
        return ProposedOrder(
            market_id=state.market_id,
            side=close_side,
            price_cents=price_cents,
            quantity=qty,
            tif="GTC",
        )

    @staticmethod
    def _mark_price(state: MarketState) -> int | None:
        if state.last_trade_cents is not None:
            return state.last_trade_cents
        if state.bid_cents is not None and state.ask_cents is not None:
            return (state.bid_cents + state.ask_cents) // 2
        return None
