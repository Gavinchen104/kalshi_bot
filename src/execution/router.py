from __future__ import annotations

from src.config import ExecutionConfig
from src.strategy.signals import Strategy
from src.types import MarketState, ProposedOrder, Signal


class OrderRouter:
    def __init__(self, config: ExecutionConfig, strategy: Strategy | None = None):
        self.config = config
        self._strategy = strategy

    def signal_to_order(
        self,
        signal: Signal,
        state: MarketState | None = None,
    ) -> ProposedOrder:
        price_cents = max(
            self.config.min_price_cents,
            min(signal.fair_value_cents, self.config.max_price_cents),
        )

        qty = self.config.default_order_size
        if self._strategy is not None and state is not None and signal.predicted_prob is not None:
            market_price = state.ask_cents if signal.side == "yes" else (100 - (state.bid_cents or 0))
            kelly_qty = self._strategy.kelly_size(signal, max(market_price or 1, 1))
            if kelly_qty > 0:
                qty = min(kelly_qty, self.config.default_order_size * 5)

        return ProposedOrder(
            market_id=signal.market_id,
            side=signal.side,
            price_cents=price_cents,
            quantity=qty,
            tif="GTC",
        )
