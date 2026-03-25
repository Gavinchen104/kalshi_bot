from __future__ import annotations

import re
from collections import defaultdict, deque

from src.api.binance_ws import BinanceDataStore
from src.config import StrategyConfig
from src.strategy.features import build_enhanced_features
from src.strategy.filters import has_acceptable_spread, has_min_depth
from src.strategy.kelly import kelly_contracts
from src.strategy.ml_model import GBMModel, LinearProbEnsemble, LinearProbModel, load_model
from src.types import MarketState, Signal


class Strategy:
    def __init__(
        self,
        config: StrategyConfig,
        btc_store: BinanceDataStore | None = None,
        bankroll_cents: int = 50_000,
    ):
        self.config = config
        self._market_re = re.compile(config.target_market_regex, re.IGNORECASE)
        self._history: dict[str, deque[MarketState]] = defaultdict(
            lambda: deque(maxlen=max(50, config.feature_lookback + 15))
        )
        self._model: GBMModel | LinearProbModel | LinearProbEnsemble | None = None
        self._btc_store = btc_store
        self._bankroll_cents = bankroll_cents
        self._model_has_edge = False
        self.reload_model()

    def set_btc_store(self, store: BinanceDataStore) -> None:
        self._btc_store = store

    def set_model_quality(self, beats_baseline: bool) -> None:
        self._model_has_edge = beats_baseline

    def reload_model(self) -> bool:
        if not self.config.use_ml_model:
            self._model = None
            return False
        try:
            self._model = load_model(self.config.model_path)
            return True
        except Exception:
            self._model = None
            return False

    def compute_signal(self, state: MarketState) -> Signal | None:
        if not self._market_re.search(state.market_id):
            return None

        history = self._history[state.market_id]
        history.append(state)

        if not has_acceptable_spread(state, self.config.max_spread_cents):
            return None
        if not has_min_depth(state, self.config.min_top_book_depth):
            return None

        if not self._model_has_edge:
            return None

        predicted_prob = None
        model_name = None
        fair: int | None = None
        reason = "midpoint_fair"

        if self._model is not None and len(history) >= self.config.feature_lookback:
            feature_map = self._build_features(list(history))
            if feature_map is not None:
                predicted_prob = self._model.predict_proba(feature_map)
                confidence = abs(predicted_prob - 0.5)
                if confidence < self.config.min_model_confidence:
                    return None
                fair = int(round(predicted_prob * 100))
                model_name = self._model.model_name
                reason = f"ml_prob={predicted_prob:.4f}"

        if fair is None:
            from src.strategy.fair_value import estimate_fair_value_cents

            fair = estimate_fair_value_cents(state)

        if fair is None or state.ask_cents is None or state.bid_cents is None:
            return None

        buy_edge = fair - state.ask_cents
        sell_edge = state.bid_cents - fair
        min_edge = max(1, self.config.min_edge_bps // 100)
        cost_floor_bps = getattr(self.config, "cost_floor_bps", 0)

        if buy_edge >= min_edge:
            edge_bps = buy_edge * 100
            if edge_bps < cost_floor_bps:
                return None
            return Signal(
                market_id=state.market_id,
                side="yes",
                edge_bps=edge_bps,
                fair_value_cents=fair,
                reason=f"{reason}:ask_below_fair",
                predicted_prob=predicted_prob,
                model_name=model_name,
            )
        if sell_edge >= min_edge:
            edge_bps = sell_edge * 100
            if edge_bps < cost_floor_bps:
                return None
            return Signal(
                market_id=state.market_id,
                side="no",
                edge_bps=edge_bps,
                fair_value_cents=fair,
                reason=f"{reason}:bid_above_fair",
                predicted_prob=predicted_prob,
                model_name=model_name,
            )
        return None

    def kelly_size(self, signal: Signal, market_price_cents: int) -> int:
        if signal.predicted_prob is None:
            return self.config.default_order_size if hasattr(self.config, "default_order_size") else 1
        return kelly_contracts(
            model_prob=signal.predicted_prob,
            market_price_cents=market_price_cents,
            bankroll_cents=self._bankroll_cents,
            min_contracts=1,
            max_contracts=getattr(self.config, "max_kelly_contracts", 10),
        )

    def _build_features(self, window: list[MarketState]) -> dict[str, float] | None:
        closes = volumes = ohlcv = None
        if self._btc_store is not None and self._btc_store.is_ready:
            closes = self._btc_store.get_closes()
            volumes = self._btc_store.get_volumes()
            ohlcv = self._btc_store.get_ohlcv()
        return build_enhanced_features(window, closes, volumes, ohlcv)
