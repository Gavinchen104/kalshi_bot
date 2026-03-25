from __future__ import annotations

from datetime import datetime, timezone

from src.api.kalshi_client import KalshiClient
from src.monitoring.logging import get_logger
from src.portfolio.pnl import PnLTracker
from src.storage.repository import Repository


logger = get_logger("paper_settler")


class PaperSettler:
    """
    Monitors Kalshi market status for markets with open paper positions.
    When a market finalizes, settles the paper position at the contract outcome.

    Settlement mapping (from YES perspective):
      result="yes"  →  100 cents  (YES contract pays out)
      result="no"   →    0 cents  (YES contract worthless)
    """

    def __init__(
        self,
        client: KalshiClient,
        pnl_tracker: PnLTracker,
        repository: Repository,
        check_interval_seconds: int = 300,
    ) -> None:
        self.client = client
        self.pnl_tracker = pnl_tracker
        self.repository = repository
        self.check_interval_seconds = check_interval_seconds
        self._settled_markets: set[str] = set()
        self._last_check_at: datetime | None = None

    async def maybe_settle(self) -> None:
        """Call periodically from the main loop to detect and apply settlements."""
        now = datetime.now(tz=timezone.utc)
        if self._last_check_at is not None:
            elapsed = (now - self._last_check_at).total_seconds()
            if elapsed < self.check_interval_seconds:
                return
        self._last_check_at = now

        open_positions = [p for p in self.pnl_tracker.ledger.all_positions() if not p.is_flat]
        if not open_positions:
            return

        for pos in open_positions:
            market_id = pos.market_id
            if market_id in self._settled_markets:
                continue
            try:
                market_data = await self.client.get_market(market_id)
                market = market_data.get("market", {})
                status = str(market.get("status", "")).lower()
                result = str(market.get("result", "")).lower()

                if status == "finalized" and result in ("yes", "no"):
                    settlement_cents = 100 if result == "yes" else 0
                    realized = self.pnl_tracker.settle_position(market_id, settlement_cents)
                    self._settled_markets.add(market_id)

                    self.repository.log_event(
                        "paper_settlement",
                        market_id=market_id,
                        payload={
                            "result": result,
                            "settlement_cents": settlement_cents,
                            "realized_pnl_cents": realized,
                            "total_realized_cents": self.pnl_tracker.realized_cents,
                        },
                    )
                    self.repository.save_position_snapshot(
                        positions=self.pnl_tracker.position_snapshot(),
                        realized_cents=self.pnl_tracker.realized_cents,
                        unrealized_cents=self.pnl_tracker.unrealized_cents,
                    )
                    logger.info(
                        "paper_settlement",
                        market_id=market_id,
                        result=result,
                        settlement_cents=settlement_cents,
                        realized_pnl_cents=realized,
                    )
            except Exception as exc:
                logger.warning(
                    "paper_settlement_check_failed",
                    market_id=market_id,
                    error=str(exc),
                )
