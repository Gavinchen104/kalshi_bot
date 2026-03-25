from collections import defaultdict


class ExposureBook:
    def __init__(self) -> None:
        self._positions: dict[str, int] = defaultdict(int)

    def get(self, market_id: str) -> int:
        return self._positions[market_id]

    def apply_fill(self, market_id: str, signed_qty: int) -> None:
        self._positions[market_id] += signed_qty

    @property
    def gross(self) -> int:
        return sum(abs(v) for v in self._positions.values())

