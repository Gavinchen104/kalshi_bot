from __future__ import annotations

from typing import Any

import httpx

from src.data.kalshi_auth import KalshiAuth


class KalshiClient:
    def __init__(self, base_url: str, api_key: str, api_secret: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.auth = KalshiAuth(api_key_id=api_key, private_key_raw=api_secret)
        self._client = httpx.AsyncClient(timeout=timeout, base_url=self.base_url)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        return self.auth.build_headers(method=method, full_url_or_path=f"{self.base_url}{path}")

    async def get_balance(self) -> dict[str, Any]:
        path = "/trade-api/v2/portfolio/balance"
        r = await self._client.get(path, headers=self._headers("GET", path))
        r.raise_for_status()
        return r.json()

    async def get_market(self, ticker: str) -> dict[str, Any]:
        path = f"/trade-api/v2/markets/{ticker}"
        r = await self._client.get(path, headers=self._headers("GET", path))
        r.raise_for_status()
        return r.json()

    async def list_open_markets_for_series(
        self,
        series_ticker: str,
        max_pages: int = 10,
    ) -> list[str]:
        """Return ticker strings of currently-open markets within a specific series."""
        out: list[str] = []
        cursor: str | None = None
        for _ in range(max_pages):
            path = f"/trade-api/v2/markets?status=open&limit=1000&series_ticker={series_ticker}"
            if cursor:
                path += f"&cursor={cursor}"
            r = await self._client.get(path, headers=self._headers("GET", path))
            r.raise_for_status()
            payload = r.json()
            for m in payload.get("markets", []) or []:
                t = str(m.get("ticker", ""))
                if t:
                    out.append(t)
            cursor = payload.get("cursor") or None
            if not cursor:
                break
        return out

    async def list_open_btc_markets(self) -> list[str]:
        """Convenience: 15m (KXBTC) + daily (KXBTCD) open BTC markets."""
        # Two separate series_ticker queries — server-side filter is much faster
        # than client-side filtering against the global open-markets list.
        # Run in parallel to halve latency.
        import asyncio as _asyncio
        a, b = await _asyncio.gather(
            self.list_open_markets_for_series("KXBTC"),
            self.list_open_markets_for_series("KXBTCD"),
        )
        # De-dupe while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for t in (a + b):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = "/trade-api/v2/portfolio/orders"
        r = await self._client.post(path, json=payload, headers=self._headers("POST", path))
        r.raise_for_status()
        return r.json()
