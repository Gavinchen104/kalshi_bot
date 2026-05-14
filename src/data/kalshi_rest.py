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

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = "/trade-api/v2/portfolio/orders"
        r = await self._client.post(path, json=payload, headers=self._headers("POST", path))
        r.raise_for_status()
        return r.json()
