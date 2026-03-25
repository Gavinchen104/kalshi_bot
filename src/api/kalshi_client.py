from __future__ import annotations

from typing import Any

import httpx

from src.api.auth import KalshiAuth

class KalshiClient:
    """Minimal REST client wrapper. Extend per official endpoint schema."""

    def __init__(self, base_url: str, api_key: str, api_secret: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.auth = KalshiAuth(api_key_id=api_key, private_key_raw=api_secret)
        self._client = httpx.AsyncClient(timeout=timeout, base_url=self.base_url)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        full_url = f"{self.base_url}{path}"
        return self.auth.build_headers(method=method, full_url_or_path=full_url)

    async def health(self) -> dict[str, Any]:
        # Placeholder endpoint. Update to a valid endpoint from docs.
        path = "/trade-api/v2/exchange/status"
        resp = await self._client.get(path, headers=self._headers("GET", path))
        resp.raise_for_status()
        return resp.json()

    async def get_balance(self) -> dict[str, Any]:
        path = "/trade-api/v2/portfolio/balance"
        resp = await self._client.get(path, headers=self._headers("GET", path))
        resp.raise_for_status()
        return resp.json()

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = "/trade-api/v2/portfolio/orders"
        resp = await self._client.post(path, json=payload, headers=self._headers("POST", path))
        resp.raise_for_status()
        return resp.json()

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        resp = await self._client.delete(path, headers=self._headers("DELETE", path))
        resp.raise_for_status()
        return resp.json()

    async def get_order(self, order_id: str) -> dict[str, Any]:
        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        resp = await self._client.get(path, headers=self._headers("GET", path))
        resp.raise_for_status()
        return resp.json()

    async def get_market(self, ticker: str) -> dict[str, Any]:
        path = f"/trade-api/v2/markets/{ticker}"
        resp = await self._client.get(path, headers=self._headers("GET", path))
        resp.raise_for_status()
        return resp.json()

