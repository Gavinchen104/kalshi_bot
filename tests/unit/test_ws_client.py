import pytest

from src.api.ws_client import KalshiWSClient


@pytest.mark.asyncio
async def test_mock_stream_reads_events(tmp_path) -> None:
    file_path = tmp_path / "stream.jsonl"
    file_path.write_text(
        '{"data":{"market_ticker":"MKT-1","yes_bid":48,"yes_ask":52,"yes_bid_size":10,"yes_ask_size":12,"last_price":50}}\n',
        encoding="utf-8",
    )
    ws = KalshiWSClient(
        "wss://example.test",
        api_key="test-key",
        api_secret="test-secret",
        market_ids=["MKT-1"],
    )

    received = []
    async for item in ws.stream_from_jsonl(str(file_path), loop_forever=False):
        received.append(item)

    assert len(received) == 1
    assert received[0].market_id == "MKT-1"


def test_parse_live_ticker_message_shape() -> None:
    ws = KalshiWSClient(
        "wss://example.test",
        api_key="test-key",
        api_secret="test-secret",
        market_ids=[],
    )
    raw = (
        '{"type":"ticker","sid":1,"msg":{"market_ticker":"KXBTCD-TEST",'
        '"yes_bid_dollars":"0.98","yes_ask_dollars":"1.00","price_dollars":"0.97",'
        '"yes_bid_size_fp":"400.00","yes_ask_size_fp":"0.00"}}'
    )
    state = ws._parse_message(raw)
    assert state is not None
    assert state.market_id == "KXBTCD-TEST"
    assert state.bid_cents == 98
    assert state.ask_cents == 100

