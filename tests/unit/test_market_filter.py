from src.main import _is_allowed_market


def test_market_filter_allows_btc_by_symbol_token() -> None:
    assert _is_allowed_market("KXBTCD-2026APR01-T52000", allowlist=[], symbol_filters=["BTC"])
    assert not _is_allowed_market("KXETHD-2026APR01-T3000", allowlist=[], symbol_filters=["BTC"])


def test_market_filter_allows_explicit_allowlist() -> None:
    assert _is_allowed_market("KXBTCD-TEST", allowlist=["KXBTCD-TEST"], symbol_filters=[])
    assert not _is_allowed_market("KXBTCD-OTHER", allowlist=["KXBTCD-TEST"], symbol_filters=[])

