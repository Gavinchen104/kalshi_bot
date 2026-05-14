CREATE TABLE IF NOT EXISTS market_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    bid_cents INTEGER, ask_cents INTEGER,
    bid_size INTEGER, ask_size INTEGER,
    last_trade_cents INTEGER,
    updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_market_state_market_time ON market_state(market_id, id);

CREATE TABLE IF NOT EXISTS prob_estimate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    prob REAL NOT NULL,
    horizon_seconds REAL NOT NULL,
    spot_usd REAL NOT NULL,
    vol_annualized REAL NOT NULL,
    source TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    -- Captured for outcome-pairing at settlement.
    market_yes_ask_cents INTEGER, market_yes_bid_cents INTEGER,
    market_mid_cents INTEGER
);
CREATE INDEX IF NOT EXISTS ix_prob_estimate_market_time ON prob_estimate(market_id, id);

CREATE TABLE IF NOT EXISTS signal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    our_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    edge REAL NOT NULL,
    fair_price_cents INTEGER NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS coinbase_candle (
    timestamp_ms INTEGER PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL, volume REAL
);

CREATE TABLE IF NOT EXISTS paper_order (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    fill_price_cents INTEGER,
    fee_cents INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pnl_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_cents INTEGER NOT NULL,
    realized_cents INTEGER NOT NULL,
    unrealized_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS position_snapshot (
    market_id TEXT PRIMARY KEY,
    net_quantity INTEGER NOT NULL,
    avg_entry_cents REAL NOT NULL,
    realized_pnl_cents INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calibration_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_size INTEGER NOT NULL,
    brier_score REAL,
    log_loss REAL,
    n_samples INTEGER NOT NULL,
    bin_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
