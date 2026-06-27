"""FIFO position-engine tests. The engine is the keystone — test it hard."""

from datetime import date

import pandas as pd

from engine.positions import reconstruct_round_trips


def _t(symbol, d, side, qty, price):
    return {"symbol": symbol, "trade_date": date(2024, 1, d),
            "side": side, "quantity": qty, "price": price}


def test_simple_round_trip():
    df = pd.DataFrame([
        _t("RELIANCE", 1, "BUY", 10, 100.0),
        _t("RELIANCE", 6, "SELL", 10, 110.0),
    ])
    rt = reconstruct_round_trips(df)
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["holding_days"] == 5
    assert row["pnl"] == 100.0          # (110-100)*10
    assert bool(row["is_winner"]) is True
    assert abs(row["pnl_pct"] - 10.0) < 1e-9


def test_partial_fill_one_buy_two_sells():
    df = pd.DataFrame([
        _t("TCS", 1, "BUY", 10, 100.0),
        _t("TCS", 3, "SELL", 4, 120.0),
        _t("TCS", 5, "SELL", 6, 90.0),
    ])
    rt = reconstruct_round_trips(df)
    assert len(rt) == 2
    assert rt.iloc[0]["quantity"] == 4
    assert rt.iloc[0]["pnl"] == 80.0    # (120-100)*4
    assert rt.iloc[1]["quantity"] == 6
    assert rt.iloc[1]["pnl"] == -60.0   # (90-100)*6


def test_fifo_matches_oldest_lot_first():
    # Two buys at different prices, one sell. FIFO must consume the FIRST buy.
    df = pd.DataFrame([
        _t("INFY", 1, "BUY", 5, 100.0),   # oldest
        _t("INFY", 2, "BUY", 5, 200.0),   # newer
        _t("INFY", 9, "SELL", 5, 150.0),
    ])
    rt = reconstruct_round_trips(df)
    assert len(rt) == 1
    # Matched against the 100 lot, not the 200 lot.
    assert rt.iloc[0]["entry_price"] == 100.0
    assert rt.iloc[0]["pnl"] == 250.0   # (150-100)*5
    assert rt.iloc[0]["entry_date"] == date(2024, 1, 1)


def test_same_day_buy_then_sell_is_matched():
    # Naive date-only sorting can place the SELL before the BUY and miss this.
    df = pd.DataFrame([
        _t("HDFC", 4, "SELL", 10, 105.0),
        _t("HDFC", 4, "BUY", 10, 100.0),
    ])
    rt = reconstruct_round_trips(df)
    assert len(rt) == 1
    assert rt.iloc[0]["holding_days"] == 0
    assert rt.iloc[0]["pnl"] == 50.0


def test_short_to_open_is_skipped_and_counted():
    df = pd.DataFrame([
        _t("WIPRO", 1, "SELL", 10, 100.0),   # no inventory -> short-to-open
        _t("WIPRO", 2, "BUY", 4, 90.0),
        _t("WIPRO", 3, "SELL", 4, 95.0),     # long round-trip
    ])
    rt = reconstruct_round_trips(df)
    # Only the genuine long round-trip is recorded.
    assert len(rt) == 1
    assert rt.iloc[0]["pnl"] == 20.0         # (95-90)*4
    assert rt.attrs["skipped_short_qty"] == 10.0
    assert rt.attrs["skipped_short_count"] >= 1


def test_open_position_tracked():
    df = pd.DataFrame([
        _t("SBIN", 1, "BUY", 10, 100.0),
        _t("SBIN", 5, "SELL", 4, 110.0),
    ])
    rt = reconstruct_round_trips(df)
    assert len(rt) == 1
    assert rt.attrs["open_qty"] == 6.0       # 6 shares never sold


def test_empty_input():
    rt = reconstruct_round_trips(pd.DataFrame(
        columns=["symbol", "trade_date", "side", "quantity", "price"]))
    assert len(rt) == 0
    assert rt.attrs["skipped_short_qty"] == 0.0
