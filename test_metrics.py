"""Metric tests. The headline claim: a trader with a PLANTED disposition effect
is recovered by the detector, and a clean trader is not flagged."""

from datetime import date

import pandas as pd

from engine.metrics import disposition_holding_period, disposition_pgr_plr
from engine.positions import reconstruct_round_trips
from engine.pricing import make_synthetic_price_source
from synthetic.generate import (
    generate_synthetic_trader,
    generate_disposition_trader_with_prices,
    price_anchor_from_trades,
)


# ---------------------------------------------------------------------------
# v1a — holding-period method
# ---------------------------------------------------------------------------
def test_engine_recovers_known_disposition():
    df = generate_synthetic_trader(n=400, win_hold=3, loss_hold=20, seed=42)
    rt = reconstruct_round_trips(df)
    result = disposition_holding_period(rt)
    assert result["status"] == "ok"
    assert result["verdict"] is True
    assert 5 < result["disposition_ratio"] < 9     # expect ≈ 20/3 ≈ 6.7
    assert result["p_value"] < 0.05
    assert result["effect_size"] > 0               # losers held longer


def test_no_false_positive_on_unbiased_trader():
    # Winners and losers held for the SAME duration -> no disposition effect.
    df = generate_synthetic_trader(n=400, win_hold=10, loss_hold=10, seed=7)
    rt = reconstruct_round_trips(df)
    result = disposition_holding_period(rt)
    assert result["status"] == "ok"
    assert result["verdict"] is False              # ratio ≈ 1, not significant


def test_insufficient_data_guard():
    df = pd.DataFrame([
        {"symbol": "X", "trade_date": date(2024, 1, 1), "side": "BUY",
         "quantity": 1, "price": 100.0},
        {"symbol": "X", "trade_date": date(2024, 1, 3), "side": "SELL",
         "quantity": 1, "price": 110.0},
    ])
    rt = reconstruct_round_trips(df)
    result = disposition_holding_period(rt)
    assert result["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# v1b — PGR/PLR with a deterministic synthetic price source (no network)
# ---------------------------------------------------------------------------
def test_pgr_plr_runs_with_synthetic_prices():
    df = generate_synthetic_trader(n=400, win_hold=3, loss_hold=20, seed=3)
    anchor = price_anchor_from_trades(df)
    # Gentle upward drift so held positions get marked to plausible prices.
    price_on = make_synthetic_price_source(anchor, drift_per_day=0.001, seed=1)
    result = disposition_pgr_plr(df, price_on)
    assert result["status"] == "ok"
    assert 0.0 <= result["pgr"] <= 1.0
    assert 0.0 <= result["plr"] <= 1.0
    assert result["rg"] > 0 and result["rl"] > 0
    # spread is defined; sign depends on the price path, so we only assert bounds
    assert -1.0 <= result["spread"] <= 1.0


def test_both_metrics_recover_bias_from_consistent_simulator():
    """The strong claim: one price-consistent behavioral process produces a
    disposition effect that BOTH the holding-period proxy AND the canonical
    Odean PGR/PLR measure recover."""
    trades, price_on = generate_disposition_trader_with_prices(seed=11)
    rt = reconstruct_round_trips(trades)

    hp = disposition_holding_period(rt)
    assert hp["status"] == "ok"
    assert hp["disposition_ratio"] > 1.0      # losers held longer
    assert hp["p_value"] < 0.05
    assert hp["verdict"] is True

    odean = disposition_pgr_plr(trades, price_on)
    assert odean["status"] == "ok"
    assert odean["pgr"] > odean["plr"]        # gains realized faster than losses
    assert odean["spread"] > 0.0
    assert odean["verdict"] is True
