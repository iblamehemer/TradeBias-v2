"""Synthetic validation — build this WITH the engine, not after.

You cannot trust the engine on real, messy data until you have proven it
recovers a bias you *planted*. This generates a trader who, by construction,
holds winners ~``win_hold`` days and losers ~``loss_hold`` days. The engine
should then recover ``disposition_ratio ≈ loss_hold / win_hold``.

Being able to say "I generate a trader with a planted bias and prove my detector
recovers it" is a genuinely sophisticated claim — lead with it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_trader(n: int = 200, win_hold: int = 3, loss_hold: int = 20,
                              win_rate: float = 0.45, seed: int = 42) -> pd.DataFrame:
    """A trader with a KNOWN disposition effect.

    Winners are sold after ~``win_hold`` days, losers after ~``loss_hold`` days,
    so the engine should recover ``disposition_ratio ≈ loss_hold / win_hold``.

    Returns a normalized tradebook: columns
    ['symbol', 'trade_date', 'side', 'quantity', 'price'].
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    start = date(2024, 1, 1)

    # Each round-trip gets its OWN symbol. This is a validation harness, not a
    # realism exercise: unique symbols guarantee that FIFO matches each planted
    # buy to its planted sell, so the engine recovers the bias we injected
    # instead of blending holding periods across re-used tickers.
    for k in range(n):
        sym = f"SYN{k:04d}"
        entry = start + timedelta(days=int(rng.integers(0, 300)))
        entry_price = float(rng.uniform(100, 1000))
        is_winner = rng.random() < win_rate  # win rate < 50% is realistic

        if is_winner:
            hold = max(1, int(rng.normal(win_hold, 1)))
            exit_price = entry_price * (1 + rng.uniform(0.02, 0.15))
        else:
            hold = max(1, int(rng.normal(loss_hold, 4)))
            exit_price = entry_price * (1 - rng.uniform(0.02, 0.15))

        qty = float(rng.integers(1, 50))
        rows.append({"symbol": sym, "trade_date": entry,
                     "side": "BUY", "quantity": qty, "price": round(entry_price, 2)})
        rows.append({"symbol": sym, "trade_date": entry + timedelta(days=hold),
                     "side": "SELL", "quantity": qty, "price": round(exit_price, 2)})

    return (pd.DataFrame(rows)
            .sort_values("trade_date")
            .reset_index(drop=True))


def price_anchor_from_trades(trades: pd.DataFrame) -> dict[str, tuple[date, float]]:
    """Build a price anchor (symbol -> (first_date, first_price)) for the
    deterministic synthetic price source, so PGR/PLR can be tested offline.
    """
    anchor: dict[str, tuple[date, float]] = {}
    first = (trades.sort_values("trade_date")
             .groupby("symbol", as_index=False)
             .first())
    for r in first.itertuples(index=False):
        anchor[r.symbol] = (r.trade_date, float(r.price))
    return anchor


def generate_disposition_trader_with_prices(
    n_symbols: int = 140,
    horizon: int = 60,
    p_sell_win: float = 0.45,
    p_sell_loss: float = 0.04,
    mu: float = 0.0004,
    sigma: float = 0.02,
    seed: int = 11,
):
    """A behaviorally-driven trader on a CONSISTENT price path.

    Unlike ``generate_synthetic_trader`` (which plants holding periods directly),
    this simulates an actual daily price path per symbol and lets a *disposition
    rule* drive the sell decision:

        * while a position is UP, sell it with high daily probability
          (``p_sell_win``)  -> winners are realized fast;
        * while it is DOWN, sell it with low daily probability
          (``p_sell_loss``) -> losers are clung to.

    Because the realized P&L and the paper marks both come from the SAME path,
    this lets us validate the canonical Odean PGR/PLR measure offline — not just
    the holding-period proxy. The same data should show:

        disposition_ratio > 1   (losers held longer)   and   PGR - PLR > 0.

    Returns
    -------
    (trades, price_on)
        trades   : normalized tradebook DataFrame.
        price_on : deterministic ``price_on(symbol, date) -> float | None`` built
                   from the simulated paths (None outside a symbol's window).
    """
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows: list[dict] = []
    paths: dict[str, pd.Series] = {}

    for k in range(n_symbols):
        sym = f"SIM{k:04d}"
        entry = start + timedelta(days=int(rng.integers(0, 120)))
        p0 = float(rng.uniform(100, 1000))

        # Simulate the daily close path from entry.
        prices = np.empty(horizon + 1)
        prices[0] = p0
        rets = rng.normal(mu, sigma, horizon)
        for d in range(1, horizon + 1):
            prices[d] = prices[d - 1] * (1.0 + rets[d - 1])

        # Disposition-driven exit: sell winners fast, hold losers.
        exit_day = horizon  # force-close at the horizon if never triggered
        for d in range(1, horizon + 1):
            up = prices[d] > p0
            prob = p_sell_win if up else p_sell_loss
            if rng.random() < prob:
                exit_day = d
                break

        # Record the full path (for the price source) and the two trades.
        idx = pd.to_datetime([entry + timedelta(days=d) for d in range(horizon + 1)])
        paths[sym] = pd.Series(prices, index=idx).sort_index()

        qty = float(rng.integers(1, 50))
        rows.append({"symbol": sym, "trade_date": entry,
                     "side": "BUY", "quantity": qty, "price": round(float(p0), 2)})
        rows.append({"symbol": sym, "trade_date": entry + timedelta(days=exit_day),
                     "side": "SELL", "quantity": qty,
                     "price": round(float(prices[exit_day]), 2)})

    trades = (pd.DataFrame(rows)
              .sort_values("trade_date")
              .reset_index(drop=True))

    def price_on(symbol: str, when):
        s = paths.get(symbol)
        if s is None:
            return None
        ts = pd.Timestamp(when).normalize()
        pos = s.index.searchsorted(ts, side="right") - 1
        if pos < 0 or ts > s.index[-1]:
            return None
        return float(s.iloc[pos])

    return trades, price_on


if __name__ == "__main__":
    # Quick offline demo: generate, save a sample CSV.
    df = generate_synthetic_trader(n=300, win_hold=3, loss_hold=20)
    df.to_csv("sample_data/synthetic_tradebook.csv", index=False)
    print(f"Wrote {len(df)} rows to sample_data/synthetic_tradebook.csv")
