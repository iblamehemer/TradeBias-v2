"""Stage 3/6 — Disposition metrics (CORE).

The disposition effect = selling winners too early and clinging to losers.
Two complementary measures live here:

  * disposition_holding_period  (v1a)  — CSV only, the first real result.
        Fingerprint: losers are held longer than winners. Quantified and
        significance-tested with Mann-Whitney U.

  * disposition_pgr_plr         (v1b)  — Odean (1998), needs a price source.
        PGR = RG / (RG + PG),  PLR = RL / (RL + PL),  spread = PGR - PLR.
        The price source is a pluggable callable so synthetic prices feed it in
        tests and yfinance feeds it in production, with no change to this code.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PriceFn = Callable[[str, object], Optional[float]]  # (symbol, date) -> price | None

_EPS = 1e-9
_MIN_PER_GROUP = 5  # need >=5 winners AND >=5 losers to judge


# ---------------------------------------------------------------------------
# v1a — holding-period method (no external data)
# ---------------------------------------------------------------------------
def disposition_holding_period(rt: pd.DataFrame) -> dict:
    """Holding-period disposition test on FIFO round-trips.

    Returns a dict with the averages, the disposition ratio
    (avg loser hold / avg winner hold; > 1 => disposition effect), the
    one-sided Mann-Whitney p-value (losers held stochastically longer), an
    effect size (rank-biserial correlation), and a boolean verdict.
    """
    if rt is None or rt.empty:
        return {"status": "insufficient_data", "n_winners": 0, "n_losers": 0}

    winners = rt[rt["is_winner"]]
    losers = rt[~rt["is_winner"]]

    if len(winners) < _MIN_PER_GROUP or len(losers) < _MIN_PER_GROUP:
        return {
            "status": "insufficient_data",
            "n_winners": int(len(winners)),
            "n_losers": int(len(losers)),
            "min_required": _MIN_PER_GROUP,
        }

    win_hold = winners["holding_days"].astype(float)
    loss_hold = losers["holding_days"].astype(float)
    avg_win, avg_loss = win_hold.mean(), loss_hold.mean()
    ratio = avg_loss / avg_win if avg_win > 0 else float("inf")

    # One-sided Mann-Whitney U: are losers held longer than winners?
    # Non-parametric on purpose — holding periods are heavily right-skewed, so a
    # t-test's normality assumption would be inappropriate.
    u_stat, p = mannwhitneyu(loss_hold, win_hold, alternative="greater")

    # Rank-biserial effect size from U (0 = no effect, 1 = complete separation).
    n1, n2 = len(loss_hold), len(win_hold)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)
    # For alternative="greater", losers-longer corresponds to negative rb above;
    # flip sign so a positive value means "losers held longer".
    effect_size = -rank_biserial

    return {
        "status": "ok",
        "n_winners": int(n1 := len(winners)),
        "n_losers": int(len(losers)),
        "avg_hold_winners": round(float(avg_win), 1),
        "avg_hold_losers": round(float(avg_loss), 1),
        "median_hold_winners": float(winners["holding_days"].median()),
        "median_hold_losers": float(losers["holding_days"].median()),
        "disposition_ratio": round(float(ratio), 2),     # > 1 => disposition effect
        "p_value": round(float(p), 4),                   # < 0.05 => significant
        "effect_size": round(float(effect_size), 3),     # rank-biserial, signed
        "u_statistic": float(u_stat),
        "verdict": bool(ratio > 1 and p < 0.05),
    }


# ---------------------------------------------------------------------------
# v1b — PGR / PLR, Odean's method (needs a price source)
# ---------------------------------------------------------------------------
def disposition_pgr_plr(trades: pd.DataFrame, price_on: PriceFn) -> dict:
    """Proportion of Gains/Losses Realized (Odean, 1998).

    On every day the trader sells something, classify every position:
        RG realized gain   — sold that day at a profit
        RL realized loss   — sold that day at a loss
        PG paper gain      — still held, currently up vs cost
        PL paper loss      — still held, currently down vs cost

    Modeling choices (documented, defensible):
      * Realized result is tallied per *symbol per selling day* (sum of the
        FIFO-matched lots sold that day), so FIFO lot-splitting can't inflate
        counts.
      * A still-held symbol is marked against that day's CLOSE via ``price_on``;
        held value uses the weighted-average cost of the remaining open lots.
      * If ``price_on`` returns None (e.g. an option/future yfinance can't
        price), that held position is skipped from the paper tally and counted
        as ``unpriced_paper``. Realized results never need prices.

    Returns PGR, PLR, the disposition spread, the raw RG/RL/PG/PL totals, and
    diagnostics. ``status='insufficient_data'`` if there is nothing to mark.
    """
    if trades is None or trades.empty:
        return {"status": "insufficient_data"}

    t = trades.copy()
    t["_side_rank"] = (t["side"] == "SELL").astype(int)
    t = t.sort_values(["trade_date", "_side_rank"], kind="mergesort").reset_index(drop=True)

    open_lots: dict[str, deque] = defaultdict(deque)  # symbol -> deque[[qty, price]]
    rg = rl = pg = pl = 0
    selling_days = 0
    unpriced_paper = 0

    for tdate, day in t.groupby("trade_date", sort=True):
        day_has_sell = (day["side"] == "SELL").any()

        realized_pnl_by_symbol: dict[str, float] = defaultdict(float)

        # Process the day's trades in (buy-before-sell) order.
        for row in day.itertuples(index=False):
            sym, side = row.symbol, row.side
            qty, price = float(row.quantity), float(row.price)
            if side == "BUY":
                open_lots[sym].append([qty, price])
            else:  # SELL — FIFO match, accumulate realized P&L for the symbol
                remaining = qty
                while remaining > _EPS and open_lots[sym]:
                    lot = open_lots[sym][0]
                    matched = min(remaining, lot[0])
                    realized_pnl_by_symbol[sym] += (price - lot[1]) * matched
                    lot[0] -= matched
                    remaining -= matched
                    if lot[0] <= _EPS:
                        open_lots[sym].popleft()
                # leftover 'remaining' is a short-to-open; ignored in v1 (long-only)

        if not day_has_sell:
            continue
        selling_days += 1

        # Tally realized gains/losses for this selling day (per symbol).
        for sym, pnl in realized_pnl_by_symbol.items():
            if pnl > _EPS:
                rg += 1
            elif pnl < -_EPS:
                rl += 1

        # Tally paper gains/losses across all still-open positions, marked to
        # this day's close.
        for sym, dq in open_lots.items():
            held_qty = sum(lot[0] for lot in dq)
            if held_qty <= _EPS:
                continue
            mkt = price_on(sym, tdate)
            if mkt is None or (isinstance(mkt, float) and np.isnan(mkt)):
                unpriced_paper += 1
                continue
            cost = sum(lot[0] * lot[1] for lot in dq) / held_qty  # wtd-avg cost
            if mkt > cost * (1 + _EPS):
                pg += 1
            elif mkt < cost * (1 - _EPS):
                pl += 1

    realized = rg + rl
    paper = pg + pl
    if (rg + pg) == 0 or (rl + pl) == 0 or realized == 0:
        return {
            "status": "insufficient_data",
            "rg": rg, "rl": rl, "pg": pg, "pl": pl,
            "selling_days": selling_days, "unpriced_paper": unpriced_paper,
        }

    pgr = rg / (rg + pg)
    plr = rl / (rl + pl)
    spread = pgr - plr

    return {
        "status": "ok",
        "pgr": round(pgr, 4),
        "plr": round(plr, 4),
        "spread": round(spread, 4),      # > 0 => disposition effect
        "rg": rg, "rl": rl, "pg": pg, "pl": pl,
        "selling_days": selling_days,
        "unpriced_paper": unpriced_paper,
        "verdict": bool(spread > 0),
    }
