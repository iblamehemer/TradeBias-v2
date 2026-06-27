"""Stage 2 — Position engine (CORE).

FIFO round-trip reconstruction. Every downstream metric is computed off the
output of this module, so it is the keystone: get it right, validate it against
the synthetic generator, *then* trust it on real data.

FIFO matching: a sell is matched against the oldest open buy lots first. This
handles partial fills and scaling in/out correctly, which a naive average-price
approach does not.

v1 scope: LONG round-trips only (buy-then-sell). A sell with no matching long
inventory is a short-to-open; we count and skip these in v1 and surface the
count to the user. Shorts are a documented v2 extension (track *net position*
instead of long-only inventory — see README).
"""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

_EPS = 1e-9


def _ordered(trades: pd.DataFrame) -> pd.DataFrame:
    """Sort trades into execution order.

    Critical correctness detail the naive version misses: on a single day a BUY
    must be processed before a SELL, otherwise a same-day buy→sell round-trip
    can't be matched. We sort by (date, [time if present], side) with BUY ranked
    before SELL, using a stable sort so equal keys keep input order.
    """
    t = trades.copy()
    t["_side_rank"] = (t["side"] == "SELL").astype(int)  # BUY=0 before SELL=1

    sort_cols = ["trade_date"]
    if "trade_time" in t.columns and t["trade_time"].notna().all():
        # Build a sortable seconds-since-midnight when every row has a time.
        t["_secs"] = t["trade_time"].apply(
            lambda x: x.hour * 3600 + x.minute * 60 + x.second
        )
        sort_cols.append("_secs")
    sort_cols.append("_side_rank")

    t = t.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return t.drop(columns=[c for c in ("_side_rank", "_secs") if c in t.columns])


def reconstruct_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Turn a normalized trade log into FIFO-matched LONG round-trips.

    Parameters
    ----------
    trades : normalized DataFrame with columns
        ['symbol', 'trade_date', 'side', 'quantity', 'price'] (+ optional
        'trade_time').

    Returns
    -------
    DataFrame of closed round-trips with columns:
        symbol, entry_date, exit_date, holding_days, entry_price, exit_price,
        quantity, pnl, pnl_pct, is_winner

    Diagnostics are attached on ``df.attrs``:
        skipped_short_qty   total sell quantity with no long inventory (shorts)
        skipped_short_count number of such sell events
        open_qty            quantity still open at end of data (never sold)
    """
    cols = ["symbol", "entry_date", "exit_date", "holding_days", "entry_price",
            "exit_price", "quantity", "pnl", "pnl_pct", "is_winner"]

    if trades is None or trades.empty:
        empty = pd.DataFrame(columns=cols)
        empty.attrs.update(skipped_short_qty=0.0, skipped_short_count=0, open_qty=0.0)
        return empty

    t = _ordered(trades)

    open_lots: dict[str, deque] = defaultdict(deque)  # symbol -> deque[[qty, price, date]]
    round_trips: list[dict] = []
    skipped_short_qty = 0.0
    skipped_short_count = 0

    for row in t.itertuples(index=False):
        sym = row.symbol
        side = row.side
        qty = float(row.quantity)
        price = float(row.price)
        tdate = row.trade_date

        if side == "BUY":
            open_lots[sym].append([qty, price, tdate])
            continue

        # SELL: match FIFO against oldest open long lots.
        remaining = qty
        matched_any = False
        while remaining > _EPS and open_lots[sym]:
            lot = open_lots[sym][0]  # oldest lot
            matched = min(remaining, lot[0])
            pnl = (price - lot[1]) * matched
            round_trips.append({
                "symbol": sym,
                "entry_date": lot[2],
                "exit_date": tdate,
                "holding_days": (tdate - lot[2]).days,
                "entry_price": lot[1],
                "exit_price": price,
                "quantity": matched,
                "pnl": pnl,
                "pnl_pct": (price - lot[1]) / lot[1] * 100.0,
                "is_winner": pnl > 0,
            })
            lot[0] -= matched
            remaining -= matched
            matched_any = True
            if lot[0] <= _EPS:
                open_lots[sym].popleft()

        if remaining > _EPS:  # short-to-open (no long inventory left): skip in v1
            skipped_short_qty += remaining
            if not matched_any or remaining == qty:
                skipped_short_count += 1
            else:
                skipped_short_count += 1  # partial: the leftover opened a short

    open_qty = sum(lot[0] for dq in open_lots.values() for lot in dq)

    rt = pd.DataFrame(round_trips, columns=cols)
    rt.attrs["skipped_short_qty"] = round(skipped_short_qty, 4)
    rt.attrs["skipped_short_count"] = int(skipped_short_count)
    rt.attrs["open_qty"] = round(float(open_qty), 4)
    return rt
