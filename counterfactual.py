"""Stage 5/6 — The rupee counterfactual (v1b).

The line that makes people lean in:
    "Holding your winners with the same patience you gave your losers would
     have earned you roughly ₹X more."

It needs forward prices (what each winner did *after* you sold it), so it is a
v1b feature that depends on ``price_on``. Reported as an ESTIMATE with the
assumption stated plainly — the honesty is part of what makes it credible.

Method
------
1. Take the trader's MEDIAN LOSER holding period as "the patience they clearly
   had."
2. For each WINNING round-trip held for fewer days than that, re-price it as if
   it had been exited ``(median_loser_hold - this_winner_hold)`` calendar days
   later, using ``price_on``.
3. delta = (counterfactual_exit_price - actual_exit_price) * quantity.
4. Headline = sum of POSITIVE deltas (trades where the extra patience actually
   helped). We also report the NET (deltas of both signs) so the number isn't
   cherry-picked.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

PriceFn = Callable[[str, object], Optional[float]]


def rupee_counterfactual(rt: pd.DataFrame, price_on: PriceFn) -> dict:
    """Estimate the rupees left on the table by selling winners early.

    Parameters
    ----------
    rt : closed round-trips from ``reconstruct_round_trips``.
    price_on : ``price_on(symbol, date) -> price | None``.

    Returns a dict with the headline missed-upside (positive deltas), the net
    figure (all deltas), per-trade detail, and coverage diagnostics.
    """
    if rt is None or rt.empty:
        return {"status": "insufficient_data"}

    winners = rt[rt["is_winner"]].copy()
    losers = rt[~rt["is_winner"]]
    if winners.empty or losers.empty:
        return {"status": "insufficient_data",
                "n_winners": int(len(winners)), "n_losers": int(len(losers))}

    median_loser_hold = float(losers["holding_days"].median())

    rows = []
    priced = 0
    unpriced = 0
    for r in winners.itertuples(index=False):
        extra_days = int(round(median_loser_hold - r.holding_days))
        if extra_days <= 0:
            continue  # already held at least as long as the median loser
        cf_date = r.exit_date + timedelta(days=extra_days)
        cf_price = price_on(r.symbol, cf_date)
        if cf_price is None or (isinstance(cf_price, float) and np.isnan(cf_price)):
            unpriced += 1
            continue
        priced += 1
        delta = (float(cf_price) - float(r.exit_price)) * float(r.quantity)
        rows.append({
            "symbol": r.symbol,
            "entry_date": r.entry_date,
            "exit_date": r.exit_date,
            "actual_hold": int(r.holding_days),
            "counterfactual_hold": int(r.holding_days + extra_days),
            "actual_exit_price": round(float(r.exit_price), 2),
            "counterfactual_exit_price": round(float(cf_price), 2),
            "quantity": float(r.quantity),
            "delta": round(delta, 2),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "status": "insufficient_data",
            "median_loser_hold": median_loser_hold,
            "priced": priced, "unpriced": unpriced,
            "note": "No winners could be re-priced (likely derivatives or no price data).",
        }

    missed_upside = float(detail.loc[detail["delta"] > 0, "delta"].sum())
    net_delta = float(detail["delta"].sum())

    return {
        "status": "ok",
        "median_loser_hold": round(median_loser_hold, 1),
        "missed_upside": round(missed_upside, 2),   # headline: positive deltas only
        "net_delta": round(net_delta, 2),           # honesty: both signs
        "n_winners_repriced": int(len(detail)),
        "n_winners_helped": int((detail["delta"] > 0).sum()),
        "priced": priced,
        "unpriced": unpriced,
        "detail": detail.sort_values("delta", ascending=False).reset_index(drop=True),
        "assumption": (
            f"Each winner re-exited so its holding period matches the median "
            f"loser hold ({median_loser_hold:.0f} days). Estimate only; ignores "
            f"charges, taxes, and capital that would have been tied up."
        ),
    }
