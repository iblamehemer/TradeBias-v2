"""Stage 6 — Price source (v1b).

A pluggable historical-close provider. Both the PGR/PLR metric and the rupee
counterfactual depend on a callable ``price_on(symbol, date) -> float | None``.
This module builds that callable from yfinance for production; tests inject a
synthetic callable instead, so the metric code never changes.

NSE equities map to yfinance via the ``.NS`` suffix (e.g. RELIANCE -> RELIANCE.NS).
Derivatives (NIFTY24JUL..., option strikes) have no yfinance series; the lookup
returns None for them and the caller surfaces the unpriced count honestly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

PriceFn = Callable[[str, object], Optional[float]]

# Cheap heuristic: treat a symbol as an F&O contract (unpriceable via the equity
# feed) if it carries an expiry/strike fingerprint. Conservative on purpose.
_DERIV_HINTS = ("FUT", "CE", "PE")


def looks_like_derivative(symbol: str) -> bool:
    s = symbol.upper()
    if any(s.endswith(h) for h in ("CE", "PE")):
        return True
    if "FUT" in s:
        return True
    # NIFTY/BANKNIFTY weekly/monthly contracts usually embed a year+month token.
    if any(idx in s for idx in ("NIFTY", "BANKNIFTY", "FINNIFTY")) and any(ch.isdigit() for ch in s):
        return True
    return False


def to_yahoo_symbol(symbol: str, suffix: str = ".NS") -> str:
    """Map an internal symbol to a Yahoo Finance ticker (NSE by default)."""
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}{suffix}"


class YFinancePriceSource:
    """Bulk-downloads daily closes once per symbol, then answers as-of lookups.

    For a requested date it returns that day's close, or the most recent prior
    trading day's close (markets are shut on weekends/holidays). Results and
    failures are cached in-process so repeated lookups are free.
    """

    def __init__(self, suffix: str = ".NS", pad_days: int = 10):
        self.suffix = suffix
        self.pad_days = pad_days  # fetch a little extra history for as-of lookups
        self._series: dict[str, Optional[pd.Series]] = {}  # symbol -> close series | None

    # -- internal ---------------------------------------------------------
    def _load(self, symbol: str, start: date, end: date) -> Optional[pd.Series]:
        if symbol in self._series:
            return self._series[symbol]

        if looks_like_derivative(symbol):
            self._series[symbol] = None
            return None

        try:
            import yfinance as yf
        except Exception:  # noqa: BLE001
            self._series[symbol] = None
            return None

        ystart = (pd.Timestamp(start) - pd.Timedelta(days=self.pad_days)).date()
        yend = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
        try:
            df = yf.download(
                to_yahoo_symbol(symbol, self.suffix),
                start=str(ystart), end=str(yend),
                progress=False, auto_adjust=True, threads=False,
            )
        except Exception:  # noqa: BLE001 - network/symbol failures are non-fatal
            self._series[symbol] = None
            return None

        if df is None or df.empty or "Close" not in df.columns:
            self._series[symbol] = None
            return None

        close = df["Close"]
        if isinstance(close, pd.DataFrame):       # multiindex safety
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).normalize()
        close = close.dropna().sort_index()
        self._series[symbol] = close if not close.empty else None
        return self._series[symbol]

    # -- public callable --------------------------------------------------
    def price_on(self, symbol: str, when, start: date, end: date) -> Optional[float]:
        series = self._load(symbol, start, end)
        if series is None or series.empty:
            return None
        ts = pd.Timestamp(when).normalize()
        # as-of: latest close at or before `when`
        idx = series.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return float(series.iloc[idx])

    def bind(self, start: date, end: date) -> PriceFn:
        """Return a 2-arg ``price_on(symbol, date)`` closed over the date window."""
        return lambda symbol, when: self.price_on(symbol, when, start, end)


def make_price_source(start: date, end: date, suffix: str = ".NS") -> PriceFn:
    """Convenience: a ready-to-use ``price_on(symbol, date)`` backed by yfinance."""
    return YFinancePriceSource(suffix=suffix).bind(start, end)


# ---------------------------------------------------------------------------
# Deterministic synthetic price source (for tests / offline demos)
# ---------------------------------------------------------------------------
def make_synthetic_price_source(anchor: dict[str, tuple[date, float]],
                                drift_per_day: float = 0.0,
                                seed: int = 0) -> PriceFn:
    """Build a deterministic price function for testing PGR/PLR offline.

    ``anchor`` maps symbol -> (anchor_date, anchor_price). Price grows by a fixed
    daily drift from the anchor, plus a small deterministic per-symbol wiggle.
    No randomness across calls — same (symbol, date) always returns the same
    number, which is what the metric needs.
    """
    rng = np.random.default_rng(seed)
    phase = {sym: float(rng.uniform(0, 6.28)) for sym in anchor}

    def price_on(symbol: str, when) -> Optional[float]:
        if symbol not in anchor:
            return None
        a_date, a_price = anchor[symbol]
        days = (pd.Timestamp(when).date() - a_date).days
        wiggle = 1.0 + 0.01 * np.sin(days / 5.0 + phase[symbol])
        return float(a_price * (1.0 + drift_per_day) ** days * wiggle)

    return price_on
