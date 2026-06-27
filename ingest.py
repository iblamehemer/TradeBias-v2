"""Stage 1 — Ingest.

Load a broker CSV, map its columns onto the internal normalized schema, and
clean the values. Brokers all differ, so columns are *mapped*, never hardcoded.

Internal normalized schema (the output of this module):

    symbol      str    e.g. RELIANCE, NIFTY24JUL...
    trade_date  date   parsed to datetime.date
    trade_time  time   optional in v1; used for intraday ordering when present
    side        str    'BUY' or 'SELL'
    quantity    float  always positive
    price       float  per-share / per-contract execution price
"""

from __future__ import annotations

import io
from datetime import date, time

import numpy as np
import pandas as pd


class IngestError(ValueError):
    """Raised when a CSV cannot be mapped onto the internal schema."""


# ---------------------------------------------------------------------------
# Broker presets
# ---------------------------------------------------------------------------
# Each preset maps an *internal* field -> the broker's exported column header.
# Headers below reflect common Indian retail tradebook exports. Real exports
# drift over time, so "custom" lets the user map manually in the UI.
BROKER_PRESETS: dict[str, dict[str, str]] = {
    "zerodha": {
        "symbol": "Symbol",
        "trade_date": "Trade Date",
        "trade_time": "Order Execution Time",
        "side": "Trade Type",
        "quantity": "Quantity",
        "price": "Price",
    },
    "groww": {
        "symbol": "Stock name",
        "trade_date": "Trade date",
        "side": "Type",
        "quantity": "Quantity",
        "price": "Price",
    },
    "upstox": {
        "symbol": "Company",
        "trade_date": "Date",
        "side": "Transaction Type",
        "quantity": "Quantity",
        "price": "Price",
    },
    "angelone": {
        "symbol": "Symbol",
        "trade_date": "Trade Date",
        "trade_time": "Trade Time",
        "side": "Buy/Sell",
        "quantity": "Quantity",
        "price": "Trade Price",
    },
    "dhan": {
        "symbol": "Name",
        "trade_date": "Date",
        "side": "Buy/Sell",
        "quantity": "Quantity",
        "price": "Trade Price",
    },
    "custom": {},  # user maps every field manually in the UI
}

# Required internal fields (trade_time is optional).
REQUIRED_FIELDS = ["symbol", "trade_date", "side", "quantity", "price"]

# How various brokers spell the buy/sell flag.
_BUY_TOKENS = {"buy", "b", "bought", "buy ", "1", "+1", "long"}
_SELL_TOKENS = {"sell", "s", "sold", "sell ", "-1", "short", "sl"}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def _clean_number(series: pd.Series) -> pd.Series:
    """Coerce a column of money/quantity strings to float.

    Handles thousands separators, currency symbols, stray spaces and blanks.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u20b9", "", regex=False)  # ₹
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.replace("\u20a8", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_side(series: pd.Series) -> pd.Series:
    """Map a column of buy/sell flags to canonical 'BUY' / 'SELL'."""
    s = series.astype(str).str.strip().str.lower()

    def _map(v: str) -> str | float:
        if v in _BUY_TOKENS or v.startswith("buy"):
            return "BUY"
        if v in _SELL_TOKENS or v.startswith("sell"):
            return "SELL"
        return np.nan

    return s.map(_map)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column. Indian exports are day-first (DD-MM-YYYY)."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.5:  # fall back to month-first if day-first failed
        alt = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if alt.isna().mean() < parsed.isna().mean():
            parsed = alt
    return parsed.dt.date


def _parse_times(series: pd.Series) -> pd.Series:
    """Best-effort parse of an execution-time column to datetime.time."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_frame(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Map a raw broker DataFrame onto the internal normalized schema.

    Parameters
    ----------
    raw : the broker's exported table, columns as exported.
    mapping : internal_field -> broker_column_header (a BROKER_PRESETS entry or
        a user-supplied custom mapping).

    Returns
    -------
    A clean DataFrame with columns [symbol, trade_date, side, quantity, price]
    (+ trade_time when available). Rows that cannot be parsed are dropped, and a
    short report is attached on ``df.attrs['ingest_report']``.
    """
    if raw is None or raw.empty:
        raise IngestError("The uploaded file has no rows.")

    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    missing = [
        f for f in REQUIRED_FIELDS
        if f not in mapping or mapping[f] not in raw.columns
    ]
    if missing:
        raise IngestError(
            "Could not find columns for: "
            + ", ".join(missing)
            + ". Available columns are: "
            + ", ".join(map(str, raw.columns))
            + ". Pick the right broker preset or map these fields manually."
        )

    out = pd.DataFrame()
    out["symbol"] = raw[mapping["symbol"]].astype(str).str.strip().str.upper()
    out["trade_date"] = _parse_dates(raw[mapping["trade_date"]])
    out["side"] = _normalize_side(raw[mapping["side"]])
    out["quantity"] = _clean_number(raw[mapping["quantity"]]).abs()
    out["price"] = _clean_number(raw[mapping["price"]])

    if mapping.get("trade_time") and mapping["trade_time"] in raw.columns:
        out["trade_time"] = _parse_times(raw[mapping["trade_time"]])

    rows_in = len(out)

    # Drop unusable rows and record why.
    bad_mask = (
        out["symbol"].isna()
        | (out["symbol"].str.len() == 0)
        | out["trade_date"].isna()
        | out["side"].isna()
        | out["quantity"].isna()
        | (out["quantity"] <= 0)
        | out["price"].isna()
        | (out["price"] <= 0)
    )
    dropped = int(bad_mask.sum())
    out = out.loc[~bad_mask].reset_index(drop=True)

    if out.empty:
        raise IngestError(
            "Every row failed validation after mapping. The column mapping is "
            "probably wrong — check that side/quantity/price point at real columns."
        )

    out.attrs["ingest_report"] = {
        "rows_in": rows_in,
        "rows_used": len(out),
        "rows_dropped": dropped,
        "n_symbols": out["symbol"].nunique(),
        "date_min": out["trade_date"].min(),
        "date_max": out["trade_date"].max(),
        "n_buys": int((out["side"] == "BUY").sum()),
        "n_sells": int((out["side"] == "SELL").sum()),
    }
    return out


def load_and_normalize(file, broker: str = "zerodha",
                       mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a CSV (path / bytes / file-like) and normalize it.

    ``mapping`` overrides the broker preset (used by the Custom mapper in the UI).
    """
    raw = _read_csv_any(file)
    if mapping is None:
        if broker not in BROKER_PRESETS:
            raise IngestError(f"Unknown broker preset '{broker}'.")
        mapping = BROKER_PRESETS[broker]
        if not mapping:
            raise IngestError(
                "The 'custom' preset needs a column mapping supplied by the UI."
            )
    return normalize_frame(raw, mapping)


def _read_csv_any(file) -> pd.DataFrame:
    """Read a CSV from a path, raw bytes, or a Streamlit UploadedFile."""
    try:
        if isinstance(file, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(file))
        if isinstance(file, str):
            return pd.read_csv(file)
        # Streamlit UploadedFile / any file-like
        if hasattr(file, "seek"):
            file.seek(0)
        return pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise IngestError(f"Could not parse the CSV: {exc}") from exc
