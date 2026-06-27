"""Behavioral Risk Scorecard — Streamlit UI.

Wires the deterministic engine (stages 1-6) into a demoable app.

One rule: the engine is the product. Every headline number is computed by the
engine from the tradebook. The optional AI debrief at the bottom only *describes*
those numbers — it never computes them. That distinction is exactly what a risk
professional looks for.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.ingest import load_and_normalize, normalize_frame, BROKER_PRESETS, IngestError, REQUIRED_FIELDS
from engine.positions import reconstruct_round_trips
from engine.metrics import disposition_holding_period, disposition_pgr_plr
from engine.counterfactual import rupee_counterfactual
from engine.pricing import YFinancePriceSource
from synthetic.generate import generate_disposition_trader_with_prices

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Behavioral Risk Scorecard", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1200px;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .verdict-bad   {background:#fef2f2; border:1px solid #fecaca; color:#991b1b;
                      padding:1rem 1.2rem; border-radius:12px; font-size:1.05rem;}
      .verdict-good  {background:#f0fdf4; border:1px solid #bbf7d0; color:#166534;
                      padding:1rem 1.2rem; border-radius:12px; font-size:1.05rem;}
      .source-note   {color:#64748b; font-size:0.85rem;}
      .pill {display:inline-block; background:#eff6ff; color:#1d4ed8; border-radius:999px;
             padding:2px 10px; font-size:0.78rem; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inr(x: float) -> str:
    """Format a number as Indian rupees with lakh/crore comma grouping."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    whole = int(round(x))
    s = str(whole)
    if len(s) <= 3:
        grouped = s
    else:
        head, last3 = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}₹{grouped}"


@st.cache_resource(show_spinner=False)
def get_price_source(start: date, end: date) -> YFinancePriceSource:
    """One yfinance source per (start, end), cached across reruns."""
    return YFinancePriceSource()


def ratio_gauge(ratio: float, verdict: bool) -> go.Figure:
    shown = min(ratio, 10.0)  # cap the needle for readability
    color = "#dc2626" if verdict else "#16a34a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=shown,
        number={"suffix": "×", "font": {"size": 40},
                "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 1], "color": "#dcfce7"},
                {"range": [1, 3], "color": "#fef9c3"},
                {"range": [3, 10], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#0f172a", "width": 3},
                          "thickness": 0.75, "value": 1},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def holding_boxplot(rt: pd.DataFrame) -> go.Figure:
    win = rt[rt["is_winner"]]["holding_days"]
    los = rt[~rt["is_winner"]]["holding_days"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=win, name="Winners", marker_color="#16a34a",
                         boxpoints="all", jitter=0.4, pointpos=0))
    fig.add_trace(go.Box(y=los, name="Losers", marker_color="#dc2626",
                         boxpoints="all", jitter=0.4, pointpos=0))
    fig.update_layout(height=360, yaxis_title="Holding period (days)",
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def plain_english_debrief(d: dict, cf: dict | None, pgr: dict | None,
                          net_pnl: float) -> str:
    """Deterministic narrative built FROM the engine's numbers (no LLM)."""
    lines = []
    if d.get("verdict"):
        lines.append(
            f"Your tradebook shows a **textbook disposition effect**. You held "
            f"losing positions on average **{d['avg_hold_losers']} days** but "
            f"cut winners after just **{d['avg_hold_winners']} days** — that's "
            f"**{d['disposition_ratio']}× longer** on losers. The gap is "
            f"statistically significant (Mann-Whitney p = {d['p_value']}), so "
            f"this is a real pattern, not a handful of unlucky trades."
        )
    else:
        lines.append(
            f"No statistically significant disposition effect was detected. "
            f"You held losers {d['avg_hold_losers']} days vs winners "
            f"{d['avg_hold_winners']} days (ratio {d['disposition_ratio']}×, "
            f"p = {d['p_value']}) — not a meaningful gap on this data."
        )
    if pgr and pgr.get("status") == "ok":
        verdict_word = "confirms" if pgr["verdict"] else "does not confirm"
        lines.append(
            f"The canonical Odean cross-check {verdict_word} it: you realised "
            f"**{pgr['pgr']:.0%}** of your winning positions but only "
            f"**{pgr['plr']:.0%}** of your losing ones (spread "
            f"{pgr['spread']:+.2f})."
        )
    if cf and cf.get("status") == "ok" and cf["missed_upside"] > 0:
        lines.append(
            f"Sizing the cost: had you given your winners the same patience you "
            f"gave your losers (~{cf['median_loser_hold']:.0f} days), they would "
            f"have returned roughly **{inr(cf['missed_upside'])}** more — an "
            f"estimate across the {cf['n_winners_helped']} winners where holding "
            f"longer actually helped."
        )
    lines.append(
        f"Net realised P&L on closed round-trips: **{inr(net_pnl)}**."
    )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📉 Behavioral Risk Scorecard")
st.markdown(
    "<span class='pill'>v1b</span> SEBI's July 2025 study found **~91% of individual "
    "F&O traders lost money in FY25**, with net losses of **₹1.05 lakh crore** "
    "(₹1.8 lakh crore over FY22–FY24). The *disposition effect* — selling winners "
    "early and clinging to losers — is one mechanical reason why. This tool measures "
    "it in **your own** tradebook.",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — data source & options
# ---------------------------------------------------------------------------
if "trades" not in st.session_state:
    st.session_state.trades = None
    st.session_state.injected_price_on = None
    st.session_state.source_label = None

with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Upload tradebook", "Demo trader (offline)"],
                    help="No data? The demo trader has a built-in disposition "
                         "effect and a self-contained price feed, so every "
                         "feature works without uploading anything.")

    if mode == "Upload tradebook":
        broker = st.selectbox("Broker format", list(BROKER_PRESETS.keys()), index=0)
        up = st.file_uploader("Tradebook CSV", type="csv")
        mapping = None

        if up is not None and broker == "custom":
            up.seek(0)
            raw_cols = list(pd.read_csv(up, nrows=5).columns)
            st.caption("Map your columns:")
            mapping = {}
            for field in REQUIRED_FIELDS + ["trade_time"]:
                opts = ["—"] + [str(c) for c in raw_cols]
                pick = st.selectbox(field, opts, key=f"map_{field}")
                if pick != "—":
                    mapping[field] = pick

        if up is not None:
            try:
                up.seek(0)
                trades = load_and_normalize(up, broker, mapping=mapping)
                st.session_state.trades = trades
                st.session_state.injected_price_on = None
                st.session_state.source_label = f"Uploaded ({broker})"
                rep = trades.attrs.get("ingest_report", {})
                st.success(f"Loaded {rep.get('rows_used', len(trades))} trades, "
                           f"{rep.get('n_symbols', '?')} symbols.")
                if rep.get("rows_dropped"):
                    st.caption(f"{rep['rows_dropped']} unparseable rows skipped.")
            except IngestError as e:
                st.error(str(e))

    else:  # Demo trader
        st.caption("A simulated trader who sells winners fast and clings to "
                   "losers, on a consistent price path.")
        if st.button("Load demo trader", type="primary"):
            tdf, price_on = generate_disposition_trader_with_prices(seed=11)
            st.session_state.trades = tdf
            st.session_state.injected_price_on = price_on
            st.session_state.source_label = "Demo trader (offline)"
        st.caption("Tip: real Zerodha-format sample lives in "
                   "`sample_data/sample_zerodha_format.csv`.")

    st.divider()
    fetch_prices = st.checkbox(
        "Enable price-based metrics (₹ counterfactual + PGR/PLR)",
        value=(st.session_state.injected_price_on is not None),
        help="Uses yfinance for uploaded equities (NSE), or the demo's built-in "
             "feed. F&O contracts can't be priced and are surfaced as such.",
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
trades = st.session_state.trades

if trades is None or trades.empty:
    st.info("⬅️ Upload a tradebook or load the demo trader to begin.")
    st.markdown(
        "**What you'll get**\n"
        "- A FIFO reconstruction of every closed round-trip from your raw trades\n"
        "- A statistically-tested verdict on whether you hold losers longer than winners\n"
        "- (with prices) the canonical Odean PGR/PLR measure and a rupee cost estimate"
    )
    st.stop()

# Date-range filter
dmin, dmax = trades["trade_date"].min(), trades["trade_date"].max()
with st.sidebar:
    st.divider()
    st.caption("Date filter")
    dr = st.date_input("Range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(dr, tuple) and len(dr) == 2:
    lo, hi = dr
    trades = trades[(trades["trade_date"] >= lo) & (trades["trade_date"] <= hi)]

st.caption(f"Source: {st.session_state.source_label} · "
           f"{trades['trade_date'].min()} → {trades['trade_date'].max()}")

# --- Engine run -------------------------------------------------------------
rt = reconstruct_round_trips(trades)
if rt.empty:
    st.warning("No closed round-trips in this date range. "
               "Every position is either still open or a short-to-open.")
    st.stop()

d = disposition_holding_period(rt)
net_pnl = float(rt["pnl"].sum())

# §1 Summary cards -----------------------------------------------------------
st.subheader("Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", len(trades))
c2.metric("Round-trips", len(rt))
c3.metric("Win rate", f"{rt['is_winner'].mean()*100:.0f}%")
c4.metric("Net P&L", inr(net_pnl))
if d["status"] == "ok":
    c5.metric("Avg hold W / L", f"{d['avg_hold_winners']} / {d['avg_hold_losers']} d")
else:
    c5.metric("Avg hold W / L", "—")

skipped = rt.attrs.get("skipped_short_qty", 0)
if skipped:
    st.caption(f"ℹ️ {skipped:g} units of sell-to-open (short) activity skipped — "
               f"v1 reconstructs long round-trips only (shorts are a v2 extension).")

st.divider()

# §2 The verdict -------------------------------------------------------------
st.subheader("The verdict")
if d["status"] == "insufficient_data":
    st.warning(f"Need at least 5 winning and 5 losing round-trips to judge. "
               f"You have {d.get('n_winners', 0)} winners and "
               f"{d.get('n_losers', 0)} losers in this range.")
else:
    gcol, tcol = st.columns([1, 1.3])
    with gcol:
        st.plotly_chart(ratio_gauge(d["disposition_ratio"], d["verdict"]),
                        use_container_width=True)
        st.caption("How many times longer you hold losers than winners. "
                   "Needle past **1×** = disposition effect.")
    with tcol:
        if d["verdict"]:
            st.markdown(
                f"<div class='verdict-bad'>📌 <b>Textbook disposition effect.</b><br>"
                f"You hold losers <b>{d['disposition_ratio']}× longer</b> than "
                f"winners — {d['avg_hold_losers']}d vs {d['avg_hold_winners']}d. "
                f"The difference is statistically significant "
                f"(Mann-Whitney U, p = {d['p_value']}; effect size "
                f"{d['effect_size']}).</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='verdict-good'>✅ <b>No significant disposition "
                f"effect.</b><br>Losers {d['avg_hold_losers']}d vs winners "
                f"{d['avg_hold_winners']}d (ratio {d['disposition_ratio']}×, "
                f"p = {d['p_value']}).</div>",
                unsafe_allow_html=True)
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Median hold — winners", f"{d['median_hold_winners']:.0f} d")
        m2.metric("Median hold — losers", f"{d['median_hold_losers']:.0f} d")
        m3.metric("p-value", f"{d['p_value']}")

st.divider()

# §3 Evidence ----------------------------------------------------------------
st.subheader("Evidence")
e1, e2 = st.columns([1, 1])
with e1:
    st.markdown("**Holding-period distribution**")
    st.plotly_chart(holding_boxplot(rt), use_container_width=True)
with e2:
    st.markdown("**Your 8 longest-held losers**")
    losers = (rt[~rt["is_winner"]]
              .sort_values("holding_days", ascending=False)
              .head(8)[["symbol", "holding_days", "entry_date", "exit_date", "pnl"]]
              .copy())
    losers["pnl"] = losers["pnl"].map(inr)
    losers = losers.rename(columns={"holding_days": "days", "pnl": "P&L"})
    st.dataframe(losers, use_container_width=True, hide_index=True)

    st.markdown("**Your 8 fastest-cut winners**")
    winners = (rt[rt["is_winner"]]
               .sort_values("holding_days", ascending=True)
               .head(8)[["symbol", "holding_days", "entry_date", "exit_date", "pnl"]]
               .copy())
    winners["pnl"] = winners["pnl"].map(inr)
    winners = winners.rename(columns={"holding_days": "days", "pnl": "P&L"})
    st.dataframe(winners, use_container_width=True, hide_index=True)

# --- Build price source for v1b --------------------------------------------
price_on = None
if fetch_prices:
    if st.session_state.injected_price_on is not None:
        price_on = st.session_state.injected_price_on
    else:
        src = get_price_source(dmin, dmax)
        price_on = src.bind(dmin, dmax)

pgr = None
cf = None
if price_on is not None:
    with st.spinner("Marking positions against historical prices…"):
        pgr = disposition_pgr_plr(trades, price_on)
        cf = rupee_counterfactual(rt, price_on)

    st.divider()
    st.subheader("Price-based measures (v1b)")

    pcol, ccol = st.columns([1, 1])
    with pcol:
        st.markdown("**Odean PGR / PLR**  ·  *the canonical academic measure*")
        if pgr and pgr.get("status") == "ok":
            b1, b2, b3 = st.columns(3)
            b1.metric("PGR", f"{pgr['pgr']:.0%}", help="Proportion of winning "
                      "positions you realised.")
            b2.metric("PLR", f"{pgr['plr']:.0%}", help="Proportion of losing "
                      "positions you realised.")
            b3.metric("Spread (PGR−PLR)", f"{pgr['spread']:+.2f}",
                      help="> 0 ⇒ disposition effect.")
            st.caption(f"Tallied over {pgr['selling_days']} selling days · "
                       f"RG {pgr['rg']} / RL {pgr['rl']} / PG {pgr['pg']} / "
                       f"PL {pgr['pl']}"
                       + (f" · {pgr['unpriced_paper']} held positions unpriced"
                          if pgr.get("unpriced_paper") else ""))
        else:
            st.info("Not enough priced positions to compute PGR/PLR "
                    "(often because the book is F&O contracts yfinance can't price).")
    with ccol:
        st.markdown("**Rupee left on the table**")
        if cf and cf.get("status") == "ok":
            st.metric("Estimated missed upside", inr(cf["missed_upside"]))
            st.caption(cf["assumption"])
        else:
            st.info("Couldn't estimate the rupee counterfactual on this data "
                    f"({cf.get('note', 'no priced winners')}).")

    if cf and cf.get("status") == "ok" and not cf["detail"].empty:
        with st.expander("Per-trade counterfactual detail"):
            det = cf["detail"].copy()
            for col in ("actual_exit_price", "counterfactual_exit_price", "delta"):
                det[col] = det[col].map(lambda v: inr(v) if col == "delta" else f"₹{v:,.2f}")
            st.dataframe(det, use_container_width=True, hide_index=True)
else:
    st.divider()
    st.info("Enable **price-based metrics** in the sidebar to add the Odean "
            "PGR/PLR measure and the rupee counterfactual (v1b).")

# §5 AI debrief (deterministic; LLM-optional) --------------------------------
st.divider()
st.subheader("Debrief")
st.markdown(plain_english_debrief(d if d["status"] == "ok" else
                                  {"verdict": False, "avg_hold_losers": "—",
                                   "avg_hold_winners": "—", "disposition_ratio": "—",
                                   "p_value": "—"},
                                  cf, pgr, net_pnl))
st.caption("This debrief is generated deterministically from the engine's "
           "numbers. An optional LLM layer could rephrase it for tone — but it "
           "would only describe these figures, never compute them.")

with st.expander("Methodology & honest caveats"):
    st.markdown(
        "- **FIFO round-trips.** A sell is matched against the oldest open buy "
        "lots first, handling partial fills and scaling in/out. Same-day "
        "buy→sell is ordered correctly (buy before sell).\n"
        "- **Holding-period method (v1a).** Disposition ratio = avg loser hold / "
        "avg winner hold. Significance via a one-sided Mann-Whitney U test "
        "(non-parametric, because holding periods are heavily right-skewed). "
        "Effect size is the rank-biserial correlation.\n"
        "- **PGR/PLR (v1b, Odean 1998).** On each selling day, realised gains/"
        "losses come from the tradebook; still-held positions are marked to that "
        "day's close to classify paper gains/losses. Needs external prices — "
        "that's why it's v1b.\n"
        "- **Rupee counterfactual (v1b).** Re-prices each winner as if held to "
        "the median-loser holding period. An estimate; ignores charges, taxes, "
        "and tied-up capital.\n"
        "- **Shorts.** v1 reconstructs long round-trips only; sell-to-open is "
        "counted and skipped. Net-position tracking (both directions) is a "
        "documented v2 extension.\n"
        "- **Charges.** Gross of brokerage/STT, which slightly overstates P&L."
    )
