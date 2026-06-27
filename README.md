# Behavioral Risk Scorecard

**Reads an Indian retail trader's tradebook and measures one bias that is costing them money — the disposition effect — rigorously, from a CSV, with a rupee figure attached.**

A Streamlit app over a deterministic, tested analytics engine. It reconstructs every closed round-trip from a raw broker export (FIFO lot matching), then detects whether the trader systematically *sells winners early and clings to losers* — and quantifies it two independent ways, one of which is the canonical academic measure.

---

## Why this exists

> In **July 2025**, SEBI reported that roughly **91% of individual F&O traders lost money in FY25**, with **aggregate net losses of ₹1.05 lakh crore** — up 41% year-on-year — across **9.6 million** traders, an average of about ₹1.1 lakh lost per person. An earlier **September 2024** SEBI study found **~93% lost money over FY22–FY24**, with **aggregate losses exceeding ₹1.8 lakh crore**.

Those numbers are the motivation. *Why* do so many lose? Part of the answer is mechanical and behavioral, not just bad luck. The **disposition effect** — the tendency to realize gains too quickly and hold losses too long — is one of the most robust findings in behavioral finance (Shefrin & Statman 1985; Odean 1998). It quietly converts a winning strategy into a losing P&L.

This tool measures that bias **in an individual's own trades**. Not a survey, not a generic warning — your tradebook, your number.

---

## What it does

Upload a tradebook CSV (or load the built-in demo trader) and the app produces:

- **A FIFO reconstruction** of every closed round-trip — handles partial fills, scaling in/out, and same-day trades correctly, which a naïve average-price approach does not.
- **The verdict (v1a):** how much longer you hold losers than winners, with a **Mann-Whitney U test** establishing the gap is statistically real rather than noise from a handful of trades. Reported with a signed **rank-biserial effect size**.
- **The canonical measure (v1b):** **PGR / PLR** — Odean's (1998) Proportion of Gains Realized vs Proportion of Losses Realized, and the **disposition spread** between them. This requires daily market prices (explained below), so it is the upgrade layer.
- **A rupee counterfactual (v1b):** an honest estimate of what holding your winners *with the same patience you already gave your losers* would have earned — money left on the table.
- **A plain-English debrief** built entirely from the engine's numbers (no LLM in the computation path).

---

## The headline claim

**I plant a known bias and prove the detector recovers it — on both measures, from a single consistent simulated process.**

`synthetic/generate.py` builds a trader whose sells are driven by a disposition *rule* over a simulated daily price path: winners get sold fast (≈45%/day once in profit), losers get clung to (≈4%/day). Because the price process and the trades come from the same simulation, **both** the holding-period proxy **and** the price-dependent Odean measure should light up — and they do:

| Measure | Planted direction | Recovered (demo, seed 11) |
|---|---|---|
| Holding-period ratio (v1a) | losers held longer | **3.18×**, p ≈ 0, effect size 0.52 |
| PGR / PLR spread (v1b, Odean) | gains realized more readily | **+0.374** (PGR 0.43 vs PLR 0.06) |
| Rupee counterfactual | upside forgone | **≈ ₹23,000** missed upside |

A separate test asserts the engine **does *not* fire on an unbiased trader** (no false positive). Being able to say *"I generate a trader with a planted bias and prove my detector recovers it, and that it stays silent when there's no bias"* is the part that makes a risk professional or an admissions reader take it seriously.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. run
streamlit run app.py
```

Then in the sidebar either:
- **Load demo trader (offline)** — runs the full v1a + v1b pipeline with a built-in price source, no internet needed. Best for a first look.
- **Upload tradebook** — pick your broker preset (Zerodha / Groww / Upstox / Angel One / Dhan), or map columns manually for any other CSV.

A sample Zerodha-format file lives in `sample_data/sample_zerodha_format.csv`.

### Run the tests

```bash
pip install pytest
python -m pytest -q
```

12 tests cover the FIFO engine (partial fills, FIFO ordering, same-day buy→sell, short-skipping, open-position tracking, empty input) and the metrics (bias recovery, no-false-positive, the insufficient-data guard, PGR/PLR well-formedness, and the both-metrics-recover claim above).

---

## Build order

Built deliberately in stages, each one working before the next was started:

1. **Ingest** — load a broker CSV, map columns to an internal schema, normalize (₹ symbols, commas, BUY/SELL token variants, Indian `DD-MM-YYYY` dates).
2. **Position engine** — FIFO round-trip reconstruction. The keystone: every downstream number is computed off this.
3. **Disposition metric (CSV-only)** — holding-period analysis + Mann-Whitney U. The first real result, needs nothing but the tradebook.
4. **Streamlit UI** — stages 1–3 wired into a demoable app.
5. **Synthetic validation** — built *alongside* the engine, not after, to prove it recovers a planted bias.

Stages 1–5 are **v1a** — a complete, self-contained portfolio piece.

6. **PGR/PLR + rupee counterfactual** — adds a price-data dependency. This is **v1b**, the upgrade.

---

## Architecture

```
behavioral-scorecard/
  app.py                      Streamlit UI (stages 1–6 wired together)
  engine/
    ingest.py                 CSV load, broker column mapping, normalize
    positions.py              FIFO round-trip reconstruction       [CORE]
    metrics.py                disposition metrics + statistical tests [CORE]
    pricing.py                yfinance close prices + synthetic source (v1b)
    counterfactual.py         rupee-left-on-the-table estimate     (v1b)
  synthetic/
    generate.py               synthetic traders with a known bias  [VALIDATION]
  tests/
    test_positions.py         7 engine tests
    test_metrics.py           5 metric/validation tests
  sample_data/                a ready-to-upload Zerodha-format CSV + a synthetic one
  .streamlit/config.toml      theme
```

**The price source is a pluggable callable** — `price_on(symbol, date) -> float | None`. Production uses `YFinancePriceSource` (bulk daily closes, `.NS` suffix for NSE, as-of lookup to the nearest prior trading day, in-process caching, graceful `None` on derivatives/failures). Tests and the offline demo feed a deterministic synthetic price source instead. The metric code never changes between the two.

---

## Methodology — two measures, and why one needs prices

**Holding-period method (v1a).** The disposition fingerprint is simple: losers are held longer than winners. We compute the ratio of average holding periods and test it with **Mann-Whitney U** (one-sided: are loser holding periods stochastically *greater*?). Mann-Whitney rather than a t-test because holding periods are heavily right-skewed, so the normality a t-test assumes doesn't hold. We report a signed **rank-biserial effect size** so the magnitude is visible, not just the p-value. This needs only the tradebook.

**PGR/PLR — Odean's method (v1b).** On every day the trader sells *something*, classify every position they hold: a **Realized Gain/Loss** for what was sold, a **Paper Gain/Loss** for what's still held. Then:

```
PGR = RG / (RG + PG)      proportion of winning positions actually realized
PLR = RL / (RL + PL)      proportion of losing positions actually realized
Disposition Spread = PGR − PLR        > 0  ⇒  disposition effect
```

**The data wrinkle (stated honestly):** to know whether a *held* position is a paper gain or loss on a given selling day, you need that day's *market* price. A tradebook only contains prices at *your own* transaction times. That's precisely why PGR/PLR is v1b and the holding-period method is v1a — one needs external price data, the other doesn't. Presenting both, and being able to explain *why* they differ in their data requirements, is the point.

---

## Limitations & honest caveats

- **Shorts.** v1 reconstructs **long** round-trips only (buy-then-sell). A sell-to-open is a short; the engine counts and surfaces skipped short quantity rather than hiding it. Intraday/F&O traders short constantly, so v2 generalizes to **net-position tracking** — a trade moving you away from zero opens, toward zero closes — which covers both directions. Stated as a known, scoped limitation.
- **Charges.** Brokerage / STT / fees are ignored, which slightly overstates gross P&L. Fine for measuring *behavior*; noted for honesty.
- **Synthetic data is for validation, not headlines.** The planted-bias trader exists to prove the engine works. Headline results should come from real tradebooks — your own, or anonymized ones.
- **The counterfactual is an estimate.** It re-prices winners as if held to the median loser holding period and sums only where extra holding actually helped, with the assumption stated in the UI. It's directional evidence, not a precise claim.

**The engine is the product. Any narrative layer describes the numbers — it never computes them.** The rigor lives in the math, not in prose.

---

## Deploying it live

A live URL beats a screen-share. This runs as-is on **Streamlit Community Cloud**: push the repo to GitHub, point Streamlit Cloud at `app.py`, and the demo-trader path works with zero configuration (it needs no internet). Share the link directly.

---

## References

- Shefrin, H. & Statman, M. (1985). *The Disposition to Sell Winners Too Early and Ride Losers Too Long.* Journal of Finance.
- Odean, T. (1998). *Are Investors Reluctant to Realize Their Losses?* Journal of Finance.
- SEBI (Jul 2025; Sep 2024). Studies on individual trader P&L in the equity F&O segment.
