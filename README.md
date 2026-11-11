# Builderr Trading Round 2 — Regime Momentum agent

Agent name: **zkgvn** · Entry for the Builderr Trading Round 2 challenge
($1,000 prize pool: $600 / $250 / $150 to the top 3 entrants who beat
Arnav, the Round 1 winner, over the Round 2 forward window).

This repo contains one submission file: **`agent.py`** — a single
`decide(market_state, portfolio_state, cash)` function, Python stdlib
only, no network, no LLM, no API keys, no file access.

## Strategy in one paragraph

Rank a fixed pool of liquid broad/sector ETFs and mega-cap leaders by
42-day momentum, 21-day momentum, distance above the 50-day average, and
20-day realised volatility; hold the top 6 positive-score names equally
(max 22% each). Stay risk-on only while SPY and QQQ hold their 50-day
trends and QQQ volatility is calm; otherwise rotate into defensive ETFs
(XLP/XLU/XLV/XLE), and go 100% to cash when both SPY and QQQ fall below
their 200-day averages. A small 2× sleeve (QLD/SSO) is allowed only in
calm QQQ uptrends. Protection: 8% stop-loss per position, drawdown taper
(halve exposure below -6% from peak equity, cash below -10%), 3-day
rebalance cadence plus drift trim above 27%. Beta-adjusted gross is
capped at 1.40× (limit 1.50×); every position stays under the 30% cap.

## Test it

```bash
# Official admission check (needs the template's preview.py + sample data):
cp agent.py /path/to/builderr-trading-template/candidate.py
cd /path/to/builderr-trading-template && python preview.py candidate.py

# Independent harness (same engine) + sensitivity sweep:
python3 backtest.py            # 3 sample windows
python3 backtest.py --sweep    # every parameter +/-20%

# Unit tests (edge cases + the 2x-overlay branch, synthetic data):
python3 tests/test_agent.py
```

## Verified results (honest summary)

| Check | Result |
|---|---|
| Official `preview.py` (3 sample windows) | PASS — 0 errors, peak gross 0.93×, peak concentration 23%, worst drawdown 10.6% |
| Sample windows | calm +8.7% / selloff -4.1% / COVID crash -7.0% |
| Yahoo 2y out-of-sample (Sep 2024–Nov 2026, 30 tickers) | +100.9%, MaxDD 21.2%, Sharpe 1.51 |
| Live Round 2 window (7 Jul–11 Nov 2026, same engine) | **-9.1%** — worse than Arnav (-7.29%), the template (-0.67%) and buy-and-hold (+2.95%). Momentum rotation churns in chop; see `backtest_results.md` §3 |
| Sensitivity | no ±20% parameter move collapses the strategy |

Full numbers, methods and caveats: [`backtest_results.md`](backtest_results.md).
Verified live spec: [`SPEC.md`](SPEC.md).

## Rules compliance

- Long-only; no shorts. Gross leverage ≤ 1.5× (we cap at 1.40×).
- Single position < 30% (targets ≤ 22%, trim at 27%).
- ≤ 50 trades/day (we cap at 45 orders/day). Minimum hold 60 s (daily decisions).
- `decide()` runs in ~6 ms (limit 5 s).
- No lookahead: only the bars handed to `decide()` are used. No external
  data, no network calls, no hidden files, no API keys, no `.env`.
- The `data/yahoo/` CSVs in this workdir are offline validation data only;
  they are not read by `agent.py` and are not part of the submission.

## Submission

Not submitted yet. See [`SUBMIT.md`](SUBMIT.md) for the exact remaining
steps (PR to `builderr-ai/builderr-trading-template` with `agent.py`, or
email to `submit@builderr.ai`). Deadline: **15 Nov 2026, 23:59 IST**
(16 Nov 01:29 WIB). Up to 4 revisions before the cutoff.
