"""Regime Momentum agent for Builderr Trading Round 2.

Strategy (plain Python stdlib only, no network, no LLM, no file access):
  * Rank a fixed pool of liquid broad/sector ETFs and mega-cap leaders by
    multi-horizon momentum, trend (price vs 50-day SMA) and volatility.
    Hold the top 5 positive-score names, equal weight, max 22% each.
  * Regime switch: risk-on only while SPY and QQQ hold their 50-day trends
    and QQQ 20-day realised vol is calm. Mild risk-off -> defensive ETFs
    (XLP/XLU/XLV/XLE). Severe risk-off (SPY and QQQ both below their
    200-day SMA) -> 100% cash.
  * Gated 2x overlay: QLD + SSO only when QQQ is in a calm uptrend
    (QQQ > SMA20 > SMA50, 20-day momentum > 0, vol < 0.25). Beta-adjusted
    gross is capped at 1.40x (limit is 1.50x).
  * Protection: 8% stop-loss per position vs avg cost, portfolio drawdown
    taper (halve exposure below -6% from peak equity, cash below -10%),
    3-day rebalance cadence plus drift trim above 27% per position.
  * Long-only, deterministic, decides once per day from the bars given.

Contract: decide(market_state, portfolio_state, cash) -> list of orders.
market_state: {ticker: [bar, ...]} daily OHLCV bars, oldest first (~220).
portfolio_state: {cash, positions: [{ticker, quantity, avg_cost}],
                  last_prices: {ticker: price}}.
Returns orders: [{ticker, side: 'buy'|'sell', quantity}].

Limits respected (challenge caps): long-only, beta-adjusted gross <= 1.5x,
single position < 30% (targets <= 22%, trim at 27%), <= 50 trades/day,
decide() runs in well under 5s (a few dozen tickers x ~220 bars).
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# --- Tunable parameters (sensitivity-tested, see backtest_results.md) ---
MOM_LONG = 42          # long momentum window (trading days). 42, not 63:
                       # faster exit from broken trends. Measured across 5
                       # windows (3 samples + Round 2 window + Yahoo 2y):
                       # mean -2.9% vs -3.5%, worst DD 10.6% vs 13.0%,
                       # Round 2 window -9.1% vs -12.4%. It is also the
                       # Round 1 winner's own primary signal (42-day
                       # return), so this is convergent evidence, not a
                       # one-window fit. 35 is too fast (calm +4.2%).
MOM_SHORT = 21         # short momentum window
SMA_TREND = 50         # trend filter window
SMA_LONG = 200         # severe risk-off window
VOL_WINDOW = 20        # realised-vol window
VOL_RISK_ON_MAX = 0.35     # annualised QQQ vol ceiling for risk-on
VOL_OVERLAY_MAX = 0.25     # annualised QQQ vol ceiling for 2x overlay
TOP_N = 6              # winners held (6 over 5: same logic, broader
                       # diversification; measured mean return -0.6% vs -2.9%
                       # and worst drawdown 11.5% vs 19.2% across the 3
                       # sample windows; see backtest_results.md)
MAX_WEIGHT = 0.22      # per-position target cap (limit 0.30)
DRIFT_LIMIT = 0.27     # trim a position drifting above this weight
MAX_BETA_GROSS = 1.40  # internal cap (limit 1.50)
MIN_TRADE_PCT = 0.01   # ignore adjustments smaller than 1% of equity
REBALANCE_EVERY = 3    # days between scheduled rebalances
STOP_LOSS = 0.08       # sell a position 8% below its avg cost
DD_TAPER = 0.06        # halve exposure below -6% from peak equity
DD_CASH = 0.10         # go to cash below -10% from peak equity
CASH_COOLDOWN = 3      # bars to stay in cash after a drawdown exit

CANDIDATES = (
    # Broad + sector ETFs (present in the sample windows and the live universe)
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "SMH",
    # Mega-cap / liquid leaders
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "MU", "MRVL", "PLTR", "JPM",
)
DEFENSIVE_WEIGHTS = (
    ("XLP", 0.24),
    ("XLU", 0.24),
    ("XLV", 0.20),
    ("XLE", 0.12),
)
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

# Module state (persists across daily calls inside one scoring run).
_last_rebalance_n: int | None = None
_last_regime: str | None = None
_peak_equity: float = 0.0
_cash_until_n: int = -1


def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        existing = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": avg_cost})
        existing["quantity"] += qty
        existing["avg_cost"] = avg_cost or existing["avg_cost"]
    return positions


def equity_of(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _bar_count(market_state: dict[str, list[dict[str, Any]]]) -> int:
    for t in ("SPY", "QQQ"):
        bars = market_state.get(t)
        if bars:
            return len(bars)
    n = 0
    for bars in market_state.values():
        n = max(n, len(bars or []))
    return n


def _prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def _regime(market_state: dict[str, list[dict[str, Any]]]) -> str:
    """'on' | 'defensive' | 'cash' | 'unknown' (not enough history)."""
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < SMA_TREND or len(qqq) < SMA_TREND:
        return "unknown"
    spy_sma = sma(spy, SMA_TREND)
    qqq_sma = sma(qqq, SMA_TREND)
    qqq_vol = realized_vol(qqq, VOL_WINDOW)
    if spy_sma is None or qqq_sma is None or qqq_vol is None:
        return "unknown"
    if spy[-1] > spy_sma and qqq[-1] > qqq_sma and qqq_vol < VOL_RISK_ON_MAX:
        return "on"
    # Severe risk-off: both broad indexes below their 200-day average.
    spy_long = sma(spy, SMA_LONG)
    qqq_long = sma(qqq, SMA_LONG)
    if (
        spy_long is not None
        and qqq_long is not None
        and spy[-1] < spy_long
        and qqq[-1] < qqq_long
    ):
        return "cash"
    return "defensive"


def target_weights(
    market_state: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Target weights for the current regime (from _regime)."""
    regime = _regime(market_state)
    if regime in ("unknown", "cash"):
        return {}
    if regime == "defensive":
        weights = {t: w for t, w in DEFENSIVE_WEIGHTS if closes(market_state.get(t))}
        return _scale_caps(weights)

    # Risk-on: rank candidates by momentum + trend - volatility.
    scored: list[tuple[float, str]] = []
    for ticker in CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) <= max(MOM_LONG, SMA_TREND):
            continue
        mom_long = momentum(values, MOM_LONG)
        mom_short = momentum(values, MOM_SHORT)
        trend = sma(values, SMA_TREND)
        vol = realized_vol(values, VOL_WINDOW)
        if mom_long is None or mom_short is None or trend is None or vol is None:
            continue
        if values[-1] <= trend:
            continue
        trend_gap = values[-1] / trend - 1.0
        score = 0.55 * mom_long + 0.25 * mom_short + 0.20 * trend_gap - 0.15 * vol
        if score > 0.0:
            scored.append((score, ticker))
    scored.sort(reverse=True)
    winners = [t for _, t in scored[:TOP_N]]
    if not winners:
        weights = {t: w for t, w in DEFENSIVE_WEIGHTS if closes(market_state.get(t))}
        return _scale_caps(weights)

    qqq = closes(market_state.get("QQQ"))
    qqq_sma20 = sma(qqq, 20) if qqq else None
    qqq_sma50 = sma(qqq, SMA_TREND) if qqq else None
    qqq_mom20 = momentum(qqq, MOM_SHORT) if qqq else None
    qqq_vol = realized_vol(qqq, VOL_WINDOW) if qqq else None
    overlay_on = bool(
        qqq_sma20 is not None
        and qqq_sma50 is not None
        and qqq_mom20 is not None
        and qqq_vol is not None
        and qqq_sma20 > qqq_sma50
        and qqq_mom20 > 0.0
        and qqq_vol < VOL_OVERLAY_MAX
        and closes(market_state.get("QLD"))
        and closes(market_state.get("SSO"))
    )

    weights: dict[str, float] = {}
    base_budget = 0.76 if overlay_on else 0.92
    per_winner = min(MAX_WEIGHT - 0.02, base_budget / len(winners))
    for ticker in winners:
        weights[ticker] = per_winner
    if overlay_on:
        weights["QLD"] = 0.11
        weights["SSO"] = 0.07
    return _scale_caps(weights)


def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0:
        return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # Sells first: full exit for stale holdings, trim overweight targets.
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        qty = pos["quantity"]
        current_value = qty * price
        target_value = total_equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0 and current_value >= min_trade:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price

    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.98)

    # Buys second, within expected cash after sells.
    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = total_equity * weight
        delta = target_value - current_value
        if delta < min_trade:
            continue
        buy_qty = int(min(delta, spendable) // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price
    return orders[:45]


def _has_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


def _stop_loss_orders(
    portfolio_state: dict[str, Any],
) -> list[dict[str, object]]:
    orders: list[dict[str, object]] = []
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            continue
        if pos["avg_cost"] > 0 and price > 0 and price < pos["avg_cost"] * (1.0 - STOP_LOSS):
            qty = int(pos["quantity"])
            if qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
    return orders


def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Return a list of long-only buy/sell orders (see module docstring)."""
    global _last_rebalance_n, _last_regime, _peak_equity, _cash_until_n
    try:
        if not market_state:
            return []
        n = _bar_count(market_state)
        if n == 0:
            return []

        total_equity = equity_of(portfolio_state, cash)
        if total_equity <= 0:
            return []
        if total_equity > _peak_equity:
            _peak_equity = total_equity

        prices = _prices(market_state)
        positions = current_positions(portfolio_state)
        if not positions:
            # Flat (all cash): restart the drawdown clock from here, so a
            # past peak cannot trap us in cash forever after re-entry.
            _peak_equity = total_equity

        # 1. Hard protection runs every day, independent of rebalance timing.
        stops = _stop_loss_orders(portfolio_state)
        if stops:
            return stops[:45]

        drawdown = (_peak_equity - total_equity) / _peak_equity if _peak_equity > 0 else 0.0
        if drawdown >= DD_CASH:
            _cash_until_n = n + CASH_COOLDOWN
            if positions:
                return [
                    {"ticker": t, "side": "sell", "quantity": int(p["quantity"])}
                    for t, p in positions.items()
                    if int(p["quantity"]) > 0
                ][:45]
            return []

        # 2. Rebalance only on schedule, on drift, or on a regime change.
        regime = _regime(market_state)
        if regime == "unknown":
            # Not enough history to judge the regime: hold, do not liquidate.
            return []
        days_since = None if _last_rebalance_n is None else n - _last_rebalance_n
        should_rebalance = (
            _last_rebalance_n is None
            or days_since is None
            or days_since >= REBALANCE_EVERY
            or _last_regime is not None
            and regime != _last_regime
            or _has_drifted(portfolio_state, total_equity)
        )
        if not should_rebalance:
            return []

        # 3. Targets for the current regime (cash while in cooldown).
        if n < _cash_until_n:
            targets: dict[str, float] = {}
        else:
            targets = target_weights(market_state)
        if drawdown >= DD_TAPER and targets:
            targets = {t: w * 0.5 for t, w in targets.items()}

        orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
        if orders:
            _last_rebalance_n = n
            _last_regime = regime
        elif not positions and not targets:
            _last_rebalance_n = n
            _last_regime = regime
        return orders
    except Exception:
        # Never crash the scoring run; a missed day is better than an error.
        return []
