"""
tools.py
--------
Pure-Python mock tools for the Multi-Agent Trading Analysis System.
No external API calls are made; data is deterministically generated from
the ticker string so results are reproducible and stable across runs.

Each tool returns a rich dictionary that the corresponding specialist agent
formats into a natural-language report.

Supports both traditional equities (e.g., AAPL, RELIANCE, MSFT) and
crypto assets (e.g., BTC, ETH, SOL).
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Crypto ticker registry — used to branch between equity and crypto logic.
CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "MATIC", "LINK", "LTC", "ATOM", "UNI", "NEAR",
}


def _seed_from_ticker(ticker: str) -> int:
    """Derive a deterministic integer seed from a ticker string."""
    digest = hashlib.md5(ticker.upper().encode()).hexdigest()
    return int(digest[:8], 16)


def _seeded_random(ticker: str, offset: int = 0) -> random.Random:
    """Return a seeded Random instance for reproducible mock data."""
    rng = random.Random()
    rng.seed(_seed_from_ticker(ticker) + offset)
    return rng


def _is_crypto(ticker: str) -> bool:
    """Return True if the ticker is a known crypto asset."""
    return ticker.upper() in CRYPTO_TICKERS


# ---------------------------------------------------------------------------
# Tool 1 — Fundamental Analysis
# ---------------------------------------------------------------------------

def mock_fundamental_analysis(ticker: str) -> Dict[str, Any]:
    """
    Perform a mock fundamental analysis for a given ticker.

    For traditional stocks this returns classic equity metrics (P/E ratio,
    EPS, revenue growth, debt-to-equity, etc.).  For crypto assets it
    returns on-chain / network health metrics (market cap, NVT ratio,
    active addresses, developer activity, etc.).

    Parameters
    ----------
    ticker : str
        Asset symbol, e.g. "AAPL", "RELIANCE", "BTC", "SOL".

    Returns
    -------
    Dict[str, Any]
        Dictionary of fundamental metrics and a qualitative summary.
    """
    rng = _seeded_random(ticker, offset=1)
    t = ticker.upper()
    is_crypto = _is_crypto(t)

    if is_crypto:
        market_cap_b = round(rng.uniform(5, 800), 2)
        nvt_ratio = round(rng.uniform(15, 120), 2)
        active_addresses_7d = rng.randint(50_000, 5_000_000)
        dev_commits_30d = rng.randint(20, 500)
        staking_yield_pct = round(rng.uniform(0, 12), 2)
        circulating_supply_pct = round(rng.uniform(40, 100), 1)
        protocol_revenue_m = round(rng.uniform(0.5, 300), 2)
        tvl_b = round(rng.uniform(0.1, 50), 2)  # Total Value Locked (DeFi)

        score = _score(
            [nvt_ratio < 50, dev_commits_30d > 200, staking_yield_pct > 4,
             circulating_supply_pct > 60, protocol_revenue_m > 20],
            weights=[3, 2, 1, 1, 2]
        )

        return {
            "ticker": t,
            "asset_type": "Crypto",
            "market_cap_billion_usd": market_cap_b,
            "nvt_ratio": nvt_ratio,
            "active_addresses_7d": active_addresses_7d,
            "developer_commits_30d": dev_commits_30d,
            "staking_yield_pct": staking_yield_pct,
            "circulating_supply_pct": circulating_supply_pct,
            "protocol_revenue_30d_million_usd": protocol_revenue_m,
            "total_value_locked_billion_usd": tvl_b,
            "fundamental_score": score,           # 0-100
            "summary": _crypto_fundamental_summary(t, nvt_ratio, dev_commits_30d, score),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    else:
        pe_ratio = round(rng.uniform(8, 55), 2)
        eps = round(rng.uniform(1.5, 35), 2)
        revenue_growth_pct = round(rng.uniform(-5, 45), 2)
        profit_margin_pct = round(rng.uniform(5, 40), 2)
        debt_to_equity = round(rng.uniform(0.1, 3.5), 2)
        roe_pct = round(rng.uniform(5, 45), 2)
        current_ratio = round(rng.uniform(0.8, 4.0), 2)
        dividend_yield_pct = round(rng.uniform(0, 5), 2)
        fcf_yield_pct = round(rng.uniform(1, 12), 2)
        book_value_per_share = round(rng.uniform(10, 500), 2)

        score = _score(
            [pe_ratio < 25, revenue_growth_pct > 10, profit_margin_pct > 15,
             debt_to_equity < 1.5, roe_pct > 15, current_ratio > 1.5],
            weights=[2, 3, 2, 2, 2, 1]
        )

        return {
            "ticker": t,
            "asset_type": "Equity",
            "pe_ratio": pe_ratio,
            "eps": eps,
            "revenue_growth_pct": revenue_growth_pct,
            "profit_margin_pct": profit_margin_pct,
            "debt_to_equity": debt_to_equity,
            "roe_pct": roe_pct,
            "current_ratio": current_ratio,
            "dividend_yield_pct": dividend_yield_pct,
            "fcf_yield_pct": fcf_yield_pct,
            "book_value_per_share": book_value_per_share,
            "fundamental_score": score,            # 0-100
            "summary": _equity_fundamental_summary(t, pe_ratio, revenue_growth_pct, score),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ---------------------------------------------------------------------------
# Tool 2 — Technical Analysis
# ---------------------------------------------------------------------------

def mock_technical_analysis(ticker: str) -> Dict[str, Any]:
    """
    Perform a mock technical analysis for a given ticker.

    Returns price-action indicators including RSI, MACD histogram,
    Bollinger Band position, moving-average crossover signals, support /
    resistance levels, and an overall technical bias.

    Parameters
    ----------
    ticker : str
        Asset symbol, e.g. "MSFT", "SOL", "RELIANCE".

    Returns
    -------
    Dict[str, Any]
        Dictionary of technical indicators and a directional bias string.
    """
    rng = _seeded_random(ticker, offset=2)
    t = ticker.upper()

    # Simulate a mock current price
    base_price = rng.uniform(10, 4000)
    current_price = round(base_price * rng.uniform(0.85, 1.15), 2)

    # RSI (0-100; oversold < 30, overbought > 70)
    rsi = round(rng.uniform(20, 85), 2)

    # MACD histogram — positive = bullish momentum
    macd_hist = round(rng.uniform(-2.5, 3.5), 4)

    # Bollinger Band position (0 = lower band, 1 = upper band)
    bb_position = round(rng.uniform(0.05, 0.95), 3)

    # EMA crossover
    ema_20 = round(current_price * rng.uniform(0.92, 1.05), 2)
    ema_50 = round(current_price * rng.uniform(0.90, 1.08), 2)
    ema_200 = round(current_price * rng.uniform(0.80, 1.15), 2)
    golden_cross = ema_20 > ema_50 > ema_200  # Bullish signal

    # Volume trend
    avg_volume_m = round(rng.uniform(1, 500), 2)  # millions
    vol_change_pct = round(rng.uniform(-40, 80), 2)

    # Support & resistance (mock)
    support_1 = round(current_price * rng.uniform(0.88, 0.96), 2)
    resistance_1 = round(current_price * rng.uniform(1.04, 1.15), 2)

    # ATR — Average True Range (volatility proxy)
    atr_pct = round(rng.uniform(0.5, 8), 2)

    # Compute overall technical score
    score = _score(
        [rsi < 60, macd_hist > 0, bb_position < 0.7,
         golden_cross, vol_change_pct > 0, current_price > ema_50],
        weights=[2, 3, 1, 3, 1, 2]
    )

    bias = "Bullish" if score >= 60 else ("Bearish" if score <= 35 else "Neutral")

    return {
        "ticker": t,
        "current_price_usd": current_price,
        "rsi_14": rsi,
        "macd_histogram": macd_hist,
        "bollinger_band_position": bb_position,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "golden_cross_active": golden_cross,
        "avg_daily_volume_million": avg_volume_m,
        "volume_change_vs_avg_pct": vol_change_pct,
        "support_level_1": support_1,
        "resistance_level_1": resistance_1,
        "atr_pct": atr_pct,
        "technical_score": score,            # 0-100
        "technical_bias": bias,
        "summary": _technical_summary(t, rsi, macd_hist, golden_cross, bias, score),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Tool 3 — Sentiment Analysis
# ---------------------------------------------------------------------------

def mock_sentiment_analysis(context_text: str) -> Dict[str, Any]:
    """
    Perform a mock sentiment analysis on the provided news / market context.

    Rather than calling an NLP API, this function uses keyword scoring
    against bullish and bearish lexicons to produce a deterministic
    sentiment profile.  The result includes a compound sentiment score,
    a categorical label, and a brief qualitative summary.

    Parameters
    ----------
    context_text : str
        Raw news headlines, analyst commentary, or market notes supplied
        by the user.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing sentiment scores, label, and a summary.
    """
    text = context_text.lower()

    BULLISH_KEYWORDS = [
        "bullish", "surge", "rally", "breakout", "buy", "upgrade",
        "positive", "growth", "beat", "record", "gain", "strong",
        "outperform", "moon", "adoption", "partnership", "revenue",
        "profit", "exceed", "launch", "expand", "optimistic", "bull",
        "support", "accumulate", "institutional", "etf", "approval",
    ]
    BEARISH_KEYWORDS = [
        "bearish", "crash", "dump", "sell", "downgrade", "negative",
        "decline", "miss", "loss", "weak", "underperform", "fear",
        "regulation", "ban", "hack", "lawsuit", "recession", "layoff",
        "concern", "risk", "volatile", "bear", "resistance", "overbought",
        "debt", "inflation", "rate hike", "warning", "correction",
    ]

    bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    total = bull_hits + bear_hits

    if total == 0:
        # Neutral with slight random tilt
        rng = random.Random(len(context_text))
        compound_score = round(rng.uniform(-0.1, 0.1), 3)
    else:
        compound_score = round((bull_hits - bear_hits) / max(total, 1), 3)
        compound_score = max(-1.0, min(1.0, compound_score))

    # Scale to 0-100
    sentiment_pct = round((compound_score + 1) / 2 * 100, 1)

    if compound_score >= 0.25:
        label = "Positive"
    elif compound_score <= -0.25:
        label = "Negative"
    else:
        label = "Neutral"

    fear_greed_index = round(sentiment_pct * 0.9 + 5, 1)  # mock proxy 5-95

    return {
        "compound_sentiment_score": compound_score,      # -1.0 to +1.0
        "sentiment_pct": sentiment_pct,                  # 0-100
        "sentiment_label": label,
        "bullish_signals_detected": bull_hits,
        "bearish_signals_detected": bear_hits,
        "fear_greed_proxy": fear_greed_index,
        "word_count": len(context_text.split()),
        "summary": _sentiment_summary(label, compound_score, bull_hits, bear_hits),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Tool 4 — Risk Score Calculator
# ---------------------------------------------------------------------------

def calculate_risk_score(
    ticker: str,
    position_size: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Calculate a composite risk score and position-sizing metrics for a
    given asset and notional position size.

    Returns volatility estimates, Value-at-Risk (VaR), max drawdown proxy,
    beta (vs. benchmark), liquidity risk rating, and recommended position
    limits derived from Kelly Criterion approximation.

    Parameters
    ----------
    ticker : str
        Asset symbol, e.g. "AAPL", "BTC", "RELIANCE".
    position_size : float, optional
        Notional value of the intended position in USD (default: $100,000).

    Returns
    -------
    Dict[str, Any]
        Dictionary of risk metrics and qualitative risk rating.
    """
    rng = _seeded_random(ticker, offset=4)
    t = ticker.upper()
    is_crypto = _is_crypto(t)

    # Crypto is inherently more volatile
    vol_base = 0.04 if is_crypto else 0.015
    daily_vol_pct = round(rng.uniform(vol_base, vol_base * 4), 4)

    annualized_vol_pct = round(daily_vol_pct * math.sqrt(252) * 100, 2)

    # Value-at-Risk (95% confidence, 1-day, parametric)
    var_95_pct = round(1.645 * daily_vol_pct * 100, 3)
    var_95_usd = round(position_size * var_95_pct / 100, 2)

    # Maximum drawdown proxy (historical simulation mock)
    max_drawdown_pct = round(rng.uniform(10, 70 if is_crypto else 45), 2)

    # Beta vs. broad market (BTC = market for crypto)
    beta = round(rng.uniform(0.5, 2.5 if is_crypto else 1.8), 3)

    # Sharpe ratio approximation
    expected_return_pct = round(rng.uniform(5, 40 if is_crypto else 25), 2)
    risk_free_rate_pct = 5.25  # mock current risk-free rate
    sharpe = round(
        (expected_return_pct - risk_free_rate_pct) / annualized_vol_pct, 3
    )

    # Liquidity risk
    liquidity_score = rng.randint(1, 10)
    liquidity_label = (
        "High" if liquidity_score >= 7
        else ("Medium" if liquidity_score >= 4 else "Low")
    )

    # Kelly Criterion approximate position sizing
    win_prob = round(rng.uniform(0.45, 0.70), 3)
    loss_prob = 1 - win_prob
    avg_win_loss_ratio = round(rng.uniform(1.2, 3.0), 2)
    kelly_fraction = round(
        (win_prob * avg_win_loss_ratio - loss_prob) / avg_win_loss_ratio, 4
    )
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # cap at 25%
    recommended_position_usd = round(position_size * kelly_fraction, 2)

    # Overall risk rating
    risk_score = _score(
        [annualized_vol_pct < 30, max_drawdown_pct < 30, beta < 1.5,
         sharpe > 0.5, liquidity_score > 5, kelly_fraction > 0.05],
        weights=[3, 2, 2, 2, 1, 1],
        invert=True  # higher raw score → higher risk
    )
    risk_level = (
        "Low" if risk_score <= 30
        else ("Moderate" if risk_score <= 60 else "High")
    )

    return {
        "ticker": t,
        "position_size_usd": position_size,
        "asset_type": "Crypto" if is_crypto else "Equity",
        "daily_volatility_pct": round(daily_vol_pct * 100, 3),
        "annualized_volatility_pct": annualized_vol_pct,
        "var_95_1day_pct": var_95_pct,
        "var_95_1day_usd": var_95_usd,
        "max_drawdown_proxy_pct": max_drawdown_pct,
        "beta": beta,
        "expected_annual_return_pct": expected_return_pct,
        "sharpe_ratio": sharpe,
        "liquidity_score": liquidity_score,
        "liquidity_risk": liquidity_label,
        "kelly_fraction": kelly_fraction,
        "recommended_position_usd": recommended_position_usd,
        "risk_score": risk_score,               # 0-100 (higher = more risky)
        "risk_level": risk_level,
        "summary": _risk_summary(t, risk_level, annualized_vol_pct, var_95_usd, sharpe),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Internal helpers for scoring & narrative generation
# ---------------------------------------------------------------------------

def _score(
    conditions: list,
    weights: list | None = None,
    invert: bool = False,
) -> int:
    """Weighted boolean scoring — returns 0-100."""
    if weights is None:
        weights = [1] * len(conditions)
    total_weight = sum(weights)
    earned = sum(w for cond, w in zip(conditions, weights) if cond)
    raw = round(earned / total_weight * 100)
    return 100 - raw if invert else raw


def _equity_fundamental_summary(ticker, pe, rev_growth, score) -> str:
    tone = "strong" if score >= 65 else ("weak" if score <= 35 else "mixed")
    return (
        f"{ticker} shows {tone} fundamentals. P/E of {pe:.1f} places it "
        f"{'below' if pe < 20 else 'above'} historical market averages, "
        f"while revenue growth of {rev_growth:.1f}% signals "
        f"{'expansion' if rev_growth > 10 else 'contraction or stagnation'}. "
        f"Overall fundamental quality score: {score}/100."
    )


def _crypto_fundamental_summary(ticker, nvt, commits, score) -> str:
    tone = "robust" if score >= 65 else ("weak" if score <= 35 else "moderate")
    return (
        f"{ticker} exhibits {tone} on-chain fundamentals. NVT ratio of {nvt:.1f} "
        f"suggests the network is {'fairly valued' if nvt < 50 else 'potentially overvalued'}. "
        f"Developer activity of {commits} commits over 30 days indicates "
        f"{'strong' if commits > 200 else 'limited'} protocol development momentum. "
        f"Fundamental score: {score}/100."
    )


def _technical_summary(ticker, rsi, macd, golden_cross, bias, score) -> str:
    rsi_label = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
    return (
        f"{ticker} is technically {bias.lower()} (score {score}/100). "
        f"RSI({rsi:.1f}) is in {rsi_label} territory. "
        f"MACD histogram ({macd:+.4f}) signals {'positive' if macd > 0 else 'negative'} momentum. "
        f"Golden cross {'is' if golden_cross else 'is not'} active."
    )


def _sentiment_summary(label, compound, bull_hits, bear_hits) -> str:
    return (
        f"Market sentiment is {label.lower()} (compound score: {compound:+.3f}). "
        f"Detected {bull_hits} bullish and {bear_hits} bearish signal(s) in the provided context. "
        f"{'Positive momentum may support upside.' if label == 'Positive' else 'Caution warranted given negative market tone.' if label == 'Negative' else 'Mixed signals; monitor closely.'}"
    )


def _risk_summary(ticker, level, vol, var_usd, sharpe) -> str:
    return (
        f"{ticker} carries a {level} risk profile. "
        f"Annualized volatility of {vol:.1f}% with a 1-day 95% VaR of ${var_usd:,.2f}. "
        f"Sharpe ratio of {sharpe:.2f} indicates "
        f"{'attractive risk-adjusted returns' if sharpe > 1 else 'marginal' if sharpe > 0 else 'poor'} "
        f"risk-adjusted performance."
    )
