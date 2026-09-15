"""Daily price history, cached locally.

Prices come from Yahoo via yfinance. Adjusted closes are used throughout so that
splits and dividends do not masquerade as volatility - an unadjusted 4:1 split
looks like a -75% return and would dominate any realized-vol estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Market proxy. Used to build a market-volatility control so the model is not
#: simply rewarded for recognizing market-wide volatile regimes.
MARKET_TICKER = "SPY"


def download_prices(
    tickers: list[str], start: str, end: str, cache_path: Path, refresh: bool = False
) -> pd.DataFrame:
    """Return tidy daily prices: columns [date, ticker, close, volume].

    Cached as Parquet; pass `refresh=True` to re-download.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        log.info("loading cached prices from %s", cache_path)
        return pd.read_parquet(cache_path)

    import yfinance as yf

    symbols = sorted(set(tickers) | {MARKET_TICKER})
    log.info("downloading %d symbols from %s to %s", len(symbols), start, end)
    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check network access and ticker list")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    volume = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    frame = (
        close.stack(future_stack=True)
        .rename("close")
        .to_frame()
        .join(volume.stack(future_stack=True).rename("volume"))
        .reset_index()
    )
    frame.columns = ["date", "ticker", "close", "volume"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    missing = sorted(set(symbols) - set(frame["ticker"].unique()))
    if missing:
        log.warning("no price data returned for: %s", ", ".join(missing))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    log.info("cached %d price rows for %d tickers", len(frame), frame["ticker"].nunique())
    return frame


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker log returns. Log returns are additive across time, which
    makes the realized-vol windows below well-defined."""
    out = prices.sort_values(["ticker", "date"]).copy()
    out["log_return"] = out.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return out


def build_panels(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-ticker frames indexed by date, for fast event-window slicing."""
    return {
        ticker: grp.set_index("date").sort_index()
        for ticker, grp in returns.groupby("ticker", sort=False)
    }


def trading_days(returns: pd.DataFrame) -> pd.DatetimeIndex:
    """The market's trading calendar, taken from the index proxy's own history.

    Deriving the calendar from observed index sessions rather than a holiday
    library keeps it correct for early closes and exchange outages without an
    extra dependency.
    """
    market = returns.loc[returns["ticker"] == MARKET_TICKER, "date"]
    if market.empty:
        market = returns["date"]
    return pd.DatetimeIndex(sorted(market.unique()))
