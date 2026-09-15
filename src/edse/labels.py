"""Event-time alignment and the prediction target.

The whole project's credibility rests on this file. Two things must hold:

1. **The event day must be the first session that can trade on the news.**
   Most earnings releases cross the wire after the 16:00 ET close, so the first
   reacting session is the *next* trading day. Treating the filing date as the
   event day would put the announcement return inside the "pre-announcement"
   baseline window for the majority of the sample - a leak that would make the
   model look far better than it is.

2. **The pre-window must end strictly before the event day**, with no shared
   observations. Overlapping windows would put post-announcement information
   into the feature that normalizes the target.

Target: realized volatility over the `post_window` sessions starting at t0,
divided by the realized volatility over the `pre_window` sessions ending the day
before t0. The binary label is whether that ratio exceeds `expansion_threshold`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd

from .ingest.edgar import parse_acceptance

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass(frozen=True)
class EventWindow:
    """Resolved event timing for one filing."""

    t0: pd.Timestamp        # first session whose return can reflect the news
    release_timing: str     # "pre_market" | "intraday" | "after_hours" | "non_session"


def resolve_event_day(
    acceptance_datetime: str, filing_date: str, calendar: pd.DatetimeIndex
) -> EventWindow | None:
    """Map a filing's acceptance timestamp to the first session that trades on it.

    - Released before 09:30 ET on a trading day -> that day's return reflects it.
    - Released during the session -> that day's return reflects it.
    - Released at/after 16:00 ET, or on a non-trading day -> the next session.
    """
    accepted = parse_acceptance(acceptance_datetime)
    if accepted is None:
        # Fall back to the filing date and assume after-hours, the modal case.
        accepted = datetime.combine(pd.Timestamp(filing_date).date(), time(17, 0))

    day = pd.Timestamp(accepted.date())
    clock = accepted.time()
    is_session = day in calendar

    if is_session and clock < MARKET_OPEN:
        timing, target = "pre_market", day
    elif is_session and clock < MARKET_CLOSE:
        timing, target = "intraday", day
    else:
        timing = "after_hours" if is_session else "non_session"
        nxt = calendar[calendar > day]
        if len(nxt) == 0:
            return None
        target = nxt[0]

    if target not in calendar:
        nxt = calendar[calendar >= target]
        if len(nxt) == 0:
            return None
        target = nxt[0]

    return EventWindow(t0=target, release_timing=timing)


def realized_vol(returns: np.ndarray) -> float:
    """Annualized realized volatility of a return series.

    ddof=1 because these are short samples (5-20 observations) where the
    population estimator is materially biased low.
    """
    clean = returns[np.isfinite(returns)]
    if len(clean) < 2:
        return np.nan
    return float(np.std(clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_event_labels(
    events: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    market_panel: pd.DataFrame,
    post_window: int,
    pre_window: int,
    expansion_threshold: float,
    min_pre_obs: int,
    min_post_obs: int,
) -> pd.DataFrame:
    """Attach event timing, the volatility target, and market-state controls.

    Returns one row per resolvable event. Events are dropped (with a counted
    reason) when price history is too sparse to measure either window.
    """
    rows: list[dict] = []
    drops: dict[str, int] = {}

    def drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for rec in events.to_dict("records"):
        ticker = rec["ticker"]
        panel = panels.get(ticker)
        if panel is None or panel.empty:
            drop("no_price_history")
            continue

        window = resolve_event_day(rec.get("acceptance_datetime", ""), rec["filing_date"], calendar)
        if window is None:
            drop("unresolvable_event_day")
            continue

        idx = panel.index
        pos = idx.searchsorted(window.t0)
        if pos >= len(idx) or idx[pos] != window.t0:
            drop("event_day_missing_for_ticker")
            continue

        # Strictly disjoint windows: pre ends at pos-1, post starts at pos.
        pre = panel["log_return"].iloc[max(0, pos - pre_window) : pos].to_numpy()
        post = panel["log_return"].iloc[pos : pos + post_window].to_numpy()

        if np.isfinite(pre).sum() < min_pre_obs:
            drop("insufficient_pre_window")
            continue
        if np.isfinite(post).sum() < min_post_obs:
            drop("insufficient_post_window")
            continue

        rv_pre, rv_post = realized_vol(pre), realized_vol(post)
        if not np.isfinite(rv_pre) or rv_pre <= 0 or not np.isfinite(rv_post):
            drop("degenerate_volatility")
            continue

        # Market volatility over the same pre-window, known at t0-1. This is a
        # control, not an outcome: it lets the model separate "this disclosure is
        # risky" from "everything was volatile that month".
        mkt_pos = market_panel.index.searchsorted(window.t0)
        mkt_pre = market_panel["log_return"].iloc[max(0, mkt_pos - pre_window) : mkt_pos].to_numpy()
        rv_market_pre = realized_vol(mkt_pre)

        prior_close = float(panel["close"].iloc[pos - 1]) if pos > 0 else np.nan
        dollar_volume = float(
            (panel["close"] * panel["volume"]).iloc[max(0, pos - pre_window) : pos].mean()
        )

        rows.append(
            {
                **rec,
                "t0": window.t0,
                "release_timing": window.release_timing,
                "rv_pre": rv_pre,
                "rv_post": rv_post,
                "vol_expansion": rv_post / rv_pre,
                "y": int((rv_post / rv_pre) > expansion_threshold),
                "rv_market_pre": rv_market_pre,
                "prior_close": prior_close,
                "avg_dollar_volume": dollar_volume,
            }
        )

    if drops:
        log.info("dropped events by reason: %s", dict(sorted(drops.items())))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("t0").reset_index(drop=True)
        log.info(
            "labeled %d events | positive rate %.3f | median expansion %.2fx",
            len(out),
            out["y"].mean(),
            out["vol_expansion"].median(),
        )
    return out
