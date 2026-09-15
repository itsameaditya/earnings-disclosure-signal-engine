"""Leakage tests.

These are the tests that matter. Every other failure makes the project wrong in
ways you can see; a leak makes it *look right* while being worthless, so the
split logic and the event-window alignment get tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edse.labels import realized_vol, resolve_event_day
from edse.model import holdout_split, purged_time_splits


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    """Weekday-only sessions for two months of 2024."""
    days = pd.bdate_range("2024-01-01", "2024-03-01")
    return pd.DatetimeIndex(days)


@pytest.fixture
def event_dates() -> pd.Series:
    return pd.Series(pd.bdate_range("2020-01-01", periods=400))


class TestPurgedSplits:
    def test_train_always_precedes_test(self, event_dates):
        for train, test in purged_time_splits(event_dates, n_splits=5, embargo_days=10):
            assert event_dates.iloc[train].max() < event_dates.iloc[test].min()

    def test_embargo_gap_is_enforced(self, event_dates):
        """No training event may fall within the embargo window before a test block.

        This is the actual leak being prevented: a training event whose 5-session
        forward label window runs into the test period.
        """
        embargo = 10
        for train, test in purged_time_splits(event_dates, n_splits=5, embargo_days=embargo):
            gap = (event_dates.iloc[test].min() - event_dates.iloc[train].max()).days
            assert gap >= embargo, f"gap of {gap}d violates the {embargo}d embargo"

    def test_train_and_test_are_disjoint(self, event_dates):
        for train, test in purged_time_splits(event_dates, n_splits=5, embargo_days=10):
            assert not set(train) & set(test)

    def test_folds_are_chronological(self, event_dates):
        splits = purged_time_splits(event_dates, n_splits=5, embargo_days=10)
        starts = [event_dates.iloc[test].min() for _, test in splits]
        assert starts == sorted(starts)

    def test_training_set_grows(self, event_dates):
        """Expanding window: later folds train on strictly more history."""
        sizes = [len(tr) for tr, _ in purged_time_splits(event_dates, 5, 10)]
        assert sizes == sorted(sizes)

    def test_unsorted_input_is_handled(self):
        """Splits must be time-ordered even when rows arrive shuffled."""
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=300)).sample(
            frac=1.0, random_state=0
        ).reset_index(drop=True)
        for train, test in purged_time_splits(dates, n_splits=4, embargo_days=10):
            assert dates.iloc[train].max() < dates.iloc[test].min()

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="too few samples"):
            purged_time_splits(pd.Series(pd.bdate_range("2024-01-01", periods=4)), 5, 10)


class TestHoldoutSplit:
    def test_holdout_is_the_future(self, event_dates):
        train, test = holdout_split(event_dates, test_fraction=0.2, embargo_days=10)
        assert event_dates.iloc[train].max() < event_dates.iloc[test].min()

    def test_holdout_size_is_approximately_the_fraction(self, event_dates):
        _, test = holdout_split(event_dates, test_fraction=0.2, embargo_days=10)
        assert 0.18 <= len(test) / len(event_dates) <= 0.22


class TestEventDayResolution:
    def test_after_hours_moves_to_next_session(self, calendar):
        """A 16:30 release cannot be traded until the next session opens."""
        window = resolve_event_day("2024-01-10T21:30:00.000Z", "2024-01-10", calendar)
        assert window.release_timing == "after_hours"
        assert window.t0 == pd.Timestamp("2024-01-11")

    def test_pre_market_trades_same_day(self, calendar):
        """07:00 ET -> 12:00 UTC, before the open, so that day's return reflects it."""
        window = resolve_event_day("2024-01-10T12:00:00.000Z", "2024-01-10", calendar)
        assert window.release_timing == "pre_market"
        assert window.t0 == pd.Timestamp("2024-01-10")

    def test_intraday_trades_same_day(self, calendar):
        window = resolve_event_day("2024-01-10T17:00:00.000Z", "2024-01-10", calendar)
        assert window.release_timing == "intraday"
        assert window.t0 == pd.Timestamp("2024-01-10")

    def test_friday_after_hours_skips_the_weekend(self, calendar):
        window = resolve_event_day("2024-01-12T22:00:00.000Z", "2024-01-12", calendar)
        assert window.t0 == pd.Timestamp("2024-01-15")  # Monday

    def test_dst_boundary(self, calendar):
        """20:30 UTC is 16:30 EDT (after close) in summer but 15:30 EST in winter.

        The same UTC clock time therefore lands on opposite sides of the close
        depending on the date - which is exactly the bug a naive parser ships.
        """
        summer = pd.DatetimeIndex(pd.bdate_range("2024-07-01", "2024-07-31"))
        after = resolve_event_day("2024-07-10T20:30:00.000Z", "2024-07-10", summer)
        assert after.release_timing == "after_hours"

        winter = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-01-31"))
        during = resolve_event_day("2024-01-10T20:30:00.000Z", "2024-01-10", winter)
        assert during.release_timing == "intraday"

    def test_missing_timestamp_assumes_after_hours(self, calendar):
        """The modal case; assuming intraday would leak the announcement return."""
        window = resolve_event_day("", "2024-01-10", calendar)
        assert window.t0 == pd.Timestamp("2024-01-11")


class TestRealizedVol:
    def test_annualizes(self):
        daily = 0.01
        returns = np.array([daily, -daily] * 10)
        assert realized_vol(returns) == pytest.approx(
            np.std(returns, ddof=1) * np.sqrt(252), rel=1e-9
        )

    def test_nan_for_degenerate_input(self):
        assert np.isnan(realized_vol(np.array([0.01])))
        assert np.isnan(realized_vol(np.array([])))

    def test_ignores_nans(self):
        assert np.isfinite(realized_vol(np.array([0.01, np.nan, -0.02, 0.005])))
