"""Label construction, feature blocks, and the extraction cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edse.extract.base import ExtractionCache, ExtractionResult
from edse.features import build_features
from edse.labels import compute_event_labels
from edse.schema import EarningsClaims


@pytest.fixture
def synthetic_market():
    """A panel that is dead calm until 2024-02-01, then violently volatile.

    The regime change is placed exactly on the event day so that any misalignment
    - off-by-one, timezone slip, overlapping windows - shows up as the pre-window
    picking up volatility it must not see.
    """
    dates = pd.bdate_range("2023-11-01", "2024-03-15")
    breakpoint_ = pd.Timestamp("2024-02-01")
    rng = np.random.default_rng(0)

    frames = {}
    for ticker in ("TEST", "SPY"):
        returns = np.where(
            dates < breakpoint_,
            rng.normal(0, 0.001, len(dates)),   # calm
            rng.normal(0, 0.050, len(dates)),   # explosive
        )
        frames[ticker] = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "close": 100 * np.exp(np.cumsum(returns)),
                "volume": 1_000_000,
                "log_return": returns,
            }
        ).set_index("date")
    return frames, pd.DatetimeIndex(dates), breakpoint_


class TestEventLabels:
    def test_pre_window_excludes_post_event_volatility(self, synthetic_market):
        """The regime break must land entirely in the post window."""
        panels, calendar, breakpoint_ = synthetic_market
        # Released after the close on the session before the break, so t0 == break.
        prior_session = calendar[calendar < breakpoint_][-1]
        events = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "accession": "test-1",
                    "filing_date": prior_session.strftime("%Y-%m-%d"),
                    "acceptance_datetime": f"{prior_session.strftime('%Y-%m-%d')}T22:00:00.000Z",
                }
            ]
        )
        out = compute_event_labels(
            events, panels, calendar, panels["SPY"],
            post_window=5, pre_window=20, expansion_threshold=1.5,
            min_pre_obs=15, min_post_obs=5,
        )
        assert len(out) == 1
        row = out.iloc[0]
        assert row["t0"] == breakpoint_
        # Calm pre-window, explosive post-window: roughly 0.001 vs 0.050 daily.
        assert row["rv_pre"] < 0.10, "pre-window absorbed post-event volatility"
        assert row["rv_post"] > 0.30
        assert row["y"] == 1

    def test_insufficient_history_is_dropped(self, synthetic_market):
        """An event too early to have a full pre-window must not be labeled."""
        panels, calendar, _ = synthetic_market
        events = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "accession": "too-early",
                    "filing_date": "2023-11-02",
                    "acceptance_datetime": "2023-11-02T22:00:00.000Z",
                }
            ]
        )
        out = compute_event_labels(
            events, panels, calendar, panels["SPY"],
            post_window=5, pre_window=20, expansion_threshold=1.5,
            min_pre_obs=15, min_post_obs=5,
        )
        assert out.empty

    def test_unknown_ticker_is_dropped(self, synthetic_market):
        panels, calendar, _ = synthetic_market
        events = pd.DataFrame(
            [{"ticker": "NOPE", "accession": "x", "filing_date": "2024-02-01",
              "acceptance_datetime": "2024-02-01T22:00:00.000Z"}]
        )
        out = compute_event_labels(
            events, panels, calendar, panels["SPY"], 5, 20, 1.5, 15, 5
        )
        assert out.empty


class TestFeatures:
    @pytest.fixture
    def events(self):
        n = 40
        base = {
            "t0": pd.bdate_range("2024-01-01", periods=n),
            "ticker": ["AAA"] * n,
            "rv_pre": np.linspace(0.1, 0.5, n),
            "rv_post": np.linspace(0.2, 0.6, n),
            "rv_market_pre": np.linspace(0.1, 0.2, n),
            "prior_close": np.linspace(50, 150, n),
            "avg_dollar_volume": np.linspace(1e6, 9e6, n),
            "release_timing": ["after_hours"] * n,
            "y": [0, 1] * (n // 2),
        }
        claims = {
            "claim_revenue_direction": ["increased"] * n,
            "claim_eps_direction": ["decreased"] * n,
            "claim_margin_direction": ["flat"] * n,
            "claim_guidance_action": ["raised"] * n,
            "claim_dividend_action": ["not_stated"] * n,
            "claim_tone": ["confident"] * n,
            "claim_announced_buyback": [True] * n,
            "claim_non_gaap_emphasis": [2] * n,
            "claim_hedging_intensity": [1] * n,
            "claim_revenue_yoy_pct": [5.0] * (n // 2) + [None] * (n // 2),
            "claim_confidence": [3] * n,
        }
        return pd.DataFrame({**base, **claims})

    def test_blocks_are_disjoint_and_cover_the_matrix(self, events):
        X, blocks = build_features(events)
        assert not set(blocks["control"]) & set(blocks["claim"])
        assert set(blocks["control"]) | set(blocks["claim"]) == set(X.columns)

    def test_control_block_has_no_claim_data(self, events):
        """The ablation is meaningless if disclosure fields leak into controls."""
        _, blocks = build_features(events)
        for column in blocks["control"]:
            assert "tone" not in column and "guidance" not in column
            assert "revenue" not in column and "buyback" not in column

    def test_missing_numeric_gets_an_indicator(self, events):
        X, _ = build_features(events)
        assert "revenue_yoy_pct_missing" in X.columns
        assert X["revenue_yoy_pct_missing"].sum() == 20

    def test_matrix_is_finite(self, events):
        X, _ = build_features(events)
        assert np.isfinite(X.to_numpy()).all()

    def test_one_hot_uses_schema_levels_not_observed_ones(self, events):
        """Every enum level gets a column even when absent, so folds align."""
        X, _ = build_features(events)
        for level in ("increased", "decreased", "flat", "not_stated"):
            assert f"revenue_direction__{level}" in X.columns


class TestExtractionCache:
    def _result(self) -> ExtractionResult:
        return ExtractionResult(
            accession="a-1", ticker="AAA", extractor="claude", model="m",
            claims=EarningsClaims(
                revenue_direction="increased", revenue_yoy_pct=5.0,
                eps_direction="increased", margin_direction="flat",
                guidance_action="maintained", guidance_horizon_quarters=4,
                announced_buyback=False, dividend_action="flat",
                announced_restructuring=False, announced_impairment=False,
                executive_transition=False, tone="neutral", non_gaap_emphasis=1,
                hedging_intensity=1, macro_headwind_cited=False,
                segment_weakness_disclosed=False, one_time_items=False, confidence=3,
            ),
            ok=True, cost_usd=0.001,
        )

    def test_round_trip(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        key = cache.key("claude", "m", "v1", "some text")
        cache.put(key, self._result())
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded.claims.revenue_yoy_pct == 5.0
        assert loaded.ok

    def test_key_changes_with_prompt_version(self, tmp_path):
        """A prompt edit must not silently reuse claims from the old prompt."""
        cache = ExtractionCache(tmp_path)
        assert cache.key("claude", "m", "v1", "t") != cache.key("claude", "m", "v2", "t")

    def test_key_changes_with_model_and_text(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        base = cache.key("claude", "m1", "v1", "t")
        assert base != cache.key("claude", "m2", "v1", "t")
        assert base != cache.key("claude", "m1", "v1", "other")

    def test_miss_returns_none(self, tmp_path):
        assert ExtractionCache(tmp_path).get("nonexistent") is None

    def test_corrupt_entry_is_discarded(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        key = cache.key("claude", "m", "v1", "t")
        cache.put(key, self._result())
        path = next(tmp_path.rglob("*.json"))
        path.write_text("{not json")
        assert cache.get(key) is None
        assert not path.exists()
