"""Turn extracted claims plus market state into a model matrix.

Features are grouped into named blocks so the headline experiment is a one-line
ablation: does the CLAIM block add anything on top of the CONTROL block? Without
that split, a good AUC says nothing about whether the LLM contributed - a model
can score well purely on "this stock was already volatile and thinly traded".

Every feature here is knowable at t0-1. Nothing is derived from the outcome
windows, and no statistic is computed across the full sample (which would leak
test-period information into training rows through the scaler).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .schema import BOOLEAN_FIELDS, CATEGORICAL_FIELDS, NUMERIC_FIELDS, ORDINAL_FIELDS

log = logging.getLogger(__name__)

#: Sentinel for a numeric the release did not state. Paired with an explicit
#: `*_missing` indicator so "not disclosed" is a learnable signal in its own
#: right rather than being silently imputed into the middle of the distribution.
MISSING_SENTINEL = 0.0

CONTROL_BLOCK = "control"
CLAIM_BLOCK = "claim"


def _one_hot(series: pd.Series, prefix: str, levels: list[str]) -> pd.DataFrame:
    """One-hot with a fixed level list.

    The levels come from the schema enums, not from the observed data, so a
    category absent from the training fold still produces a column and the
    matrix has identical shape across folds.
    """
    out = pd.DataFrame(index=series.index)
    for level in levels:
        out[f"{prefix}__{level}"] = (series.astype(str) == level).astype(int)
    return out


def build_features(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (feature_matrix, block_name -> column names).

    `events` must carry the labeled event columns from `labels.py` plus one
    column per extracted claim field, flattened as `claim_<field>`.
    """
    from .schema import Direction, GuidanceAction, Tone

    frames: list[pd.DataFrame] = []
    blocks: dict[str, list[str]] = {CONTROL_BLOCK: [], CLAIM_BLOCK: []}
    idx = events.index

    # ---------------- Control block: market state, no text ----------------
    controls = pd.DataFrame(index=idx)
    # Logs because volatility and dollar volume are right-skewed over orders of
    # magnitude; untransformed they let a handful of extreme events dominate.
    controls["log_rv_pre"] = np.log(events["rv_pre"].clip(lower=1e-4))
    controls["log_rv_market_pre"] = np.log(events["rv_market_pre"].clip(lower=1e-4))
    controls["rv_ratio_to_market"] = events["rv_pre"] / events["rv_market_pre"].clip(lower=1e-4)
    controls["log_dollar_volume"] = np.log(events["avg_dollar_volume"].clip(lower=1.0))
    controls["log_price"] = np.log(events["prior_close"].clip(lower=0.01))

    t0 = pd.to_datetime(events["t0"])
    controls["quarter"] = t0.dt.quarter
    # Calendar position, not a trend term: a raw year would let the model
    # extrapolate a time trend that cannot exist out of sample.
    controls["month"] = t0.dt.month

    # Days since this ticker's previous earnings event. Uses only prior events.
    controls["days_since_prev"] = (
        events.assign(_t0=t0)
        .groupby("ticker", sort=False)["_t0"]
        .diff()
        .dt.days.fillna(91.0)
        .clip(0, 250)
    )
    controls = pd.concat(
        [controls, _one_hot(events["release_timing"], "timing",
                            ["pre_market", "intraday", "after_hours", "non_session"])],
        axis=1,
    )
    frames.append(controls)
    blocks[CONTROL_BLOCK] = list(controls.columns)

    # ---------------- Claim block: everything from the disclosure ----------
    claims = pd.DataFrame(index=idx)
    enum_levels = {
        "revenue_direction": [e.value for e in Direction],
        "eps_direction": [e.value for e in Direction],
        "margin_direction": [e.value for e in Direction],
        "dividend_action": [e.value for e in Direction],
        "guidance_action": [e.value for e in GuidanceAction],
        "tone": [e.value for e in Tone],
    }
    for field in CATEGORICAL_FIELDS:
        col = f"claim_{field}"
        if col not in events:
            continue
        claims = pd.concat([claims, _one_hot(events[col], field, enum_levels[field])], axis=1)

    for field in BOOLEAN_FIELDS:
        col = f"claim_{field}"
        if col in events:
            claims[field] = events[col].fillna(False).astype(int)

    for field in ORDINAL_FIELDS:
        col = f"claim_{field}"
        if col in events:
            claims[field] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)

    for field in NUMERIC_FIELDS:
        col = f"claim_{field}"
        if col not in events:
            continue
        values = pd.to_numeric(events[col], errors="coerce")
        claims[f"{field}_missing"] = values.isna().astype(int)
        claims[field] = values.fillna(MISSING_SENTINEL)

    # Magnitude matters separately from direction: a 1% revenue decline and a
    # 30% decline are the same one-hot but very different disclosures.
    if "revenue_yoy_pct" in claims:
        claims["revenue_yoy_abs"] = claims["revenue_yoy_pct"].abs()

    if "claim_confidence" in events:
        claims["extractor_confidence"] = pd.to_numeric(
            events["claim_confidence"], errors="coerce"
        ).fillna(0)

    frames.append(claims)
    blocks[CLAIM_BLOCK] = list(claims.columns)

    matrix = pd.concat(frames, axis=1).astype(float)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    log.info(
        "features: %d rows x %d cols (%d control, %d claim)",
        len(matrix), matrix.shape[1], len(blocks[CONTROL_BLOCK]), len(blocks[CLAIM_BLOCK]),
    )
    return matrix, blocks


def flatten_claims(events: pd.DataFrame, extractions: dict[str, dict]) -> pd.DataFrame:
    """Join extracted claims onto events as `claim_<field>` columns.

    Events without a successful extraction are dropped: imputing an entire
    disclosure would fabricate the very signal being tested.
    """
    records, kept = [], []
    for pos, accession in enumerate(events["accession"]):
        claims = extractions.get(accession)
        if not claims:
            continue
        records.append({f"claim_{k}": v for k, v in claims.items()})
        kept.append(pos)

    if not records:
        raise ValueError("no events have a successful extraction; run `edse extract` first")

    dropped = len(events) - len(kept)
    if dropped:
        log.warning("dropped %d/%d events with no successful extraction", dropped, len(events))

    base = events.iloc[kept].reset_index(drop=True)
    claim_frame = pd.DataFrame(records).reset_index(drop=True)
    return pd.concat([base, claim_frame], axis=1)
