"""Extraction quality measurement.

The downstream model is only as trustworthy as the claims feeding it, so
extraction is graded on its own terms rather than judged by whether the volatility
model happens to work. Four complementary measurements, because no single one is
sufficient:

- **Accuracy vs gold labels** - the primary metric, but it needs hand labels.
- **Macro-F1 per field** - accuracy alone rewards always predicting the majority
  class, which for fields like `guidance_action` is a real failure mode.
- **Abstention behaviour** - does the model say "not stated" when the release
  genuinely does not say, and does it over-abstain when it does?
- **Self-consistency** - agreement across repeated extractions at temperature>0.
  Requires no gold labels, so it scales to the whole corpus and catches
  instability that a 50-document gold set would miss.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..schema import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    GRADED_FIELDS,
    NUMERIC_FIELDS,
    ORDINAL_FIELDS,
)

log = logging.getLogger(__name__)

#: Values that mean "the release does not say".
ABSTENTION_VALUES = {"not_stated", "not_provided", None}


def values_match(field: str, pred, gold, numeric_rtol: float) -> bool:
    """Field-appropriate equality.

    Numerics compare within a relative tolerance because releases state figures
    at varying precision ("up 9 percent" vs "up 9.2 percent"), and a null is only
    correct against a null - treating a missed number as a skip would let a model
    score well by declining to extract anything.
    """
    if field in NUMERIC_FIELDS:
        pred_null, gold_null = pred is None or pd.isna(pred), gold is None or pd.isna(gold)
        if pred_null or gold_null:
            return pred_null and gold_null
        denom = max(abs(float(gold)), 1.0)
        return abs(float(pred) - float(gold)) <= numeric_rtol * denom
    if pred is None or gold is None:
        return pred is gold
    return str(pred) == str(gold)


def grade_field(
    predictions: list, golds: list, field: str, numeric_rtol: float
) -> dict:
    """Accuracy (and macro-F1 / MAE where meaningful) for one field."""
    correct = [values_match(field, p, g, numeric_rtol) for p, g in zip(predictions, golds)]
    out = {
        "field": field,
        "n": len(golds),
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
    }

    if field in CATEGORICAL_FIELDS or field in BOOLEAN_FIELDS:
        pred_s = [str(p) for p in predictions]
        gold_s = [str(g) for g in golds]
        out["macro_f1"] = float(
            f1_score(gold_s, pred_s, average="macro", zero_division=0)
        )
        out["n_classes_gold"] = len(set(gold_s))

    if field in ORDINAL_FIELDS:
        # Gold labels are hand-edited JSON, so a typo can put a string in an
        # ordinal field. Skip what will not coerce rather than aborting the whole
        # evaluation - the exact-match accuracy above already counts it wrong.
        pairs = []
        for p, g in zip(predictions, golds):
            try:
                pairs.append((float(p), float(g)))
            except (TypeError, ValueError):
                continue
        # MAE matters for ordinals: predicting 2 when gold is 3 is a near-miss,
        # while exact-match accuracy scores it identically to predicting 0.
        out["mae"] = float(np.mean([abs(p - g) for p, g in pairs])) if pairs else float("nan")

    return out


def extraction_report(
    predictions: pd.DataFrame, golds: pd.DataFrame, numeric_rtol: float = 0.02
) -> tuple[pd.DataFrame, dict]:
    """Per-field and overall extraction quality on the gold set.

    Both frames must be indexed by accession. Only accessions present in both are
    graded, and the count is reported so partial coverage cannot be mistaken for
    a full evaluation.
    """
    shared = predictions.index.intersection(golds.index)
    if len(shared) == 0:
        raise ValueError("no overlapping accessions between predictions and gold labels")
    if len(shared) < len(golds):
        log.warning(
            "grading %d of %d gold documents (%d missing from predictions)",
            len(shared), len(golds), len(golds) - len(shared),
        )

    pred, gold = predictions.loc[shared], golds.loc[shared]
    rows = [
        grade_field(pred[f].tolist(), gold[f].tolist(), f, numeric_rtol)
        for f in GRADED_FIELDS
        if f in pred.columns and f in gold.columns
    ]
    table = pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)

    # Exact-match rate: every graded field correct on the same document. A harsh
    # metric, and the one that matters if a downstream consumer needs the whole
    # record to be right.
    per_doc = [
        all(
            values_match(f, pred.loc[acc, f], gold.loc[acc, f], numeric_rtol)
            for f in GRADED_FIELDS
            if f in pred.columns and f in gold.columns
        )
        for acc in shared
    ]

    summary = {
        "n_documents": len(shared),
        "n_fields": len(table),
        "mean_field_accuracy": float(table["accuracy"].mean()),
        "mean_macro_f1": float(table["macro_f1"].mean()) if "macro_f1" in table else float("nan"),
        "document_exact_match": float(np.mean(per_doc)),
        "worst_field": table.iloc[0]["field"] if len(table) else None,
        "worst_field_accuracy": float(table.iloc[0]["accuracy"]) if len(table) else float("nan"),
    }
    return table, summary


def abstention_report(
    predictions: pd.DataFrame, golds: pd.DataFrame
) -> pd.DataFrame:
    """Does the extractor abstain when it should, and only when it should?

    - `recall`: of the cases the release genuinely does not state, how many did
      the extractor correctly decline to answer.
    - `precision`: of the cases it declined, how many were genuine non-statements.
      Low precision means over-abstention - discarding real signal.
    """
    shared = predictions.index.intersection(golds.index)
    pred, gold = predictions.loc[shared], golds.loc[shared]

    rows = []
    for f in CATEGORICAL_FIELDS + NUMERIC_FIELDS:
        if f not in pred.columns or f not in gold.columns:
            continue

        def is_abstention(v):
            return (v is None) or (isinstance(v, float) and pd.isna(v)) or (v in ABSTENTION_VALUES)

        p_abs = np.array([is_abstention(v) for v in pred[f]])
        g_abs = np.array([is_abstention(v) for v in gold[f]])
        tp = int((p_abs & g_abs).sum())
        rows.append(
            {
                "field": f,
                "gold_abstention_rate": float(g_abs.mean()),
                "pred_abstention_rate": float(p_abs.mean()),
                "precision": float(tp / p_abs.sum()) if p_abs.sum() else float("nan"),
                "recall": float(tp / g_abs.sum()) if g_abs.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def consistency_report(runs: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Agreement across repeated extractions of the same documents.

    For each field and document, the modal answer's share across runs. This needs
    no gold labels, so it scales to the full corpus - and an unstable field is
    unusable downstream even if it happens to score well on a small gold set.
    """
    if len(runs) < 2:
        raise ValueError("consistency needs at least 2 runs")

    shared = runs[0].index
    for run in runs[1:]:
        shared = shared.intersection(run.index)
    if len(shared) == 0:
        raise ValueError("no documents common to all consistency runs")

    per_field: dict[str, list[float]] = defaultdict(list)
    for field in GRADED_FIELDS:
        if not all(field in r.columns for r in runs):
            continue
        for acc in shared:
            answers = [str(r.loc[acc, field]) for r in runs]
            modal = max(set(answers), key=answers.count)
            per_field[field].append(answers.count(modal) / len(answers))

    table = (
        pd.DataFrame(
            [
                {"field": f, "mean_agreement": float(np.mean(v)),
                 "unanimous_rate": float(np.mean([x == 1.0 for x in v]))}
                for f, v in per_field.items()
            ]
        )
        .sort_values("mean_agreement")
        .reset_index(drop=True)
    )
    summary = {
        "n_documents": len(shared),
        "n_runs": len(runs),
        "mean_agreement": float(table["mean_agreement"].mean()) if len(table) else float("nan"),
        "least_stable_field": table.iloc[0]["field"] if len(table) else None,
    }
    return table, summary


def confidence_calibration(
    predictions: pd.DataFrame, golds: pd.DataFrame, numeric_rtol: float = 0.02
) -> pd.DataFrame:
    """Does the extractor's self-reported confidence track its actual accuracy?

    A model whose accuracy is flat across confidence levels is not self-aware,
    and its `confidence` field should not be trusted as a filter.
    """
    shared = predictions.index.intersection(golds.index)
    pred, gold = predictions.loc[shared], golds.loc[shared]
    if "confidence" not in pred.columns:
        return pd.DataFrame()

    graded = [f for f in GRADED_FIELDS if f in pred.columns and f in gold.columns]
    rows = []
    for acc in shared:
        hits = [values_match(f, pred.loc[acc, f], gold.loc[acc, f], numeric_rtol) for f in graded]
        rows.append({"confidence": pred.loc[acc, "confidence"], "accuracy": float(np.mean(hits))})

    return (
        pd.DataFrame(rows)
        .groupby("confidence")
        .agg(n=("accuracy", "size"), mean_accuracy=("accuracy", "mean"))
        .reset_index()
    )
