"""Calibrated volatility-expansion model with leakage-aware evaluation.

Three commitments define this module:

1. **Time-ordered splits with an embargo.** Every fold trains on the past and
   tests on the future. A gap is inserted between them because each label is
   built from a 5-session forward window: without the embargo, an event near the
   end of the training block has a label that depends on prices overlapping the
   first test events, which leaks.

2. **Calibration is the deliverable, not an afterthought.** A ranking score is
   not much use for a risk question - "how likely is vol to expand?" needs a
   number that means what it says. So probabilities are isotonic-calibrated and
   scored with Brier and ECE, not AUC alone.

3. **Ablation decides whether the LLM earned its cost.** Controls-only,
   claims-only, and combined models are trained identically on the same folds;
   the difference is the measurement the project exists to make.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def purged_time_splits(
    dates: pd.Series, n_splits: int, embargo_days: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits with a purge gap between train and test.

    Training indices whose events fall within `embargo_days` of the test block's
    start are removed, because their forward-looking labels overlap the test
    period. Returns [(train_idx, test_idx), ...] in chronological order.
    """
    order = np.argsort(dates.values)
    ordered_dates = pd.to_datetime(dates.values[order])
    n = len(order)
    fold_size = n // (n_splits + 1)
    if fold_size < 2:
        raise ValueError(f"too few samples ({n}) for {n_splits} splits")

    splits = []
    for k in range(1, n_splits + 1):
        cut = fold_size * k
        test_positions = np.arange(cut, min(cut + fold_size, n))
        if len(test_positions) == 0:
            continue
        test_start = ordered_dates[cut]
        embargo_start = test_start - pd.Timedelta(days=embargo_days)
        train_positions = np.array(
            [i for i in range(cut) if ordered_dates[i] < embargo_start], dtype=int
        )
        if len(train_positions) < 20:
            continue
        splits.append((order[train_positions], order[test_positions]))
    return splits


def holdout_split(
    dates: pd.Series, test_fraction: float, embargo_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Final chronological holdout: earliest (1-f) train, latest f test, purged."""
    order = np.argsort(dates.values)
    ordered_dates = pd.to_datetime(dates.values[order])
    cut = int(len(order) * (1 - test_fraction))
    test_start = ordered_dates[cut]
    embargo_start = test_start - pd.Timedelta(days=embargo_days)
    train = order[[i for i in range(cut) if ordered_dates[i] < embargo_start]]
    return train, order[cut:]


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Binned |accuracy - confidence|, weighted by bin population.

    Uses quantile bins rather than fixed-width ones: predicted probabilities
    cluster tightly around the base rate, and fixed-width bins would leave most
    of them empty and report a misleadingly small error.
    """
    if len(y_true) == 0:
        return float("nan")
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(probs.mean() - y_true.mean()))
    bins = np.clip(np.digitize(probs, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        total += mask.sum() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(total / len(y_true))


def reliability_curve(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean_predicted, observed_frequency, count) per quantile bin."""
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([probs.mean()]), np.array([y_true.mean()]), np.array([len(y_true)])
    bins = np.clip(np.digitize(probs, edges[1:-1]), 0, len(edges) - 2)
    pred, obs, counts = [], [], []
    for b in np.unique(bins):
        mask = bins == b
        pred.append(probs[mask].mean())
        obs.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(pred), np.array(obs), np.array(counts)


@dataclass
class FitResult:
    """Scores for one (feature set, estimator) configuration."""

    name: str
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    brier: float
    brier_skill: float
    logloss: float
    ece: float
    y_true: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    probs: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def row(self) -> dict:
        return {
            "model": self.name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "base_rate": round(self.base_rate, 4),
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "brier_skill": round(self.brier_skill, 4),
            "log_loss": round(self.logloss, 4),
            "ece": round(self.ece, 4),
        }


def make_estimator(kind: str, random_state: int) -> Pipeline:
    """Imputer -> scaler -> classifier, so preprocessing is fit per fold.

    Fitting the scaler inside the pipeline is what prevents test-period
    statistics from bleeding into training rows.
    """
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown estimator: {kind}")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def fit_and_score(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    name: str,
    estimator_kind: str = "gbm",
    calibration_method: str = "isotonic",
    n_splits: int = 5,
    embargo_days: int = 10,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> FitResult:
    """Train on the chronological training block, score on the future holdout.

    Calibration uses purged time splits *inside* the training block, so the
    calibrator never sees the holdout and never sees its own fitting data.
    """
    train_idx, test_idx = holdout_split(dates, test_fraction, embargo_days)
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    inner = purged_time_splits(dates.iloc[train_idx].reset_index(drop=True), n_splits, embargo_days)
    if not inner:
        raise ValueError(f"{name}: could not build inner CV folds; need more events")

    model = CalibratedClassifierCV(
        estimator=make_estimator(estimator_kind, random_state),
        method=calibration_method,
        cv=inner,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    base_rate = float(y_train.mean())
    # Brier skill vs always predicting the training base rate. A model that
    # cannot beat the base rate has negative skill, which AUC would hide.
    reference = brier_score_loss(y_test, np.full_like(probs, base_rate))
    brier = brier_score_loss(y_test, probs)

    return FitResult(
        name=name,
        n_train=len(train_idx),
        n_test=len(test_idx),
        base_rate=base_rate,
        auc=roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan"),
        brier=brier,
        brier_skill=1.0 - brier / reference if reference > 0 else float("nan"),
        logloss=log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6), labels=[0, 1]),
        ece=expected_calibration_error(y_test, probs),
        y_true=y_test,
        probs=probs,
    )


def run_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    blocks: dict[str, list[str]],
    estimator_kind: str = "gbm",
    **kwargs,
) -> list[FitResult]:
    """Controls-only vs claims-only vs combined, on identical folds.

    The combined-minus-controls gap is the project's headline number: the
    incremental value of LLM-extracted disclosure claims over market state alone.
    """
    control_cols = blocks.get("control", [])
    claim_cols = blocks.get("claim", [])
    configs = {
        "controls_only": control_cols,
        "claims_only": claim_cols,
        "controls_plus_claims": control_cols + claim_cols,
    }
    results = []
    for name, cols in configs.items():
        usable = [c for c in cols if c in X.columns]
        if not usable:
            log.warning("skipping %s: no columns available", name)
            continue
        results.append(
            fit_and_score(X[usable], y, dates, name, estimator_kind=estimator_kind, **kwargs)
        )
        log.info("%-22s %s", name, results[-1].row())
    return results


def permutation_importance_report(
    X: pd.DataFrame, y: np.ndarray, dates: pd.Series, top_n: int = 20, random_state: int = 42,
    test_fraction: float = 0.2, embargo_days: int = 10,
) -> pd.DataFrame:
    """Permutation importance on the holdout, scored by AUC drop."""
    from sklearn.inspection import permutation_importance

    train_idx, test_idx = holdout_split(dates, test_fraction, embargo_days)
    model = make_estimator("gbm", random_state)
    model.fit(X.iloc[train_idx], y[train_idx])
    result = permutation_importance(
        model, X.iloc[test_idx], y[test_idx],
        n_repeats=10, random_state=random_state, scoring="roc_auc",
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "auc_drop": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("auc_drop", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
