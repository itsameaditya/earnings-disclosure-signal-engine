"""Text handling, the rule-based baseline, and extraction grading."""

from __future__ import annotations

import pandas as pd
import pytest

from edse.eval.extraction import (
    abstention_report,
    consistency_report,
    extraction_report,
    values_match,
)
from edse.extract.baseline import RuleBasedExtractor
from edse.ingest.edgar import html_to_text
from edse.schema import GRADED_FIELDS

RELEASE = """\
Castellan Industries Reports Third Quarter Results

Revenue declined 6 percent to $812 million. Operating margin contracted to 11.2
percent from 14.8 percent.

"Demand in our European industrial end-markets remained soft, and currency was a
meaningful headwind," said CFO Lee Park.

The Company announced a restructuring program including a workforce reduction.
The Company now expects full-year adjusted EPS of $3.10 to $3.30, down from its
prior range of $3.55 to $3.75.

Forward-Looking Statements
This release may contain forward-looking statements. We believe, expect and
anticipate that results could potentially vary and may be uncertain.
"""


class TestHtmlToText:
    def test_block_tags_become_newlines(self):
        out = html_to_text("<p>First para</p><p>Second para</p>")
        assert "First para" in out and "Second para" in out
        # Without block separation these would run together as one token.
        assert "paraSecond" not in out

    def test_table_cells_stay_separated(self):
        out = html_to_text("<table><tr><td>Revenue</td><td>$812</td></tr></table>")
        assert "Revenue" in out and "$812" in out
        assert "Revenue$812" not in out

    def test_scripts_and_styles_are_removed(self):
        out = html_to_text("<div>Keep<script>var x=1;</script><style>p{}</style></div>")
        assert "var x" not in out and "p{}" not in out
        assert "Keep" in out

    def test_entities_are_unescaped(self):
        assert "AT&T" in html_to_text("<p>AT&amp;T</p>")

    def test_edgar_preamble_is_stripped(self):
        """EDGAR repeats type/sequence/filename above the exhibit body."""
        out = html_to_text(
            "<p>EX-99.1</p><p>2</p><p>a8-kex991.htm</p><p>Document</p>"
            "<p>Exhibit 99.1</p><p>Acme reports results</p>"
        )
        assert out.startswith("Exhibit 99.1")

    def test_malformed_html_does_not_raise(self):
        assert html_to_text("<p>unclosed <div>nested <span>text") != ""


class TestBaselineExtractor:
    @pytest.fixture
    def claims(self):
        return RuleBasedExtractor().extract(RELEASE, ticker="CAST", accession="a").claims

    def test_direction_and_magnitude(self, claims):
        assert claims.revenue_direction.value == "decreased"
        assert claims.revenue_yoy_pct == -6.0
        assert claims.margin_direction.value == "decreased"

    def test_guidance_revision_without_the_word_guidance(self, claims):
        """"down from its prior range" never sits next to the word "guidance"."""
        assert claims.guidance_action.value == "lowered"

    def test_discrete_actions(self, claims):
        assert claims.announced_restructuring is True
        assert claims.announced_buyback is False

    def test_safe_harbor_is_excluded_from_hedging(self):
        """Boilerplate hedging would otherwise pin this field to a constant."""
        extractor = RuleBasedExtractor()
        with_boilerplate = extractor.extract(RELEASE, ticker="C", accession="a").claims
        without = extractor.extract(
            RELEASE.split("Forward-Looking Statements")[0], ticker="C", accession="b"
        ).claims
        assert with_boilerplate.hedging_intensity == without.hedging_intensity

    def test_output_validates_against_the_schema(self, claims):
        assert set(GRADED_FIELDS).issubset(claims.model_dump().keys())

    def test_empty_input_does_not_raise(self):
        result = RuleBasedExtractor().extract("", ticker="X", accession="y")
        assert result.ok


class TestValuesMatch:
    def test_numeric_within_tolerance(self):
        assert values_match("revenue_yoy_pct", 9.0, 9.1, 0.02)
        assert not values_match("revenue_yoy_pct", 9.0, 12.0, 0.02)

    def test_null_only_matches_null(self):
        """A missed number must not be scored as a free pass."""
        assert values_match("revenue_yoy_pct", None, None, 0.02)
        assert not values_match("revenue_yoy_pct", 5.0, None, 0.02)
        assert not values_match("revenue_yoy_pct", None, 5.0, 0.02)

    def test_small_values_use_an_absolute_floor(self):
        """Relative tolerance around zero would otherwise demand exactness."""
        assert values_match("revenue_yoy_pct", 0.0, 0.01, 0.02)

    def test_categorical_is_exact(self):
        assert values_match("tone", "confident", "confident", 0.02)
        assert not values_match("tone", "confident", "neutral", 0.02)


def _typed_values(**overrides) -> dict:
    """A schema-valid value for every graded field, so grading sees real types."""
    from edse.schema import BOOLEAN_FIELDS, NUMERIC_FIELDS, ORDINAL_FIELDS

    values: dict = {}
    for f in GRADED_FIELDS:
        if f in BOOLEAN_FIELDS:
            values[f] = False
        elif f in ORDINAL_FIELDS:
            values[f] = 1
        elif f in NUMERIC_FIELDS:
            values[f] = 4.0
        elif f == "guidance_action":
            values[f] = "maintained"
        elif f == "tone":
            values[f] = "neutral"
        else:
            values[f] = "increased"
    return {**values, **overrides}


class TestExtractionReport:
    def _frame(self, values: dict, accessions: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [{f: values.get(f) for f in GRADED_FIELDS} for _ in accessions],
            index=pd.Index(accessions, name="accession"),
        )

    def test_perfect_prediction_scores_one(self):
        accessions = ["a", "b", "c"]
        gold = self._frame(_typed_values(), accessions)
        _table, summary = extraction_report(gold.copy(), gold)
        assert summary["mean_field_accuracy"] == 1.0
        assert summary["document_exact_match"] == 1.0

    def test_malformed_ordinal_in_gold_is_scored_wrong_not_fatal(self):
        """A hand-labeling typo must not abort the whole evaluation."""
        accessions = ["a"]
        gold = self._frame(_typed_values(non_gaap_emphasis="two"), accessions)
        pred = self._frame(_typed_values(), accessions)
        table, _ = extraction_report(pred, gold)
        row = table.loc[table["field"] == "non_gaap_emphasis"].iloc[0]
        assert row["accuracy"] == 0.0

    def test_disjoint_indices_raise(self):
        a = self._frame(_typed_values(), ["a"])
        b = self._frame(_typed_values(), ["z"])
        with pytest.raises(ValueError, match="no overlapping accessions"):
            extraction_report(a, b)

    def test_abstention_recall_detects_over_answering(self):
        accessions = ["a", "b"]
        gold = self._frame({f: "not_stated" for f in GRADED_FIELDS}, accessions)
        pred = self._frame(_typed_values(), accessions)
        report = abstention_report(pred, gold)
        tone = report.loc[report["field"] == "tone"].iloc[0]
        assert tone["gold_abstention_rate"] == 1.0
        assert tone["recall"] == 0.0  # never abstained when it should have


class TestConsistency:
    def test_identical_runs_are_unanimous(self):
        frame = pd.DataFrame(
            [_typed_values()], index=pd.Index(["a"], name="accession")
        )
        table, summary = consistency_report([frame.copy(), frame.copy(), frame.copy()])
        assert summary["mean_agreement"] == 1.0
        assert (table["unanimous_rate"] == 1.0).all()

    def test_disagreement_lowers_the_score(self):
        accession = pd.Index(["a"], name="accession")
        base = _typed_values()
        run_a = pd.DataFrame([base], index=accession)
        run_b = pd.DataFrame([{**base, "tone": "cautious"}], index=accession)
        table, _ = consistency_report([run_a, run_b])
        tone = table.loc[table["field"] == "tone"].iloc[0]
        assert tone["mean_agreement"] == 0.5

    def test_single_run_raises(self):
        frame = pd.DataFrame([_typed_values()], index=pd.Index(["a"]))
        with pytest.raises(ValueError, match="at least 2 runs"):
            consistency_report([frame])
