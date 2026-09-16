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


class TestTextPrep:
    """Narrative trimming: the fix for silent context overflow."""

    def test_trims_at_statement_header(self):
        from edse.textprep import prepare

        narrative = "Acme reports results. " * 120  # ~2.6k chars
        doc = narrative + "\nCONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n" + "1 2 3 " * 5000
        out = prepare(doc)
        assert out.trimmed_at_marker
        assert "CONDENSED CONSOLIDATED" not in out.text
        assert "Acme reports results" in out.text
        assert out.chars < out.original_chars / 2

    def test_does_not_trim_an_early_marker(self):
        """A marker in the first paragraph is a contents entry, not the break."""
        from edse.textprep import prepare

        doc = "CONSOLIDATED BALANCE SHEETS\n" + "real narrative content. " * 200
        out = prepare(doc)
        assert not out.trimmed_at_marker
        assert out.text  # the release survives intact

    def test_no_marker_leaves_text_alone(self):
        from edse.textprep import prepare

        doc = "Acme reports results. " * 100
        out = prepare(doc)
        assert not out.trimmed_at_marker
        assert not out.hard_truncated
        assert out.chars == len(doc.strip())

    def test_hard_cap_is_recorded_not_silent(self):
        """A truncation that drops content must be visible downstream."""
        from edse.textprep import prepare

        out = prepare("word " * 10000, max_chars=5000)
        assert out.hard_truncated
        assert out.chars <= 5000
        assert out.meta()["hard_truncated"] is True

    def test_cap_disabled_by_none(self):
        from edse.textprep import prepare

        out = prepare("word " * 10000, max_chars=None)
        assert not out.hard_truncated

    def test_metadata_round_trip(self):
        from edse.textprep import prepare

        out = prepare("Acme. " * 500)
        meta = out.meta()
        assert meta["original_chars"] == len("Acme. " * 500)
        assert set(meta) == {
            "original_chars", "prepared_chars", "trimmed_at_marker",
            "hard_truncated", "guidance_recovered",
        }

    def test_guidance_after_the_marker_is_recovered(self):
        """Some filers put a non-GAAP reconciliation before their outlook section.

        Trimming at the first statement header removes the guidance with the
        tables. Measured on the real corpus this destroyed the outlook in 91
        filings, and `guidance_action` would have been wrong for all of them with
        no error raised - so the block is carried back explicitly.
        """
        from edse.textprep import prepare

        doc = (
            "Acme reports second quarter results. " * 60
            + "\nRECONCILIATION OF GAAP TO NON-GAAP MEASURES\n"
            + "1.0 2.0 3.0 " * 400
            + "\nFull Year 2019 Revenues:\n"
            + "The Company now expects full-year organic revenue growth of 5 percent.\n"
            + "\nCONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n"
            + "9 9 9 " * 400
        )
        out = prepare(doc)
        assert out.trimmed_at_marker
        assert out.guidance_recovered
        assert "now expects full-year organic revenue growth of 5 percent" in out.text
        assert "RECONCILIATION" not in out.text

    def test_no_recovery_when_guidance_is_already_in_the_narrative(self):
        """Avoid duplicating an outlook the narrative already states."""
        from edse.textprep import prepare

        doc = (
            "Acme reports results. The Company reaffirms its full-year guidance. " * 40
            + "\nCONSOLIDATED BALANCE SHEETS\n"
            + "1 2 3 " * 400
        )
        out = prepare(doc)
        assert out.trimmed_at_marker
        assert not out.guidance_recovered

    def test_recovery_is_bounded(self):
        """A guidance match inside the tables must not drag the tables back."""
        from edse.textprep import GUIDANCE_RECOVERY_CHARS, prepare

        doc = (
            "Acme reports results. " * 80
            + "\nCONSOLIDATED STATEMENTS OF OPERATIONS\n"
            + "1 2 3 " * 200
            + "\nOutlook\n"
            + "X" * 50000
        )
        out = prepare(doc)
        assert out.guidance_recovered
        assert out.chars < 2000 + GUIDANCE_RECOVERY_CHARS + 500


class TestConsistencyUsesPreparedText:
    """Consistency must extract the same text `edse extract` extracts.

    Regression: `edse consistency` read documents straight off disk while
    `edse extract` trimmed statement tables first. The stability numbers would
    then describe a pipeline nobody runs - and because the untrimmed documents
    are the long ones, the server's context shift would drop text silently, so
    a field could look unstable purely because separate runs were answering
    about different halves of the filing.
    """

    def _corpus(self, tmp_path):
        narrative = "Acme reports record revenue, up 12% year over year. " * 60
        tables = "1 2 3 4 5 " * 5000
        doc = narrative + "\nCONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n" + tables
        (tmp_path / "0000000000-00-000000.txt").write_text(doc)
        return [{"accession": "0000000000-00-000000", "ticker": "ACME"}], doc

    def test_text_is_trimmed_before_extraction(self, tmp_path):
        from edse.cli_eval import prepared_texts

        records, raw = self._corpus(tmp_path)
        texts = prepared_texts(records, tmp_path, None)

        sent = texts["0000000000-00-000000"]
        assert "Acme reports record revenue" in sent
        assert "CONDENSED CONSOLIDATED" not in sent
        assert len(sent) < len(raw) / 2

    def test_matches_what_extract_would_send(self, tmp_path):
        """The helper must agree with `textprep.prepare` exactly, not approximately."""
        from edse.cli_eval import prepared_texts
        from edse.textprep import prepare

        records, raw = self._corpus(tmp_path)
        texts = prepared_texts(records, tmp_path, 70000)

        assert texts["0000000000-00-000000"] == prepare(raw, max_chars=70000).text

    def test_missing_document_is_skipped_not_fatal(self, tmp_path):
        from edse.cli_eval import prepared_texts

        records, _ = self._corpus(tmp_path)
        records.append({"accession": "9999999999-99-999999", "ticker": "GONE"})

        texts = prepared_texts(records, tmp_path, None)
        assert set(texts) == {"0000000000-00-000000"}


class TestReportedExtractors:
    """The report must find every extractor that has artifacts, not a fixed list.

    Regression: the report looped over the literal tuple ("claude", "baseline"),
    so `local` -- the default extractor -- was silently absent from both the
    extraction and prediction sections. The symptom reads as "that stage never
    ran", which is indistinguishable from the truth in a report whose whole
    contract is to omit stages that did not run.
    """

    def test_discovers_local_and_orders_control_first(self, tmp_path, monkeypatch):
        import edse.cli_eval as ce

        for name in ("local", "baseline", "claude"):
            (tmp_path / f"model_results_{name}.json").write_text("{}")
        monkeypatch.setattr(ce, "REPORTS_DIR", tmp_path)

        assert ce._reported_extractors("model_results_") == ["baseline", "local", "claude"]

    def test_unknown_extractor_is_kept_not_dropped(self, tmp_path, monkeypatch):
        import edse.cli_eval as ce

        (tmp_path / "model_results_baseline.json").write_text("{}")
        (tmp_path / "model_results_someone-elses-model.json").write_text("{}")
        monkeypatch.setattr(ce, "REPORTS_DIR", tmp_path)

        assert ce._reported_extractors("model_results_") == [
            "baseline",
            "someone-elses-model",
        ]

    def test_ignores_other_report_json(self, tmp_path, monkeypatch):
        """`extraction_summary.json` must not be read as an extractor named 'summary'."""
        import edse.cli_eval as ce

        (tmp_path / "extraction_stats_local.json").write_text("{}")
        (tmp_path / "extraction_summary.json").write_text("{}")
        monkeypatch.setattr(ce, "REPORTS_DIR", tmp_path)

        assert ce._reported_extractors("extraction_stats_") == ["local"]


class TestGuidanceHeadingRecovery:
    """Outlook sections placed after the tables must survive the trim.

    Regression: recovery keyed on a bare `Outlook`/`Guidance` heading with at
    most one qualifier, so the most common heading in this corpus -- "Business
    Outlook" -- did not match. 72 filings (6.1% of trimmed ones), concentrated
    in QCOM, MRK, LLY and LMT, silently lost their guidance section, and
    `guidance_action` would have read `not_provided` for every one of them with
    nothing raised anywhere.
    """

    def _release(self, heading: str) -> str:
        narrative = "Acme reported quarterly revenue of $1.0 billion. " * 40
        outlook = (
            f"\n{heading}\n"
            "Acme now expects full-year revenue of $4.2 billion to $4.4 billion, "
            "raising the prior range, and adjusted EPS of $3.10 to $3.20 for the year.\n"
        )
        tables = "\nSegment Results\n" + "1 2 3 4 5 " * 800
        return narrative + tables + outlook

    @pytest.mark.parametrize(
        "heading",
        [
            "Business Outlook",
            "Financial Outlook",
            "2025 Financial Guidance",
            "Full-Year 2025 Financial Outlook",
            "Fiscal 2024 full year guidance",
            "2022Outlook",  # html-to-text runs the year into the next word
            "Outlook:",
        ],
    )
    def test_heading_variants_are_recovered(self, heading):
        from edse.textprep import prepare

        out = prepare(self._release(heading))
        assert out.trimmed_at_marker
        assert out.guidance_recovered, f"{heading!r} was not recovered"
        assert "full-year revenue of $4.2 billion" in out.text

    def test_reconciliation_caption_is_not_guidance(self):
        """A table caption containing "OUTLOOK" must not drag the tables back in."""
        from edse.textprep import prepare

        narrative = "Acme reported quarterly revenue of $1.0 billion. " * 40
        tables = (
            "\nSegment Results\n"
            + "1 2 3 4 5 " * 800
            + "\nRECONCILIATION OF GAAP TO NON-GAAP OUTLOOK\n"
            + "9 8 7 6 5 " * 800
        )
        out = prepare(narrative + tables)
        assert out.trimmed_at_marker
        assert not out.guidance_recovered
        assert "RECONCILIATION" not in out.text
        assert out.chars < out.original_chars / 2


class TestAvailableExtractions:
    """`edse eval-extraction` must default to the extractors that exist.

    Regression: the default was the literal string "claude,baseline", so `local`
    -- the default extractor -- was skipped and the extraction-quality table
    omitted the very run the gold set was labeled for. Third instance of the
    same hard-coded-extractor-list bug in this codebase, after the report's two.
    """

    def test_discovers_and_orders_control_first(self, tmp_path, monkeypatch):
        import edse.cli_eval as ce

        for name in ("local", "claude", "baseline"):
            (tmp_path / f"extractions_{name}.parquet").touch()
        monkeypatch.setattr(ce, "PROCESSED_DIR", tmp_path)

        assert ce._available_extractions() == ["baseline", "local", "claude"]

    def test_ignores_unrelated_parquets(self, tmp_path, monkeypatch):
        import edse.cli_eval as ce

        (tmp_path / "extractions_baseline.parquet").touch()
        (tmp_path / "events_labeled.parquet").touch()
        monkeypatch.setattr(ce, "PROCESSED_DIR", tmp_path)

        assert ce._available_extractions() == ["baseline"]


class TestPairedBootstrap:
    """The headline ablation number needs an interval, not just a sign."""

    def _pair(self, n=400, seed=0):
        import numpy as np

        from edse.model import FitResult

        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        # `weak` is near-random; `strong` genuinely separates the classes.
        weak = rng.uniform(0, 1, n)
        strong = np.clip(y * 0.45 + rng.uniform(0, 0.55, n), 0, 1)

        def mk(name, probs):
            return FitResult(
                name=name, n_train=n, n_test=n, base_rate=float(y.mean()),
                auc=0.0, brier=0.0, brier_skill=0.0, logloss=0.0, ece=0.0,
                y_true=y, probs=probs,
            )

        return mk("controls_only", weak), mk("controls_plus_claims", strong)

    def test_detects_a_real_improvement(self):
        from edse.model import paired_bootstrap_delta

        weak, strong = self._pair()
        out = paired_bootstrap_delta(weak, strong, n_boot=400)
        lo, _hi = out["delta_auc_ci95"]
        assert lo > 0, "a genuine separation should give a CI strictly above zero"
        assert out["delta_auc_p_gt_0"] > 0.95

    def test_does_not_invent_an_effect_between_identical_models(self):
        """Same predictions twice must give a zero delta and a CI containing zero."""
        from edse.model import paired_bootstrap_delta

        weak, _ = self._pair()
        twin = type(weak)(**{**weak.__dict__, "name": "controls_plus_claims"})
        out = paired_bootstrap_delta(weak, twin, n_boot=400)
        lo, hi = out["delta_auc_ci95"]
        assert lo == 0.0 and hi == 0.0, f"identical models gave a nonzero CI: {lo},{hi}"

    def test_rejects_unpaired_inputs(self):
        """Pairing is the point; mismatched holdouts must fail loudly."""
        import numpy as np
        import pytest as _pytest

        from edse.model import paired_bootstrap_delta

        weak, strong = self._pair()
        strong.y_true = np.append(strong.y_true, 1)
        strong.probs = np.append(strong.probs, 0.5)
        with _pytest.raises(ValueError, match="identical held-out"):
            paired_bootstrap_delta(weak, strong, n_boot=50)
