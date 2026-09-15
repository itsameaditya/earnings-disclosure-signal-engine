"""Structured claim schema extracted from 8-K earnings press releases.

Design constraints, in priority order:

1. **Every field must be answerable from the press release alone.** No field may
   require knowledge of analyst consensus, the stock reaction, or anything dated
   after the filing. That is what keeps the downstream model leakage-free.
2. **Explicit abstention.** Categoricals carry a `NOT_STATED` member and numerics
   are nullable, so "the release does not say" is a first-class answer rather
   than something the model has to fabricate. Abstention rate is itself an eval
   metric (see eval/extraction.py).
3. **Gradeable.** Categoricals are closed enums (exact match) and numerics are
   scalars (tolerance match), so extraction quality is measurable field by field
   against gold labels without a human in the loop at eval time.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Direction of a reported figure against the comparison the release itself uses."""

    INCREASED = "increased"
    DECREASED = "decreased"
    FLAT = "flat"
    NOT_STATED = "not_stated"


class GuidanceAction(str, Enum):
    """What the release did to forward guidance, if anything."""

    RAISED = "raised"
    LOWERED = "lowered"
    MAINTAINED = "maintained"
    INITIATED = "initiated"
    WITHDRAWN = "withdrawn"
    NOT_PROVIDED = "not_provided"


class Tone(str, Enum):
    """Management's characterization of the quarter, as stated - not as inferred."""

    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    NOT_STATED = "not_stated"


class EarningsClaims(BaseModel):
    """Claims extracted from a single 8-K Item 2.02 earnings press release."""

    # --- Headline results -------------------------------------------------
    revenue_direction: Direction = Field(
        description=(
            "Direction of reported revenue versus the prior-year period, as stated in "
            "the release. Use NOT_STATED if the release gives no year-over-year comparison."
        )
    )
    revenue_yoy_pct: float | None = Field(
        default=None,
        description=(
            "Year-over-year revenue change in percent, as an explicit number stated in "
            "the release (e.g. 'revenue grew 12%' -> 12.0; 'revenue fell 3%' -> -3.0). "
            "Null if the release does not state a percentage. Do not compute it yourself "
            "from dollar figures."
        ),
    )
    eps_direction: Direction = Field(
        description="Direction of reported EPS versus the prior-year period, as stated."
    )
    margin_direction: Direction = Field(
        description=(
            "Direction of gross or operating margin versus the prior-year period, as "
            "stated. If the release discusses several margins, use the operating margin."
        )
    )

    # --- Forward guidance -------------------------------------------------
    guidance_action: GuidanceAction = Field(
        description=(
            "What the release does to forward guidance. RAISED/LOWERED require an "
            "explicit change versus previously issued guidance. INITIATED means guidance "
            "is given for a period not previously guided. WITHDRAWN means previously "
            "issued guidance is being removed. NOT_PROVIDED means the release contains "
            "no forward guidance at all."
        )
    )
    guidance_horizon_quarters: int | None = Field(
        default=None,
        description=(
            "Number of quarters the forward guidance covers (1 for next-quarter-only, "
            "4 for full-year). Null when no guidance is provided."
        ),
    )

    # --- Discrete corporate actions --------------------------------------
    announced_buyback: bool = Field(
        description="True only if the release announces a new or expanded share repurchase authorization."
    )
    dividend_action: Direction = Field(
        description=(
            "Direction of any dividend change announced in this release. NOT_STATED if "
            "the release does not mention a dividend action (a routine declaration at an "
            "unchanged rate is FLAT)."
        )
    )
    announced_restructuring: bool = Field(
        description="True if the release announces restructuring, layoffs, or a cost-reduction program."
    )
    announced_impairment: bool = Field(
        description="True if the release discloses a goodwill or asset impairment charge."
    )
    executive_transition: bool = Field(
        description="True if the release announces a CEO, CFO, or other named-executive change."
    )

    # --- Disclosure texture ----------------------------------------------
    # These are the fields that a keyword baseline cannot reproduce, and they are
    # where the LLM is expected to earn its cost.
    tone: Tone = Field(
        description=(
            "Management's characterization of the quarter in the quoted commentary. "
            "CONFIDENT for unhedged positive framing ('record', 'exceptional', 'strong "
            "momentum'), CAUTIOUS for acknowledged difficulty ('challenging', 'headwinds', "
            "'softness'), NEUTRAL for flat reporting. Judge the quoted executive "
            "commentary, not the numbers."
        )
    )
    non_gaap_emphasis: int = Field(
        ge=0,
        le=3,
        description=(
            "How heavily the release leans on non-GAAP/adjusted figures, 0-3. "
            "0 = GAAP only; 1 = non-GAAP mentioned alongside GAAP; 2 = non-GAAP leads the "
            "headline numbers; 3 = headline results are non-GAAP with GAAP relegated to "
            "the tables only."
        ),
    )
    hedging_intensity: int = Field(
        ge=0,
        le=3,
        description=(
            "Density of hedging and uncertainty language in the narrative sections "
            "(excluding the boilerplate safe-harbor paragraph), 0-3. 0 = none; "
            "3 = pervasive conditionals and qualifiers."
        ),
    )
    macro_headwind_cited: bool = Field(
        description=(
            "True if the release attributes results to macroeconomic conditions "
            "(FX, inflation, rates, tariffs, demand environment) rather than to "
            "company-specific execution."
        )
    )
    segment_weakness_disclosed: bool = Field(
        description="True if the release identifies a specific business segment or geography as underperforming."
    )
    one_time_items: bool = Field(
        description="True if results include charges or gains the release labels one-time, unusual, or non-recurring."
    )

    # --- Extraction self-report ------------------------------------------
    # Not a feature. Used by the eval harness to test whether the model knows
    # when it is guessing; see eval/extraction.py::abstention_report.
    confidence: int = Field(
        ge=0,
        le=3,
        description=(
            "Your confidence that the fields above are correct, 0-3. Use 0-1 when the "
            "release is unusually short, is not a standard earnings release, or is "
            "largely tabular with little narrative."
        ),
    )


#: Fields graded by exact match against gold labels.
CATEGORICAL_FIELDS = (
    "revenue_direction",
    "eps_direction",
    "margin_direction",
    "guidance_action",
    "dividend_action",
    "tone",
)

#: Boolean fields, graded by exact match (and reported with precision/recall).
BOOLEAN_FIELDS = (
    "announced_buyback",
    "announced_restructuring",
    "announced_impairment",
    "executive_transition",
    "macro_headwind_cited",
    "segment_weakness_disclosed",
    "one_time_items",
)

#: Ordinal 0-3 fields, graded by exact match and by mean absolute error.
ORDINAL_FIELDS = ("non_gaap_emphasis", "hedging_intensity")

#: Nullable numerics, graded within a relative tolerance, with null treated as a
#: distinct correct-or-not answer rather than silently skipped.
NUMERIC_FIELDS = ("revenue_yoy_pct", "guidance_horizon_quarters")

#: Every graded field. `confidence` is deliberately excluded - it is a
#: self-report about the extraction, not a claim about the filing.
GRADED_FIELDS = CATEGORICAL_FIELDS + BOOLEAN_FIELDS + ORDINAL_FIELDS + NUMERIC_FIELDS
