# Gold label provenance and conventions

## Provenance — read this first

**These labels were produced by `claude-opus-5` reading the filing text, not by a
human.** Every row carries `_labeled_by` recording that.

What that does and does not license:

- Grading the **`local` (qwen2.5:7b)** or **`baseline` (rule-based)** extractors
  against this set is a *strong-model reference* comparison. That is a real
  measurement — the reference model is far more capable than either — but it is
  **not accuracy against human ground truth**, and `edse eval-extraction` output
  should be read as agreement with a reference reading.
- Grading the **`claude` extractor** against this set is **circular** and means
  nothing. Do not report it.
- The prompt-portability finding in the README is supported by this set only in
  the same qualified sense.

A human-labeled set would strictly dominate this one. Nothing here prevents
re-reviewing a row by hand; change `_labeled_by` when you do.

## How the labels were made

- **From the prepared text, not the raw filing.** Each label is based on
  `textprep.prepare()` output — the identical text the extractor receives — so a
  label never rests on text the model could not see. The corollary is that a
  trimming bug is invisible to this eval, which is why it is worth measuring
  trimming separately. (One such bug surfaced while labeling: QCOM prepared to
  2,345 of 26,633 chars, dropping the guidance section. Fixed before labeling.)
- **Without the prefill.** Rows are pre-filled from the rule-based extractor.
  Those values were not displayed while judging; anchoring to them would make
  the baseline score against a gold set that is partly a copy of itself.
  Measured afterwards, the labels agree with the prefill on **58.8%** of graded
  fields — lowest on `non_gaap_emphasis` (20%), `revenue_yoy_pct` (20%) and
  `hedging_intensity` (30%), which is where a regex should be expected to fail.
- `confidence` is **not** labeled. It is excluded from `GRADED_FIELDS` because
  it is a self-report about the extraction, not a claim about the filing.

## Field conventions

Judgement calls that recur, resolved once so the set is internally consistent.

| Situation | Convention |
|---|---|
| GAAP and non-GAAP figures disagree | Use the **reported (GAAP)** figure. Applies to `eps_direction`, `margin_direction`, `revenue_yoy_pct`. |
| Release covers Q4 **and** the full year | Label the **quarter**; the event is the quarterly announcement. |
| `revenue_yoy_pct` | Only a percentage **explicitly stated** in the text. Never computed from dollar figures. `flat`/`—%` is `0.0`. Null when only segment growth rates are given. |
| `margin_direction` | Prefer the **operating** margin. `not_stated` unless a direction versus the prior year is given for a consolidated margin; conflicting segment margins alone are `not_stated`. |
| `guidance_action` when one line is maintained and another changed | The **explicit change** wins over the reiteration. |
| `guidance_horizon_quarters` | `4` for full-year guidance, `1` for next-quarter-only, null when no guidance. |
| `announced_buyback` | Only a **new or expanded authorization**. Reporting repurchases executed under an existing program is `false`. |
| `dividend_action` | Only a change **announced in this release**. Reporting dividends paid is `not_stated`. |
| `announced_restructuring` | `true` if the release **discloses** a restructuring, severance or cost-reduction program, whether newly announced or in progress — the extractor cannot know which. |
| `macro_headwind_cited` | `true` only when macro conditions are cited as a **cause of results**. A constant-currency reconciliation on its own is `false`: otherwise nearly every multinational is `true` and the field carries no signal. |
| `one_time_items` | `true` when the release **isolates and quantifies** a discrete non-recurring charge or gain. The literal words "one-time"/"unusual" are not required; quantification is. |
| `non_gaap_emphasis` | `0` GAAP only · `1` both presented, GAAP at least equal billing · `2` non-GAAP leads the headline numbers · `3` headline is non-GAAP, GAAP only in the tables. |
| `hedging_intensity` | Judged on the narrative and outlook sections, **excluding** the safe-harbor boilerplate — which is near-identical across filings and would flatten the field. |
| `tone` | Judge the **quoted executive commentary**, not the numbers. An explicit acknowledgement of difficulty ("challenging", "headwinds", "reset year") makes it `cautious` even alongside positive framing. |
