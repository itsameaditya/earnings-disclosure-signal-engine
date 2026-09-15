# Earnings Disclosure Signal Engine

An LLM extracts structured claims from SEC 8-K earnings releases; those claims become
features in a calibrated model predicting post-announcement volatility, with an eval
harness that measures extraction quality independently of downstream performance.

**Runs end-to-end for free.** The default extractor is a local model via Ollama, so
the whole pipeline reproduces on a laptop at zero API cost. A Claude backend is fully
implemented behind the same interface for anyone with a key, and comparing the two is
itself one of the results.

The question the project actually answers: **does an LLM reading an earnings release
tell you anything about upcoming volatility that the market's own state doesn't
already?** That framing forces two things most "LLM + finance" projects skip — a
measured extraction step, and an ablation that can come back negative.

---

## Pipeline

```
SEC EDGAR                Claude                   scikit-learn
─────────                ──────                   ────────────
8-K Item 2.02   ──┐
(earnings)        │   structured claims      calibrated classifier
                  ├──▶  (18 fields,     ──▶  P(vol expansion)  ──▶  ablation +
Exhibit 99.1    ──┘    JSON-schema                                   calibration
(press release)         constrained)                                 report
                                                      ▲
Yahoo Finance ──▶ realized-vol target ────────────────┘
                  + market-state controls
```

Two evaluation loops run independently:

| Loop | Question | Metrics |
|---|---|---|
| **Extraction** | Are the claims correct? | field accuracy, macro-F1, document exact-match, abstention precision/recall, self-consistency, confidence calibration |
| **Prediction** | Do the claims carry signal? | AUC, Brier skill, ECE, reliability curve, controls-only vs. claims-only vs. combined |

Three extractors implement one interface, so all of it runs unchanged on any of them:

| Extractor | Cost | What it is |
|---|---|---|
| `local` (default) | free | `qwen2.5:7b` via Ollama, schema-constrained decoding |
| `claude` | ~$18 for the corpus | Claude API with structured outputs |
| `baseline` | free | Rule-based regex/keyword extractor - the control |

Keeping them separate matters: a model can score well on volatility while the
extraction is mostly wrong (the features act as noisy sector dummies), and
extraction can be near-perfect while carrying no predictive signal at all. Only
measuring both tells you which situation you're in.

---

## What makes this non-trivial

**Event timing is a real leak, and it's the default bug.** Most earnings releases
cross the wire *after* the 16:00 ET close, so the first session that can trade on
the news is the *next* one. EDGAR reports acceptance time in UTC; the naive
`filing_date` treatment puts the announcement's own return inside the
"pre-announcement" baseline window for the majority of the sample. The pipeline
parses acceptance timestamps, converts UTC→Eastern with DST handled, and classifies
each release as pre-market / intraday / after-hours before choosing `t0`.
`tests/test_leakage.py::test_dst_boundary` pins the case where the same UTC clock
time falls on opposite sides of the close in summer and winter.

**The windows are strictly disjoint.** The target is realized vol over sessions
`[t0, t0+5)` divided by the baseline over `[t0-20, t0)`. Not one observation is
shared between them.

**Splits are time-ordered with an embargo.** Every fold trains on the past and
tests on the future, with a 10-day purge gap — because each label depends on a
5-session forward window, so a training event near the boundary would otherwise
have a label built from prices that overlap the test period.

**The baseline is a real attempt, not a strawman.** A rule-based extractor with
genuine regexes and curated cue lists runs the identical pipeline. If the LLM
can't beat keyword matching, the project should say so.

**Abstention is first-class.** Every categorical field has a `not_stated` member
and numerics are nullable, so "the release doesn't say" is a real answer rather
than something the model has to invent. Abstention precision and recall are
measured separately: over-abstaining discards signal, under-abstaining fabricates it.

---

## Two findings from building it

**Prompts are model-specific artifacts, and the failure is silent.** The system
prompt written for Claude is ~1,800 tokens and leans hard on abstention ("a wrong
confident answer is far more costly than an abstention"). On `qwen2.5:7b` that
framing dominated the model: measured across 12 releases that state a revenue
percentage in plain text, it produced a **null `revenue_yoy_pct` on 100% of them**,
and `margin_direction: not_stated` on 100%. A 40-token prompt recovered **100%
recall** on the same documents. An intermediate version carrying about half the
rules scored 75% - for this model, more instruction text monotonically increased
abstention.

The important part is *how* it failed. Every response was valid JSON, schema-conformant,
and confidently structured. Nothing downstream would have flagged it; the feature would
simply have been a constant column, and the ablation would have quietly reported that
LLM claims add no signal. Schema validation cannot catch this. Only measuring
extraction against known answers can. Both prompts are kept side by side in the repo
(`extract/llm.py`, `extract/local.py`).

**Half of an earnings release is not an earnings release.** Press releases are a
narrative followed by financial statement tables. Every field in the schema is
answerable from the narrative; none needs the tables. Trimming at the statement
header (`textprep.py`) cut the corpus **54%**, and - more importantly - took the
share of documents overflowing a 16k context window from **11.2% to zero**. Those
would have been truncated silently, producing confident claims based on partial text.

A third, smaller lesson: while measuring the above, a throwaway regex written to
approximate ground truth was itself wrong twice (`[^.]` doesn't match inside
"$109.4 billion"), first understating the model's recall and then overstating its
fabrication rate. That is the argument for a hand-labeled gold set rather than a
convenient proxy.

---

## Results

> **Status.** The dataset, the rule-based control, and the methodology below are
> final. Two results are still open, and both are open for stated reasons rather
> than as placeholders:
>
> - **The local-LLM ablation row.** Extraction over all 1,447 filings runs at a
>   measured ~13 s/document on one laptop GPU — serial by design, since concurrent
>   requests just queue inside Ollama. It is cached per document and resumable, so
>   it costs nothing but wall-clock. Partial results are deliberately not reported:
>   the corpus is ordered by issuer, so any prefix is a handful of tech mega-caps
>   rather than a sample, and an AUC computed on it would be a tech-sector number
>   wearing a corpus-wide label.
> - **The extraction-quality table.** It needs hand labels (`make label`), and
>   those are the one input in this project that cannot be generated by the
>   project. Grading the LLM against labels an LLM produced would measure
>   agreement, not accuracy.
>
> Every number below is regenerated from pipeline artifacts by `edse report` and
> none is hand-edited; `reports/REPORT.md` omits any section whose stage has not
> been run, which is why it is shorter than this file.

### Dataset

**1,445 labeled earnings events**, 40 large-cap issuers, 2018-01-12 to 2026-08-20.
Positive rate (post-announcement vol > 1.5x the pre-announcement baseline): **46.9%**.
Median vol expansion across all events: **1.44x** - earnings do raise volatility,
which is the sanity check the target has to pass before anything else is worth reading.

Release timing, which is the whole reason event-day alignment gets its own module:

| timing | events | share |
|---|---:|---:|
| pre-market | 942 | 65.2% |
| after-hours | 489 | 33.8% |
| intraday | 13 | 0.9% |

A third of this sample cannot trade on the news until the next session. Treating
`filing_date` as the event day would put the announcement's own return inside the
"pre-announcement" baseline for every one of those 489 events.

### Prediction - rule-based control

| feature set | AUC | Brier | Brier skill | ECE |
|---|---:|---:|---:|---:|
| controls only | 0.686 | 0.225 | +0.092 | 0.061 |
| claims only | 0.553 | 0.246 | +0.007 | 0.080 |
| controls + claims | 0.685 | 0.226 | +0.087 | 0.064 |

**Keyword-extracted claims add nothing** (-0.001 AUC, -0.004 Brier skill). Market
state alone reaches 0.686 AUC, and pre-announcement realized volatility is by far the
strongest single feature (+0.091 AUC drop under permutation).

That is the bar. It is also the result that makes the ablation worth running: an
extractor can produce 1,447 clean, schema-valid records and still contribute zero
signal. Whether a real LLM clears this bar is the open question this repo exists
to answer, and a negative answer there would be reported just as plainly.

![ablation](reports/figures/ablation_baseline.png)

The calibration is genuinely good - predictions track observed frequency closely -
but the histogram underneath is the honest part: predictions span roughly 0.14-0.62,
so the model is separating events weakly even where it is well-calibrated.

![reliability](reports/figures/reliability_baseline.png)

---

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # add EDGAR_USER_AGENT (required) and ANTHROPIC_API_KEY
```

`EDGAR_USER_AGENT` must contain a contact email — SEC blocks anonymous clients,
and the failure mode is silent (empty result sets), so the code refuses to start
without one.

For the free local extractor:

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:7b
```

```bash
edse universe                 # resolve tickers -> CIKs via SEC's own mapping
edse ingest                   # download 8-K Item 2.02 press-release exhibits
edse prices                   # daily OHLCV for the universe + SPY
edse label                    # event-time alignment and the volatility target

edse extract --extractor local        # free, local model (default)
edse extract --extractor baseline     # free, rules only
edse extract --extractor claude       # needs ANTHROPIC_API_KEY

edse train --extractor local          # ablation, calibration, figures
edse report                           # assemble reports/REPORT.md
```

`ANTHROPIC_API_KEY` is only needed for `--extractor claude`. Everything else, including
every figure and metric in this README, runs without it.

Extraction quality (needs hand labels):

```bash
edse gold --sample 60                 # stratified labeling template
python scripts/label_gold.py          # review it field by field (see below)
edse eval-extraction                  # grade the extractors against it
edse consistency --runs 3             # stability, no gold labels needed
```

`scripts/label_gold.py` walks the template one filing at a time, showing the *same
trimmed narrative the extractor is given* — so a gold label is never based on text
the model could not see — with the rubric for each field pulled from `schema.py` and
the prefill from the rule-based extractor. It never displays the LLM's own answer:
the gold set is the independent measurement, and anchoring it to the system under
evaluation would inflate every score. Progress saves after each document.

Extraction is cached on disk, keyed by `(extractor, model, prompt_version,
document_text)`. A re-run after a crash re-extracts only what's missing, and
editing the prompt invalidates the cache instead of silently mixing prompt
versions into one dataset.

---

## Design decisions

**Why 8-K Item 2.02 rather than full-text search.** EDGAR tags each 8-K with the
Items it reports; Item 2.02 *is* "Results of Operations and Financial Condition."
Filtering on the tag gives a precise, reproducible event set with no keyword
guesswork. The narrative itself is almost never in the 8-K body — it's furnished as
Exhibit 99.1, resolved from the filing's SGML document table. (`index.json` is not
usable for this: its `type` field holds the icon filename, not the exhibit type.)

**Why volatility expansion rather than direction.** Direction prediction from public
filings at the moment of release is close to a market-efficiency claim. Volatility
is the honest target: it's genuinely uncertain, it's what options markets price, and
disclosure characteristics plausibly inform it. A ratio to the pre-announcement
baseline also normalizes away persistent cross-sectional differences — without it
the model would mostly learn which tickers are volatile.

**Why the schema descriptions are the prompt.** Field descriptions in
`src/edse/schema.py` are the extraction instructions, so there's one source of
truth. The system prompt carries only cross-cutting rules and the judgment calls a
schema can't express.

**Why Haiku 4.5 by default.** A bulk extractor over ~1.4k documents is the case
where a cheaper model earns its place. `configs/config.yaml` sets `quality_model:
claude-opus-5` for a subset comparison — run `edse extract --model claude-opus-5
--limit 100` and grade both against the same gold set to see what the cheap model
costs you in accuracy.

**On prompt caching — measured, not assumed.** The system prompt is a frozen prefix
marked with `cache_control`, with per-document text after the breakpoint. Whether
the cache actually engages is model-dependent: the API only caches prefixes above a
minimum length (~1024 tokens for Sonnet/Opus, 2048 for Haiku). This prompt is ~1.7k
tokens, so it caches on Sonnet/Opus and most likely **does not** on Haiku 4.5, the
default. The prompt is deliberately not padded to cross that line — padding would
burn real tokens on every uncached call to manufacture a metric. Instead
`reports/extraction_stats_*.json` reports measured `cache_hit_rate`, so the answer
is observed.

**On gold-label anchoring.** `edse gold` pre-fills rows from the *rule-based*
extractor, never the LLM. Pre-filling from the system under evaluation would anchor
the labeler toward its answers and inflate every downstream score. `--prefill none`
gives blank rows.

---

## Layout

```
src/edse/
  schema.py            18-field claim schema; descriptions double as prompt text
  config.py            config + env loading
  ingest/edgar.py      8-K discovery, exhibit resolution, HTML→text
  ingest/prices.py     daily prices, returns, trading calendar
  labels.py            event-time alignment + volatility target      ← leakage-critical
  features.py          claims + market state → matrix, in named blocks
  model.py             purged time splits, calibration, ablation     ← leakage-critical
  textprep.py          narrative trimming + guidance recovery         ← see findings
  extract/llm.py       Claude extractor (structured outputs)
  extract/local.py     Ollama extractor, schema-constrained decoding
  extract/baseline.py  rule-based extractor
  eval/extraction.py   accuracy, abstention, consistency, confidence
  eval/plots.py        report figures
  cli.py, cli_eval.py  command-line pipeline
scripts/label_gold.py  interactive gold-label review
tests/                 63 tests; test_leakage.py covers the split/timing logic
```

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

`tests/test_leakage.py` is the set that matters — a leak makes the project *look*
right while being worthless, so the embargo, the chronological ordering, the
event-day resolution, and the DST boundary are all asserted directly.

---

## Limitations

- **Yahoo Finance data.** Free, occasionally revised, and not survivorship-bias-free.
  The universe is chosen from companies that exist today, which biases toward
  survivors. Fine for a methods demonstration; not a backtest you'd trade.
- **40 large caps.** Mega-cap disclosure practice is not small-cap disclosure
  practice, and the results shouldn't be extrapolated there.
- **Realized vol from daily closes** is a coarse estimator over a 5-day window.
  Intraday or options-implied vol would be sharper.
- **The gold set is small.** Field accuracy on ~60 documents carries wide confidence
  intervals; self-consistency is reported partly because it scales to the full corpus.
- **This is not trading advice or a trading system.** There is no execution model, no
  transaction costs, and no position sizing.
