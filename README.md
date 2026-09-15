# Earnings Disclosure Signal Engine

An LLM extracts structured claims from SEC 8-K earnings releases; those claims become
features in a calibrated model predicting post-announcement volatility, with an eval
harness that measures extraction quality independently of downstream performance.

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

## Results

<!-- RESULTS -->

---

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # add EDGAR_USER_AGENT (required) and ANTHROPIC_API_KEY
```

`EDGAR_USER_AGENT` must contain a contact email — SEC blocks anonymous clients,
and the failure mode is silent (empty result sets), so the code refuses to start
without one.

```bash
edse universe                 # resolve tickers -> CIKs via SEC's own mapping
edse ingest                   # download 8-K Item 2.02 press-release exhibits
edse prices                   # daily OHLCV for the universe + SPY
edse label                    # event-time alignment and the volatility target

edse extract --extractor baseline     # free, no API key
edse extract --extractor claude       # requires ANTHROPIC_API_KEY
edse extract --extractor claude --dry-run   # coverage + cost, no API calls

edse train --extractor claude         # ablation, calibration, figures
```

Extraction quality (needs hand labels):

```bash
edse gold --sample 60                 # stratified labeling template
# ... edit data/gold/gold_labels.jsonl, flip "_reviewed": true ...
edse eval-extraction                  # grade claude vs. baseline
edse consistency --runs 3             # stability, no gold labels needed
```

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
  extract/llm.py       Claude extractor (structured outputs)
  extract/baseline.py  rule-based extractor
  eval/extraction.py   accuracy, abstention, consistency, confidence
  eval/plots.py        report figures
  cli.py, cli_eval.py  command-line pipeline
tests/                 54 tests; test_leakage.py covers the split/timing logic
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
