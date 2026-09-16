"""Evaluation subcommands: gold labeling, extraction grading, consistency."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import FIGURES_DIR, GOLD_DIR, PROCESSED_DIR, REPORTS_DIR, Config, ensure_dirs
from .schema import GRADED_FIELDS

log = logging.getLogger("edse.eval")

GOLD_PATH = GOLD_DIR / "gold_labels.jsonl"


def _load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise SystemExit(f"no gold labels at {GOLD_PATH} - run `edse gold init` first")
    records = [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]
    reviewed = [r for r in records if r.get("_reviewed")]
    if not reviewed:
        raise SystemExit(
            f"{GOLD_PATH} has {len(records)} rows but none marked `\"_reviewed\": true`.\n"
            "Edit the file, correct each field, and flip _reviewed to true on the rows you finish."
        )
    log.info("loaded %d reviewed gold labels (of %d rows)", len(reviewed), len(records))
    return pd.DataFrame(reviewed).set_index("accession")


def cmd_gold_init(args, cfg: Config) -> None:
    """Create a gold-label template, stratified across tickers and years.

    Rows are optionally pre-filled from the **rule-based** extractor, never from
    the LLM. Pre-filling from the system under evaluation would anchor the
    labeler toward its answers and inflate every score that follows; the baseline
    is a neutral starting point that still saves typing. `--prefill none` gives
    blank rows if you want to avoid anchoring entirely.
    """
    from .cli import DOCUMENTS_DIR, FILINGS_PATH

    ensure_dirs()
    if not FILINGS_PATH.exists():
        raise SystemExit("no filings found - run `edse ingest` first")

    filings = pd.read_parquet(FILINGS_PATH)
    filings["year"] = pd.to_datetime(filings["filing_date"]).dt.year

    # Stratify by filing year, sampling randomly within each year. Disclosure
    # style drifts over eight years, so year is the axis worth balancing; with 60
    # draws from 1.4k filings across 40 issuers, ticker diversity follows on its
    # own. (Stratifying by ticker instead gives 1-2 filings each, which forces
    # every pick to the same few positions in each issuer's history.)
    years = sorted(filings["year"].unique())
    per_year = max(1, -(-args.sample // len(years)))
    picked: list = []
    for _, group in filings.groupby("year", sort=True):
        picked.extend(
            group.sample(min(len(group), per_year), random_state=args.seed).index.tolist()
        )

    sample = filings.loc[picked].sample(frac=1.0, random_state=args.seed).head(args.sample)

    existing = set()
    if GOLD_PATH.exists():
        existing = {
            json.loads(line)["accession"]
            for line in GOLD_PATH.read_text().splitlines()
            if line.strip()
        }

    prefill = None
    if args.prefill == "baseline":
        from .extract.baseline import RuleBasedExtractor

        prefill = RuleBasedExtractor()

    written = 0
    with GOLD_PATH.open("a") as fh:
        for rec in sample.to_dict("records"):
            if rec["accession"] in existing:
                continue
            doc = DOCUMENTS_DIR / f"{rec['accession']}.txt"
            if not doc.exists():
                continue

            values = {f: None for f in GRADED_FIELDS}
            if prefill is not None:
                result = prefill.extract(
                    doc.read_text(), ticker=rec["ticker"], accession=rec["accession"]
                )
                dumped = result.claims.model_dump(mode="json")
                values = {f: dumped.get(f) for f in GRADED_FIELDS}

            fh.write(
                json.dumps(
                    {
                        "accession": rec["accession"],
                        "ticker": rec["ticker"],
                        "filing_date": rec["filing_date"],
                        "document": str(doc.relative_to(doc.parents[3])),
                        "_reviewed": False,
                        "_prefill": args.prefill,
                        **values,
                    }
                )
                + "\n"
            )
            written += 1

    print(
        f"added {written} template rows -> {GOLD_PATH}\n\n"
        f"Next: open each document under {DOCUMENTS_DIR}, correct the fields, and set\n"
        f'"_reviewed": true on the rows you finish. Only reviewed rows are graded.\n'
        f"Field definitions and the labeling rubric are in src/edse/schema.py."
    )


def cmd_eval_extraction(args, cfg: Config) -> None:
    """Grade extractors against the gold labels and plot the comparison."""
    from .eval.extraction import (
        abstention_report,
        confidence_calibration,
        extraction_report,
    )
    from .eval.plots import plot_extraction_quality

    ensure_dirs()
    gold = _load_gold()
    rtol = cfg.eval["numeric_rtol"]

    tables, summaries = {}, {}
    for extractor in args.extractors.split(","):
        path = PROCESSED_DIR / f"extractions_{extractor}.parquet"
        if not path.exists():
            log.warning("skipping %s: %s not found", extractor, path.name)
            continue
        preds = pd.read_parquet(path).set_index("accession")
        table, summary = extraction_report(preds, gold, rtol)
        tables[extractor], summaries[extractor] = table, summary

        print(f"\n=== {extractor} ===")
        print(table.to_string(index=False))
        print(json.dumps(summary, indent=2))

        table.to_csv(REPORTS_DIR / f"extraction_quality_{extractor}.csv", index=False)
        abst = abstention_report(preds, gold)
        abst.to_csv(REPORTS_DIR / f"abstention_{extractor}.csv", index=False)
        print("\nabstention behaviour:")
        print(abst.to_string(index=False))

        conf = confidence_calibration(preds, gold, rtol)
        if not conf.empty:
            conf.to_csv(REPORTS_DIR / f"confidence_calibration_{extractor}.csv", index=False)
            print("\nself-reported confidence vs. measured accuracy:")
            print(conf.to_string(index=False))

    if not tables:
        raise SystemExit("no extractions found to grade - run `edse extract` first")

    plot_extraction_quality(
        tables.get("claude", next(iter(tables.values()))),
        tables.get("baseline"),
        FIGURES_DIR / "extraction_quality.png",
    )
    (REPORTS_DIR / "extraction_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nfigures -> {FIGURES_DIR}")


def prepared_texts(
    records: list[dict], documents_dir: Path, max_chars: int | None
) -> dict[str, str]:
    """Document text prepared the way `edse extract` prepares it, keyed by accession.

    Consistency has to send the extractor the *same* text the real extraction
    stage sends, or it measures a pipeline that does not exist. `edse extract`
    trims financial-statement tables first; the untrimmed documents are also the
    ones long enough for the server's context shift to drop text silently, so
    measuring stability on raw text would report fields as unstable partly
    because different runs were answering about text the model never saw.

    Missing documents are skipped rather than raising: the corpus and the
    document directory are allowed to disagree, and the caller reports on what
    it actually has.
    """
    from .textprep import prepare

    texts = {}
    for rec in records:
        doc = documents_dir / f"{rec['accession']}.txt"
        if doc.exists():
            texts[rec["accession"]] = prepare(doc.read_text(), max_chars=max_chars).text
    return texts


def cmd_consistency(args, cfg: Config) -> None:
    """Repeat extraction at temperature>0 and measure agreement across runs.

    This needs no gold labels, so it is the one extraction-quality signal that
    scales to the whole corpus. An unstable field is unusable downstream even if
    it scores well on a small gold sample.
    """
    from .cli import DOCUMENTS_DIR, FILINGS_PATH, _build_extractor
    from .eval.extraction import consistency_report
    from .eval.plots import plot_consistency

    ensure_dirs()
    if not FILINGS_PATH.exists():
        raise SystemExit("no filings found - run `edse ingest` first")

    filings = pd.read_parquet(FILINGS_PATH).sample(
        min(cfg.eval["consistency_sample_size"], len(pd.read_parquet(FILINGS_PATH))),
        random_state=cfg.model["random_state"],
    )
    n_runs = args.runs or cfg.eval["consistency_runs"]
    temperature = cfg.eval["consistency_temperature"]

    prepared = prepared_texts(
        filings.to_dict("records"),
        DOCUMENTS_DIR,
        cfg.extraction.get("max_document_chars"),
    )

    runs = []
    for run_index in range(n_runs):
        extractor = _build_extractor(cfg, args.extractor, args.model, temperature)
        rows = []
        print(f"run {run_index + 1}/{n_runs} ...", flush=True)
        for rec in filings.to_dict("records"):
            text = prepared.get(rec["accession"])
            if text is None:
                continue
            result = extractor.extract(
                text, ticker=rec["ticker"], accession=rec["accession"],
                filing_date=rec.get("filing_date", ""),
            )
            if result.ok and result.claims:
                rows.append({"accession": result.accession, **result.claims.model_dump(mode="json")})
        runs.append(pd.DataFrame(rows).set_index("accession"))

    table, summary = consistency_report(runs)
    print("\n" + table.to_string(index=False))
    print(json.dumps(summary, indent=2))

    table.to_csv(REPORTS_DIR / "consistency.csv", index=False)
    (REPORTS_DIR / "consistency_summary.json").write_text(json.dumps(summary, indent=2))
    plot_consistency(table, FIGURES_DIR / "consistency.png")


def register(sub) -> None:
    """Attach the evaluation subcommands to the CLI parser."""
    p = sub.add_parser("gold", help="create a hand-labeling template for extraction eval")
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--prefill", choices=["baseline", "none"], default="baseline",
                   help="pre-fill from the rule-based extractor (never the LLM) to reduce typing")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_gold_init)

    p = sub.add_parser("eval-extraction", help="grade extractors against gold labels")
    p.add_argument("--extractors", default="claude,baseline")
    p.set_defaults(func=cmd_eval_extraction)

    p = sub.add_parser("consistency", help="measure extraction stability across repeated runs")
    p.add_argument("--extractor", choices=["claude", "baseline"], default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--runs", type=int, default=None)
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("report", help="assemble reports/REPORT.md from pipeline artifacts")
    p.set_defaults(func=cmd_report)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cmd_report(args, cfg: Config) -> None:
    """Assemble reports/REPORT.md from whatever artifacts exist.

    Deliberately tolerant of missing stages: it reports what has actually been
    run rather than failing, so a partial pipeline still produces a readable
    summary that names its own gaps.
    """
    ensure_dirs()
    lines: list[str] = [
        "# Earnings Disclosure Signal Engine - results",
        "",
        "Generated by `edse report`. Every number below is produced by the pipeline;",
        "sections are omitted when the corresponding stage has not been run.",
        "",
    ]

    events_path = PROCESSED_DIR / "events_labeled.parquet"
    if events_path.exists():
        events = pd.read_parquet(events_path)
        timing = events["release_timing"].value_counts()
        lines += [
            "## Dataset",
            "",
            (
                f"- **{len(events):,} labeled earnings events** across "
                f"{events['ticker'].nunique()} companies"
            ),
            f"- {events['t0'].min().date()} to {events['t0'].max().date()}",
            (
                f"- Positive rate (vol expansion > {cfg.labels['expansion_threshold']}x): "
                f"**{events['y'].mean():.1%}**"
            ),
            f"- Median vol expansion: {events['vol_expansion'].median():.2f}x",
            "",
            "Release timing (this is why event-day alignment matters):",
            "",
            "| timing | events | share |",
            "|---|---:|---:|",
        ]
        for name, count in timing.items():
            lines.append(f"| {name} | {count:,} | {count / len(events):.1%} |")
        lines.append("")

    for extractor in ("claude", "baseline"):
        stats = _read_json(REPORTS_DIR / f"extraction_stats_{extractor}.json")
        if not stats:
            continue
        lines += [
            f"## Extraction - {extractor}",
            "",
            f"- Documents extracted: **{stats['n_ok']:,}** ({stats['n_failed']} failed)",
            (
                f"- Total cost: **${stats['total_cost_usd']:.2f}** "
                f"(${stats['cost_per_doc_usd']:.5f}/doc)"
            ),
            f"- Mean latency: {stats['mean_latency_s']:.2f}s",
            (
                f"- Measured cache hit rate: **{stats['cache_hit_rate']:.1%}** "
                f"({stats['cache_read_tokens']:,} of {stats['input_tokens']:,} input tokens)"
            ),
            "",
        ]

    quality = _read_json(REPORTS_DIR / "extraction_summary.json")
    if quality:
        lines += [
            "## Extraction quality vs. hand-labeled gold",
            "",
            "| extractor | docs | mean field accuracy | mean macro-F1 | document exact match |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, summary in quality.items():
            lines.append(
                f"| {name} | {summary['n_documents']} | "
                f"{summary['mean_field_accuracy']:.3f} | "
                f"{summary['mean_macro_f1']:.3f} | "
                f"{summary['document_exact_match']:.3f} |"
            )
        lines += ["", "![extraction quality](figures/extraction_quality.png)", ""]

    consistency = _read_json(REPORTS_DIR / "consistency_summary.json")
    if consistency:
        lines += [
            "## Extraction stability",
            "",
            (
                f"Across {consistency['n_runs']} runs at temperature "
                f"{cfg.eval['consistency_temperature']} on {consistency['n_documents']} docs: "
                f"mean modal agreement **{consistency['mean_agreement']:.3f}**, "
                f"least stable field `{consistency['least_stable_field']}`."
            ),
            "",
            "![consistency](figures/consistency.png)",
            "",
        ]

    for extractor in ("claude", "baseline"):
        results = _read_json(REPORTS_DIR / f"model_results_{extractor}.json")
        if not results:
            continue
        lines += [
            f"## Prediction - {extractor} claims",
            "",
            f"{results['n_events']:,} events, {results['estimator']} + isotonic calibration.",
            "",
            "| feature set | AUC | Brier | Brier skill | ECE |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in results["results"]:
            lines.append(
                f"| {row['model'].replace('_', ' ')} | {row['auc']:.4f} | "
                f"{row['brier']:.4f} | {row['brier_skill']:+.4f} | {row['ece']:.4f} |"
            )
        by_name = {r["model"]: r for r in results["results"]}
        if "controls_only" in by_name and "controls_plus_claims" in by_name:
            delta_auc = by_name["controls_plus_claims"]["auc"] - by_name["controls_only"]["auc"]
            delta_bs = (
                by_name["controls_plus_claims"]["brier_skill"]
                - by_name["controls_only"]["brier_skill"]
            )
            verdict = "adds signal" if delta_auc > 0 and delta_bs > 0 else "does not add signal"
            lines += [
                "",
                (
                    f"**Incremental value of the claim block: {delta_auc:+.4f} AUC, "
                    f"{delta_bs:+.4f} Brier skill - the claim block {verdict} "
                    f"over market-state controls alone.**"
                ),
            ]
        lines += [
            "",
            f"![ablation](figures/ablation_{extractor}.png)",
            "",
            f"![reliability](figures/reliability_{extractor}.png)",
            "",
            "Top features by permutation importance (AUC drop on the holdout):",
            "",
            "| feature | AUC drop |",
            "|---|---:|",
        ]
        for row in results["top_features"][:10]:
            lines.append(f"| `{row['feature']}` | {row['auc_drop']:+.4f} |")
        lines.append("")

    out = REPORTS_DIR / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(lines)} lines)")
