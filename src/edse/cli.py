"""Command-line entry point. Run `edse --help` for the pipeline stages."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from .config import (
    CONFIGS_DIR,
    FIGURES_DIR,
    GOLD_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    Config,
    ensure_dirs,
)

log = logging.getLogger("edse")

DOCUMENTS_DIR = RAW_DIR / "documents"
FILINGS_PATH = RAW_DIR / "filings.parquet"
PRICES_PATH = RAW_DIR / "prices.parquet"
EVENTS_PATH = PROCESSED_DIR / "events_labeled.parquet"
GOLD_PATH = GOLD_DIR / "gold_labels.jsonl"

#: Large caps across sectors with long, uninterrupted 8-K histories. Breadth
#: matters because volatility dynamics differ sharply by sector, and a
#: tech-only universe would not generalize.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "INTC", "CSCO", "ORCL", "ADBE", "CRM", "TXN", "QCOM",
    "JPM", "BAC", "GS", "WFC", "AXP", "MS",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY",
    "WMT", "HD", "MCD", "NKE", "PG", "KO", "PEP", "COST",
    "BA", "CAT", "GE", "HON", "UPS", "LMT",
    "CVX", "T", "VZ", "DIS",
]


def _universe_path() -> Path:
    return CONFIGS_DIR / "universe.yaml"


def _load_universe() -> list[dict]:
    path = _universe_path()
    if not path.exists():
        raise SystemExit("universe.yaml not found - run `edse universe` first")
    return yaml.safe_load(path.read_text())["companies"]


# ---------------------------------------------------------------- universe
def cmd_universe(args, cfg: Config) -> None:
    """Resolve tickers to CIKs via SEC's own mapping and record filing counts."""
    from .ingest.edgar import EdgarClient

    client = EdgarClient(cfg.edgar_user_agent, cfg.edgar["requests_per_second"])
    mapping = client.ticker_map()
    tickers = args.tickers.split(",") if args.tickers else DEFAULT_UNIVERSE

    companies, skipped = [], []
    for ticker in tickers:
        entry = mapping.get(ticker)
        if entry is None:
            skipped.append((ticker, "no CIK in SEC mapping"))
            continue
        cik, name = entry
        count = sum(
            1
            for _ in client.iter_filings(
                ticker, cik, cfg.edgar["required_item"],
                cfg.edgar["start_date"], cfg.edgar["end_date"],
            )
        )
        # A CIK with almost no Item 2.02 history usually signals a reorganization
        # that moved the filing entity - including it would silently shrink the
        # per-ticker sample without explanation.
        if count < args.min_filings:
            skipped.append((ticker, f"only {count} Item 2.02 filings"))
            continue
        companies.append({"ticker": ticker, "cik": cik, "name": name, "n_filings": count})
        print(f"  {ticker:6s} CIK={cik:<10d} {count:3d} filings  {name}")

    _universe_path().write_text(
        yaml.safe_dump({"companies": companies}, sort_keys=False, default_flow_style=False)
    )
    total = sum(c["n_filings"] for c in companies)
    print(f"\n{len(companies)} companies, {total} earnings 8-Ks -> {_universe_path()}")
    for ticker, reason in skipped:
        print(f"  skipped {ticker}: {reason}")


# ------------------------------------------------------------------ ingest
def cmd_ingest(args, cfg: Config) -> None:
    """Discover earnings 8-Ks and download their press-release exhibits."""
    from .ingest.edgar import EdgarClient

    ensure_dirs()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    companies = _load_universe()
    client = EdgarClient(cfg.edgar_user_agent, cfg.edgar["requests_per_second"])

    filings = []
    for company in companies:
        found = list(
            client.iter_filings(
                company["ticker"], company["cik"], cfg.edgar["required_item"],
                cfg.edgar["start_date"], cfg.edgar["end_date"],
            )
        )
        filings.extend(found)
        log.info("%s: %d filings", company["ticker"], len(found))

    if args.limit:
        filings = filings[: args.limit]

    kept, missing = [], 0
    for i, filing in enumerate(filings, 1):
        doc_path = DOCUMENTS_DIR / f"{filing.accession}.txt"
        if doc_path.exists() and not args.refresh:
            kept.append(filing.to_dict())
            continue
        result = client.fetch_press_release(filing, cfg.edgar["max_exhibit_chars"])
        if result is None:
            missing += 1
            continue
        exhibit, text = result
        doc_path.write_text(text)
        kept.append({**filing.to_dict(), "exhibit": exhibit, "n_chars": len(text)})
        if i % 50 == 0:
            log.info("fetched %d/%d exhibits", i, len(filings))

    frame = pd.DataFrame(kept).drop_duplicates(subset=["accession"])
    frame.to_parquet(FILINGS_PATH, index=False)
    print(
        f"{len(frame)} filings with press releases -> {FILINGS_PATH}\n"
        f"  {missing} filings had no usable EX-99 exhibit\n"
        f"  documents -> {DOCUMENTS_DIR}"
    )


def cmd_prices(args, cfg: Config) -> None:
    """Download daily prices for the universe plus the market proxy."""
    from .ingest.prices import download_prices

    ensure_dirs()
    tickers = [c["ticker"] for c in _load_universe()]
    # Pad the start so the earliest event still has a full pre-window.
    start = (pd.Timestamp(cfg.edgar["start_date"]) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    frame = download_prices(tickers, start, cfg.edgar["end_date"], PRICES_PATH, refresh=args.refresh)
    print(f"{len(frame)} rows, {frame['ticker'].nunique()} tickers -> {PRICES_PATH}")


# ----------------------------------------------------------------- extract
def _build_extractor(cfg: Config, kind: str, model: str | None, temperature: float | None):
    if kind == "baseline":
        from .extract.baseline import RuleBasedExtractor

        return RuleBasedExtractor()

    if kind == "local":
        from .extract.local import OllamaExtractor

        extractor = OllamaExtractor(
            model=model or cfg.extraction["local_model"],
            temperature=temperature,
            num_ctx=cfg.extraction["local_num_ctx"],
        )
        extractor.health_check()  # fail now, not 400 documents in
        return extractor

    from .extract.llm import ClaudeExtractor

    if not cfg.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or run with `--extractor baseline` to use the rule-based extractor."
        )
    return ClaudeExtractor(
        model=model or cfg.extraction["model"],
        max_tokens=cfg.extraction["max_tokens"],
        temperature=temperature,
    )


def cmd_extract(args, cfg: Config) -> None:
    """Extract structured claims from every downloaded press release."""
    from .extract.base import ExtractionCache

    ensure_dirs()
    if not FILINGS_PATH.exists():
        raise SystemExit("no filings found - run `edse ingest` first")

    filings = pd.read_parquet(FILINGS_PATH)
    if args.limit:
        filings = filings.head(args.limit)

    extractor = _build_extractor(cfg, args.extractor, args.model, args.temperature)
    cache = ExtractionCache(INTERIM_DIR / "extractions")

    from .textprep import prepare

    max_chars = cfg.extraction.get("max_document_chars")
    tasks, prepared_docs = [], []
    for rec in filings.to_dict("records"):
        doc = DOCUMENTS_DIR / f"{rec['accession']}.txt"
        if not doc.exists():
            continue
        # Trim financial-statement tables before extraction; every schema field
        # is answerable from the narrative, and the tables are what push
        # documents past the context window.
        prepped = prepare(doc.read_text(), max_chars=max_chars)
        prepared_docs.append(prepped)
        key = cache.key(
            extractor.name, extractor.model, extractor.prompt_version, prepped.text
        )
        tasks.append((rec, prepped.text, key))

    if prepared_docs:
        trimmed = sum(d.trimmed_at_marker for d in prepared_docs)
        hard = sum(d.hard_truncated for d in prepared_docs)
        saved = 1 - sum(d.chars for d in prepared_docs) / sum(
            d.original_chars for d in prepared_docs
        )
        print(
            f"document prep: {trimmed}/{len(prepared_docs)} trimmed at a statement "
            f"header, {saved:.0%} of characters removed"
            + (f", {hard} hard-truncated at the cap" if hard else "")
        )

    cached = {k: cache.get(k) for _, _, k in tasks}
    pending = [t for t in tasks if cached.get(t[2]) is None]
    print(
        f"{len(tasks)} documents | {len(tasks) - len(pending)} already cached | "
        f"{len(pending)} to extract with {extractor.model}"
    )
    if args.dry_run:
        return

    results = [r for r in cached.values() if r is not None]

    def run(task):
        rec, text, key = task
        result = extractor.extract(
            text, ticker=rec["ticker"], accession=rec["accession"],
            filing_date=rec.get("filing_date", ""),
        )
        if result.ok:  # never cache a failure - it must be retryable on re-run
            cache.put(key, result)
        return result

    if pending:
        # Local inference is GPU-bound on one machine: concurrent requests queue
        # inside Ollama and add no throughput, so keep it serial.
        workers = cfg.extraction["max_workers"] if args.extractor == "claude" else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, t): t for t in pending}
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if i % 25 == 0 or i == len(pending):
                    spend = sum(r.cost_usd for r in results)
                    print(f"  {i}/{len(pending)} extracted | ${spend:.2f} so far", flush=True)

    _write_extractions(results, extractor.name)


def _write_extractions(results: list, extractor_name: str) -> None:
    ok = [r for r in results if r.ok and r.claims]
    failed = [r for r in results if not r.ok]

    rows = [{"accession": r.accession, "ticker": r.ticker, **r.claims.model_dump(mode="json")}
            for r in ok]
    out = PROCESSED_DIR / f"extractions_{extractor_name}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)

    total_cost = sum(r.cost_usd for r in results)
    cache_read = sum(r.cache_read_tokens for r in results)
    total_in = sum(r.input_tokens + r.cache_read_tokens + r.cache_write_tokens for r in results)
    stats = {
        "extractor": extractor_name,
        "n_ok": len(ok),
        "n_failed": len(failed),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_doc_usd": round(total_cost / max(len(ok), 1), 5),
        "mean_latency_s": round(
            sum(r.latency_s for r in results) / max(len(results), 1), 2
        ),
        "input_tokens": total_in,
        "output_tokens": sum(r.output_tokens for r in results),
        "cache_read_tokens": cache_read,
        # Measured, not assumed: if the system prompt is below the model's
        # minimum cacheable length this stays at 0.0, and that is the honest
        # answer rather than a claim that caching "is enabled".
        "cache_hit_rate": round(cache_read / total_in, 4) if total_in else 0.0,
    }
    (REPORTS_DIR / f"extraction_stats_{extractor_name}.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{len(ok)} extractions -> {out}")
    print(json.dumps(stats, indent=2))
    if failed:
        reasons = pd.Series([r.error.split(":")[0] for r in failed]).value_counts()
        print(f"\n{len(failed)} failures:\n{reasons.to_string()}")


# ------------------------------------------------------------------- label
def cmd_label(args, cfg: Config) -> None:
    """Join filings to prices and build the volatility-expansion target."""
    from .ingest.prices import MARKET_TICKER, build_panels, to_returns, trading_days
    from .labels import compute_event_labels

    ensure_dirs()
    for path, hint in ((FILINGS_PATH, "edse ingest"), (PRICES_PATH, "edse prices")):
        if not path.exists():
            raise SystemExit(f"missing {path.name} - run `{hint}` first")

    filings = pd.read_parquet(FILINGS_PATH)
    returns = to_returns(pd.read_parquet(PRICES_PATH))
    panels = build_panels(returns)
    market = panels.get(MARKET_TICKER)
    if market is None:
        raise SystemExit(f"no price history for market proxy {MARKET_TICKER}")

    labeled = compute_event_labels(
        filings, panels, trading_days(returns), market,
        post_window=cfg.labels["post_window"],
        pre_window=cfg.labels["pre_window"],
        expansion_threshold=cfg.labels["expansion_threshold"],
        min_pre_obs=cfg.labels["min_pre_obs"],
        min_post_obs=cfg.labels["min_post_obs"],
    )
    if labeled.empty:
        raise SystemExit("no events could be labeled - check price coverage")

    labeled.to_parquet(EVENTS_PATH, index=False)
    timing = labeled["release_timing"].value_counts()
    print(
        f"{len(labeled)} labeled events -> {EVENTS_PATH}\n"
        f"  date range      : {labeled['t0'].min().date()} .. {labeled['t0'].max().date()}\n"
        f"  positive rate   : {labeled['y'].mean():.3f} "
        f"(vol expansion > {cfg.labels['expansion_threshold']}x)\n"
        f"  median expansion: {labeled['vol_expansion'].median():.2f}x\n"
        f"  release timing  :\n{timing.to_string()}"
    )


# ------------------------------------------------------------------- train
def cmd_train(args, cfg: Config) -> None:
    """Build features, run the ablation, and write the model report."""
    from .eval.plots import plot_ablation, plot_reliability
    from .features import build_features, flatten_claims
    from .model import permutation_importance_report, run_ablation

    ensure_dirs()
    if not EVENTS_PATH.exists():
        raise SystemExit("no labeled events - run `edse label` first")

    events = pd.read_parquet(EVENTS_PATH)
    extraction_path = PROCESSED_DIR / f"extractions_{args.extractor}.parquet"
    if not extraction_path.exists():
        raise SystemExit(f"missing {extraction_path.name} - run `edse extract` first")

    extractions = pd.read_parquet(extraction_path)
    claim_map = {
        rec["accession"]: {k: v for k, v in rec.items() if k not in ("accession", "ticker")}
        for rec in extractions.to_dict("records")
    }

    joined = flatten_claims(events, claim_map)
    X, blocks = build_features(joined)
    y = joined["y"].to_numpy()
    dates = joined["t0"]

    results = run_ablation(
        X, y, dates, blocks,
        estimator_kind=args.estimator,
        calibration_method=cfg.model["calibration_method"],
        n_splits=cfg.model["n_splits"],
        embargo_days=cfg.model["embargo_days"],
        test_fraction=cfg.model["test_fraction"],
        random_state=cfg.model["random_state"],
    )
    table = pd.DataFrame([r.row() for r in results])
    print("\n" + table.to_string(index=False))

    claim_source = {"claude": "Claude", "local": "local-LLM"}.get(args.extractor, "rule-based")
    plot_ablation(results, FIGURES_DIR / f"ablation_{args.extractor}.png", claim_source)
    combined = next((r for r in results if r.name == "controls_plus_claims"), results[-1])
    plot_reliability(
        combined.y_true, combined.probs,
        FIGURES_DIR / f"reliability_{args.extractor}.png",
        title=f"Calibration - controls + {args.extractor} claims",
    )

    importance = permutation_importance_report(
        X, y, dates,
        test_fraction=cfg.model["test_fraction"],
        embargo_days=cfg.model["embargo_days"],
        random_state=cfg.model["random_state"],
    )
    importance.to_csv(REPORTS_DIR / f"feature_importance_{args.extractor}.csv", index=False)

    payload = {
        "extractor": args.extractor,
        "estimator": args.estimator,
        "n_events": len(joined),
        "results": table.to_dict("records"),
        "top_features": importance.head(10).to_dict("records"),
    }
    (REPORTS_DIR / f"model_results_{args.extractor}.json").write_text(json.dumps(payload, indent=2))

    controls = next((r for r in results if r.name == "controls_only"), None)
    if controls is not None:
        print(
            f"\nIncremental value of {args.extractor} claims over controls alone:\n"
            f"  AUC         {controls.auc:.4f} -> {combined.auc:.4f} "
            f"({combined.auc - controls.auc:+.4f})\n"
            f"  Brier skill {controls.brier_skill:.4f} -> {combined.brier_skill:.4f} "
            f"({combined.brier_skill - controls.brier_skill:+.4f})"
        )
    print(f"\nfigures -> {FIGURES_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edse", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("universe", help="resolve tickers to CIKs and count filings")
    p.add_argument("--tickers", default=None, help="comma-separated; defaults to the built-in list")
    p.add_argument("--min-filings", type=int, default=8)
    p.set_defaults(func=cmd_universe)

    p = sub.add_parser("ingest", help="download earnings 8-K press releases")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("prices", help="download daily price history")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_prices)

    p = sub.add_parser("extract", help="extract structured claims")
    p.add_argument(
        "--extractor", choices=["claude", "local", "baseline"], default="local",
        help="local = free Ollama model (default); claude = hosted API; baseline = rules",
    )
    p.add_argument("--model", default=None, help="override the configured model")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="report cost/coverage without calling the API")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("label", help="build the volatility-expansion target")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="run the ablation and write the model report")
    p.add_argument("--extractor", default="local")
    p.add_argument("--estimator", choices=["gbm", "logistic"], default="gbm")
    p.set_defaults(func=cmd_train)

    from .cli_eval import register as register_eval

    register_eval(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    args.func(args, Config.load(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
