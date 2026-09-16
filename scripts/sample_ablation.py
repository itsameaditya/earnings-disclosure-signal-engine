#!/usr/bin/env python3
"""Extract a stratified sample of the corpus, for an ablation before the full run finishes.

Why this exists, and what it is careful about.

Local extraction over all 1,447 filings is hours of laptop GPU time, and the
tempting shortcut -- report the ablation on whatever has been extracted so far --
is invalid here. `data/raw/filings.parquet` is ordered by issuer, so any prefix
is a handful of tech mega-caps; an AUC computed on it would be a sector number
wearing a corpus-wide label.

A stratified random sample is a different object. Sampling proportionally within
issuer, with a fixed seed, gives an unbiased estimate with wider confidence
intervals -- honest as long as it is labeled a sample, which is what
`--extractor local-sample` keeps distinct downstream.

Two properties worth stating because they are easy to get wrong:

- **Extractions land in the shared cache**, so nothing here is throwaway work: a
  document extracted for the sample is already done when the full-corpus run
  reaches it.
- **The sample is drawn from the whole corpus**, not from the un-extracted
  remainder. Drawing from the remainder and unioning with the already-extracted
  prefix would reproduce exactly the issuer bias this is meant to avoid.

    python scripts/sample_ablation.py --n 600
    edse train --extractor local-sample
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from edse.cli import DOCUMENTS_DIR, FILINGS_PATH, _build_extractor, _write_extractions
from edse.config import Config, ensure_dirs
from edse.extract.base import ExtractionCache
from edse.textprep import prepare

SAMPLE_NAME = "local-sample"


def stratified_sample(filings: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional-allocation random sample, stratified by issuer.

    Proportional rather than equal allocation so the sample keeps the corpus's
    issuer mix instead of inventing a balanced one the population does not have.
    Every issuer with at least one filing contributes at least one event, so no
    issuer silently drops out of the ablation.
    """
    share = n / len(filings)
    parts = []
    for _, group in filings.groupby("ticker", sort=True):
        take = max(1, round(len(group) * share))
        parts.append(group.sample(min(take, len(group)), random_state=seed))
    return pd.concat(parts).sort_values("accession")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="target sample size")
    args = ap.parse_args()

    ensure_dirs()
    cfg = Config.load()
    if not FILINGS_PATH.exists():
        raise SystemExit("no filings found - run `edse ingest` first")

    filings = pd.read_parquet(FILINGS_PATH)
    sample = stratified_sample(filings, args.n, cfg.model["random_state"])

    extractor = _build_extractor(cfg, "local", None, None)
    cache = ExtractionCache(Path(cfg.extraction["cache_dir"]))
    max_chars = cfg.extraction.get("max_document_chars")

    tasks = []
    for rec in sample.to_dict("records"):
        doc = DOCUMENTS_DIR / f"{rec['accession']}.txt"
        if not doc.exists():
            continue
        text = prepare(doc.read_text(), max_chars=max_chars).text
        key = cache.key(extractor.name, extractor.model, extractor.prompt_version, text)
        tasks.append((rec, text, key))

    years = sorted({str(r["filing_date"])[:4] for r, _, _ in tasks})
    cached = {k: cache.get(k) for _, _, k in tasks}
    pending = [t for t in tasks if cached.get(t[2]) is None]
    print(
        f"stratified sample: {len(tasks)} events | "
        f"{sample['ticker'].nunique()} issuers | {years[0]}-{years[-1]}\n"
        f"{len(tasks) - len(pending)} already cached | {len(pending)} to extract",
        flush=True,
    )

    results = [r for r in cached.values() if r is not None]

    def run(task):
        rec, text, key = task
        result = extractor.extract(
            text,
            ticker=rec["ticker"],
            accession=rec["accession"],
            filing_date=rec.get("filing_date", ""),
        )
        if result.ok:
            cache.put(key, result)
        return result

    if pending:
        workers = cfg.extraction.get("local_max_workers", 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(run, pending), 1):
                results.append(result)
                if i % 25 == 0 or i == len(pending):
                    print(f"  {i}/{len(pending)} extracted", flush=True)

    # Written under the sample's own name so it can never be confused with, or
    # overwrite, the full-corpus artifacts.
    _write_extractions(results, SAMPLE_NAME)
    print(f"\nnext: edse train --extractor {SAMPLE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
