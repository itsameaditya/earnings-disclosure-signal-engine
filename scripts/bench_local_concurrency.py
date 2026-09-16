#!/usr/bin/env python3
"""Measure local-extractor throughput at a given client concurrency.

This is the script behind the "On local concurrency" table in the README. That
table replaced a claim this project had asserted without measuring -- that
concurrent requests just queue inside Ollama -- so the measurement lives in the
repo rather than in a commit message.

Protocol, two runs:

    pkill -f "ollama serve"
    OLLAMA_NUM_PARALLEL=1 ollama serve &
    python scripts/bench_local_concurrency.py --arm a --workers 1

    pkill -f "ollama serve"
    OLLAMA_NUM_PARALLEL=2 ollama serve &
    python scripts/bench_local_concurrency.py --arm b --workers 2

The server restart is deliberately left to you: OLLAMA_NUM_PARALLEL is read at
server start, and a script in this repo has no business killing a daemon it did
not launch. Client concurrency above the server's OLLAMA_NUM_PARALLEL measures
queuing, not batching -- which is a legitimate thing to measure, but know which
one you are doing.

Why the measurement is shaped this way:

- **Disjoint documents per arm** (`--arm a` / `--arm b`). Re-using documents
  would let the second arm hit Ollama's KV prefix cache and look fast for the
  wrong reason.
- **Arms balanced on length.** Latency is dominated by prompt prefill, so a
  length imbalance between arms reads directly as a throughput difference. A
  first version of this sorted the sample and alternated, which quietly handed
  one arm a document one rank longer every time -- a 24% length advantage. This
  takes adjacent pairs and alternates which arm gets the longer member, so the
  offsets cancel.
- **Normalised on prompt tokens.** Ollama reports `prompt_eval_count`, which is
  what prefill time actually scales with; s/doc is only comparable when the arms
  match exactly, and tokens/s survives residual imbalance.
- **Results are written to the real extraction cache.** Both arms extract
  genuine uncached corpus documents through the real extractor at temperature 0,
  so the GPU time is not wasted -- a subsequent `edse extract` finds them done.
  Concurrency changes scheduling, not per-document output.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from edse.config import Config
from edse.extract.base import ExtractionCache
from edse.extract.local import OllamaExtractor
from edse.textprep import prepare

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "raw" / "documents"
FILINGS = ROOT / "data" / "raw" / "filings.parquet"
CACHE = ROOT / "data" / "interim" / "extractions"


def uncached_tasks(cfg: Config, extractor: OllamaExtractor, cache: ExtractionCache):
    """Every corpus document with no cache entry, with its prepared text."""
    if not FILINGS.exists():
        raise SystemExit("no filings found - run `edse ingest` first")
    filings = pd.read_parquet(FILINGS)
    max_chars = cfg.extraction.get("max_document_chars")
    pending = []
    for rec in filings.to_dict("records"):
        doc = DOCS / f"{rec['accession']}.txt"
        if not doc.exists():
            continue
        prepped = prepare(doc.read_text(), max_chars=max_chars)
        key = cache.key(
            extractor.name, extractor.model, extractor.prompt_version, prepped.text
        )
        if cache.get(key) is None:
            pending.append((rec, prepped.text, key))
    return pending


def split_arms(pending, n_per_arm: int):
    """Two disjoint, length-balanced arms.

    Sample across the whole length distribution, then hand out adjacent pairs in
    ABBA order so neither arm systematically draws the longer document.
    """
    pool = sorted(pending, key=lambda t: len(t[1]))
    need = n_per_arm * 2
    step = max(1, len(pool) // need)
    picked = pool[::step][:need]

    arm_a, arm_b = [], []
    for i in range(0, len(picked) - 1, 2):
        lo, hi = picked[i], picked[i + 1]
        if (i // 2) % 2 == 0:
            arm_a.append(lo)
            arm_b.append(hi)
        else:
            arm_a.append(hi)
            arm_b.append(lo)
    return arm_a, arm_b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("a", "b"), required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--n", type=int, default=12, help="documents per arm")
    args = ap.parse_args()

    cfg = Config.load()
    extractor = OllamaExtractor(
        model=cfg.extraction["local_model"],
        num_ctx=cfg.extraction["local_num_ctx"],
    )
    extractor.health_check()
    cache = ExtractionCache(CACHE)

    pending = uncached_tasks(cfg, extractor, cache)
    if len(pending) < args.n * 2:
        raise SystemExit(
            f"need {args.n * 2} uncached documents for two disjoint arms, "
            f"have {len(pending)}. Lower --n, or clear some cache entries."
        )

    arm_a, arm_b = split_arms(pending, args.n)
    arm = arm_a if args.arm == "a" else arm_b

    chars = sum(len(t[1]) for t in arm)
    print(
        f"arm {args.arm}: n={len(arm)} workers={args.workers} "
        f"mean_chars={chars / len(arm):.0f}",
        flush=True,
    )

    def run(task):
        rec, text, key = task
        started = time.perf_counter()
        result = extractor.extract(
            text,
            ticker=rec["ticker"],
            accession=rec["accession"],
            filing_date=rec.get("filing_date", ""),
        )
        elapsed = time.perf_counter() - started
        if result.ok:
            cache.put(key, result)
        return result, elapsed

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(run, arm))
    wall = time.perf_counter() - t0

    ok = [r for r, _ in outcomes if r.ok]
    if not ok:
        raise SystemExit("every extraction failed - check the Ollama server")
    prompt_tok = sum(r.input_tokens for r in ok)
    out_tok = sum(r.output_tokens for r in ok)
    latencies = sorted(dt for _, dt in outcomes)

    print(
        f"\nworkers={args.workers} wall={wall:.1f}s docs={len(arm)} ok={len(ok)}\n"
        f"  prompt tokens    {prompt_tok} ({prompt_tok / wall:.1f}/s)\n"
        f"  output tokens    {out_tok} ({out_tok / wall:.1f}/s)\n"
        f"  s/doc            {wall / len(arm):.2f}\n"
        f"  latency median   {latencies[len(latencies) // 2]:.1f}s\n"
        f"  latency max      {latencies[-1]:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
