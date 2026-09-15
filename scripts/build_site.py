#!/usr/bin/env python3
"""Render the results site published at

    https://itsameaditya.github.io/earnings-disclosure-signal-engine

The site is two pages of rendered Markdown: `README.md` (the narrative, the
methodology and the caveats) and `reports/REPORT.md` (the machine-generated
numbers). Nothing here computes a result or reformats one -- if a metric is
wrong on the site it is wrong in the artifact, which is the property we want.

Figures are copied to two paths because the two source documents link to them
differently: README.md is read from the repo root (`reports/figures/x.png`) and
REPORT.md from inside `reports/` (`figures/x.png`). Both must resolve on the
site without rewriting either file, since both are also read on GitHub.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "build" / "site"

PAGES = [
    {
        "slug": "index",
        "source": ROOT / "README.md",
        "title": "Earnings Disclosure Signal Engine",
        "description": (
            "An LLM extracts structured claims from SEC 8-K earnings releases; "
            "those claims become features in a calibrated model predicting "
            "post-announcement volatility."
        ),
    },
    {
        "slug": "report",
        "source": ROOT / "reports" / "REPORT.md",
        "title": "Results - Earnings Disclosure Signal Engine",
        "description": "Pipeline-generated results: dataset, extraction stats and ablation.",
    },
]

EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list"]


def render(md_text: str) -> str:
    html = markdown.markdown(md_text, extensions=EXTENSIONS)
    # Wrap tables so a wide results table scrolls on a phone instead of
    # forcing the whole page sideways.
    return re.sub(
        r"<table>.*?</table>",
        lambda m: f'<div class="table-wrap">{m.group(0)}</div>',
        html,
        flags=re.DOTALL,
    )


def main() -> int:
    missing = [p["source"] for p in PAGES if not p["source"].exists()]
    if missing:
        names = ", ".join(str(m.relative_to(ROOT)) for m in missing)
        print(f"error: missing source document(s): {names}", file=sys.stderr)
        print("run `edse report` before building the site", file=sys.stderr)
        return 1

    figures = ROOT / "reports" / "figures"
    pngs = sorted(figures.glob("*.png"))
    if not pngs:
        print(
            "error: reports/figures/ has no PNGs - the site would render with "
            "broken images. Run the pipeline (`make pipeline`) first.",
            file=sys.stderr,
        )
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for dest in (OUT / "figures", OUT / "reports" / "figures"):
        dest.mkdir(parents=True, exist_ok=True)
        for png in pngs:
            shutil.copy2(png, dest / png.name)

    shutil.copy2(SITE / "style.css", OUT / "style.css")

    template = (SITE / "template.html").read_text()
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for page in PAGES:
        body = render(page["source"].read_text())
        html = (
            template.replace("__CONTENT__", body)
            .replace("__TITLE__", page["title"])
            .replace("__DESCRIPTION__", page["description"])
            .replace("__SOURCE__", str(page["source"].relative_to(ROOT)))
            .replace("__BUILT__", built)
            .replace("__ROOT__", "")
            .replace(
                "__NAV_INDEX__", ' aria-current="page"' if page["slug"] == "index" else ""
            )
            .replace(
                "__NAV_REPORT__", ' aria-current="page"' if page["slug"] == "report" else ""
            )
        )
        (OUT / f"{page['slug']}.html").write_text(html)
        print(f"  {page['slug']}.html  <- {page['source'].relative_to(ROOT)}")

    # Jekyll would otherwise swallow any path starting with an underscore.
    (OUT / ".nojekyll").write_text("")

    print(f"\nsite -> {OUT.relative_to(ROOT)} ({len(pngs)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
