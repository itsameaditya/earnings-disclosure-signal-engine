#!/usr/bin/env python
"""Interactive review tool for the hand-labeled gold set.

Grading extraction quality needs labels a human made. `edse gold` writes the
template and pre-fills it from the *rule-based* extractor; this walks you through
the rows field by field, showing the same trimmed narrative the extractor is given
so a label is never based on text the model could not see.

The LLM's own extraction is deliberately never displayed: the gold set is the
independent measurement, and anchoring it to the system under evaluation would
inflate every score in `edse eval-extraction`.

    ./.venv/bin/python scripts/label_gold.py              # next unreviewed row
    ./.venv/bin/python scripts/label_gold.py --list       # progress summary
    ./.venv/bin/python scripts/label_gold.py --accession 0000320187-18-000034
    ./.venv/bin/python scripts/label_gold.py --redo       # revisit reviewed rows too

Progress is written after every document, so quitting mid-set loses nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edse.schema import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    GRADED_FIELDS,
    NUMERIC_FIELDS,
    ORDINAL_FIELDS,
    EarningsClaims,
)
from edse.textprep import prepare

GOLD_PATH = REPO / "data" / "gold" / "gold_labels.jsonl"
MAX_DOCUMENT_CHARS = 70000  # matches configs/config.yaml extraction.max_document_chars

HELP = """
commands at any field prompt
  <enter>      accept the value shown in [brackets]
  1 2 3 ...    pick an option by number
  null | -     set the field to null (nullable numerics only)
  ?            full rubric text for this field, from schema.py
  t            reopen the filing narrative in the pager
  /pattern     search the narrative, with surrounding context
  b            back up one field
  s            skip this filing, leave it unreviewed
  q            save and quit
"""


@dataclass
class Field:
    """One graded field, with the prompt metadata derived from the schema."""

    name: str
    description: str
    kind: str  # enum | bool | ordinal | numeric
    choices: tuple[str, ...] = ()
    nullable: bool = False

    @property
    def hint(self) -> str:
        if self.kind == "ordinal":
            return "0-3"
        if self.kind == "numeric":
            return "number or null"
        return " ".join(f"{i + 1}={c}" for i, c in enumerate(self.choices))


def build_fields() -> list[Field]:
    """Derive the field list from the schema, so there is one source of truth."""
    fields = []
    for name in GRADED_FIELDS:
        info = EarningsClaims.model_fields[name]
        description = " ".join((info.description or "").split())
        if name in CATEGORICAL_FIELDS:
            enum_cls = info.annotation
            assert isinstance(enum_cls, type) and issubclass(enum_cls, Enum), name
            fields.append(
                Field(name, description, "enum", tuple(m.value for m in enum_cls))
            )
        elif name in BOOLEAN_FIELDS:
            fields.append(Field(name, description, "bool", ("false", "true")))
        elif name in ORDINAL_FIELDS:
            fields.append(Field(name, description, "ordinal"))
        elif name in NUMERIC_FIELDS:
            fields.append(Field(name, description, "numeric", nullable=True))
    return fields


def load_rows() -> list[dict]:
    if not GOLD_PATH.exists():
        raise SystemExit(f"no gold template at {GOLD_PATH} - run `edse gold --sample 60` first")
    rows = [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{GOLD_PATH} is empty - run `edse gold --sample 60` first")
    return rows


def save_rows(rows: list[dict]) -> None:
    """Rewrite the whole file atomically: a crash mid-write must not truncate labels."""
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    fd, tmp = tempfile.mkstemp(dir=str(GOLD_PATH.parent), prefix=".gold-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, GOLD_PATH)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def narrative_for(row: dict) -> tuple[str, str]:
    """Return the trimmed narrative plus a one-line note about what prep did."""
    path = REPO / row["document"]
    if not path.exists():
        raise SystemExit(f"missing document {path} - run `edse ingest` first")
    prepped = prepare(path.read_text(), max_chars=MAX_DOCUMENT_CHARS)
    meta = prepped.meta()
    bits = [f"{meta['prepared_chars']:,} chars shown of {meta['original_chars']:,}"]
    if meta["trimmed_at_marker"]:
        bits.append("trimmed at statement header")
    if meta["guidance_recovered"]:
        bits.append("outlook section recovered from the trimmed tail")
    if meta["hard_truncated"]:
        bits.append("HARD TRUNCATED at the char cap - claims may rest on partial text")
    return prepped.text, "; ".join(bits)


def page(text: str, header: str) -> None:
    """Show the narrative in the pager, falling back to stdout when piped."""
    body = f"{header}\n{'=' * len(header)}\n\n{text}\n"
    if not sys.stdout.isatty():
        print(body)
        return
    pager = os.environ.get("PAGER", "less -R")
    try:
        subprocess.run([*pager.split(), "-"], input=body, text=True, check=False)
    except FileNotFoundError:
        print(body)


def search(text: str, pattern: str) -> None:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        print(f"  bad pattern: {exc}")
        return
    hits = list(rx.finditer(text))
    if not hits:
        print(f"  no match for /{pattern}/")
        return
    for hit in hits[:8]:
        lo, hi = max(0, hit.start() - 120), min(len(text), hit.end() + 120)
        snippet = " ".join(text[lo:hi].split())
        print(f"  ...{snippet}...")
    if len(hits) > 8:
        print(f"  ({len(hits) - 8} more matches)")


def show(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse(field: Field, raw: str):
    """Parse one answer. Returns (ok, value); ok is False when input was invalid."""
    text = raw.strip().lower()
    if field.nullable and text in {"null", "none", "-"}:
        return True, None

    if field.kind in {"enum", "bool"}:
        if text.isdigit() and 1 <= int(text) <= len(field.choices):
            text = field.choices[int(text) - 1]
        if text not in field.choices:
            print(f"  not a valid value. options: {field.hint}")
            return False, None
        return True, text == "true" if field.kind == "bool" else text

    if field.kind == "ordinal":
        if text.isdigit() and 0 <= int(text) <= 3:
            return True, int(text)
        print("  expected an integer 0-3")
        return False, None

    try:
        number = float(text)
    except ValueError:
        print("  expected a number, or null")
        return False, None
    if field.name == "guidance_horizon_quarters":
        if number != int(number):
            print("  expected a whole number of quarters, or null")
            return False, None
        return True, int(number)
    return True, number


def review(row: dict, fields: list[Field], position: str) -> str:
    """Walk one filing's fields. Returns 'done', 'skip', or 'quit'."""
    text, note = narrative_for(row)
    header = f"{row['ticker']}  {row['filing_date']}  {row['accession']}"
    print(f"\n{'─' * 78}\n{position}  {header}\n  {note}\n{'─' * 78}")
    page(text, header)

    values = {f.name: row.get(f.name) for f in fields}
    index = 0
    while index < len(fields):
        field = fields[index]
        current = show(values[field.name])
        print(f"\n[{index + 1}/{len(fields)}] {field.name}   ({field.hint})")
        print(f"      {field.description[:150]}{'...' if len(field.description) > 150 else ''}")
        try:
            raw = input(f"      [{current}] > ")
        except EOFError:
            print()
            return "quit"

        command = raw.strip().lower()
        if command == "":
            index += 1
            continue
        if command == "?":
            print(f"\n  {field.description}\n")
            continue
        if command == "t":
            page(text, header)
            continue
        if command.startswith("/"):
            search(text, raw.strip()[1:])
            continue
        if command == "b":
            index = max(0, index - 1)
            continue
        if command == "s":
            return "skip"
        if command == "q":
            return "quit"
        if command in {"h", "help"}:
            print(HELP)
            continue

        ok, value = parse(field, raw)
        if ok:
            values[field.name] = value
            index += 1

    changed = [
        f"{name}: {show(row.get(name))} -> {show(value)}"
        for name, value in values.items()
        if row.get(name) != value
    ]
    print(f"\n  {len(changed)} field(s) changed from the prefill:")
    for line in changed or ["      (none)"]:
        print(f"      {line}")
    confirm = input("\n  mark reviewed? [Y/n/s=skip/q=quit] ").strip().lower()
    if confirm in {"q", "quit"}:
        return "quit"
    if confirm in {"n", "no", "s", "skip"}:
        return "skip"

    row.update(values)
    row["_reviewed"] = True
    row["_reviewed_at"] = datetime.now().astimezone().date().isoformat()
    return "done"


def summarize(rows: list[dict]) -> None:
    done = [r for r in rows if r.get("_reviewed")]
    print(f"gold set: {len(done)} of {len(rows)} reviewed")
    for row in rows:
        mark = "x" if row.get("_reviewed") else " "
        print(f"  [{mark}] {row['ticker']:<6} {row['filing_date']}  {row['accession']}")
    if len(done) < len(rows):
        print(f"\n{len(rows) - len(done)} left. Run without --list to label the next one.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show progress and exit")
    ap.add_argument("--accession", help="label one specific filing")
    ap.add_argument("--redo", action="store_true", help="include already-reviewed rows")
    ap.add_argument("--limit", type=int, help="stop after this many filings this session")
    args = ap.parse_args()

    rows = load_rows()
    if args.list:
        summarize(rows)
        return

    fields = build_fields()
    if args.accession:
        queue = [r for r in rows if r["accession"] == args.accession]
        if not queue:
            raise SystemExit(f"{args.accession} is not in the gold set")
    else:
        queue = [r for r in rows if args.redo or not r.get("_reviewed")]
    if not queue:
        print("every row is already reviewed - `edse eval-extraction` is ready to run.")
        return
    if args.limit:
        queue = queue[: args.limit]

    print(f"{len(queue)} filing(s) to review. `h` for commands, `q` saves and quits.")
    completed = 0
    for offset, row in enumerate(queue, start=1):
        outcome = review(row, fields, f"({offset}/{len(queue)})")
        if outcome == "done":
            completed += 1
            save_rows(rows)
            print("  saved.")
        elif outcome == "quit":
            break

    save_rows(rows)
    total = sum(1 for r in rows if r.get("_reviewed"))
    print(f"\nlabeled {completed} this session. gold set now {total}/{len(rows)} reviewed.")
    if total >= 2:
        print("grade with: ./.venv/bin/edse eval-extraction")


if __name__ == "__main__":
    main()
