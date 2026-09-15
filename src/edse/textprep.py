"""Document preparation for extraction.

An earnings press release is two documents stapled together: a narrative (the
headline, management commentary, and forward guidance) followed by the financial
statement tables. Every field in `edse.schema` is answerable from the narrative;
none requires the tables.

Trimming the tables is therefore not a cost hack, it is a relevance filter, and
it buys three things at once:

- **No silent truncation.** Untrimmed, 11% of this corpus overflows a 16k context
  window and the tail is dropped without any error - the worst kind of data bug,
  because the extractor still returns confident-looking output.
- **Less distractor text.** Pages of numeric tables give a model plenty of
  opportunity to report a figure from the wrong column or period.
- **Roughly half the tokens**, which is what makes a full local run practical.

Trimming happens at extraction time, not at ingest: the raw documents stay
complete on disk so this decision stays reversible and auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Headers that mark the start of the financial-statement section. Ordered
#: loosely from most to least specific; the earliest match in the document wins.
_TABLE_MARKERS = (
    r"CONDENSED\s+CONSOLIDATED\s+(?:INTERIM\s+)?(?:STATEMENTS?|BALANCE)",
    r"CONSOLIDATED\s+(?:STATEMENTS?\s+OF|BALANCE\s+SHEETS?)",
    r"CONSOLIDATING\s+STATEMENTS?",
    r"RECONCILIATION\s+OF\s+(?:GAAP|NON-GAAP)",
    r"SUPPLEMENTAL\s+(?:FINANCIAL|INFORMATION|DATA)",
    r"SEGMENT\s+(?:INFORMATION|RESULTS|DATA)\s*$",
)
_MARKER_RE = re.compile("|".join(_TABLE_MARKERS), re.IGNORECASE | re.MULTILINE)

#: Never trim to less than this. A marker appearing very early usually means a
#: table-of-contents entry or an unusual layout, not the real section break -
#: cutting there would throw away the entire release.
MIN_NARRATIVE_CHARS = 1500

#: Forward-guidance language. Some filers (Coca-Cola and Caterpillar are the
#: clearest cases here) put a non-GAAP reconciliation *before* their outlook
#: section, so a naive trim at the first statement header removes the guidance
#: along with the tables. Measured on this corpus, that silently destroyed the
#: guidance section in 91 documents - 7.7% of trimmed filings - and
#: `guidance_action` would have been wrong for every one of them with no error
#: raised anywhere. Anything matching this is recovered from the removed text.
_GUIDANCE_RE = re.compile(
    r"(full[- ]year \d{4}\s+(?:revenues?|outlook|guidance|earnings)"
    r"|we (?:now )?expect[^.]{0,80}(?:full[- ]year|fiscal \d{4})"
    r"|(?:raising|lowering|reaffirm\w*|updating|reiterat\w*) (?:its |our )?"
    r"(?:full[- ]year |fiscal )?(?:guidance|outlook)"
    r"|^\s*(?:\d{4} |fiscal \d{4} |full[- ]year )?(?:outlook|guidance)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

#: How much text to carry back when a guidance section is found in the removed
#: portion. Enough for a full outlook block, bounded so a match inside the tables
#: cannot drag the statements back in.
GUIDANCE_RECOVERY_CHARS = 5000


@dataclass(frozen=True)
class PreparedDocument:
    """A document ready for extraction, with what happened to it recorded."""

    text: str
    original_chars: int
    trimmed_at_marker: bool
    hard_truncated: bool
    guidance_recovered: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)

    def meta(self) -> dict:
        return {
            "original_chars": self.original_chars,
            "prepared_chars": self.chars,
            "trimmed_at_marker": self.trimmed_at_marker,
            # True means the safety net fired and content was dropped without a
            # semantic boundary. Surfaced in the extraction report rather than
            # swallowed, because it means claims may be based on partial text.
            "hard_truncated": self.hard_truncated,
            "guidance_recovered": self.guidance_recovered,
        }


def prepare(text: str, max_chars: int | None = None) -> PreparedDocument:
    """Trim a release to its narrative, with a hard cap as a safety net.

    `max_chars` is the backstop for documents where no marker is found (about
    18% of this corpus - filers with non-standard layouts). Pass None to disable.
    """
    original = len(text)
    trimmed = False

    recovered = False
    match = _MARKER_RE.search(text)
    if match and match.start() >= MIN_NARRATIVE_CHARS:
        kept, removed = text[: match.start()].rstrip(), text[match.start() :]
        trimmed = True
        # Carry the outlook section back if the trim took it and the narrative
        # does not already contain guidance language.
        if not _GUIDANCE_RE.search(kept):
            block = _extract_guidance(removed)
            if block:
                kept = f"{kept}\n\n{block}"
                recovered = True
        text = kept

    hard = False
    if max_chars is not None and len(text) > max_chars:
        # Cut on a paragraph boundary when one is close, so a sentence is not
        # sliced mid-clause.
        window = text[:max_chars]
        boundary = window.rfind("\n\n")
        text = window[:boundary] if boundary > max_chars * 0.8 else window
        hard = True

    return PreparedDocument(
        text=text.strip(),
        original_chars=original,
        trimmed_at_marker=trimmed,
        hard_truncated=hard,
        guidance_recovered=recovered,
    )


def _extract_guidance(removed: str) -> str | None:
    """Pull the forward-guidance block out of text that trimming removed."""
    match = _GUIDANCE_RE.search(removed)
    if not match:
        return None
    # Back up to the start of the line so a section header is not cut in half.
    start = removed.rfind("\n", 0, match.start()) + 1
    return removed[start : start + GUIDANCE_RECOVERY_CHARS].strip() or None
