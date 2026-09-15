"""SEC EDGAR ingestion: discover earnings 8-Ks and pull their press-release exhibits.

Why 8-K Item 2.02 specifically: EDGAR tags each 8-K with the Items it reports, and
Item 2.02 is "Results of Operations and Financial Condition" - the earnings release.
Filtering on the tag rather than on text search gives a precise, reproducible event
set with no keyword guesswork.

The actual earnings narrative is almost never in the 8-K body; it is furnished as
Exhibit 99.1. The body is a two-paragraph cover page pointing at the exhibit, so we
resolve the exhibit from the filing's SGML document table and fetch that.
"""

from __future__ import annotations

import html
import logging
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Preference order for the press-release exhibit. EX-99.1 is the near-universal
# convention; the looser forms are fallbacks for filers who label differently.
_EXHIBIT_PREFERENCE = ("EX-99.1", "EX-99", "EX-99.2")

_DOC_RE = re.compile(r"<TYPE>([^<\n]+)\n<SEQUENCE>(\d+)\n<FILENAME>([^<\n]+)")


@dataclass(frozen=True)
class Filing:
    """One earnings 8-K event."""

    ticker: str
    cik: int
    accession: str
    filing_date: str          # YYYY-MM-DD, the date EDGAR recorded the filing
    report_date: str          # YYYY-MM-DD, the fiscal period end
    acceptance_datetime: str  # ISO 8601 UTC from EDGAR; parse_acceptance() -> Eastern
    items: str
    primary_document: str

    @property
    def nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def archive_url(self) -> str:
        return ARCHIVE_BASE.format(cik=self.cik, nodash=self.nodash)

    def to_dict(self) -> dict:
        return asdict(self)


class RateLimiter:
    """Thread-safe minimum-interval limiter.

    SEC's published fair-access ceiling is 10 requests/second. We default to 6
    and serialize across threads, because being throttled by EDGAR shows up as
    truncated result sets rather than a clean error.
    """

    def __init__(self, per_second: float):
        self._min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last = time.monotonic()


class EdgarClient:
    def __init__(self, user_agent: str, requests_per_second: float = 6.0):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )
        self.limiter = RateLimiter(requests_per_second)

    @retry(
        retry=retry_if_exception_type((requests.RequestException,)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str) -> requests.Response:
        self.limiter.wait()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    # -- ticker -> CIK ----------------------------------------------------

    def ticker_map(self) -> dict[str, tuple[int, str]]:
        """SEC's own ticker->CIK table. Never hardcode CIKs; they change on reorgs."""
        data = self._get(COMPANY_TICKERS_URL).json()
        return {v["ticker"]: (int(v["cik_str"]), v["title"]) for v in data.values()}

    # -- filing discovery -------------------------------------------------

    def iter_filings(
        self, ticker: str, cik: int, required_item: str, start_date: str, end_date: str
    ) -> Iterator[Filing]:
        """Yield every 8-K for `cik` tagged with `required_item` inside the date range.

        The submissions endpoint splits long histories: `filings.recent` holds the
        most recent ~1000 entries and `filings.files` lists older shards. Companies
        that file frequently will silently lose their early years if you only read
        `recent`, so both are walked.
        """
        payload = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        blocks = [payload["filings"]["recent"]]
        for extra in payload["filings"].get("files", []):
            blocks.append(self._get(f"https://data.sec.gov/submissions/{extra['name']}").json())

        for block in blocks:
            yield from self._filings_from_block(
                block, ticker, cik, required_item, start_date, end_date
            )

    @staticmethod
    def _filings_from_block(
        block: dict, ticker: str, cik: int, required_item: str, start_date: str, end_date: str
    ) -> Iterator[Filing]:
        n = len(block.get("accessionNumber", []))

        def col(name: str) -> list:
            vals = block.get(name) or []
            return list(vals) + [""] * (n - len(vals))

        for form, acc, fdate, rdate, accepted, items, primary in zip(
            col("form"),
            col("accessionNumber"),
            col("filingDate"),
            col("reportDate"),
            col("acceptanceDateTime"),
            col("items"),
            col("primaryDocument"),
        ):
            if form != "8-K" or required_item not in (items or ""):
                continue
            if not (start_date <= fdate <= end_date):
                continue
            yield Filing(
                ticker=ticker,
                cik=cik,
                accession=acc,
                filing_date=fdate,
                report_date=rdate or fdate,
                acceptance_datetime=accepted,
                items=items,
                primary_document=primary,
            )

    # -- exhibit retrieval ------------------------------------------------

    def _document_table(self, filing: Filing) -> list[tuple[str, int, str]]:
        """(type, sequence, filename) for every document in the submission.

        Parsed from `-index-headers.html`, which embeds the raw SGML document
        table. `index.json` is not usable for this: its `type` field holds the
        icon filename ("text.gif"), not the exhibit type.
        """
        url = f"{filing.archive_url}/{filing.accession}-index-headers.html"
        text = html.unescape(self._get(url).text)
        return [(t.strip(), int(seq), fn.strip()) for t, seq, fn in _DOC_RE.findall(text)]

    def fetch_press_release(self, filing: Filing, max_chars: int) -> tuple[str, str] | None:
        """Return (exhibit_filename, plain_text) for the earnings release, or None.

        Returns None rather than raising when a filing has no usable exhibit -
        some Item 2.02 filings only incorporate results by reference, and those
        are legitimately not part of the event set.
        """
        try:
            docs = self._document_table(filing)
        except requests.RequestException as exc:
            log.warning("%s %s: document table unavailable (%s)", filing.ticker, filing.accession, exc)
            return None

        chosen: tuple[str, int, str] | None = None
        for pref in _EXHIBIT_PREFERENCE:
            candidates = [d for d in docs if d[0].upper() == pref]
            if candidates:
                # Largest by sequence tie-break is wrong; pick the earliest sequence,
                # which is the primary press release when several 99.x exhibits exist.
                chosen = min(candidates, key=lambda d: d[1])
                break
        if chosen is None:
            log.info("%s %s: no EX-99 exhibit", filing.ticker, filing.accession)
            return None

        _, _, filename = chosen
        try:
            raw = self._get(f"{filing.archive_url}/{filename}").text
        except requests.RequestException as exc:
            log.warning("%s %s: exhibit fetch failed (%s)", filing.ticker, filing.accession, exc)
            return None

        text = html_to_text(raw)
        if len(text) < 500:
            log.info("%s %s: exhibit too short (%d chars)", filing.ticker, filing.accession, len(text))
            return None
        return filename, text[:max_chars]


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\xa0]+")
_NL_RE = re.compile(r"\n{3,}")

# Block-level tags get a newline appended so paragraphs and table rows stay
# separated. Without this, lxml's text_content() runs everything together
# ("DocumentExhibit 99.1Apple reports third quarter results"), which costs the
# extractor the sentence boundaries it needs.
_NEWLINE_TAGS = frozenset(
    {"p", "div", "br", "tr", "table", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}
)
_CELL_TAGS = frozenset({"td", "th"})

# EDGAR serves archived exhibits with a short preamble repeating the document
# type, sequence number and filename. It is not part of the press release and
# would otherwise be the first thing the extractor reads.
_PREAMBLE_TOKEN_RE = re.compile(
    r"^(?:EX-\d+(?:\.\d+)*|\d{1,3}|[\w\-.]+\.(?:html?|txt)|Document)$", re.IGNORECASE
)


def html_to_text(raw: str) -> str:
    """Flatten filing HTML to text, preserving paragraph and table structure.

    Uses lxml when available because filing HTML is frequently malformed in ways
    that defeat naive tag stripping (unclosed tags, nested tables, Word export
    artifacts). The regex path is a dependency-free fallback with the same
    block-level newline behavior.
    """
    try:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(raw)
        for bad in doc.xpath("//script|//style"):
            bad.getparent().remove(bad)
        for el in doc.iter():
            tag = el.tag.lower() if isinstance(el.tag, str) else ""
            if tag in _NEWLINE_TAGS:
                el.tail = (el.tail or "") + "\n"
            elif tag in _CELL_TAGS:
                el.tail = (el.tail or "") + "\t"
        text = doc.text_content()
    except Exception:  # noqa: BLE001 - malformed filings are expected; fall back quietly
        cleaned = _SCRIPT_RE.sub(" ", raw)
        cleaned = re.sub(r"</(p|div|tr|h[1-6]|li|table)>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</(td|th)>", "\t", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        text = _TAG_RE.sub(" ", cleaned)

    text = html.unescape(text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _NL_RE.sub("\n\n", text).strip()
    return _strip_edgar_preamble(text)


def _strip_edgar_preamble(text: str) -> str:
    """Drop EDGAR's document-header lines from the top of an exhibit."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and i < 8:
        stripped = lines[i].strip()
        if not stripped or _PREAMBLE_TOKEN_RE.match(stripped):
            i += 1
            continue
        break
    return "\n".join(lines[i:]).strip()


def parse_acceptance(acceptance: str) -> datetime | None:
    """Parse EDGAR's acceptance timestamp into an Eastern-time datetime.

    The submissions API reports acceptance in **UTC** with a trailing Z
    ("2026-07-30T20:30:28.000Z"), while the SGML filing header reports the same
    instant in Eastern ("20260730163028"). Both are handled here and normalized
    to US/Eastern, because the only thing downstream cares about is where the
    release falls relative to the 09:30-16:00 ET session - which decides whether
    the market can react the same day or the next. Getting this wrong shifts the
    event window by a full day for every after-hours release, and the large
    majority of earnings releases are after-hours.
    """
    if not acceptance:
        return None

    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            utc = datetime.strptime(acceptance, fmt).replace(tzinfo=timezone.utc)
            return utc.astimezone(EASTERN).replace(tzinfo=None)
        except ValueError:
            continue

    # Already Eastern (SGML header form, or an API response without the Z).
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            # Deliberately naive: the return value is wall-clock Eastern, compared
            # against Eastern session boundaries. Attaching a tzinfo here would
            # imply UTC to every caller and silently shift the event day.
            return datetime.strptime(acceptance, fmt)  # noqa: DTZ007
        except ValueError:
            continue

    log.warning("unparseable acceptance timestamp: %r", acceptance)
    return None
