"""Extractor interface, result record, and the on-disk extraction cache."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from ..schema import EarningsClaims

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """One extraction attempt, with the accounting needed by the eval harness."""

    accession: str
    ticker: str
    extractor: str
    model: str
    claims: EarningsClaims | None
    ok: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = self.claims.model_dump(mode="json") if self.claims else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExtractionResult:
        d = dict(d)
        raw = d.pop("claims", None)
        return cls(claims=EarningsClaims.model_validate(raw) if raw else None, **d)


class Extractor(Protocol):
    """Anything that turns press-release text into structured claims.

    The LLM extractor and the rule-based baseline implement the same interface so
    the pipeline and the eval harness can run either without branching.
    """

    name: str
    model: str

    def extract(self, text: str, *, ticker: str, accession: str) -> ExtractionResult: ...


class ExtractionCache:
    """Content-addressed cache of extraction results.

    The key covers the extractor, the model, the prompt version and the document
    text, so changing the prompt or swapping models produces a new key instead of
    silently serving stale claims. This is what makes a 1,200-document run
    resumable: re-running after a crash re-extracts only what is missing, at no
    additional API cost.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(extractor: str, model: str, prompt_version: str, text: str) -> str:
        h = hashlib.sha256()
        for part in (extractor, model, prompt_version, text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def _path(self, key: str) -> Path:
        # Shard by prefix; a flat directory of thousands of files is slow to list.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ExtractionResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return ExtractionResult.from_dict(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001 - a corrupt entry must not kill the run
            log.warning("discarding unreadable cache entry %s (%s)", path.name, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, result: ExtractionResult) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run cannot leave a half-written file.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.to_dict(), indent=2))
        tmp.replace(path)

    def stats(self) -> dict:
        files = list(self.root.rglob("*.json"))
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}
