"""Local-model extraction via Ollama.

Exists so the whole project can be reproduced for free: same `Extractor`
interface as the Claude backend, same schema, same eval harness. The point is
not that a 7B local model matches a frontier model - it is that the eval harness
can *measure* the gap, which is the question a reader should be asking.

Structured output uses Ollama's `format` parameter with a JSON schema, which
constrains decoding rather than merely requesting JSON in the prompt. Two
practical differences from the hosted API drive the code below:

- `$ref`/`$defs` are inlined. Ollama compiles the schema to a GBNF grammar and
  does not reliably resolve references.
- `num_ctx` is set explicitly. Ollama's default context is 4096 tokens, which is
  less than half a typical earnings release - the tail would be silently dropped
  and the extractor would score badly for reasons that have nothing to do with
  the model.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..schema import EarningsClaims
from .base import ExtractionResult
from .llm import USER_TEMPLATE

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"

#: The local model gets its own, much shorter system prompt. This is a measured
#: decision, not a stylistic one.
#:
#: The Claude prompt in `llm.py` is ~1.8k tokens and leans hard on abstention
#: ("a wrong confident answer is far more costly than an abstention"). On
#: qwen2.5:7b that framing dominates: measured over 12 releases that state a
#: revenue percentage in plain text, the Claude prompt produced a null
#: `revenue_yoy_pct` on 100% of them, and `margin_direction: not_stated` on 100%.
#: The prompt below recovered 100% recall on the same documents. An intermediate
#: version carrying about half the rules scored 75% - for this model, more
#: instruction text monotonically increased abstention.
#:
#: The generalizable point, and the reason both prompts are kept side by side in
#: this repo: prompts are model-specific artifacts. A prompt tuned for a frontier
#: model can silently destroy a smaller model's recall while still returning
#: perfectly schema-valid output - which is exactly the failure an eval harness
#: exists to catch, and which no amount of JSON-validity checking would reveal.
LOCAL_SYSTEM_PROMPT = """You extract structured claims from SEC 8-K earnings press releases.

Base every field only on what this release says. Use the not-stated / null value
when the release does not state something. Return one JSON object matching the schema."""
PROMPT_VERSION = "local-v3"

#: Context window. Median release is ~7.8k tokens and the system prompt ~1.8k;
#: 16k leaves headroom for the long tail without exhausting unified memory.
DEFAULT_NUM_CTX = 16384


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` into a self-contained schema.

    Ollama's grammar compiler does not follow references, and an unresolved
    `$ref` silently yields an unconstrained field rather than an error.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            target = resolve(copy.deepcopy(defs.get(name, {})))
            # Keep any sibling keys (e.g. a field-level description) alongside
            # the resolved definition.
            extras = {k: resolve(v) for k, v in node.items() if k != "$ref"}
            return {**target, **extras}
        return {k: resolve(v) for k, v in node.items()}

    return resolve(schema)


def build_schema() -> dict[str, Any]:
    """JSON schema for the claim record, flattened for Ollama."""
    return inline_refs(EarningsClaims.model_json_schema())


class OllamaExtractor:
    """Extracts `EarningsClaims` using a locally hosted model. Free to run."""

    name = "local"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = DEFAULT_HOST,
        temperature: float | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._schema = build_schema()
        self.session = requests.Session()

    @property
    def prompt_version(self) -> str:
        suffix = "" if self.temperature is None else f"-t{self.temperature}"
        return f"{PROMPT_VERSION}-ctx{self.num_ctx}{suffix}"

    def health_check(self) -> None:
        """Fail early and clearly if the server or model is missing."""
        try:
            tags = self.session.get(f"{self.host}/api/tags", timeout=10).json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.host} ({exc}).\n"
                "Start it with:  ollama serve"
            ) from exc

        available = {m["name"] for m in tags.get("models", [])}
        if self.model not in available and f"{self.model}:latest" not in available:
            raise RuntimeError(
                f"model {self.model!r} is not pulled. Run:  ollama pull {self.model}\n"
                f"available: {', '.join(sorted(available)) or '(none)'}"
            )

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _invoke(self, user: str) -> dict:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        options["temperature"] = 0.0 if self.temperature is None else self.temperature

        response = self.session.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "format": self._schema,  # constrains decoding to the schema
                "stream": False,
                "options": options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def extract(
        self, text: str, *, ticker: str, accession: str, filing_date: str = ""
    ) -> ExtractionResult:
        user = USER_TEMPLATE.format(ticker=ticker, filing_date=filing_date, text=text)
        started = time.monotonic()

        def failure(err: str) -> ExtractionResult:
            return ExtractionResult(
                accession=accession, ticker=ticker, extractor=self.name, model=self.model,
                claims=None, ok=False, error=err, latency_s=time.monotonic() - started,
            )

        try:
            payload = self._invoke(user)
        except requests.RequestException as exc:
            log.error("%s %s: %s", ticker, accession, exc)
            return failure(f"{type(exc).__name__}: {exc}")

        content = (payload.get("message") or {}).get("content", "")
        if not content:
            return failure("empty_response")

        try:
            claims = EarningsClaims.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001 - malformed output is a normal failure here
            # Grammar-constrained decoding makes this rare but not impossible
            # (e.g. the model hits the token cap mid-object).
            log.warning("%s %s: schema validation failed (%s)", ticker, accession, exc)
            return failure(f"validation_error: {str(exc)[:200]}")

        return ExtractionResult(
            accession=accession, ticker=ticker, extractor=self.name, model=self.model,
            claims=claims, ok=True,
            input_tokens=payload.get("prompt_eval_count", 0) or 0,
            output_tokens=payload.get("eval_count", 0) or 0,
            latency_s=time.monotonic() - started,
            cost_usd=0.0,  # local inference: electricity only
            meta={
                "prompt_version": self.prompt_version,
                "load_ms": round((payload.get("load_duration", 0) or 0) / 1e6),
                "prompt_eval_ms": round((payload.get("prompt_eval_duration", 0) or 0) / 1e6),
                "eval_ms": round((payload.get("eval_duration", 0) or 0) / 1e6),
            },
        )
