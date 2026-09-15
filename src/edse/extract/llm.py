"""Claude-backed claim extraction.

Uses structured outputs (`output_config.format` / `messages.parse`) so the model
returns schema-valid JSON rather than prose that has to be salvaged with a
regex. The schema in `edse.schema` is the single source of truth: its field
descriptions are the extraction instructions, so the prompt below carries only
the cross-cutting rules and the judgment calls the schema cannot express.

Caching: the system prompt is a frozen, document-independent prefix marked with
`cache_control`, and per-document text goes in the user turn after the breakpoint,
which is what keeps the prefix stable.

Whether that cache actually engages is model-dependent and worth stating plainly:
the API only caches a prefix above a minimum length (roughly 1024 tokens for
Sonnet/Opus, 2048 for Haiku). This system prompt is around 1.7k tokens, so it
caches on Sonnet/Opus and most likely does NOT on Haiku 4.5, the default bulk
model. The prompt is not padded to cross that line - padding would cost real
tokens on every uncached call to manufacture a metric. Instead the extraction
report prints measured `cache_read_tokens`, so the effect is observed rather than
assumed. See `edse.cli.extract`'s summary and reports/extraction_report.md.
"""

from __future__ import annotations

import logging
import time

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..schema import EarningsClaims
from .base import ExtractionResult

log = logging.getLogger(__name__)

#: Bump on any prompt change. Part of the cache key, so an edit here
#: invalidates cached extractions instead of mixing prompt versions in one dataset.
PROMPT_VERSION = "v2"

#: USD per million tokens. Cache writes bill at 1.25x input, reads at 0.10x.
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}

SYSTEM_PROMPT = """\
You extract structured claims from SEC Form 8-K Item 2.02 earnings press releases.

Your output feeds a quantitative model. That imposes three rules that override any
instinct to be helpful or complete:

1. REPORT, DO NOT INFER. Every field must be supported by something the release
   actually says. If the release does not state it, use the not-stated / null /
   not-provided value for that field. A wrong confident answer is far more costly
   than an abstention, because abstentions are measured and modeled while errors
   silently corrupt the training data.

2. USE ONLY THIS DOCUMENT. You may know this company, this quarter, or how the
   stock reacted. Ignore all of it. Do not compare results to analyst consensus
   unless the release itself states the comparison. The model that consumes your
   output must see only information that existed at the moment of filing; anything
   else is lookahead bias.

3. THE RELEASE'S OWN FRAMING WINS. When the release and the tables disagree, or
   when GAAP and non-GAAP figures point different directions, report what the
   release's narrative presents as the headline result, and record the GAAP /
   non-GAAP emphasis separately in `non_gaap_emphasis`.

FIELD GUIDANCE

Directional fields (revenue, EPS, margin): these compare against the PRIOR-YEAR
period, which is the standard comparison in earnings releases. If a release only
gives a sequential (quarter-over-quarter) comparison, that is not a year-over-year
statement - use `not_stated`. "Roughly flat", "essentially unchanged", and changes
under half a percent are `flat`.

`revenue_yoy_pct`: only fill this when the release states a percentage directly.
Do not divide two dollar figures to derive one. If the release gives a range
across segments, use the total-company figure. Negative for declines.

`guidance_action`: this is about FORWARD guidance, not reported results.
- `raised` / `lowered` require an explicit change against previously issued guidance,
  including phrasings like "now expects" or "increasing our full-year outlook".
- `initiated` is first-time guidance for a period, including the routine case of a
  company giving next-year guidance for the first time on its Q4 release.
- `maintained` covers "reaffirms", "continues to expect", and restating the same
  numbers without comment.
- `withdrawn` covers suspending or pulling existing guidance.
- `not_provided` means the release contains no forward-looking numeric outlook at
  all. Do not treat safe-harbor boilerplate as guidance.

`tone`: judge the QUOTED EXECUTIVE COMMENTARY, not the numbers and not the
headline. A release can post excellent results in flat, procedural language
(`neutral`) or weak results with defiant optimism (`confident`). `cautious`
requires acknowledged difficulty, not merely the absence of enthusiasm.

`hedging_intensity`: count conditionals and qualifiers in the narrative only.
EXCLUDE the forward-looking-statements safe-harbor paragraph - it is legally
mandated boilerplate, it appears in essentially every release, and counting it
would collapse this field to a constant.

`macro_headwind_cited`: true when results are attributed to conditions outside the
company's control (currency, inflation, interest rates, tariffs, the demand
environment, weather). Merely mentioning the economy is not enough; the release
must connect it to results.

`announced_buyback`: a NEW or EXPANDED authorization announced in this release.
Reporting shares repurchased during the quarter under an existing program is not
an announcement - that is routine capital-return reporting.

`dividend_action`: `increased`/`decreased` for a changed rate, `flat` for a routine
declaration at an unchanged rate, `not_stated` when no dividend is mentioned.

`confidence`: your own reliability on THIS document. Use 0-1 when the release is
mostly tables with little narrative, is a pre-announcement or partial-results
release, is a non-standard filing (a REIT supplemental, a fund report), or is
truncated. This is a self-report used to audit your calibration; it is not a
feature, so report it honestly rather than optimistically.


WORKED EXAMPLES

These are synthetic releases illustrating the judgment calls above. Study the
reasoning, not the specific numbers.

--- Example A ---
"Northwind Systems Reports Record Fourth Quarter. Revenue of $2.41 billion rose 9
percent. Adjusted EPS of $1.88 compares to $1.52. On a GAAP basis, diluted EPS was
$0.94, reflecting $0.71 of acquisition-related amortization. 'We executed against
our plan and delivered a solid finish to the year,' said CEO Dana Roth. The Company
continues to expect full-year revenue of $10.2-$10.4 billion. The Board declared a
quarterly dividend of $0.30 per share, unchanged."

revenue_direction=increased, revenue_yoy_pct=9.0, eps_direction=increased,
margin_direction=not_stated (no margin comparison given), guidance_action=maintained
("continues to expect"), guidance_horizon_quarters=4, dividend_action=flat (declared,
rate unchanged), tone=neutral ("solid", "executed against our plan" is procedural,
not confident language, despite the "Record" headline), non_gaap_emphasis=2 (adjusted
EPS leads, GAAP is reconciled after), hedging_intensity=0.

Note: the headline says "Record" but tone follows the QUOTED COMMENTARY, which is
flat. And margin is not_stated even though you could compute one from other figures.

--- Example B ---
"Castellan Industries Reports Third Quarter Results. Revenue declined 6 percent to
$812 million. Operating margin contracted to 11.2 percent from 14.8 percent. 'Demand
in our European industrial end-markets remained soft, and currency was a meaningful
headwind,' said CFO Lee Park. 'While we believe conditions should gradually improve,
the timing remains uncertain, and we are taking action on costs.' The Company
announced a restructuring program expected to yield $40 million in annualized
savings, including a workforce reduction. The Company now expects full-year adjusted
EPS of $3.10-$3.30, down from its prior range of $3.55-$3.75."

revenue_direction=decreased, revenue_yoy_pct=-6.0, eps_direction=not_stated (the
release gives no reported-EPS comparison; the EPS figure is forward guidance),
margin_direction=decreased, guidance_action=lowered ("now expects ... down from its
prior range"), guidance_horizon_quarters=4, announced_restructuring=true,
macro_headwind_cited=true (currency and end-market demand, tied to results),
segment_weakness_disclosed=true (European industrial), tone=cautious,
hedging_intensity=2 ("believe", "should gradually", "timing remains uncertain").

Note: eps_direction is not_stated. Guidance EPS is not reported EPS - do not let a
forward number fill a backward-looking field.

Return exactly one JSON object matching the schema. No commentary."""

USER_TEMPLATE = """\
Company: {ticker}
Filing date: {filing_date}

<press_release>
{text}
</press_release>

Extract the structured claims."""


def estimate_cost(model: str, usage) -> float:
    """Dollar cost of one call, including the cache-rate adjustments."""
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    regular = getattr(usage, "input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    output = getattr(usage, "output_tokens", 0) or 0
    return (
        regular * in_rate
        + cache_read * in_rate * 0.10
        + cache_write * in_rate * 1.25
        + output * out_rate
    ) / 1_000_000


class ClaudeExtractor:
    """Extracts `EarningsClaims` from press-release text via the Claude API."""

    name = "claude"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 4096,
        temperature: float | None = None,
        client: anthropic.Anthropic | None = None,
        use_cache_control: bool = True,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_cache_control = use_cache_control
        self.client = client or anthropic.Anthropic()

    @property
    def prompt_version(self) -> str:
        # Temperature belongs in the cache key: a consistency run at temperature
        # 1.0 must not overwrite the deterministic extraction of the same filing.
        suffix = "" if self.temperature is None else f"-t{self.temperature}"
        return f"{PROMPT_VERSION}{suffix}"

    def _system(self) -> list[dict]:
        block: dict = {"type": "text", "text": SYSTEM_PROMPT}
        if self.use_cache_control:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
        ),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _invoke(self, system: list[dict], user: str):
        """One API call. Uses `messages.parse` unless a temperature is pinned.

        `messages.parse` is the supported structured-output path but does not
        expose `temperature`; the self-consistency eval needs temperature > 0, so
        that case falls through to `messages.create` with the schema transformed
        by the same SDK helper `parse` uses. Both paths therefore send byte-identical
        schemas, which also keeps them on the same prompt cache prefix.
        """
        if self.temperature is None:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=EarningsClaims,
            )
            parsed = next(
                (b.parsed_output for b in resp.content if getattr(b, "parsed_output", None)), None
            )
            return resp, parsed

        from anthropic.lib._parse._transform import transform_schema

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": transform_schema(EarningsClaims)}
            },
        )
        text = next((b.text for b in resp.content if b.type == "text"), None)
        parsed = EarningsClaims.model_validate_json(text) if text else None
        return resp, parsed

    def extract(
        self, text: str, *, ticker: str, accession: str, filing_date: str = ""
    ) -> ExtractionResult:
        user = USER_TEMPLATE.format(ticker=ticker, filing_date=filing_date, text=text)
        started = time.monotonic()

        def failure(err: str) -> ExtractionResult:
            return ExtractionResult(
                accession=accession,
                ticker=ticker,
                extractor=self.name,
                model=self.model,
                claims=None,
                ok=False,
                error=err,
                latency_s=time.monotonic() - started,
            )

        try:
            resp, claims = self._invoke(self._system(), user)
        except anthropic.BadRequestError as exc:
            # Non-retryable: an oversized document or a schema the API rejects.
            log.error("%s %s: bad request (%s)", ticker, accession, exc)
            return failure(f"bad_request: {exc}")
        except anthropic.APIStatusError as exc:
            log.error("%s %s: api error %s", ticker, accession, exc.status_code)
            return failure(f"api_error_{exc.status_code}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad filing must not kill the run
            log.error("%s %s: %s", ticker, accession, exc)
            return failure(f"{type(exc).__name__}: {exc}")

        if claims is None:
            return failure("no_parsed_output")

        usage = resp.usage
        return ExtractionResult(
            accession=accession,
            ticker=ticker,
            extractor=self.name,
            model=self.model,
            claims=claims,
            ok=True,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            latency_s=time.monotonic() - started,
            cost_usd=estimate_cost(self.model, usage),
            meta={"stop_reason": resp.stop_reason, "prompt_version": self.prompt_version},
        )
