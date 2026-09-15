"""Rule-based extractor: the baseline the LLM has to beat.

This exists to answer the question a reviewer should ask first: *does the LLM
actually add anything over keyword matching?* Without this comparison, strong
downstream model performance proves nothing about the extraction step - it could
be driven entirely by "the word 'record' appears".

The rules below are a genuine attempt, not a strawman: real regexes for the
numeric and directional fields, curated cue-phrase lists for the categorical
ones. Fields that require reading comprehension rather than pattern matching
(`tone`, `hedging_intensity`, `non_gaap_emphasis`) are handled with the best
lexical proxy available, which is precisely where the gap should show up.
"""

from __future__ import annotations

import re
import time

from ..schema import Direction, EarningsClaims, GuidanceAction, Tone
from .base import ExtractionResult

# Safe-harbor boilerplate appears in nearly every release; counting its hedging
# language would swamp the narrative signal, so it is cut before analysis.
_SAFE_HARBOR_RE = re.compile(
    r"(forward[- ]looking statements|safe harbor|private securities litigation reform)",
    re.IGNORECASE,
)

_REV_PCT_RE = re.compile(
    r"revenue[^.]{0,80}?\b(increased|decreased|grew|declined|rose|fell|up|down)\b[^.]{0,40}?"
    r"(\d+(?:\.\d+)?)\s*(?:percent|%)",
    re.IGNORECASE,
)

_UP_WORDS = r"(increased|grew|rose|up|higher|improved|expanded|growth)"
_DOWN_WORDS = r"(decreased|declined|fell|down|lower|contracted|decline)"
_FLAT_WORDS = r"(flat|unchanged|essentially unchanged|roughly flat|in line with)"

_CONFIDENT_CUES = (
    "record", "strong", "excellent", "outstanding", "exceptional", "robust",
    "momentum", "pleased", "proud", "accelerating", "best-ever", "milestone",
)
_CAUTIOUS_CUES = (
    "challenging", "headwind", "soft", "softness", "difficult", "pressure",
    "disappointing", "weak", "slowdown", "uncertain", "cautious",
)
_HEDGE_CUES = (
    "believe", "expect", "anticipate", "may", "could", "should", "potential",
    "uncertain", "approximately", "estimate", "likely", "intend", "plan to",
)
_MACRO_CUES = (
    "macroeconomic", "inflation", "interest rate", "foreign exchange", "currency",
    "tariff", "recession", "demand environment", "geopolitical", "supply chain",
)


def _strip_safe_harbor(text: str) -> str:
    match = _SAFE_HARBOR_RE.search(text)
    return text[: match.start()] if match else text


def _direction_near(text: str, subject: str, window: int = 120) -> Direction:
    """Direction of the first `subject` mention, from cue words in its vicinity."""
    for m in re.finditer(subject, text, re.IGNORECASE):
        span = text[m.end() : m.end() + window]
        if re.search(_FLAT_WORDS, span, re.IGNORECASE):
            return Direction.FLAT
        if re.search(_UP_WORDS, span, re.IGNORECASE):
            return Direction.INCREASED
        if re.search(_DOWN_WORDS, span, re.IGNORECASE):
            return Direction.DECREASED
    return Direction.NOT_STATED


def _count_cues(text: str, cues: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(cue) for cue in cues)


def _bucket(count: int, thresholds: tuple[int, int, int] = (1, 4, 9)) -> int:
    """Map a raw cue count onto the schema's 0-3 ordinal scale."""
    lo, mid, hi = thresholds
    if count < lo:
        return 0
    if count < mid:
        return 1
    if count < hi:
        return 2
    return 3


class RuleBasedExtractor:
    """Keyword/regex extractor with the same interface as the LLM extractor."""

    name = "baseline"
    model = "rules-v1"
    prompt_version = "rules-v1"

    def extract(
        self, text: str, *, ticker: str, accession: str, filing_date: str = ""
    ) -> ExtractionResult:
        started = time.monotonic()
        # Collapse whitespace before matching. Press-release HTML wraps lines
        # mid-phrase, so "down from its\nprior range" would defeat any pattern
        # written with literal spaces - a silent, systematic miss.
        body = re.sub(r"\s+", " ", _strip_safe_harbor(text)).strip()
        low = body.lower()

        revenue_pct: float | None = None
        m = _REV_PCT_RE.search(body)
        if m:
            verb, value = m.group(1).lower(), float(m.group(2))
            revenue_pct = -value if verb in {"decreased", "declined", "fell", "down"} else value

        # Revision phrasings are checked before the generic "mentions guidance"
        # fallback, and include the common "X, down from its prior range" form
        # that never puts a revision verb next to the word "guidance".
        raised_re = (
            r"\b(raising|raised|increasing|increased)\b[^.]{0,60}(guidance|outlook)"
            r"|\b(up|above|increased)\b[^.]{0,30}from (its |our |the )?prior"
        )
        lowered_re = (
            r"\b(lowering|lowered|reducing|reduced|cutting|cut)\b[^.]{0,60}(guidance|outlook)"
            r"|\b(down|below|reduced)\b[^.]{0,30}from (its |our |the )?prior"
        )
        if re.search(lowered_re, low):
            guidance = GuidanceAction.LOWERED
        elif re.search(raised_re, low):
            guidance = GuidanceAction.RAISED
        elif re.search(r"\b(reaffirm|maintain|continues to expect|reiterat)", low):
            guidance = GuidanceAction.MAINTAINED
        elif re.search(r"\b(withdraw|suspend)\w*\b[^.]{0,60}(guidance|outlook)", low):
            guidance = GuidanceAction.WITHDRAWN
        elif re.search(r"\b(guidance|outlook|expects? .{0,30}(full[- ]year|fiscal))", low):
            guidance = GuidanceAction.INITIATED
        else:
            guidance = GuidanceAction.NOT_PROVIDED

        confident = _count_cues(body, _CONFIDENT_CUES)
        cautious = _count_cues(body, _CAUTIOUS_CUES)
        if confident > cautious * 2 and confident >= 2:
            tone = Tone.CONFIDENT
        elif cautious >= max(2, confident):
            tone = Tone.CAUTIOUS
        elif confident or cautious:
            tone = Tone.NEUTRAL
        else:
            tone = Tone.NOT_STATED

        if re.search(r"dividend", low):
            dividend = _direction_near(body, "dividend")
            if dividend is Direction.NOT_STATED:
                dividend = Direction.FLAT  # a bare declaration implies an unchanged rate
        else:
            dividend = Direction.NOT_STATED

        gaap = low.count("gaap")
        non_gaap = low.count("non-gaap") + low.count("adjusted")

        claims = EarningsClaims(
            revenue_direction=_direction_near(body, "revenue"),
            revenue_yoy_pct=revenue_pct,
            eps_direction=_direction_near(body, r"(earnings per share|EPS)"),
            margin_direction=_direction_near(body, "margin"),
            guidance_action=guidance,
            guidance_horizon_quarters=(
                None
                if guidance is GuidanceAction.NOT_PROVIDED
                else (4 if re.search(r"full[- ]year|fiscal year", low) else 1)
            ),
            announced_buyback=bool(
                re.search(r"(repurchase|buyback)[^.]{0,80}(authoriz|program|approved|increase)", low)
            ),
            dividend_action=dividend,
            announced_restructuring=bool(
                re.search(r"restructur|workforce reduction|layoff|cost[- ]reduction", low)
            ),
            announced_impairment=bool(re.search(r"impairment|write[- ]down|goodwill charge", low)),
            executive_transition=bool(
                re.search(r"(appoint|nam(e|ing)|step(ping)? down|resign|transition)[^.]{0,60}"
                          r"(chief executive|chief financial|ceo|cfo)", low)
            ),
            tone=tone,
            non_gaap_emphasis=_bucket(non_gaap, (1, 6, 15)) if non_gaap or gaap else 0,
            hedging_intensity=_bucket(_count_cues(body, _HEDGE_CUES), (2, 6, 14)),
            macro_headwind_cited=_count_cues(body, _MACRO_CUES) >= 2,
            segment_weakness_disclosed=bool(
                re.search(r"(segment|region|division|business)[^.]{0,80}"
                          r"(declin|weak|soft|challeng|pressure)", low)
            ),
            one_time_items=bool(
                re.search(r"one[- ]time|non[- ]recurring|unusual item|discrete item", low)
            ),
            confidence=2,  # constant by construction; the baseline cannot self-assess
        )

        return ExtractionResult(
            accession=accession,
            ticker=ticker,
            extractor=self.name,
            model=self.model,
            claims=claims,
            ok=True,
            latency_s=time.monotonic() - started,
            cost_usd=0.0,
            meta={"prompt_version": self.prompt_version},
        )
