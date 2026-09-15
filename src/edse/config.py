"""Configuration loading and shared paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Parsed `configs/config.yaml`, plus environment-derived secrets."""

    edgar: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        path = Path(path) if path else CONFIGS_DIR / "config.yaml"
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)

    @property
    def edgar_user_agent(self) -> str:
        """SEC requires a descriptive UA with a contact address on every request.

        Refusing to fall back to a generic string is deliberate: an anonymous UA
        gets the whole project rate-limited or blocked, and the failure mode
        (empty result sets) is silent and confusing.
        """
        ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
        if not ua or "@" not in ua:
            raise RuntimeError(
                "EDGAR_USER_AGENT must be set to a descriptive string containing a "
                "contact email, e.g. 'edse-research you@example.com'. "
                "Copy .env.example to .env and fill it in. SEC blocks anonymous clients."
            )
        return ua

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY") or None


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GOLD_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
