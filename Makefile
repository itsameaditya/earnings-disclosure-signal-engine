# Earnings Disclosure Signal Engine
PY := ./.venv/bin/python
EDSE := ./.venv/bin/edse

.PHONY: help setup test lint data extract-baseline extract-llm train-baseline train-llm eval pipeline clean-derived

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## create the venv and install the package
	python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## ruff check
	./.venv/bin/ruff check src tests

data:  ## resolve the universe, download filings and prices, build labels
	$(EDSE) universe
	$(EDSE) ingest
	$(EDSE) prices
	$(EDSE) label

extract-baseline:  ## run the rule-based extractor (free)
	$(EDSE) extract --extractor baseline

extract-llm:  ## run the Claude extractor (needs ANTHROPIC_API_KEY)
	$(EDSE) extract --extractor claude

train-baseline:  ## ablation using baseline claims
	$(EDSE) train --extractor baseline

train-llm:  ## ablation using LLM claims
	$(EDSE) train --extractor claude

eval:  ## grade extractors against gold labels
	$(EDSE) eval-extraction

pipeline: data extract-baseline extract-llm train-llm train-baseline  ## everything end to end

clean-derived:  ## drop derived artifacts, keep raw downloads and gold labels
	rm -rf data/processed/* reports/figures/*.png reports/*.json reports/*.csv
