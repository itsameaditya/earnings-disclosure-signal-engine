# Earnings Disclosure Signal Engine
PY := ./.venv/bin/python
EDSE := ./.venv/bin/edse

.PHONY: help setup test lint data extract-baseline extract-local extract-llm \
        train-baseline train-local train-llm eval label pipeline pipeline-claude clean-derived

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

extract-local:  ## run the free local model via Ollama (the default extractor)
	$(EDSE) extract --extractor local

extract-llm:  ## run the Claude extractor (needs ANTHROPIC_API_KEY)
	$(EDSE) extract --extractor claude

train-baseline:  ## ablation using baseline claims
	$(EDSE) train --extractor baseline

train-local:  ## ablation using local-model claims
	$(EDSE) train --extractor local

train-llm:  ## ablation using Claude claims
	$(EDSE) train --extractor claude

eval:  ## grade extractors against gold labels
	$(EDSE) eval-extraction

label:  ## review the gold-label template one filing at a time
	$(PY) scripts/label_gold.py

# Default path is free: local model, no API key. `make pipeline-claude` is the
# hosted-extractor equivalent and is the only target that needs a key.
pipeline: data extract-baseline extract-local train-local train-baseline  ## everything end to end, free
	$(EDSE) report

pipeline-claude: data extract-llm train-llm  ## same via the Claude extractor (needs ANTHROPIC_API_KEY)
	$(EDSE) report

clean-derived:  ## drop derived artifacts, keep raw downloads and gold labels
	rm -rf data/processed/* reports/figures/*.png reports/*.json reports/*.csv
