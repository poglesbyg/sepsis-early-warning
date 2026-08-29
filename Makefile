VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := uv pip install --python $(PY)

.PHONY: help setup data features explore train evaluate replay experiments card html all quick regress regress-update test clean clean-all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:      ## create the venv and install everything
	uv venv --python 3.12 $(VENV)
	$(PIP) -e ".[dev]"

data:       ## download the PhysioNet 2019 training sets (~40k files, ~310 MB)
	$(PY) -m sepsis.cli data

features:   ## build the causal feature tables
	$(PY) -m sepsis.cli features

explore:    ## univariate screen, collinearity, cross-site drift
	$(PY) -m sepsis.cli explore

train:      ## fit baseline, logistic regression, XGBoost and the causal GRU
	$(PY) -m sepsis.cli train

evaluate:   ## calibrate, blend, score every split, write reports/REPORT.md
	$(PY) -m sepsis.cli evaluate

replay:     ## pick the admissions the report replays, hour by hour
	$(PY) -m sepsis.cli replay

experiments: ## leakage, feature-block ablation and the two shift experiments
	$(PY) -m sepsis.cli experiments

card:       ## regenerate MODEL_CARD.md, with performance broken out by subgroup
	$(PY) -m sepsis.cli card

html:       ## render the report and the essay as self-contained HTML pages
	$(PY) scripts/build_html_report.py
	$(PY) scripts/build_essay.py

all:        ## the whole pipeline, end to end
	$(PY) -m sepsis.cli all

quick:      ## same pipeline with a small search budget, for a fast check
	$(PY) -m sepsis.cli all --trials 8 --timeout 300

regress:    ## check every published number against configs/regression_baseline.json
	$(PY) -m sepsis.cli regress

regress-update: ## move the baseline deliberately, as a reviewable diff
	$(PY) -m sepsis.cli regress --update

test:       ## run the test suite
	$(PY) -m pytest -q

clean:      ## remove derived features, artifacts and reports (keeps raw data)
	rm -rf data/processed data/interim artifacts reports/figures reports/*.csv reports/*.json reports/REPORT.md

clean-all: clean  ## also remove the downloaded raw data
	rm -rf data/raw
