VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := uv pip install --python $(PY)

.PHONY: help setup data features explore train evaluate experiments html all quick regress regress-update test clean clean-all

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

experiments: ## leakage, feature-block ablation and the two shift experiments
	$(PY) -m sepsis.cli experiments

html:       ## render reports/REPORT.md as a single self-contained HTML page
	$(PY) scripts/build_html_report.py

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
