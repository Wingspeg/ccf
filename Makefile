# CCF (Converging Computing Framework Scheduling) -- Makefile
# Convenience targets for the most common workflows.
#
# Usage:
#   make help         - show this help
#   make install      - install the package in editable mode with dev deps
#   make test         - run the unit test suite
#   make smoke        - quick smoke test of the package import
#   make grid         - run the 9-scale x 9-variant x 5-seed grid
#   make sensitivity  - run the convergence / omega / figure pipeline
#   make figures      - regenerate figures from existing CSVs
#   make all          - grid + sensitivity (full reproduction)
#   make clean        - remove generated artifacts

PYTHON      ?= python3
INSTALL_CMD ?= $(PYTHON) -m pip install -e ".[dev]"

.PHONY: help install test smoke grid sensitivity figures all clean

help:
	@echo "CCF (Converging Computing Framework Scheduling) -- Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install      install the package in editable mode with dev deps"
	@echo "  test         run the unit test suite (pytest)"
	@echo "  smoke        quick import + variant sanity test"
	@echo "  grid         run the 9-scale x 9-variant x 5-seed grid"
	@echo "  sensitivity  run convergence / omega / figure pipeline"
	@echo "  figures      regenerate figures from existing CSVs"
	@echo "  all          grid + sensitivity (full reproduction)"
	@echo "  clean        remove generated CSVs and figures"

install:
	$(INSTALL_CMD)

test:
	$(PYTHON) -m pytest tests/

smoke:
	$(PYTHON) -c "import ccf; print('ccf v' + ccf.__version__ + ' OK')"

grid:
	$(PYTHON) -m experiments.main_experiment

sensitivity:
	$(PYTHON) -m experiments.sensitivity

figures:
	$(PYTHON) -c "from experiments.sensitivity import fig_solvetime, fig_components, OUT; fig_solvetime(OUT / 'scales.csv'); fig_components(OUT / 'scales.csv')"

all: grid sensitivity

clean:
	rm -rf results/*.csv results/figures/*.png
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
