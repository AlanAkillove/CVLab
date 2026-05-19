# ── CVLab Makefile ────────────────────────────────────────

.DEFAULT_GOAL := help

# Detect OS
UNAME_S := $(shell uname -s)

# ── Variables ─────────────────────────────────────────────
PYTHON   ?= python3
UV       ?= uv
PIP      ?= pip
PYTEST   ?= pytest
RUFF     ?= ruff

# ── Help ──────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────
.PHONY: setup
setup: ## Install project in editable mode
	$(UV) sync --dev

.PHONY: install
install: ## Install production dependencies
	$(UV) sync --no-dev

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: clean-all
clean-all: clean ## Clean everything including .cvlab data
	rm -rf .cvlab/

# ── Development ───────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	$(UV) run $(RUFF) check cvlab/

.PHONY: lint-fix
lint-fix: ## Fix auto-fixable lint issues
	$(UV) run $(RUFF) check --fix cvlab/

.PHONY: format
format: ## Run ruff formatter
	$(UV) run $(RUFF) format cvlab/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(UV) run mypy cvlab/ --ignore-missing-imports --python-version 3.10

.PHONY: check
check: lint format typecheck ## Run all checks

# ── Testing ───────────────────────────────────────────────
.PHONY: test
test: ## Run tests
	$(UV) run $(PYTEST) cvlab/tests/ -v --tb=short

.PHONY: test-full
test-full: ## Run all tests (including slow)
	$(UV) run $(PYTEST) cvlab/tests/ -v --tb=short --slow

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(UV) run $(PYTEST) cvlab/tests/ --cov=cvlab --cov-report=term --cov-report=html

.PHONY: test-quick
test-quick: ## Run quick smoke tests only
	$(UV) run $(PYTEST) cvlab/tests/ -v --tb=short -k "not slow"
	$(UV) run python -c "from cvlab import Tracker, __version__; print(f'CVLab {__version__}: import OK')"

# ── Docker ────────────────────────────────────────────────
.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t cvlab:latest -f Dockerfile .

.PHONY: docker-gpu
docker-gpu: ## Build GPU Docker image
	docker build -t cvlab:gpu -f Dockerfile.gpu .

.PHONY: docker-run
docker-run: ## Run Docker container
	docker run --rm -v $(PWD):/workspace cvlab:latest help

# ── Quality ───────────────────────────────────────────────
.PHONY: pre-commit
pre-commit: ## Install pre-commit hooks
	pre-commit install

.PHONY: pre-commit-run
pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# ── CLI quick access ─────────────────────────────────────
.PHONY: init
init: ## Initialize CVLab in current directory
	$(UV) run cvlab init

.PHONY: list
list: ## List experiments
	$(UV) run cvlab list

.PHONY: ui
ui: ## Launch Streamlit UI
	$(UV) run streamlit run cvlab/ui/app.py
