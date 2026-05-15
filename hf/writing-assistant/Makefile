VENV    = .venv
PYTHON  = $(VENV)/bin/python
PYTEST  = $(VENV)/bin/pytest
RUFF    = $(VENV)/bin/ruff
UVICORN = $(VENV)/bin/uvicorn

# Override any of these at the command line, e.g. make run MODEL=my-org/my-model
MODEL ?= mlx-community/gemma-3-4b-it-4bit
HOST  ?= 127.0.0.1
PORT  ?= 8000

.PHONY: help install run test lint format

help:
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Create .venv and install all dependencies"
	@echo "  run       Start the dev server  (override: MODEL, HOST, PORT)"
	@echo "  test      Run the test suite"
	@echo "  lint      Check code with ruff"
	@echo "  format    Auto-format code with ruff"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run MODEL=/path/to/local/mlx-model PORT=9000"
	@echo "  make test"
	@echo "  make lint"

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

run:
	WA_MODEL_PATH=$(MODEL) WA_HOST=$(HOST) WA_PORT=$(PORT) \
		$(UVICORN) main:app --host $(HOST) --port $(PORT) --reload

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
