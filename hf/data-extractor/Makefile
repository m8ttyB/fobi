.PHONY: install run test lint format

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff

MODEL ?= $(shell echo $${EXTRACTOR_MODEL_PATH:-mlx-community/gemma-3-4b-it-4bit})

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py $(FILE) --model $(MODEL)

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
