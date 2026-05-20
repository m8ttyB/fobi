.DEFAULT_GOAL := help

MODEL ?= mlx-community/gemma-3-4b-it-4bit
FILE  ?=

.PHONY: help install ingest chat reset test lint format

help:
	@echo "Usage:"
	@echo "  make install            Create .venv and install dependencies"
	@echo "  make ingest FILE=...    Ingest a single PDF document"
	@echo "  make ingest-dir DIR=... Ingest all PDFs in a directory (recursive)"
	@echo "  make chat               Start a Q&A session"
	@echo "  make reset              Delete all indexes"
	@echo "  make test               Run the test suite"
	@echo "  make lint               Check code with ruff"
	@echo "  make format             Auto-format code with ruff"

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

ingest:
	@if [ -z "$(FILE)" ]; then echo "Error: FILE is required. Usage: make ingest FILE=path/to/doc.pdf"; exit 1; fi
	QA_MODEL_PATH=$(MODEL) .venv/bin/python main.py --ingest $(FILE)

ingest-dir:
	@if [ -z "$(DIR)" ]; then echo "Error: DIR is required. Usage: make ingest-dir DIR=path/to/pdfs/"; exit 1; fi
	QA_MODEL_PATH=$(MODEL) .venv/bin/python main.py --ingest-dir $(DIR)

chat:
	QA_MODEL_PATH=$(MODEL) .venv/bin/python main.py --chat

reset:
	.venv/bin/python main.py --reset

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .
