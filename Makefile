.PHONY: install dev full run ncp test lint fmt docker build publish

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

full:
	pip install -e ".[dev,full]"

run:
	python -m nvd_claude_proxy.main

ncp:
	ncp

test:
	pytest -q

lint:
	ruff check src tests
	mypy src

fmt:
	ruff format src tests
	ruff check --fix src tests

docker:
	docker compose up --build

# ── PyPI release ──────────────────────────────────────────────────────────────

build:
	rm -rf dist/
	python -m build

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*
