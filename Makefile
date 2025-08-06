.PHONY: help setup test clean lint format build all

help:
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and setup pre-commit"
	@echo "  test         - Run all tests (examples + src)"
	@echo "  test-reset   - Clean and run tests"
	@echo "  lint         - Run pre-commit hooks on all files"
	@echo "  format       - Run formatting tools (ruff)"
	@echo "  build        - Build the package"
	@echo "  publish      - Publish the package to PyPI"
	@echo "  clean        - Clean up temp files and build artifacts"
	@echo "  all          - Setup and test"

setup:
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff format .
	uv run ruff check .

test:
	uv run pytest tests/ -v

test-debug:
	uv run pytest tests/ -v --durations=10 --pdb

test-reset: clean test

build: clean
	@echo "Building package..."
	uv build
	@echo "Build complete!"


publish: build
	@echo "Publishing package..."
	@uv publish
	@echo "Publish complete!"


clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__/ .pytest_cache/
	@rm -rf dist/ build/
	@find . -name "*.pyc" -delete
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"

all: setup test
