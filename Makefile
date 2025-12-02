.PHONY: help setup test clean lint format build all test35 test40 lock


help:
	@echo "Available targets:"
	@echo "  setup            - Install dependencies and setup pre-commit"
	@echo "  test             - Run all tests"
	@echo "  test-s3-isolated - Run isolated s3 hanging test (5s timeout)"
	@echo "  test-reset       - Clean and run tests"
	@echo "  lint             - Run pre-commit hooks on all files"
	@echo "  format           - Run formatting tools (ruff)"
	@echo "  build            - Build the package"
	@echo "  publish          - Publish the package to PyPI"
	@echo "  clean            - Clean up temp files and build artifacts"
	@echo "  all              - Setup and test"


setup:
	uv sync --group dev
	uv run pre-commit install


lint:
	uv run pre-commit run --all-files


format:
	uv run ruff format .
	uv run ruff check .


lock:
	uv lock


test: test35 test40


test35:
	uv run --with "pyspark==3.5.*" pytest -m "not spark40_only" tests/


test40:
	uv run --with "pyspark==4.0.*" pytest -m "not spark35_only" tests/


test-s3-isolated:
	@echo "Running isolated S3 signature test..."
	@uv run python tests/run_s3_timeout_test.py


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
