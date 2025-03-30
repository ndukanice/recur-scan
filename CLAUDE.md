# CLAUDE.md - Recur-Scan Development Guide

## Build & Test Commands

- Install dependencies: `make install`
- Run all tests: `make test`
- Run single test: `uv run python -m pytest tests/path_to_test.py::test_name -v`
- Check code quality: `make check`
- View all commands: `make help`

## Code Style Guidelines

- **Python Version**: 3.12+ (uses new features)
- **Formatting**:
  - Line length: 120 characters
  - Double quotes for strings
  - Ruff for formatting/linting
- **Imports**:
  - Sorted with isort (via Ruff)
  - Known first-party: `recur_scan`
- **Types**:
  - Strict typing with mypy
  - All functions require type annotations
  - No implicit optionals
- **Error Handling**:
  - Use loguru for logging
- **Quality Tools**:
  - pre-commit hooks for automated checks
  - mypy, ruff, deptry, pytest
