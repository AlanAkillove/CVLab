# Contributing to CVLab

Thanks for considering contributing! Here's how to get started.

## Development Setup

```bash
git clone <repo-url>
cd CVLab
uv venv
uv sync
```

## Running Tests

```bash
uv run pytest
```

## Code Style

- Use type hints for all public functions
- Include docstrings for public APIs (Args/Returns style)
- Run tests before submitting a PR

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the full test suite
4. Submit a PR with a clear description of what and why

## Reporting Issues

Include the full error message, Python version, and how to reproduce.
