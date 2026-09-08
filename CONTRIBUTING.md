# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Run checks before opening a pull request:

```bash
ruff check .
pytest -q
```

## Design principles

- Keep the data plane zero-copy.
- Keep control-plane messages small and language-neutral.
- Do not introduce mandatory ownership transfer or locking into the base token model.
- New synchronization features should be optional layers.
- Preserve the same public Python API on Linux and Windows wherever OS semantics allow it.
- Add cross-process tests for changes touching mappings, lifecycle, or token enforcement.
