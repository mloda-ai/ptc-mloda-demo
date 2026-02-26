# CLAUDE.md

Must read [AGENTS.md](AGENTS.md) first.

## Environment

```bash
source .venv/bin/activate
```

## Dependencies

Use `uv` to install dependencies:
```bash
uv sync --all-extras
```

## Running checks

Use `tox` to run all checks:
```bash
tox
```

Run checks from the project virtualenv:
```bash
source .venv/bin/activate && tox
```

`tox` is the required final verification step after code or dependency changes.
Running only `pytest` is not sufficient for completion.

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Plugin Development Guides

Consult the mloda-registry guides before building new plugins:
- Local path: `/home/tom/project/mloda-registry/docs/guides/`
- Key patterns: `feature-group-patterns/01-root-features.md`, `01-use-existing-plugin.md`

## Commit messages

Use Conventional Commit format for all commits so semantic versioning/release tooling can parse intent.

Examples:
- `fix: handle empty feature set`
- `chore(deps): bump mloda to 0.4.6`
