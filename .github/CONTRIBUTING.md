# Contributing

Thanks for your interest in improving PlumeRank.

This repo has two parts: a Python pipeline + FastAPI backend at the root, and a
Next.js frontend in `web/`. The Makefile is deliberately fixed at five targets
(see its header comment) — contributions that need a new build step add a
step to CI or a script under `scripts/`, not a sixth Makefile target.

## Getting started — backend (root)
1. Fork the repo and branch from `main`: `git checkout -b feat/your-feature`
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock`
3. `.venv/bin/pip install -e ".[dev]"` (ruff, mypy, pytest, pytest-cov, hypothesis, pip-audit)
4. `make test` — offline suite + coverage, no network, no credentials, ~95s

## Getting started — frontend (`web/`)
1. `cd web && npm install`
2. Point it at a running API: `NEXT_PUBLIC_PLUMERANK_API=http://127.0.0.1:8000 npm run dev`
   (start the API first: `.venv/bin/uvicorn api.main:app --reload` from the repo root)

## Before you open a PR
- `ruff check .` and `mypy plumerank api --ignore-missing-imports` are clean.
- `make test` passes. `make verify` (≤3 min; does **not** reproduce the published
  headline — see the Makefile header and `README.md`'s reproduce section) passes
  from a clean clone.
- `cd web && npm run lint && npm run typecheck && npm run build` pass.
- Add or update tests for any behavior change. Name regression tests after the
  defect they fix (see `tests/` for the convention) rather than a counter.
- Keep commits [Angular-convention](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
  (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`, `refactor:`, `test:`, `style:`, `ci:`) —
  `.github/workflows/release.yml` derives the next version tag from these on every
  push to `main`.
- If you touch anything under `plumerank/`, `api/`, or `scripts/`, run
  `scripts/check_claims.py` (invoked by `make test`) — it mechanically fails if a
  document and the source disagree about a route, an OpenCV symbol, a dependency,
  a Makefile target, or a claimed file path.

## Reporting Bugs / Requesting Features
Open an issue using the provided templates. Include repro steps, expected vs.
actual behavior, and environment details.
