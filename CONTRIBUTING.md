# Contributing

## High-risk areas (extra checklist)

If your change touches any of these:
- `agent_loop.py` (trading loop scheduling, timeouts, callbacks)
- FastAPI routing (`src/web_app.py`, `src/web/routes_*`)
- anything that uses `app.state.*`

Then the PR must include:

### Safety checklist
- [ ] No per-request mutation of `app.state.*` (especially `app.state.settings`).
- [ ] No `asyncio.run()` inside hot loops.
- [ ] External calls (LLM/exchange) have timeouts.
- [ ] Diagnostic endpoints are behind `enable_diagnostic_endpoints` and not enabled by default.
- [ ] No secrets/log files committed (`.env*`, `logs/`).

### Verification checklist
- [ ] `pytest` passes
- [ ] `ruff check .` passes
- [ ] `python -m py_compile` passes for touched modules

## Local dev quickstart

```bash
pip install -r requirements.txt
ruff check .
pytest -q
```
