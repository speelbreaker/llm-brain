# PROMPT_004: P0 Safety Tests (Phase 3)

## Objective
Complete P0 safety validation before real money deployment.

## Prerequisites
- [ ] PR #20 merged (supervisor loop)
- [ ] Phase 2 complete (vault infrastructure)

## Deliverables

### Phase 3A: Trade Permission Unit Tests
Create `tests/test_trade_permission.py`:
- Test normal mode allows all actions
- Test kill switch blocks all except DO_NOTHING
- Test CLOSE_ONLY blocks OPEN/ROLL, allows CLOSE
- Test can_trade=False blocks OPEN/ROLL, allows CLOSE
- Test priority: kill_switch > halt > close_only > can_trade
- Test can_trade=None defaults to allowed

### Phase 3B: One-Tick Integration Tests
Create `tests/test_agent_loop_one_tick.py`:
- Test gate enforcement in CLOSE_ONLY mode
- Test gate enforcement when can_trade=False
- Test risk_engine respects trade_mode
- Use mocks for DeribitClient (no network calls)

### Phase 3C: Enhanced Fidelity Suite
Modify `src/fidelity/run_suite.py`:
- Add preflight checks as Stage 0 (runs before full suite)
- Checks: data_available, calibration_present, ops_health_contract
- Fail-fast: if preflight FAIL, skip subsequent stages

Modify `scripts/run_fidelity_suite.py`:
- Add `--mode preflight|quick|full` argument
- preflight = Stage 0 only (fast, offline)
- quick = Stage 0 + fast suite
- full = Stage 0 + comprehensive comparison

Create `tests/test_fidelity_suite_contract.py`:
- Test output has required keys (underlying, overall_status, can_trade, checks, summary)
- Test overall_status is OK|WARN|FAIL
- Test can_trade is boolean
- Test fail-closed: missing data → can_trade=False

### Phase 3D: Dashboard Endpoints
Create `src/web/routes_ops.py`:
- GET `/ops/health` → returns OPS_HEALTH_latest.json
- GET `/ops/fidelity/{underlying}` → returns FIDELITY_{underlying}_latest.json
- GET `/ops/gates` → returns combined can_trade with Truth/Trust/Trade breakdown

Register router in `src/web_app.py`

Create `tests/test_ops_endpoints.py`:
- Test each endpoint returns correct status codes
- Test /ops/gates has can_trade boolean and breakdown dict

## Acceptance Criteria
- [ ] `pytest tests/test_trade_permission.py -v` passes
- [ ] `pytest tests/test_agent_loop_one_tick.py -v` passes
- [ ] `python scripts/run_fidelity_suite.py --underlying BTC --mode preflight` works
- [ ] `pytest tests/test_fidelity_suite_contract.py -v` passes
- [ ] `pytest tests/test_ops_endpoints.py -v` passes

## Tests / Verification
```bash
pytest tests/test_trade_permission.py tests/test_agent_loop_one_tick.py \
       tests/test_fidelity_suite_contract.py tests/test_ops_endpoints.py -v
```

## On Completion
- Update QUEUE.md: move to DONE
- Update CHANGELOG.md
- Archive prompt to 99_ARCHIVE/

