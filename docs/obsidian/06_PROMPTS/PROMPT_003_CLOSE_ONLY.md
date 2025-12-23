# PROMPT_003: Close-Only Mode

## Objective
Implement strict Close-Only mode enforcement across the trading engine.

## Acceptance Criteria
- [ ] Risk engine rejects ALL opening trades in Close-Only mode.
- [ ] Only reducing positions is allowed.
- [ ] Configuration flag `TRADING_MODE=CLOSE_ONLY` respected.

## Tests / Verification
-   Unit tests for `check_trade_allowed` with `mode=CLOSE_ONLY`.
