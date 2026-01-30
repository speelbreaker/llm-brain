#!/usr/bin/env bash
set -euo pipefail

# Local pre-push gate: tests + static checks + OPTIONAL Codex diff review
#
# SECURITY: The Codex step sends staged diffs to an external LLM provider.
# It is OFF by default. Enable explicitly with: CODE_REVIEW_ENABLE=1
# Bypass: SKIP_PREPUSH_REVIEW=1

if [[ "${SKIP_PREPUSH_REVIEW:-}" == "1" ]]; then
  echo "[pre-push] SKIP_PREPUSH_REVIEW=1 set; skipping gate." >&2
  exit 0
fi

# Ensure we have environment for OpenAI key
if [[ -f /etc/llmagentbrain/platform.env ]]; then
  # shellcheck disable=SC1091
  source /etc/llmagentbrain/platform.env || true
fi

cd "$(git rev-parse --show-toplevel)"

# 1) Quick python compile (fast)
if [[ -x ./.venv/bin/python ]]; then
  pyfiles=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)
  if [[ -n "${pyfiles}" ]]; then
    ./.venv/bin/python -m py_compile ${pyfiles}
  fi
fi

# 2) Run unit tests (fast subset) if present
if [[ -x ./.venv/bin/python ]]; then
  if [[ -d tests ]]; then
    echo "[pre-push] Running pytest (focused)..." >&2
    # Keep this fast but meaningful. Add more suites as they stabilize.
    ./.venv/bin/python -m pytest -q \
      tests/web/test_routes_parity.py \
      tests/web/test_telegram_webhook.py
  fi
fi

# 3) Codex review of staged diff
if [[ -x ./.venv/bin/python ]]; then
  if [[ "${CODE_REVIEW_ENABLE:-}" == "1" ]]; then
    echo "[pre-push] Running Codex diff review (CODE_REVIEW_ENABLE=1)..." >&2
    git diff --cached | ./.venv/bin/python tools/codex_review_diff.py --threshold "${CODE_REVIEW_THRESHOLD:-HIGH}"
  else
    echo "[pre-push] Skipping Codex diff review (set CODE_REVIEW_ENABLE=1 to enable)" >&2
  fi
else
  echo "[pre-push] WARNING: .venv not found; skipping Codex review" >&2
fi

exit 0
