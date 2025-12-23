#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTEXT_FILE="$REPO_ROOT/docs/CLI_AGENT_CONTEXT.md"
WRAPPER_FILE="$REPO_ROOT/docs/CODEX_PROMPT_WRAPPER.txt"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <task description>"
  exit 1
fi

if [[ ! -f "$CONTEXT_FILE" ]]; then
  echo "Missing context file: $CONTEXT_FILE"
  exit 1
fi

if [[ ! -f "$WRAPPER_FILE" ]]; then
  echo "Missing wrapper file: $WRAPPER_FILE"
  exit 1
fi

TASK_TEXT="$*"

PROMPT_PATH="$(mktemp /tmp/codexw_prompt.XXXXXX)"
cleanup() {
  [[ -f "$PROMPT_PATH" ]] && rm -f "$PROMPT_PATH"
}
trap cleanup EXIT

{
  cat "$CONTEXT_FILE"
  echo
  cat "$WRAPPER_FILE"
  printf "\nTASK: %s\n" "$TASK_TEXT"
} > "$PROMPT_PATH"

CODER_BIN="${CODER_BIN:-npx}"
CODER_ARGS=("@openai/codex" "exec" "--dangerously-bypass-approvals-and-sandbox" "--json")

if ! command -v "$CODER_BIN" >/dev/null 2>&1; then
  echo "Codex executable '$CODER_BIN' not found; please install Node/npm or set CODER_BIN."
  echo "Prompt stored at: $PROMPT_PATH"
  cat "$PROMPT_PATH"
  exit 1
fi

set +e
(
  cd "$REPO_ROOT"
  "$CODER_BIN" "${CODER_ARGS[@]}" < "$PROMPT_PATH"
)
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -ne 0 ]]; then
  echo "Codex command failed (exit code $EXIT_CODE); please paste the prompt manually."
  echo "Prompt path: $PROMPT_PATH"
  cat "$PROMPT_PATH"
  exit "$EXIT_CODE"
fi

exit 0
