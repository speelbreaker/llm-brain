#!/usr/bin/env bash
set -euo pipefail

API_BASE=${API_BASE:-http://127.0.0.1:8080}
MAX_POLLS=${MAX_POLLS:-120}
SLEEP_SECONDS=${SLEEP_SECONDS:-3}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"

echo "[smoke] health check"
curl -fsS "$API_BASE/health" >/dev/null

echo "[smoke] diag check"
DIAG=$(curl -fsS "$API_BASE/api/diag")
DIAG_EVAL=$(DIAG="$DIAG" python3 - <<'PY'
import json, os, sys
bad_keys = ["api_key", "token", "secret", "password", "authorization", "bearer"]
diag = json.loads(os.environ["DIAG"])
blob = json.dumps(diag).lower()
bad = [k for k in bad_keys if k in blob]
build_id = diag.get("build_id")
if not build_id:
    print("missing build_id", file=sys.stderr)
    sys.exit(2)
if bad:
    print("forbidden substrings: " + ",".join(bad), file=sys.stderr)
    sys.exit(3)
print(build_id)
PY
) || { echo "[smoke] diag validation failed"; exit 1; }
echo "[smoke] diag ok (build_id=$DIAG_EVAL)"

echo "[smoke] creating empty commit to trigger supervisor"
BRANCH=${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
MSG="smoke: supervisor $(date -u +%FT%TZ)"
git commit --allow-empty -m "$MSG"
git push origin "$BRANCH"
HEAD_SHA=$(git rev-parse HEAD)

echo "[smoke] polling /jobs"
for i in $(seq 1 "$MAX_POLLS"); do
  JOB_JSON=$(curl -fsS "$API_BASE/jobs" || true)
  set +e
  RESULT=$(printf "%s" "$JOB_JSON" | HEAD_SHA="$HEAD_SHA" python3 -c 'import json, os, sys
raw = sys.stdin.read()
if not raw.strip():
    sys.exit(4)
try:
    data = json.loads(raw)
except Exception:
    sys.exit(5)
jobs = data.get("jobs") or []
if not jobs:
    sys.exit(6)
head_sha = os.environ.get("HEAD_SHA")
job = next((j for j in jobs if j.get("head_sha") == head_sha), None)
if not job:
    sys.exit(7)
job_id = job.get("job_id", "")
status = job.get("status", "")
print(f"{job_id} {status}")
terminal_success = {"checks_passed"}
terminal_fail = {"checks_failed", "needs_human", "error", "fixed", "skipped"}
if status in terminal_success:
    sys.exit(0)
if status in terminal_fail:
    sys.exit(10)
sys.exit(1)')
  rc=$?
  set -e
  echo "[smoke] $RESULT"
  if [ $rc -eq 0 ]; then
    echo "[smoke] SUCCESS"
    exit 0
  elif [ $rc -eq 10 ]; then
    echo "[smoke] FAILURE" >&2
    exit 1
  fi
  sleep "$SLEEP_SECONDS"
done

echo "[smoke] TIMEOUT" >&2
exit 1
