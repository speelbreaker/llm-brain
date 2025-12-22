#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

output_path="docs/RECENT_DIFF.md"
mkdir -p "docs"

generated_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
branch="$(git rev-parse --abbrev-ref HEAD)"
head_sha="$(git rev-parse HEAD)"

if git show-ref --verify --quiet refs/remotes/origin/main; then
  base="origin/main"
else
  base="HEAD~10"
fi

if ! git rev-parse --verify --quiet "$base" >/dev/null; then
  base="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
fi

tmp_file="$(mktemp)"
redacted_file="$(mktemp)"
trap 'rm -f "$tmp_file" "$redacted_file"' EXIT

{
  echo "# Recent Diff"
  echo
  echo "generated_at_utc: $generated_at_utc"
  echo "branch: $branch"
  echo "head_sha: $head_sha"
  echo "base: $base"
  echo
  echo "## git log --oneline -n 25"
  git log --oneline -n 25
  echo
  echo "## git diff --stat $base..HEAD"
  git diff --stat "$base..HEAD"
  echo
  echo "## git diff $base..HEAD"
  git diff "$base..HEAD"
} >"$tmp_file"

python3 - "$tmp_file" "$redacted_file" <<'PY'
import re
import sys

pattern = re.compile(
    r"(OPENAI_API_KEY|DERIBIT_SECRET|DERIBIT_API_KEY|DERIBIT_API_SECRET|"
    r"ANTHROPIC_API_KEY|GEMINI_API_KEY|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|"
    r"GITHUB_TOKEN|SLACK_TOKEN|PRIVATE_KEY)"
)

input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path, encoding="utf-8", errors="ignore") as handle, open(
    output_path, "w", encoding="utf-8"
) as output:
    for line in handle:
        if pattern.search(line):
            output.write("[REDACTED SECRET]\n")
        else:
            output.write(line)
PY

line_count="$(wc -l <"$redacted_file" | tr -d " ")"
if [ "$line_count" -gt 2000 ]; then
  {
    head -n 2000 "$redacted_file"
    echo
    echo "TRUNCATED"
  } >"$output_path"
else
  mv "$redacted_file" "$output_path"
  redacted_file=""
fi
