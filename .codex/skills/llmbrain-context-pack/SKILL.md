---
name: llmbrain-context-pack
description: Generate and publish context-pack artifacts and verify Drive folder contents.
metadata:
  short-description: Generate/publish/verify context pack
---

## What to generate
- `REPO_MANIFEST_latest.json` or `.md`
- `RECENT_DIFF_latest.md`
- `ROADMAP_BACKLOG_latest.md`
- `TEST_SUMMARY_latest.txt`
- `OPS_HEALTH_latest.json`
- All other `*_latest.*` fidelity docs if they exist.

## Guardrails
- Do not run this skill outside the expected repo root; refuse the request if the repository is wrong.
- Avoid requiring a web server as part of the process.
- If ops-health generation fails, fail closed and capture the error details in the artifact so downstream consumers know what went wrong.

## Verification
- Verify `rclone lsf gdrive:llm-brain_context_pack | sort` contains the expected files.
