---
name: pr-discipline
description: "PR discipline guide: use when preparing commits so the Local→GitHub→VPS pipeline stays intact and repository rules are honored."
---

# PR Discipline Skill

## Use when
- A feature branch is being created or updated.
- Conflicts or merges need careful handling before pushing to GitHub.

## Instructions
1. Always perform development locally, open a GitHub PR, merge via GitHub, and let the VPS pull the new `main`—do not treat the VPS checkout as primary.
2. Before pushing, ensure `scripts/ensure_not_on_vps_main.sh` passes so you are not working on the VPS main branch.
3. Keep PR descriptions aligned with the queue/changelog entries you created in the vault so reviewers understand the workflow context.
4. When resolving conflicts, prefer the details captured in the vault prompts and review templates so the merged change reflects the documented plan.
