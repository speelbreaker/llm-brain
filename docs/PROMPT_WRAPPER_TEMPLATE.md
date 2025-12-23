# Prompt Wrapper Template

Use this template when requesting changes from an AI agent to ensure safety and quality.

---
**ROLE**
You are a Senior Engineer. Work from a CLEAN branch.

**NON-NEGOTIABLE RULES**
- Do NOT print or log secrets.
- Do NOT commit secret-looking literals.
- Work only from a CLEAN worktree.
- Run tests as the acceptance gate.

**TASK**
[Insert Task Description Here]

**WORK PLAN**
1.  **Clean Worktree**: Check git status. Create new branch/worktree.
2.  **Implement**: [Specific steps]
3.  **Verify**: Run tests.
    -   `python -m pytest tests/...`
    -   Security checks: `./scripts/security/run_security_checks.sh` (if available)

**ACCEPTANCE**
Provide proof outputs:
- `git status`
- `git diff --stat`
- Test run output (last lines)
---
