# PROMPT_001: Merge PR #17 (ops-health-contract)

## Objective
Resolve conflict and merge PR #17 to establish ops-health contract enforcement.

## Prerequisites
- [ ] `gh` CLI authenticated
- [ ] Clean working directory (`git status` shows no uncommitted changes)

## Acceptance Criteria
1. PR #17 rebased onto current main without conflicts
2. `pytest tests/test_ops_health_contract.py -v` passes
3. `scripts/gen_ops_health_latest.py` produces valid JSON with:
   - `overall_status` in {OK, WARN, FAIL}
   - `can_trade` is boolean
   - `gate_overall` present
4. PR merged to main

## Steps
1. `git fetch origin && git checkout vps-salvage/ops-health-contract`
2. `git rebase origin/main` - resolve conflicts
3. Run tests: `pytest tests/test_ops_health_contract.py -v`
4. Push: `git push --force-with-lease origin vps-salvage/ops-health-contract`
5. Merge: `gh pr merge 17 --squash --body "Merged: ops-health contract enforcement"`

## On Failure
- If tests fail: fix issues, amend commit, re-push
- If rebase is complex: consider `git merge origin/main` instead

## Verification
```bash
git checkout main && git pull origin main
python scripts/gen_ops_health_latest.py --help  # or --dry-run
```
