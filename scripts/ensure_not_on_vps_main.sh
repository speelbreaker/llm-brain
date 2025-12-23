#!/usr/bin/env bash
# ============================================================================
# ensure_not_on_vps_main.sh - Guard against direct commits to main on VPS
# ============================================================================
#
# PURPOSE:
#   Enforce "Local → GitHub → VPS" workflow by blocking git commit/push
#   operations on the main branch when running on the production VPS.
#
# USAGE:
#   Called by git hooks (pre-commit, pre-push) or directly:
#     ./scripts/ensure_not_on_vps_main.sh
#
# EXIT CODES:
#   0 - OK to proceed (not on VPS, or not on main branch)
#   1 - BLOCKED (on VPS and on main branch)
#
# ENVIRONMENT VARIABLES (for testing):
#   VPS_GUARD_FORCE_VPS=1    - Simulate being on VPS
#   VPS_GUARD_FORCE_BRANCH=x - Override detected branch name
#
# ============================================================================
set -euo pipefail

# --- Configuration ---
VPS_REPO_ROOT="/opt/llm-brain/llm-brain"
PROTECTED_BRANCH="main"

# --- Detection functions ---

is_on_vps() {
    # Check 1: Repo root matches VPS path
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
    
    if [[ "$repo_root" == "$VPS_REPO_ROOT" ]]; then
        return 0
    fi
    
    # Check 2: Hostname contains common VPS patterns
    local hostname
    hostname="$(hostname 2>/dev/null || echo "")"
    
    # Add your VPS hostname patterns here
    if [[ "$hostname" == *"vps"* ]] || \
       [[ "$hostname" == *"prod"* ]] || \
       [[ "$hostname" == *"llm-brain"* ]]; then
        return 0
    fi
    
    # Check 3: Environment override for testing
    if [[ "${VPS_GUARD_FORCE_VPS:-}" == "1" ]]; then
        return 0
    fi
    
    return 1
}

get_current_branch() {
    # Allow override for testing
    if [[ -n "${VPS_GUARD_FORCE_BRANCH:-}" ]]; then
        echo "$VPS_GUARD_FORCE_BRANCH"
        return
    fi
    
    git rev-parse --abbrev-ref HEAD 2>/dev/null || echo ""
}

is_protected_branch() {
    local branch="$1"
    [[ "$branch" == "$PROTECTED_BRANCH" ]]
}

# --- Main guard logic ---

main() {
    local branch
    branch="$(get_current_branch)"
    
    # Skip if not in a git repo
    if [[ -z "$branch" ]]; then
        exit 0
    fi
    
    # Skip if detached HEAD
    if [[ "$branch" == "HEAD" ]]; then
        exit 0
    fi
    
    # Check if we're on VPS AND on protected branch
    if is_on_vps && is_protected_branch "$branch"; then
        cat >&2 <<EOF

╔══════════════════════════════════════════════════════════════════════════════╗
║  ⛔  BLOCKED: Direct changes to '$PROTECTED_BRANCH' on production VPS        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This repository follows the "Local → GitHub → VPS" workflow.                ║
║  The VPS copy is DEPLOY-ONLY to prevent production drift.                    ║
║                                                                              ║
║  WHAT TO DO:                                                                 ║
║                                                                              ║
║  1. For urgent hotfixes on VPS:                                              ║
║     git checkout -b hotfix/<name>                                            ║
║     # make changes, commit, push                                             ║
║     git push -u origin hotfix/<name>                                         ║
║     # then create PR to merge into main                                      ║
║                                                                              ║
║  2. For normal development:                                                  ║
║     Work locally, push to GitHub, then deploy to VPS.                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
        exit 1
    fi
    
    # All checks passed
    exit 0
}

main "$@"

