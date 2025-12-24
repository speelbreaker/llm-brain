#!/usr/bin/env python3
"""
Supervisor Loop Script

Orchestrates the workflow by interacting with the Obsidian vault and GitHub.
Enforces strict queue discipline and "No Pile-Ups".

Modes:
- dispatch_only (default): Claims tasks, creates branches/PRs, but does not implement code.
- execute_local: Allowed for specific automation tasks (not fully implemented yet).

Environment Variables:
- SUPERVISOR_MODE: "dispatch_only" or "execute_local"
- SUPERVISOR_REPO_DIR: Path to repo root
- SUPERVISOR_AGENT_NAME: Name of the agent (e.g., vps-supervisor)
- SUPERVISOR_REMOTE: Git remote (default: origin)
- SUPERVISOR_BASE_BRANCH: Main branch (default: main)
- DRY_RUN: If set, no git pushes or PR creations.
"""

import os
import sys
import re
import json
import time
import shutil
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Configuration ---
MODE = os.environ.get("SUPERVISOR_MODE", "dispatch_only")
APP_REPO_DIR = Path(
    os.environ.get("SUPERVISOR_APP_REPO_DIR")
    or os.environ.get("SUPERVISOR_REPO_DIR")
    or "."
).resolve()
REPO_DIR = APP_REPO_DIR
AGENT_NAME = os.environ.get("SUPERVISOR_AGENT_NAME", "vps-supervisor")
REMOTE = os.environ.get("SUPERVISOR_REMOTE", "origin")
BASE_BRANCH = os.environ.get("SUPERVISOR_BASE_BRANCH", "main")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

VAULT_REPO_DIR = Path(os.environ.get("SUPERVISOR_VAULT_REPO_DIR", APP_REPO_DIR)).resolve()
VAULT_ROOT = VAULT_REPO_DIR / "docs" / "obsidian"
QUEUE_FILE = VAULT_ROOT / "02_QUEUE" / "QUEUE.md"
ACTIVE_FILE = VAULT_ROOT / "06_PROMPTS" / "_ACTIVE.md"
ARCHIVE_DIR = VAULT_ROOT / "99_ARCHIVE"
CHANGELOG_FILE = VAULT_ROOT / "03_LOGS" / "CHANGELOG.md"
RUN_LOG_FILE = VAULT_ROOT / "03_LOGS" / "supervisor_runs.md"
INCIDENT_LOG_FILE = VAULT_ROOT / "04_OPS" / "Incident_Log.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("supervisor")

# --- Helpers ---

def run_cmd(cmd: List[str], cwd: Path = REPO_DIR, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    logger.info(f"CMD: {' '.join(cmd)}")
    if DRY_RUN and cmd[0] in ("git", "gh") and any(x in cmd for x in ("push", "create", "merge")):
        logger.info("DRY_RUN: Skipping write operation")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        
    try:
        return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise

def log_run(action: str, task_id: str, result: str, details: str = ""):
    """Append to supervisor_runs.md."""
    if not RUN_LOG_FILE.parent.exists():
        RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"| {timestamp} | {action} | {task_id} | {result} | {details} |\n"
    
    # Ensure header if new
    if not RUN_LOG_FILE.exists():
        RUN_LOG_FILE.write_text("| Timestamp | Action | Task ID | Result | Details |\n|---|---|---|---|---|", encoding="utf-8")
        
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def log_incident(title: str, description: str):
    """Append to Incident_Log.md."""
    if not INCIDENT_LOG_FILE.parent.exists():
        INCIDENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"\n### {timestamp} - {title}\n\n{description}\n"
    
    with open(INCIDENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def validate_vault():
    """Run the vault validator script."""
    validator_script = REPO_DIR / "scripts" / "validate_vault_workflow.py"
    if not validator_script.exists():
        raise FileNotFoundError(f"Validator script not found: {validator_script}")
    
    try:
        run_cmd([sys.executable, str(validator_script)], cwd=VAULT_REPO_DIR)
    except subprocess.CalledProcessError:
        log_incident("Vault Validation Failed", "scripts/validate_vault_workflow.py returned non-zero exit code.")
        sys.exit(1)

def git_preflight():
    """Ensure clean state and latest main."""
    logger.info("Running git preflight...")
    run_cmd(["git", "fetch", REMOTE])
    # Check for uncommitted changes
    status = run_cmd(["git", "status", "--porcelain"], check=False)
    if status.stdout.strip():
        logger.error("Dirty worktree detected. Aborting.")
        log_incident("Dirty Worktree", f"git status:\n{status.stdout}")
        sys.exit(1)
        
    # We don't force checkout main here to allow running on feature branches for testing,
    # but in prod env, it should ideally start from main.
    # For now, we fetch and pull if on main.
    branch = run_cmd(["git", "branch", "--show-current"]).stdout.strip()
    if branch == BASE_BRANCH:
        run_cmd(["git", "pull", "--rebase", REMOTE, BASE_BRANCH])

# --- Queue Logic ---

def parse_queue() -> Dict[str, List[str]]:
    """Parse QUEUE.md into sections."""
    content = QUEUE_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    sections = {"READY": [], "IN_PROGRESS": [], "IN_REVIEW": [], "DONE": []}
    current_section = None
    
    for line in lines:
        if line.startswith("## "):
            sec = line.strip("# ").strip()
            if sec in sections:
                current_section = sec
            else:
                current_section = None
        elif current_section and line.strip().startswith("- "):
            sections[current_section].append(line.strip())
            
    return sections

def write_queue(sections: Dict[str, List[str]]):
    """Write sections back to QUEUE.md."""
    # We need to preserve the header content (e.g. title) if possible, 
    # but for simplicity we will reconstruct the file based on known structure 
    # assuming standard format.
    # To be safer, we should read the file and replace sections. 
    
    content = QUEUE_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    current_section = None
    
    # We will skip lines inside known sections and inject our new lists
    
    # Simple state machine to rebuild file
    # Warning: This replaces comments inside list sections.
    
    skip = False
    
    for line in lines:
        if line.startswith("## "):
            sec = line.strip("# ").strip()
            if sec in sections:
                new_lines.append(line)
                # Inject new items
                for item in sections[sec]:
                    new_lines.append(item)
                new_lines.append("") # Spacer
                skip = True
                current_section = sec
            else:
                new_lines.append(line)
                skip = False
                current_section = None
        elif skip:
            if not line.strip().startswith("- ") and line.strip():
                # Non-list item inside section (e.g. comment?), keep it?
                # For strict machine parsing, we might discard or try to keep.
                # Spec says "machine-parsable section". Let's assume purely list items.
                pass
            elif not line.strip():
                # Empty line, keep?
                pass
        else:
            new_lines.append(line)
            
    QUEUE_FILE.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")

def parse_item(item: str) -> Dict[str, str]:
    """Parse a queue item line."""
    # Format: "- <ID> | <Task> | key: value | ..."
    # Remove leading "- "
    raw = item.lstrip("- ").strip()
    parts = [p.strip() for p in raw.split("|")]
    
    data = {}
    if len(parts) >= 2:
        data["id"] = parts[0]
        data["task"] = parts[1]
    
    for part in parts[2:]:
        if ":" in part:
            k, v = part.split(":", 1)
            data[k.strip()] = v.strip()
            
    data["raw"] = item
    return data

def build_item(data: Dict[str, str]) -> str:
    """Reconstruct item string."""
    # Start with ID | Task
    s = f"- {data['id']} | {data['task']}"
    
    # Add other keys in specific order or alphabetical
    # Order: branch, prompt, agent, started, pr, updated, completed
    order = ["branch", "prompt", "agent", "started", "pr", "updated", "completed"]
    
    for k in order:
        if k in data:
            s += f" | {k}: {data[k]}"
            
    # Any leftovers
    for k, v in data.items():
        if k not in ["id", "task", "raw"] and k not in order:
            s += f" | {k}: {v}"
            
    return s

def slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-').lower()

# --- Actions ---

def claim_task(ready_item_str: str, sections: Dict[str, List[str]]):
    """Move READY -> IN_PROGRESS, create branch, update state."""
    item = parse_item(ready_item_str)
    task_id = item["id"]
    task_name = item["task"]
    
    logger.info(f"Claiming task: {task_id} - {task_name}")
    
    # 1. Create Branch
    slug = slugify(task_name)
    branch_name = f"feat/{task_id}__{slug}"
    
    run_cmd(["git", "checkout", "-b", branch_name])
    
    # 2. Update Item Data
    item["branch"] = branch_name
    item["agent"] = AGENT_NAME
    item["started"] = datetime.now(timezone.utc).isoformat()
    
    new_item_str = build_item(item)
    
    # 3. Update Queue Sections
    sections["READY"].remove(ready_item_str)
    sections["IN_PROGRESS"] = [new_item_str] # Enforce max 1 by list replacement
    
    write_queue(sections)
    
    # 4. Update _ACTIVE.md
    active_content = f"# Active Task: {task_id}\n\n- **Task**: {task_name}\n- **Status**: IN_PROGRESS\n- **Branch**: {branch_name}\n- **Claimed By**: {AGENT_NAME}\n- **Prompt**: {item.get('prompt', 'N/A')}\n- **Started**: {item['started']}\n"
    ACTIVE_FILE.write_text(active_content, encoding="utf-8")
    
    # 5. Commit and Push
    run_cmd(["git", "add", str(QUEUE_FILE), str(ACTIVE_FILE)])
    run_cmd(["git", "commit", "-m", f"queue: start {task_id}"])
    run_cmd(["git", "push", "-u", REMOTE, branch_name])
    
    log_run("CLAIM", task_id, "SUCCESS", f"Branch: {branch_name}")

def process_in_progress(item_str: str, sections: Dict[str, List[str]]):
    """Check if we can move to IN_REVIEW (create PR)."""
    item = parse_item(item_str)
    task_id = item["id"]
    
    # Ensure we are on the right branch
    branch = item.get("branch")
    if not branch:
        logger.error(f"Item {task_id} missing branch!")
        return

    current_branch = run_cmd(["git", "branch", "--show-current"]).stdout.strip()
    if current_branch != branch:
        run_cmd(["git", "fetch", REMOTE])
        run_cmd(["git", "checkout", branch])
        run_cmd(["git", "pull", "--rebase", REMOTE, branch])

    # Check if GH is available
    gh_version = run_cmd(["gh", "--version"], check=False)
    if gh_version.returncode != 0:
        logger.warning("gh CLI not found. Cannot create PR automatically.")
        # We leave it IN_PROGRESS with a note?
        # Requirement says: "do NOT fail; instead write a log entry... keep it IN_PROGRESS but include note"
        if "pr" not in item:
            item["pr"] = "pending_manual"
            sections["IN_PROGRESS"] = [build_item(item)]
            write_queue(sections)
            run_cmd(["git", "add", str(QUEUE_FILE)])
            run_cmd(["git", "commit", "-m", f"queue: update {task_id} pr pending"])
            run_cmd(["git", "push"])
        return

    # Check if PR exists
    pr_check = run_cmd(["gh", "pr", "list", "--head", branch, "--json", "url,state"], check=False)
    pr_url = None
    
    if pr_check.returncode == 0:
        prs = json.loads(pr_check.stdout)
        if prs:
            pr_url = prs[0]["url"]
            logger.info(f"Existing PR found: {pr_url}")
    
    if not pr_url:
        # Create PR
        logger.info(f"Creating PR for {task_id}...")
        pr_create = run_cmd([
            "gh", "pr", "create", 
            "--title", f"{task_id}: {item['task']}",
            "--body", f"Automated PR for task {task_id}.\n\nAgent: {AGENT_NAME}\nMode: {MODE}",
            "--base", BASE_BRANCH
        ], check=False)
        
        if pr_create.returncode == 0:
            pr_url = pr_create.stdout.strip()
            logger.info(f"PR Created: {pr_url}")
        else:
            logger.error("Failed to create PR")
            log_incident("PR Create Failed", pr_create.stderr)
            return

    # Move to IN_REVIEW
    if pr_url:
        item["pr"] = pr_url
        item["updated"] = datetime.now(timezone.utc).isoformat()
        new_item_str = build_item(item)
        
        sections["IN_PROGRESS"] = []
        sections["IN_REVIEW"] = [new_item_str] # Max 1
        
        write_queue(sections)
        
        # Update ACTIVE
        active_content = ACTIVE_FILE.read_text(encoding="utf-8")
        active_content = active_content.replace("Status: IN_PROGRESS", "Status: IN_REVIEW")
        active_content += f"- **PR**: {pr_url}\n"
        ACTIVE_FILE.write_text(active_content, encoding="utf-8")
        
        run_cmd(["git", "add", str(QUEUE_FILE), str(ACTIVE_FILE)])
        run_cmd(["git", "commit", "-m", f"queue: review {task_id}"])
        run_cmd(["git", "push"])
        
        log_run("PR_OPEN", task_id, "SUCCESS", f"PR: {pr_url}")

def process_in_review(item_str: str, sections: Dict[str, List[str]]):
    """Check merge status and finalize."""
    item = parse_item(item_str)
    task_id = item["id"]
    pr_url = item.get("pr")
    
    if not pr_url or "github.com" not in pr_url:
        logger.info(f"Task {task_id} in review but no valid PR url. Skipping check.")
        return

    # Check status
    gh_check = run_cmd(["gh", "pr", "view", pr_url, "--json", "mergedAt,state"], check=False)
    if gh_check.returncode != 0:
        logger.warning(f"Could not check PR status for {pr_url}")
        return
        
    data = json.loads(gh_check.stdout)
    if data.get("state") == "MERGED":
        logger.info(f"PR {pr_url} is MERGED. Finalizing...")
        finalize_task(item, sections)
    else:
        logger.info(f"PR {pr_url} state is {data.get('state')}. Waiting.")

def finalize_task(item: Dict[str, str], sections: Dict[str, List[str]]):
    """Archive and mark DONE."""
    task_id = item["id"]
    branch = item.get("branch")
    
    # We need to operate on main or a finalize branch?
    # Spec says: "Prefer: commit to the same feature branch and update PR if it’s still open; 
    # if merged, open a new PR for archival/log updates."
    # Since PR is merged, we can't push to the same feature branch effectively for merge (it's closed).
    # We must create a new branch off main.
    
    run_cmd(["git", "checkout", BASE_BRANCH])
    run_cmd(["git", "pull", "--rebase", REMOTE, BASE_BRANCH])
    
    finalize_branch = f"ops/{task_id}__finalize"
    run_cmd(["git", "checkout", "-b", finalize_branch])
    
    # 1. Archive Prompt
    prompt_path = item.get("prompt")
    if prompt_path:
        src = REPO_DIR / prompt_path
        if src.exists():
            dst = ARCHIVE_DIR / src.name
            if not dst.parent.exists():
                dst.parent.mkdir(parents=True)
            shutil.move(str(src), str(dst))
            logger.info(f"Archived {src} to {dst}")
    
    # 2. Update Changelog
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_entry = f"- {task_id}: {item['task']} (PR: {item.get('pr')})"
    
    # Simple append to end for now, or find date header.
    # Appending to top of file after header is better but complex without robust parser.
    # Let's append to bottom or just write specific format.
    # Spec requirement: "update CHANGELOG.md"
    
    if CHANGELOG_FILE.exists():
        content = CHANGELOG_FILE.read_text(encoding="utf-8")
        # Try to find today's header
        header = f"## {today}"
        if header in content:
            content = content.replace(header, f"{header}\n{log_entry}")
        else:
            # Insert at top after title? Assuming title is first line
            lines = content.splitlines()
            if lines and lines[0].startswith("# "):
                lines.insert(1, "")
                lines.insert(2, header)
                lines.insert(3, log_entry)
                content = "\n".join(lines)
            else:
                content += f"\n\n{header}\n{log_entry}"
        CHANGELOG_FILE.write_text(content, encoding="utf-8")
        
    # 3. Update Queue
    item["completed"] = datetime.now(timezone.utc).isoformat()
    new_item_str = build_item(item)
    
    # We assume 'sections' passed in is stale since we switched branches/pulled.
    # Re-parse queue from current disk state
    current_sections = parse_queue()
    # Remove from IN_REVIEW
    current_sections["IN_REVIEW"] = [x for x in current_sections["IN_REVIEW"] if item["id"] not in x]
    # Add to DONE
    current_sections["DONE"].insert(0, new_item_str)
    
    write_queue(current_sections)
    
    # 4. Clear Active
    ACTIVE_FILE.write_text("# Active Task: None\n", encoding="utf-8")
    
    # 5. Commit and PR
    run_cmd(["git", "add", "."])
    run_cmd(["git", "commit", "-m", f"queue: done {task_id} (finalize)"])
    run_cmd(["git", "push", "-u", REMOTE, finalize_branch])
    
    # Open PR for finalize
    run_cmd([
        "gh", "pr", "create",
        "--title", f"{task_id}: Finalize/Archive",
        "--body", "Archiving prompts and updating changelog/queue.",
        "--base", BASE_BRANCH
    ], check=False)
    
    log_run("FINALIZE", task_id, "SUCCESS", "Archived and Updated Logs")


# --- Main Loop ---

def main():
    logger.info(f"Supervisor Loop Starting. Agent: {AGENT_NAME}, Mode: {MODE}")
    logger.info(f"App repo: {APP_REPO_DIR}")
    logger.info(f"Vault repo: {VAULT_REPO_DIR}")

    if not VAULT_REPO_DIR.is_dir():
        logger.error(f"Vault repo missing: {VAULT_REPO_DIR}")
        sys.exit(1)

    app_vault_path = REPO_DIR / "docs" / "obsidian"
    if app_vault_path.exists():
        log_incident(
            "Unsafe Vault Location",
            f"Detected {app_vault_path} in the app repo while configuration points to {VAULT_REPO_DIR}. Aborting to prevent leaks.",
        )
        logger.error("App repo still contains docs/obsidian; refusing to run to avoid vault leakage.")
        sys.exit(1)

    if not VAULT_ROOT.exists():
        logger.error(f"Vault data missing at {VAULT_ROOT}")
        sys.exit(1)
    
    git_preflight()
    validate_vault()
    
    sections = parse_queue()
    
    # 1. Check IN_REVIEW (Merge detection)
    if sections["IN_REVIEW"]:
        if len(sections["IN_REVIEW"]) > 1:
            logger.error("Pile-up in IN_REVIEW! Max 1 allowed.")
            sys.exit(1)
        process_in_review(sections["IN_REVIEW"][0], sections)
        return

    # 2. Check IN_PROGRESS (PR Creation)
    if sections["IN_PROGRESS"]:
        if len(sections["IN_PROGRESS"]) > 1:
            logger.error("Pile-up in IN_PROGRESS! Max 1 allowed.")
            sys.exit(1)
        process_in_progress(sections["IN_PROGRESS"][0], sections)
        return

    # 3. Check READY (Claim)
    if sections["READY"]:
        claim_task(sections["READY"][0], sections)
        return

    logger.info("No actionable items in queue.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Supervisor loop failed")
        log_incident("Supervisor Exception", str(e))
        sys.exit(1)
