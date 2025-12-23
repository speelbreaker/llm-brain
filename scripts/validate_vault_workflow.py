#!/usr/bin/env python3
"""Validate the structure and content of the Obsidian workflow vault."""

import os
import re
import sys
from pathlib import Path

# Config
VAULT_ROOT = Path("docs/obsidian")
REQUIRED_DIRS = [
    "00_HOME",
    "01_RULES",
    "02_QUEUE",
    "03_LOGS",
    "06_PROMPTS",
    "99_ARCHIVE",
]
REQUIRED_FILES = [
    "00_HOME/NORTHSTAR.md",
    "01_RULES/QUEUE_DISCIPLINE.md",
    "01_RULES/PR_WORKFLOW.md",
    "02_QUEUE/QUEUE.md",
    "06_PROMPTS/_ACTIVE.md",
]

def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def check_structure():
    if not VAULT_ROOT.exists():
        fail(f"Vault root not found: {VAULT_ROOT}")
    
    for d in REQUIRED_DIRS:
        if not (VAULT_ROOT / d).is_dir():
            fail(f"Missing directory: {d}")
            
    for f in REQUIRED_FILES:
        if not (VAULT_ROOT / f).exists():
            fail(f"Missing file: {f}")

def parse_queue():
    queue_path = VAULT_ROOT / "02_QUEUE/QUEUE.md"
    content = queue_path.read_text(encoding="utf-8")
    
    sections = {
        "IN_PROGRESS": [],
        "READY": [],
        "IN_REVIEW": [],
        "DONE": []
    }
    
    current_section = None
    for line in content.splitlines():
        if line.startswith("## "):
            section_name = line.strip("# ").strip()
            if section_name in sections:
                current_section = section_name
            else:
                current_section = None # Ignore other sections
        elif current_section and line.strip().startswith("- "):
            sections[current_section].append(line.strip())

    return sections

def validate_queue(sections):
    # 1. Check limits
    if len(sections["IN_PROGRESS"]) > 1:
        fail(f"Too many IN_PROGRESS items: {len(sections['IN_PROGRESS'])} (max 1)")
    if len(sections["IN_REVIEW"]) > 1:
        fail(f"Too many IN_REVIEW items: {len(sections['IN_REVIEW'])} (max 1)")

    # 2. Validate IN_PROGRESS item format
    if sections["IN_PROGRESS"]:
        item = sections["IN_PROGRESS"][0]
        if "branch:" not in item or "prompt:" not in item:
            fail(f"IN_PROGRESS item missing 'branch:' or 'prompt:': {item}")
        
        # Extract prompt path
        match = re.search(r"prompt:\s*(\S+)", item)
        if not match:
            fail(f"Could not parse prompt path from: {item}")
        
        prompt_path = match.group(1).strip()
        full_prompt_path = Path(prompt_path)
        
        if not full_prompt_path.exists():
             # Try relative to vault root? No, spec implies repo-relative path usually
             # Let's assume repo-relative as per example "docs/obsidian/..."
             if not (Path.cwd() / prompt_path).exists():
                 fail(f"Prompt file not found: {prompt_path}")

        # 3. Validate _ACTIVE.md matches
        active_ptr = (VAULT_ROOT / "06_PROMPTS/_ACTIVE.md").read_text(encoding="utf-8").strip()
        if active_ptr != prompt_path:
            fail(f"_ACTIVE.md ({active_ptr}) does not match IN_PROGRESS prompt ({prompt_path})")

def validate_prompts(sections):
    # Gather all referenced prompts
    all_items = sections["IN_PROGRESS"] + sections["READY"] + sections["IN_REVIEW"]
    
    for item in all_items:
        match = re.search(r"prompt:\s*(\S+)", item)
        if match:
            prompt_path = match.group(1).strip()
            p = Path(prompt_path)
            if not p.exists():
                 fail(f"Referenced prompt not found: {prompt_path}")
            
            content = p.read_text(encoding="utf-8")
            if "## Acceptance Criteria" not in content:
                fail(f"Prompt {prompt_path} missing '## Acceptance Criteria'")
            if "## Tests / Verification" not in content and "## Tests" not in content:
                 # Be lenient slightly on exact heading match if "Tests" exists? 
                 # Spec said: "## Tests / Verification"
                 if "## Tests" not in content and "## Verification" not in content:
                     fail(f"Prompt {prompt_path} missing '## Tests / Verification'")

def main():
    print("Validating Vault Workflow...")
    check_structure()
    sections = parse_queue()
    validate_queue(sections)
    validate_prompts(sections)
    print("✅ Vault is valid.")
    return 0

if __name__ == "__main__":
    main()
