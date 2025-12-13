"""
Analyzers module for the Telegram Code Review Agent.

Provides lightweight static analysis:
- File categorization
- Diff summarization
- Log collection
- Config change detection
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.change_detector import ChangedFile
from agent.config import settings


@dataclass
class FileSummary:
    """Summary of changes to a single file."""
    path: str
    category: str
    status: str
    additions: int = 0
    deletions: int = 0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "description": self.description,
        }


def classify_file(path: str) -> str:
    """Classify a file into a category based on its path and extension."""
    path_lower = path.lower()
    
    if any(x in path_lower for x in ["/api/", "routes", "endpoints", "views"]):
        return "api"
    
    if any(x in path_lower for x in ["test_", "_test.", "/tests/", "spec."]):
        return "tests"
    
    if any(x in path_lower for x in [
        "config", "settings", ".env", "pyproject.toml",
        "requirements.txt", "package.json", "replit.nix"
    ]):
        return "config"
    
    if any(x in path_lower for x in [
        "dockerfile", "docker-compose", ".github/",
        "deploy", "infra", "k8s", "terraform"
    ]):
        return "infra"
    
    if any(x in path_lower for x in ["/models/", "schema", "migration"]):
        return "models"
    
    if any(x in path_lower for x in ["/db/", "database", "storage"]):
        return "database"
    
    if any(x in path_lower for x in ["readme", "docs/", ".md"]):
        return "docs"
    
    if any(path_lower.endswith(ext) for ext in [".html", ".css", ".js", ".jsx", ".tsx", ".vue"]):
        return "frontend"
    
    return "other"


def classify_files(changed_files: List[ChangedFile]) -> Dict[str, str]:
    """Classify multiple files into categories."""
    return {cf.path: classify_file(cf.path) for cf in changed_files}


def build_diff_summary(changed_files: List[ChangedFile]) -> List[FileSummary]:
    """Build a summary of changed files."""
    summaries = []
    
    for cf in changed_files:
        category = classify_file(cf.path)
        
        desc_parts = []
        if cf.status == "A":
            desc_parts.append("new file")
        elif cf.status == "D":
            desc_parts.append("deleted")
        elif cf.status == "M":
            desc_parts.append("modified")
        elif cf.status == "R":
            desc_parts.append("renamed")
        
        if cf.additions > 0 or cf.deletions > 0:
            desc_parts.append(f"+{cf.additions}/-{cf.deletions}")
        
        summaries.append(FileSummary(
            path=cf.path,
            category=category,
            status=cf.status_label,
            additions=cf.additions,
            deletions=cf.deletions,
            description=", ".join(desc_parts),
        ))
    
    return summaries


def collect_logs(log_paths: Optional[List[str]] = None, max_lines: int = 200) -> str:
    """Collect recent log entries from configured log files."""
    if log_paths is None:
        log_paths = settings.log_paths if settings else ["logs/app.log"]
    
    collected = []
    
    for log_path in log_paths:
        path = Path(log_path)
        if not path.exists():
            continue
        
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                tail_lines = lines[-max_lines:] if len(lines) > max_lines else lines
                if tail_lines:
                    collected.append(f"=== {log_path} ===")
                    collected.extend(line.rstrip() for line in tail_lines)
        except (OSError, IOError):
            continue
    
    return "\n".join(collected)


def detect_config_changes(
    changed_files: List[ChangedFile],
    watch_files: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Detect changes to configuration files."""
    if watch_files is None:
        watch_files = settings.config_watch_files if settings else []
    
    config_changes = []
    
    for cf in changed_files:
        is_config = (
            classify_file(cf.path) == "config" or
            cf.path in watch_files or
            any(cf.path.endswith(wf) for wf in watch_files)
        )
        
        if is_config:
            config_changes.append({
                "path": cf.path,
                "status": cf.status_label,
                "category": "config",
            })
    
    return config_changes


def detect_suspicious_patterns(diff_text: str) -> List[Dict[str, Any]]:
    """Detect suspicious patterns in diff text."""
    patterns = [
        (r'["\'](?:sk-|pk_live_|pk_test_|sk_live_|sk_test_)[a-zA-Z0-9_-]+["\']', "Possible API key"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'(?:secret|token|api_key)\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'https?://[^\s"\']+\.(internal|local|localhost)', "Internal URL"),
        (r'exec\s*\(|eval\s*\(|os\.system\s*\(', "Dangerous function call"),
        (r'# ?TODO|# ?FIXME|# ?HACK|# ?XXX', "Code comment requiring attention"),
    ]
    
    findings = []
    
    for pattern, description in patterns:
        matches = re.finditer(pattern, diff_text, re.IGNORECASE)
        for match in matches:
            context = diff_text[max(0, match.start() - 50):match.end() + 50]
            if context.startswith("+") or "\n+" in context:
                findings.append({
                    "pattern": description,
                    "match": match.group()[:50],
                    "severity": "HIGH" if "key" in description.lower() or "secret" in description.lower() else "MEDIUM",
                })
    
    return findings


@dataclass
class AnalysisContext:
    """Context gathered from analyzers for LLM."""
    file_summaries: List[FileSummary] = field(default_factory=list)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    config_changes: List[Dict[str, str]] = field(default_factory=list)
    suspicious_patterns: List[Dict[str, Any]] = field(default_factory=list)
    log_excerpt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_summaries": [f.to_dict() for f in self.file_summaries],
            "categories": self.categories,
            "config_changes": self.config_changes,
            "suspicious_patterns": self.suspicious_patterns,
            "has_logs": bool(self.log_excerpt),
        }


def analyze_changes(
    changed_files: List[ChangedFile],
    diff_text: str = "",
) -> AnalysisContext:
    """Run all analyzers and return combined context."""
    file_summaries = build_diff_summary(changed_files)
    
    categories: Dict[str, List[str]] = {}
    for fs in file_summaries:
        if fs.category not in categories:
            categories[fs.category] = []
        categories[fs.category].append(fs.path)
    
    config_changes = detect_config_changes(changed_files)
    suspicious = detect_suspicious_patterns(diff_text) if diff_text else []
    logs = collect_logs()
    
    return AnalysisContext(
        file_summaries=file_summaries,
        categories=categories,
        config_changes=config_changes,
        suspicious_patterns=suspicious,
        log_excerpt=logs[:5000] if logs else "",
    )
