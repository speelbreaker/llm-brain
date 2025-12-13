"""
Chat tools for the Telegram Code Review Agent.

These are the tools available for the LLM to call during natural language chat.
Includes enhanced Repo Q&A capabilities with secret redaction.
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.change_detector import ChangeDetector
from agent.storage import get_last_review, get_meta, get_review_count


IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "dist", "build",
    ".auditor", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", "eggs", ".eggs", "*.egg-info", ".nox",
}

IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".wav",
}

PRIORITY_DIRS = ["src", "app", "agent", "server", "services", "api", "lib", "core"]

def _redact_base64(match: re.Match) -> str:
    """Redact long base64-like strings (potential secrets)."""
    val = match.group(1)
    if len(val) > 60 and not any(c in val for c in ' \n\t'):
        return "***BASE64_REDACTED***"
    return val


SECRET_PATTERNS = [
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})'), r'sk-***REDACTED***'),
    (re.compile(r'(sk_live_[a-zA-Z0-9]{20,})'), r'sk_live_***REDACTED***'),
    (re.compile(r'(sk_test_[a-zA-Z0-9]{20,})'), r'sk_test_***REDACTED***'),
    (re.compile(r'(api[_-]?key\s*[=:]\s*["\']?)([a-zA-Z0-9_\-]{16,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(secret\s*[=:]\s*["\']?)([a-zA-Z0-9_\-]{16,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(password\s*[=:]\s*["\']?)([^\s"\']{8,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(token\s*[=:]\s*["\']?)([a-zA-Z0-9_\-\.]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(Authorization:\s*Bearer\s+)([a-zA-Z0-9_\-\.]+)', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(-----BEGIN[A-Z ]+-----)[\s\S]*?(-----END[A-Z ]+-----)'), r'\1\n***PEM REDACTED***\n\2'),
    (re.compile(r'([a-zA-Z0-9+/]{40,}={0,2})'), _redact_base64),
]


def redact_secrets(text: str) -> str:
    """Redact potential secrets from text output."""
    result = text
    for pattern, replacement in SECRET_PATTERNS:
        if callable(replacement):
            result = pattern.sub(replacement, result)
        else:
            result = pattern.sub(replacement, result)
    return result


def is_path_safe(path: str) -> bool:
    """Check if path is safe (no traversal, within repo)."""
    try:
        resolved = Path(path).resolve()
        cwd = Path.cwd().resolve()
        return str(resolved).startswith(str(cwd))
    except Exception:
        return False


def should_ignore_path(path: Path) -> bool:
    """Check if path should be ignored in searches."""
    parts = path.parts
    for part in parts:
        if part in IGNORE_DIRS:
            return True
        for pattern in IGNORE_DIRS:
            if fnmatch.fnmatch(part, pattern):
                return True
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    return False


GREP_LINE_PATTERN = re.compile(r'^(.+?)[:\-](\d+)[:\-]')

def _extract_grep_path(line: str) -> Optional[str]:
    """Extract file path from grep output line (handles : and - separators).
    
    Grep output formats:
    - Match line: path/file.py:123:content
    - Context line: path/file.py-123-content
    
    Uses regex to find path:line_number: pattern to avoid splitting on hyphens in paths.
    """
    if not line:
        return None
    match = GREP_LINE_PATTERN.match(line)
    if match:
        return match.group(1)
    return None


def _is_ignored_path(path_str: str) -> bool:
    """Check if path string matches any ignored directory."""
    parts = path_str.split("/")
    for part in parts:
        if part in IGNORE_DIRS:
            return True
        for pattern in IGNORE_DIRS:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


@dataclass
class SearchHit:
    """A single search result."""
    path: str
    line_no: int
    line_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None


def get_status() -> ToolResult:
    """Get current system status including git, LLM, and review info."""
    try:
        detector = ChangeDetector()
        review = get_last_review()
        review_count = get_review_count()
        git_available = detector.is_git_available()
        current_head = detector.get_current_head() if git_available else None
        last_reviewed = get_meta("last_reviewed_commit")
        
        status_parts = []
        status_parts.append(f"Git available: {'Yes' if git_available else 'No'}")
        if current_head:
            status_parts.append(f"Current HEAD: {current_head[:8]}")
        if last_reviewed:
            status_parts.append(f"Last reviewed commit: {last_reviewed[:8]}")
        status_parts.append(f"Total reviews stored: {review_count}")
        
        if review:
            status_parts.append(f"Last review: {review.created_at} - {review.overall_severity}")
        
        return ToolResult(
            success=True,
            output="\n".join(status_parts),
            data={
                "git_available": git_available,
                "current_head": current_head,
                "last_reviewed": last_reviewed,
                "review_count": review_count,
            }
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error getting status: {e}")


def get_project_map() -> ToolResult:
    """Get a high-level map of the project structure."""
    try:
        auditor_map = Path(".auditor/PROJECT_MAP.md")
        if auditor_map.exists():
            content = auditor_map.read_text()[:3000]
            return ToolResult(success=True, output=content)
        
        important_dirs = ["src", "agent", "scripts", "tests", "data", "app", "server", "api"]
        project_map = []
        
        for d in important_dirs:
            path = Path(d)
            if path.exists() and path.is_dir():
                py_files = [f for f in path.rglob("*.py") if not should_ignore_path(f)][:15]
                project_map.append(f"📁 {d}/ ({len(py_files)} Python files)")
                for f in py_files[:7]:
                    project_map.append(f"   - {f.relative_to(path)}")
                if len(py_files) > 7:
                    project_map.append(f"   ... and {len(py_files) - 7} more")
        
        config_files = ["pyproject.toml", "requirements.txt", ".env.example", "replit.md", "package.json"]
        found_configs = [f for f in config_files if Path(f).exists()]
        if found_configs:
            project_map.append("\n📄 Config files:")
            for f in found_configs:
                project_map.append(f"   - {f}")
        
        return ToolResult(
            success=True,
            output="\n".join(project_map) if project_map else "No recognized project structure found.",
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error mapping project: {e}")


def get_latest_diff(change_id: Optional[str] = None) -> ToolResult:
    """Get the latest git diff or diff for a specific commit."""
    try:
        detector = ChangeDetector()
        if not detector.is_git_available():
            return ToolResult(success=False, output="Git is not available.")
        
        if change_id:
            result = subprocess.run(
                ["git", "show", "--stat", change_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        else:
            last_reviewed = get_meta("last_reviewed_commit")
            if last_reviewed:
                result = subprocess.run(
                    ["git", "diff", "--stat", last_reviewed, "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            else:
                result = subprocess.run(
                    ["git", "diff", "--stat", "HEAD~5", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        
        if result.returncode != 0:
            return ToolResult(success=False, output=f"Git error: {result.stderr}")
        
        output = result.stdout.strip()
        if not output:
            output = "No changes detected."
        elif len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        
        return ToolResult(success=True, output=redact_secrets(output))
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Git command timed out.")
    except Exception as e:
        return ToolResult(success=False, output=f"Error getting diff: {e}")


def run_smoke_tests() -> ToolResult:
    """Run smoke tests if available."""
    try:
        smoke_script = Path("scripts/smoke_greg_strategies.py")
        if not smoke_script.exists():
            return ToolResult(
                success=False,
                output="No smoke test script found at scripts/smoke_greg_strategies.py"
            )
        
        result = subprocess.run(
            ["python", str(smoke_script), "--quick"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path.cwd()),
        )
        
        output = result.stdout + result.stderr
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        
        return ToolResult(
            success=result.returncode == 0,
            output=redact_secrets(output) or "Smoke tests completed with no output.",
            data={"return_code": result.returncode}
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Smoke tests timed out after 2 minutes.")
    except Exception as e:
        return ToolResult(success=False, output=f"Error running smoke tests: {e}")


def run_security_scans() -> ToolResult:
    """Run security-focused pattern scans on the codebase."""
    try:
        security_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            (r"eval\s*\(", "Dangerous eval() usage"),
            (r"exec\s*\(", "Dangerous exec() usage"),
            (r"subprocess\..*shell\s*=\s*True", "Shell injection risk"),
            (r"\.format\(.*input", "Potential format string injection"),
        ]
        
        findings = []
        src_dirs = ["src", "agent", "scripts"]
        
        for pattern, description in security_patterns:
            for src_dir in src_dirs:
                if not Path(src_dir).exists():
                    continue
                try:
                    result = subprocess.run(
                        ["grep", "-rn", "-E", pattern, src_dir],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.stdout.strip():
                        matches = result.stdout.strip().split("\n")[:3]
                        findings.append(f"⚠️ {description}:")
                        for match in matches:
                            findings.append(f"   {redact_secrets(match[:100])}")
                except Exception:
                    pass
        
        if findings:
            return ToolResult(
                success=True,
                output="\n".join(findings),
                data={"finding_count": len([f for f in findings if f.startswith("⚠️")])}
            )
        else:
            return ToolResult(
                success=True,
                output="✅ No obvious security issues found in pattern scan.",
                data={"finding_count": 0}
            )
    except Exception as e:
        return ToolResult(success=False, output=f"Error running security scan: {e}")


def search_repo(query: str, limit: int = 20, include_context: bool = True) -> ToolResult:
    """
    Search the repository for a string or pattern with enhanced features.
    
    Args:
        query: Search query (case-insensitive by default)
        limit: Maximum number of matches to return (default 20)
        include_context: Whether to include surrounding context lines
    """
    try:
        if not query or len(query) < 2:
            return ToolResult(success=False, output="Query must be at least 2 characters.")
        
        limit = min(limit, 50)
        
        file_types = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.json", "*.yaml", "*.yml", "*.md", "*.html", "*.css"]
        include_args = []
        for ft in file_types:
            include_args.extend(["--include", ft])
        
        exclude_dirs = [
            ".git", ".venv", "venv", "node_modules", "dist", "build",
            ".auditor", "__pycache__", ".mypy_cache", ".pytest_cache",
            ".tox", "eggs", ".eggs", ".nox", ".cache", ".uv",
        ]
        exclude_args = []
        for d in exclude_dirs:
            exclude_args.extend(["--exclude-dir", d])
        
        context_args = ["-B", "2", "-C", "2"] if include_context else []
        
        cmd = ["grep", "-rni"] + context_args + include_args + exclude_args + [query, "."]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        output = result.stdout.strip()
        if not output:
            return ToolResult(
                success=True, 
                output=f"No matches found for '{query}'.\n\nTry:\n• Different spelling\n• Partial word\n• Use /open to browse specific files",
                data={"match_count": 0}
            )
        
        lines = output.split("\n")
        
        filtered_lines = []
        for line in lines:
            if line == "--":
                filtered_lines.append(line)
                continue
            path_part = _extract_grep_path(line)
            if path_part:
                path_clean = path_part.lstrip("./")
                if not _is_ignored_path(path_clean):
                    filtered_lines.append(line)
        
        def priority_score(line: str) -> int:
            for i, d in enumerate(PRIORITY_DIRS):
                if f"./{d}/" in line or line.startswith(f"{d}/"):
                    return i
            return len(PRIORITY_DIRS)
        
        filtered_lines.sort(key=priority_score)
        
        if len(filtered_lines) > limit:
            display_lines = filtered_lines[:limit]
            truncated = True
            total = len(filtered_lines)
        else:
            display_lines = filtered_lines
            truncated = False
            total = len(filtered_lines)
        
        formatted_output = []
        for line in display_lines:
            formatted_output.append(redact_secrets(line[:200]))
        
        result_text = "\n".join(formatted_output)
        if truncated:
            result_text += f"\n\n... and {total - limit} more matches. Refine your search or increase limit."
        
        return ToolResult(
            success=True,
            output=result_text,
            data={"match_count": total, "shown": len(display_lines)}
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Search timed out.")
    except Exception as e:
        return ToolResult(success=False, output=f"Error searching: {e}")


def open_file(path: str, start_line: int = 1, end_line: int = 50) -> ToolResult:
    """
    Read a section of a file with line numbers and secret redaction.
    
    Args:
        path: Path to the file (relative to repo root)
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (max 200 lines per request)
    """
    try:
        if not is_path_safe(path):
            return ToolResult(success=False, output="⛔ Path traversal not allowed.")
        
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(success=False, output=f"File not found: {path}")
        
        if not file_path.is_file():
            return ToolResult(success=False, output=f"Not a file: {path}")
        
        sensitive_patterns = [".env", "secrets", ".key", ".pem", ".crt", ".p12", ".pfx", "credentials"]
        if any(p in path.lower() for p in sensitive_patterns):
            return ToolResult(success=False, output="⛔ Cannot read sensitive files.")
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        start_line = max(1, start_line)
        end_line = min(total_lines, end_line)
        end_line = min(end_line, start_line + 199)
        
        selected = lines[start_line - 1:end_line]
        
        formatted_lines = []
        for i, line in enumerate(selected):
            line_no = start_line + i
            line_text = redact_secrets(line.rstrip())
            formatted_lines.append(f"{line_no:4d} | {line_text}")
        
        content = "\n".join(formatted_lines)
        
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated for Telegram)"
        
        header = f"📄 {path} (lines {start_line}-{end_line} of {total_lines})\n{'─' * 40}\n"
        footer = ""
        if end_line < total_lines:
            footer = f"\n{'─' * 40}\n💡 More lines available. Ask to see lines {end_line + 1}-{min(end_line + 50, total_lines)}"
        
        return ToolResult(
            success=True,
            output=header + content + footer,
            data={"total_lines": total_lines, "shown": f"{start_line}-{end_line}"}
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error reading file: {e}")


def list_files(root: str = ".", pattern: Optional[str] = None, limit: int = 100) -> ToolResult:
    """
    List files in a directory with optional pattern matching.
    
    Args:
        root: Root directory to list (relative to repo)
        pattern: Optional glob pattern to filter files (e.g., "*.py")
        limit: Maximum number of files to return
    """
    try:
        if not is_path_safe(root):
            return ToolResult(success=False, output="⛔ Path traversal not allowed.")
        
        root_path = Path(root)
        if not root_path.exists():
            return ToolResult(success=False, output=f"Directory not found: {root}")
        
        if not root_path.is_dir():
            return ToolResult(success=False, output=f"Not a directory: {root}")
        
        limit = min(limit, 200)
        
        files = []
        dirs = []
        
        try:
            if pattern:
                items = list(root_path.rglob(pattern))
            else:
                items = list(root_path.iterdir())
        except PermissionError:
            return ToolResult(success=False, output="Permission denied.")
        
        for item in items:
            if should_ignore_path(item):
                continue
            
            try:
                rel_path = item.relative_to(root_path) if root != "." else item
                if item.is_dir():
                    dirs.append(f"📁 {rel_path}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    files.append(f"   {rel_path} ({size_str})")
            except Exception:
                continue
        
        dirs.sort()
        files.sort()
        
        all_items = dirs[:limit // 2] + files[:limit - len(dirs[:limit // 2])]
        
        if not all_items:
            return ToolResult(
                success=True,
                output=f"No files found in {root}" + (f" matching '{pattern}'" if pattern else ""),
                data={"count": 0}
            )
        
        total = len(dirs) + len(files)
        header = f"📂 {root}/" + (f" (pattern: {pattern})" if pattern else "") + f"\n{'─' * 30}\n"
        
        output = header + "\n".join(all_items[:limit])
        if total > limit:
            output += f"\n\n... and {total - limit} more items"
        
        return ToolResult(
            success=True,
            output=output,
            data={"dirs": len(dirs), "files": len(files), "shown": min(total, limit)}
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error listing files: {e}")


def tail_logs(n_lines: int = 50) -> ToolResult:
    """Get the last N lines from application logs."""
    try:
        log_paths = ["logs/app.log", "logs/agent.log", "/tmp/logs"]
        found_logs = []
        
        for log_path in log_paths:
            path = Path(log_path)
            if path.exists():
                if path.is_file():
                    found_logs.append(path)
                elif path.is_dir():
                    log_files = sorted(path.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
                    found_logs.extend(log_files[:2])
        
        if not found_logs:
            return ToolResult(success=True, output="No log files found.")
        
        output_parts = []
        n_lines = min(n_lines, 100)
        
        for log_file in found_logs[:3]:
            try:
                result = subprocess.run(
                    ["tail", "-n", str(n_lines), str(log_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.stdout.strip():
                    output_parts.append(f"=== {log_file.name} ===")
                    output_parts.append(redact_secrets(result.stdout.strip()[:1500]))
            except Exception:
                pass
        
        output = "\n\n".join(output_parts) if output_parts else "No log content available."
        
        return ToolResult(success=True, output=output)
    except Exception as e:
        return ToolResult(success=False, output=f"Error reading logs: {e}")


AVAILABLE_TOOLS = {
    "get_status": {
        "function": get_status,
        "description": "Get current system status including git state, review count, and last review info.",
        "parameters": {},
    },
    "get_project_map": {
        "function": get_project_map,
        "description": "Get a high-level map of the project structure showing directories and key files.",
        "parameters": {},
    },
    "get_latest_diff": {
        "function": get_latest_diff,
        "description": "Get the latest git diff showing what changed since last review.",
        "parameters": {
            "change_id": {"type": "string", "description": "Optional commit hash to show specific changes", "required": False}
        },
    },
    "run_smoke_tests": {
        "function": run_smoke_tests,
        "description": "Run smoke tests to verify basic functionality.",
        "parameters": {},
    },
    "run_security_scans": {
        "function": run_security_scans,
        "description": "Scan the codebase for common security issues like hardcoded secrets.",
        "parameters": {},
    },
    "search_repo": {
        "function": search_repo,
        "description": "Search the repository for code, functions, classes, or patterns. Case-insensitive. Use for 'where is X implemented?', 'find Y', 'search for Z'.",
        "parameters": {
            "query": {"type": "string", "description": "The search query (case-insensitive)", "required": True},
            "limit": {"type": "integer", "description": "Max results (default 20, max 50)", "required": False},
            "include_context": {"type": "boolean", "description": "Include surrounding lines (default true)", "required": False},
        },
    },
    "open_file": {
        "function": open_file,
        "description": "Read a section of a file with line numbers. Use after search to see code details. Max 200 lines per request.",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file", "required": True},
            "start_line": {"type": "integer", "description": "Starting line number (default 1)", "required": False},
            "end_line": {"type": "integer", "description": "Ending line number (default 50, max start+200)", "required": False},
        },
    },
    "list_files": {
        "function": list_files,
        "description": "List files and directories. Use to explore project structure before searching or opening files.",
        "parameters": {
            "root": {"type": "string", "description": "Root directory (default '.')", "required": False},
            "pattern": {"type": "string", "description": "Glob pattern filter (e.g., '*.py')", "required": False},
            "limit": {"type": "integer", "description": "Max files to return (default 100)", "required": False},
        },
    },
    "tail_logs": {
        "function": tail_logs,
        "description": "Get recent log entries from application logs.",
        "parameters": {
            "n_lines": {"type": "integer", "description": "Number of lines to show (max 100)", "required": False}
        },
    },
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    """Execute a tool by name with the given arguments."""
    if tool_name not in AVAILABLE_TOOLS:
        return ToolResult(success=False, output=f"Unknown tool: {tool_name}")
    
    tool_info = AVAILABLE_TOOLS[tool_name]
    func = tool_info["function"]
    
    try:
        return func(**arguments)
    except TypeError as e:
        return ToolResult(success=False, output=f"Invalid arguments for {tool_name}: {e}")
    except Exception as e:
        return ToolResult(success=False, output=f"Tool execution error: {e}")


def get_tools_for_openai() -> List[Dict[str, Any]]:
    """Get tool definitions in OpenAI function calling format."""
    tools = []
    for name, info in AVAILABLE_TOOLS.items():
        properties = {}
        required = []
        for param_name, param_info in info["parameters"].items():
            properties[param_name] = {
                "type": param_info["type"],
                "description": param_info["description"],
            }
            if param_info.get("required", False):
                required.append(param_name)
        
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools
