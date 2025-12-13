"""
Chat tools for the Telegram Code Review Agent.

These are the tools available for the LLM to call during natural language chat.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.change_detector import ChangeDetector
from agent.storage import get_last_review, get_meta, get_review_count


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
        important_dirs = ["src", "agent", "scripts", "tests", "data"]
        project_map = []
        
        for d in important_dirs:
            path = Path(d)
            if path.exists() and path.is_dir():
                files = list(path.rglob("*.py"))[:10]
                project_map.append(f"📁 {d}/ ({len(files)} Python files)")
                for f in files[:5]:
                    project_map.append(f"   - {f.relative_to(path)}")
                if len(files) > 5:
                    project_map.append(f"   ... and {len(files) - 5} more")
        
        config_files = ["pyproject.toml", "requirements.txt", ".env.example", "replit.md"]
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
        
        return ToolResult(success=True, output=output)
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
            output=output or "Smoke tests completed with no output.",
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
                            findings.append(f"   {match[:100]}")
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


def search_repo(query: str) -> ToolResult:
    """Search the repository for a string or pattern."""
    try:
        if not query or len(query) < 2:
            return ToolResult(success=False, output="Query must be at least 2 characters.")
        
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", query, "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        output = result.stdout.strip()
        if not output:
            return ToolResult(success=True, output=f"No matches found for '{query}'")
        
        lines = output.split("\n")
        if len(lines) > 20:
            output = "\n".join(lines[:20]) + f"\n... and {len(lines) - 20} more matches"
        
        return ToolResult(
            success=True,
            output=output,
            data={"match_count": len(lines)}
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Search timed out.")
    except Exception as e:
        return ToolResult(success=False, output=f"Error searching: {e}")


def open_file(path: str, start_line: int = 1, end_line: int = 50) -> ToolResult:
    """Read a section of a file."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(success=False, output=f"File not found: {path}")
        
        if not file_path.is_file():
            return ToolResult(success=False, output=f"Not a file: {path}")
        
        disallowed = [".env", "secrets", ".key", ".pem", ".crt"]
        if any(d in path.lower() for d in disallowed):
            return ToolResult(success=False, output="Cannot read sensitive files.")
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line, start_line + 100)
        
        selected = lines[start_line - 1:end_line]
        content = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
        
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated)"
        
        return ToolResult(
            success=True,
            output=content,
            data={"total_lines": len(lines), "shown": f"{start_line}-{end_line}"}
        )
    except Exception as e:
        return ToolResult(success=False, output=f"Error reading file: {e}")


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
                    output_parts.append(result.stdout.strip()[:1500])
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
        "description": "Search the repository for a string or pattern in Python files.",
        "parameters": {
            "query": {"type": "string", "description": "The search query", "required": True}
        },
    },
    "open_file": {
        "function": open_file,
        "description": "Read a section of a file to see its contents.",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file", "required": True},
            "start_line": {"type": "integer", "description": "Starting line number", "required": False},
            "end_line": {"type": "integer", "description": "Ending line number", "required": False},
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
