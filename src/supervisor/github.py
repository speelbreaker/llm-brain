"""GitHub API helpers and webhook signature verification."""

import hashlib
import hmac
import logging
from typing import Any, Optional

import httpx

from .models import WebhookPayload
from .retry import with_retry

logger = logging.getLogger(__name__)

GITHUB_TIMEOUT = 30.0


def verify_signature(payload_body: bytes, signature_header: str, secret: str) -> bool:
    """Verify GitHub webhook signature (X-Hub-Signature-256).
    
    Uses HMAC SHA-256 with constant-time comparison to prevent timing attacks.
    
    Args:
        payload_body: Raw request body bytes
        signature_header: Value of X-Hub-Signature-256 header
        secret: Configured webhook secret (MUST be non-empty)
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not secret or not secret.strip():
        return False
    
    if not signature_header:
        return False
    
    if not signature_header.startswith("sha256="):
        return False
    
    expected_signature = signature_header[7:]
    if len(expected_signature) != 64:
        return False
    
    computed_signature = hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(computed_signature, expected_signature)


def parse_webhook_payload(data: dict[str, Any]) -> Optional[WebhookPayload]:
    """Parse GitHub PR webhook payload."""
    action = data.get("action", "")
    pr = data.get("pull_request", {})
    repo = data.get("repository", {})
    
    if not pr or not repo:
        return None
    
    head = pr.get("head", {})
    base = pr.get("base", {})
    
    is_fork = head.get("repo", {}).get("fork", False)
    if head.get("repo", {}).get("full_name") != repo.get("full_name"):
        is_fork = True
    
    return WebhookPayload(
        action=action,
        repo_full_name=repo.get("full_name", ""),
        pr_number=pr.get("number", 0),
        head_sha=head.get("sha", ""),
        head_ref=head.get("ref", ""),
        base_ref=base.get("ref", ""),
        pr_url=pr.get("html_url", ""),
        is_fork=is_fork,
        sender=data.get("sender", {}).get("login", "unknown"),
    )


class GitHubClient:
    """GitHub API client for PR operations with retry support."""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=httpx.Timeout(GITHUB_TIMEOUT),
            )
        return self._client
    
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        operation_name: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic."""
        async def do_request() -> httpx.Response:
            client = await self._get_client()
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        
        return await with_retry(
            do_request,
            operation_name=operation_name,
            max_retries=3,
        )
    
    async def get_pr_info(self, repo: str, pr_number: int) -> dict[str, Any]:
        """Get PR details with retry."""
        response = await self._request_with_retry(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}",
            operation_name=f"github_get_pr_{repo}_{pr_number}",
        )
        return response.json()
    
    async def get_pr_files(self, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Get list of files changed in PR with retry."""
        response = await self._request_with_retry(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}/files",
            operation_name=f"github_get_files_{repo}_{pr_number}",
        )
        return response.json()
    
    async def post_pr_comment(self, repo: str, pr_number: int, body: str) -> dict[str, Any]:
        """Post a comment on a PR with retry."""
        response = await self._request_with_retry(
            "POST",
            f"/repos/{repo}/issues/{pr_number}/comments",
            operation_name=f"github_post_comment_{repo}_{pr_number}",
            json={"body": body},
        )
        return response.json()
    
    async def update_pr_comment(self, repo: str, comment_id: int, body: str) -> dict[str, Any]:
        """Update an existing PR comment with retry."""
        response = await self._request_with_retry(
            "PATCH",
            f"/repos/{repo}/issues/comments/{comment_id}",
            operation_name=f"github_update_comment_{comment_id}",
            json={"body": body},
        )
        return response.json()
    
    async def get_repo_clone_url(self, repo: str) -> str:
        """Get the clone URL for a repository with retry."""
        response = await self._request_with_retry(
            "GET",
            f"/repos/{repo}",
            operation_name=f"github_get_repo_{repo}",
        )
        data = response.json()
        clone_url = data.get("clone_url", "")
        if self.token and clone_url.startswith("https://"):
            clone_url = clone_url.replace("https://", f"https://x-access-token:{self.token}@")
        return clone_url


def format_pr_comment(
    run_number: int,
    commit_sha: str,
    checks: list[dict[str, Any]],
    failure_summary: str = "",
    fix_started: bool = False,
    arbiter_decision: Optional[dict[str, Any]] = None,
    final_status: Optional[str] = None,
    telegram_enabled: bool = False,
) -> str:
    """Format a concise, readable PR comment with check results."""
    lines = [
        f"## 🤖 Supervisor Run #{run_number}",
        "",
        f"**Commit:** `{commit_sha[:8]}`",
        "",
    ]
    
    all_passed = all(c.get("passed", False) for c in checks)
    if all_passed:
        lines.append("### ✅ All checks passed")
    else:
        lines.append("### Check Results")
        lines.append("")
        for check in checks:
            status = "✅" if check.get("passed") else "❌"
            cmd = check.get("command", "unknown").split()[0].split("/")[-1]
            lines.append(f"- {status} `{cmd}`")
    
    lines.append("")
    
    if failure_summary and not all_passed:
        excerpt = failure_summary.strip().split("\n")[-15:]
        truncated = "\n".join(excerpt)[:800]
        lines.extend([
            "<details>",
            "<summary>Failure excerpt (click to expand)</summary>",
            "",
            "```",
            truncated,
            "```",
            "",
            "</details>",
            "",
        ])
    
    if arbiter_decision:
        allowed = arbiter_decision.get("auto_fix_allowed", False)
        risk = arbiter_decision.get("risk_level", "unknown")
        
        if allowed:
            lines.append(f"**Arbiter:** 🟢 Auto-fix approved (risk: {risk})")
        else:
            reason = arbiter_decision.get("stop_reason", "")[:100]
            lines.append(f"**Arbiter:** 🔴 Auto-fix denied — {reason}")
        lines.append("")
    
    if fix_started:
        lines.append("🔧 **Codex fix in progress...**")
        lines.append("")
    
    if final_status:
        lines.extend([
            "---",
            f"**{final_status}**",
            "",
        ])
    
    if telegram_enabled:
        lines.append("_📱 See Telegram for live timeline_")
    
    return "\n".join(lines)


def format_fallback_comment(
    run_number: int,
    commit_sha: str,
    checks: list[dict[str, Any]],
    failure_summary: str = "",
    llm_error: str = "",
    telegram_enabled: bool = False,
) -> str:
    """Format a PR comment when LLM/OpenAI is unavailable.
    
    This generates a deterministic comment containing check results
    and failure summary without arbiter analysis.
    """
    lines = [
        f"## 🤖 Supervisor Run #{run_number}",
        "",
        f"**Commit:** `{commit_sha[:8]}`",
        "",
    ]
    
    all_passed = all(c.get("passed", False) for c in checks)
    if all_passed:
        lines.append("### ✅ All checks passed")
    else:
        lines.append("### ❌ Check Results")
        lines.append("")
        for check in checks:
            status = "✅" if check.get("passed") else "❌"
            cmd = check.get("command", "unknown").split()[0].split("/")[-1]
            duration = check.get("duration_seconds", 0)
            lines.append(f"- {status} `{cmd}` ({duration:.1f}s)")
    
    lines.append("")
    
    if failure_summary and not all_passed:
        excerpt = failure_summary.strip().split("\n")[-15:]
        truncated = "\n".join(excerpt)[:800]
        lines.extend([
            "<details>",
            "<summary>Failure excerpt (click to expand)</summary>",
            "",
            "```",
            truncated,
            "```",
            "",
            "</details>",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        "⚠️ **OpenAI analysis skipped**",
        "",
    ])
    
    if llm_error:
        lines.append(f"_Reason: {llm_error[:200]}_")
        lines.append("")
    
    lines.append("Manual review required. Auto-fix not attempted.")
    lines.append("")
    
    if telegram_enabled:
        lines.append("_📱 See Telegram for live timeline_")
    
    return "\n".join(lines)
