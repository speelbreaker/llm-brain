"""Enhanced Telegram notification with status card UX for PR Supervisor.

Implements:
- Status card UX (single updateable message per PR)
- HTML escaping for dynamic content
- Plaintext fallback on formatting errors
- Retry with exponential backoff for transient failures
- Proper error handling
"""

import asyncio
import html
import logging
from datetime import datetime
from typing import Optional

import httpx

from .config import SupervisorSettings
from .models import ArbiterDecision, CheckResult, DiffStats, JobStatus, SupervisorJob
from .retry import with_retry

logger = logging.getLogger(__name__)

TELEGRAM_TIMEOUT = 20.0


def safe_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Safely truncate text to max_chars, adding suffix if truncated."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    if not text:
        return ""
    return html.escape(text, quote=True)


def strip_html_tags(text: str) -> str:
    """Strip HTML tags for plaintext fallback."""
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    return clean


class TelegramStatusCard:
    """Manages a single status card message per PR."""
    
    def __init__(
        self,
        settings: SupervisorSettings,
        repo: str,
        pr_number: int,
        message_registry: "MessageRegistry",
    ):
        self.settings = settings
        self.repo = repo
        self.pr_number = pr_number
        self.registry = message_registry
        self.base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        self.enabled = settings.telegram_enabled and bool(
            settings.telegram_bot_token and settings.telegram_chat_id
        )
        self._last_update: float = 0
        
        self.pr_title: str = ""
        self.pr_url: str = ""
        self.current_phase: str = "STARTING"
        self.commit_sha: str = ""
        self.checks: list[dict] = []
        self.arbiter_verdict: Optional[str] = None
        self.loop_info: str = ""
        self.last_error: Optional[str] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client for Telegram API with explicit timeout."""
        return httpx.AsyncClient(timeout=httpx.Timeout(TELEGRAM_TIMEOUT))
    
    def _build_card_text(self, use_html: bool = True) -> str:
        """Build the status card message text.
        
        Args:
            use_html: If True, use HTML formatting. If False, use plaintext.
        """
        lines = []
        
        title_escaped = escape_html(safe_truncate(self.pr_title, 50))
        if use_html:
            lines.append(f"<b>PR #{self.pr_number}</b> — {title_escaped}")
            if self.pr_url:
                lines.append(f'<a href="{escape_html(self.pr_url)}">View PR</a>')
        else:
            lines.append(f"PR #{self.pr_number} — {safe_truncate(self.pr_title, 50)}")
            if self.pr_url:
                lines.append(f"Link: {self.pr_url}")
        lines.append("")
        
        phase_emoji = {
            "STARTING": "🔄",
            "CHECKS": "🔍",
            "DEBATE": "💬",
            "CODEX_FIX": "🔧",
            "DONE": "✅",
            "ERROR": "❌",
            "NEEDS_HUMAN": "🛑",
        }.get(self.current_phase, "⏳")
        
        phase_text = escape_html(self.current_phase) if use_html else self.current_phase
        loop_text = escape_html(self.loop_info) if use_html else self.loop_info
        if self.loop_info:
            phase_text += f" {loop_text}"
        
        if use_html:
            lines.append(f"<b>Phase:</b> {phase_emoji} {phase_text}")
        else:
            lines.append(f"Phase: {phase_emoji} {phase_text}")
        
        if self.checks:
            lines.append("")
            lines.append("<b>Checks:</b>" if use_html else "Checks:")
            for check in self.checks:
                cmd = check.get("command", "unknown")
                cmd_short = cmd.split()[0].split("/")[-1] if cmd else "?"
                passed = check.get("passed", False)
                emoji = "✅" if passed else "❌"
                cmd_escaped = escape_html(cmd_short) if use_html else cmd_short
                lines.append(f"  {emoji} {cmd_escaped}")
        
        if self.arbiter_verdict:
            lines.append("")
            verdict_text = escape_html(self.arbiter_verdict) if use_html else self.arbiter_verdict
            if use_html:
                lines.append(f"<b>Arbiter:</b> {verdict_text}")
            else:
                lines.append(f"Arbiter: {verdict_text}")
        
        if self.commit_sha:
            lines.append("")
            if use_html:
                lines.append(f"<b>Commit:</b> <code>{escape_html(self.commit_sha[:8])}</code>")
            else:
                lines.append(f"Commit: {self.commit_sha[:8]}")
        
        lines.append("")
        timestamp = datetime.utcnow().strftime('%H:%M:%S')
        if use_html:
            lines.append(f"<i>Updated: {timestamp} UTC</i>")
        else:
            lines.append(f"Updated: {timestamp} UTC")
        
        return safe_truncate("\n".join(lines), self.settings.telegram_max_chars)
    
    async def _should_debounce(self) -> bool:
        """Check if we should debounce this update."""
        now = asyncio.get_event_loop().time()
        if now - self._last_update < self.settings.telegram_debounce_seconds:
            return True
        self._last_update = now
        return False
    
    async def update_card(self) -> bool:
        """Create or update the status card message."""
        if not self.enabled:
            return False
        
        if await self._should_debounce():
            return True
        
        key = (self.repo, self.pr_number)
        existing_msg_id = self.registry.get_message_id(key)
        
        async with await self._get_client() as client:
            if existing_msg_id:
                return await self._edit_message(client, existing_msg_id)
            else:
                return await self._send_new_message(client, key)
    
    async def _send_new_message(
        self,
        client: httpx.AsyncClient,
        key: tuple,
    ) -> bool:
        """Send a new status card message with fallback to plaintext."""
        return await self._send_with_fallback(
            client,
            self._build_card_text(use_html=True),
            self._build_card_text(use_html=False),
            key=key,
        )
    
    async def _send_with_fallback(
        self,
        client: httpx.AsyncClient,
        html_text: str,
        plain_text: str,
        key: Optional[tuple] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """Send message with HTML, falling back to plaintext on error.
        
        Uses retry with exponential backoff for transient failures.
        """
        for parse_mode, text in [("HTML", html_text), (None, plain_text)]:
            try:
                result = await self._send_message_with_retry(
                    client, text, parse_mode, key, reply_to_message_id
                )
                if result:
                    return True
                
                if parse_mode == "HTML":
                    logger.warning("HTML send failed, trying plaintext")
                    continue
                    
                return False
                
            except Exception as e:
                if parse_mode == "HTML":
                    logger.warning("HTML send error, trying plaintext: %s", type(e).__name__)
                    continue
                logger.error("Telegram send failed: %s", type(e).__name__)
                return False
        
        return False
    
    async def _send_message_with_retry(
        self,
        client: httpx.AsyncClient,
        text: str,
        parse_mode: Optional[str],
        key: Optional[tuple],
        reply_to_message_id: Optional[int],
    ) -> bool:
        """Send a single message with retry logic."""
        async def do_send() -> bool:
            payload = {
                "chat_id": self.settings.telegram_chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            
            response = await client.post(
                f"{self.base_url}/sendMessage",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("ok") and data.get("result", {}).get("message_id"):
                msg_id = data["result"]["message_id"]
                if key:
                    self.registry.set_message_id(key, msg_id)
                return True
            return False
        
        try:
            return await with_retry(
                do_send,
                operation_name="telegram_send",
                max_retries=3,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return False
            raise
    
    async def _edit_message(
        self,
        client: httpx.AsyncClient,
        message_id: int,
    ) -> bool:
        """Edit an existing status card message with fallback to plaintext.
        
        Uses retry with exponential backoff for transient failures.
        """
        for parse_mode, use_html in [("HTML", True), (None, False)]:
            try:
                result = await self._edit_message_with_retry(
                    client, message_id, use_html, parse_mode
                )
                if result:
                    return True
                
                if parse_mode == "HTML":
                    logger.warning("HTML edit failed, trying plaintext")
                    continue
                
                return False
                
            except Exception as e:
                if parse_mode == "HTML":
                    logger.warning("HTML edit error, trying plaintext: %s", type(e).__name__)
                    continue
                return False
        
        return False
    
    async def _edit_message_with_retry(
        self,
        client: httpx.AsyncClient,
        message_id: int,
        use_html: bool,
        parse_mode: Optional[str],
    ) -> bool:
        """Edit a single message with retry logic."""
        async def do_edit() -> bool:
            text = self._build_card_text(use_html=use_html)
            payload = {
                "chat_id": self.settings.telegram_chat_id,
                "message_id": message_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            response = await client.post(
                f"{self.base_url}/editMessageText",
                json=payload,
            )
            
            if response.status_code == 200:
                return True
            
            if response.status_code == 400:
                data = response.json()
                if "message is not modified" in data.get("description", ""):
                    return True
            
            response.raise_for_status()
            return False
        
        try:
            return await with_retry(
                do_edit,
                operation_name="telegram_edit",
                max_retries=3,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return False
            raise
    
    async def send_detail_reply(self, text: str) -> bool:
        """Send a detail reply to the status card (for failures, arbiter, etc)."""
        if not self.enabled:
            return False
        
        key = (self.repo, self.pr_number)
        parent_msg_id = self.registry.get_message_id(key)
        
        truncated_text = safe_truncate(text, self.settings.telegram_max_chars)
        plain_text = strip_html_tags(truncated_text)
        
        async with await self._get_client() as client:
            return await self._send_with_fallback(
                client,
                truncated_text,
                plain_text,
                reply_to_message_id=parent_msg_id,
            )


class MessageRegistry:
    """Registry for tracking Telegram message IDs per PR."""
    
    def __init__(self):
        self._messages: dict[tuple[str, int], int] = {}
    
    def get_message_id(self, key: tuple[str, int]) -> Optional[int]:
        """Get stored message ID for a PR."""
        return self._messages.get(key)
    
    def set_message_id(self, key: tuple[str, int], message_id: int) -> None:
        """Store message ID for a PR."""
        self._messages[key] = message_id
    
    def clear(self, key: tuple[str, int]) -> None:
        """Clear message ID for a PR."""
        self._messages.pop(key, None)
    
    def export(self) -> dict:
        """Export registry for persistence."""
        return {f"{repo}:{pr}": msg_id for (repo, pr), msg_id in self._messages.items()}
    
    def load(self, data: dict) -> None:
        """Load registry from persisted data."""
        for key_str, msg_id in data.items():
            try:
                repo, pr_str = key_str.rsplit(":", 1)
                self._messages[(repo, int(pr_str))] = msg_id
            except (ValueError, TypeError):
                pass


class TelegramNotifier:
    """Enhanced Telegram notifier with status card support."""
    
    def __init__(self, settings: SupervisorSettings, registry: Optional[MessageRegistry] = None):
        self.settings = settings
        self.registry = registry or MessageRegistry()
        self.enabled = settings.telegram_enabled and bool(
            settings.telegram_bot_token and settings.telegram_chat_id
        )
        self._cards: dict[tuple[str, int], TelegramStatusCard] = {}
    
    def get_card(self, job: SupervisorJob) -> TelegramStatusCard:
        """Get or create a status card for a job."""
        key = (job.repo_full_name, job.pr_number)
        if key not in self._cards:
            self._cards[key] = TelegramStatusCard(
                settings=self.settings,
                repo=job.repo_full_name,
                pr_number=job.pr_number,
                message_registry=self.registry,
            )
        return self._cards[key]
    
    async def notify_job_start(self, job: SupervisorJob, pr_title: str = "") -> None:
        """Create initial status card for job."""
        card = self.get_card(job)
        card.pr_title = pr_title or f"Branch: {job.head_ref}"
        card.pr_url = job.pr_url
        card.commit_sha = job.head_sha
        card.current_phase = "STARTING"
        await card.update_card()
    
    async def notify_checks_started(self, job: SupervisorJob) -> None:
        """Update card for checks phase."""
        card = self.get_card(job)
        card.current_phase = "CHECKS"
        await card.update_card()
    
    async def notify_checks_result(
        self,
        job: SupervisorJob,
        passed: bool,
        checks: Optional[list[CheckResult]] = None,
        failure_excerpt: str = "",
    ) -> None:
        """Update card with check results."""
        card = self.get_card(job)
        
        if checks:
            card.checks = [{"command": c.command, "passed": c.passed} for c in checks]
        
        if passed:
            card.current_phase = "DONE"
        else:
            card.current_phase = "CHECKS"
        
        await card.update_card()
        
        if not passed and failure_excerpt:
            last_lines = "\n".join(failure_excerpt.strip().split("\n")[-30:])
            escaped_excerpt = escape_html(safe_truncate(last_lines, 2500))
            detail = f"<b>Check Failure Excerpt:</b>\n<pre>{escaped_excerpt}</pre>"
            await card.send_detail_reply(detail)
    
    async def notify_debate_started(self, job: SupervisorJob) -> None:
        """Update card for debate phase."""
        card = self.get_card(job)
        card.current_phase = "DEBATE"
        await card.update_card()
    
    async def notify_arbiter_decision(
        self,
        job: SupervisorJob,
        decision: ArbiterDecision,
    ) -> None:
        """Update card with arbiter decision and send detail reply."""
        card = self.get_card(job)
        
        if decision.auto_fix_allowed:
            card.arbiter_verdict = "🟢 Auto-fix approved"
        else:
            card.arbiter_verdict = f"🔴 Denied: {safe_truncate(decision.stop_reason or 'N/A', 40)}"
        
        await card.update_card()
        
        detail_lines = [
            "<b>Arbiter Decision:</b>",
            f"• Auto-fix: {'Yes' if decision.auto_fix_allowed else 'No'}",
            f"• Risk: {escape_html(decision.risk_level or 'unknown')}",
        ]
        if decision.fix_objectives:
            objectives_str = ", ".join(decision.fix_objectives[:3])
            detail_lines.append(f"• Objectives: {escape_html(safe_truncate(objectives_str, 100))}")
        
        await card.send_detail_reply("\n".join(detail_lines))
    
    async def notify_fix_started(self, job: SupervisorJob, loop_num: int, max_loops: int) -> None:
        """Update card for fix phase."""
        card = self.get_card(job)
        card.current_phase = "CODEX_FIX"
        card.loop_info = f"(loop {loop_num}/{max_loops})"
        await card.update_card()
    
    async def notify_fix_pushed(
        self,
        job: SupervisorJob,
        commit_sha: str,
        diff_stats: Optional[DiffStats] = None,
        top_files: Optional[list[str]] = None,
    ) -> None:
        """Update card and send diff summary."""
        card = self.get_card(job)
        card.commit_sha = commit_sha
        card.current_phase = "DONE"
        card.loop_info = ""
        await card.update_card()
        
        if diff_stats:
            detail_lines = [
                "<b>Fix Pushed:</b>",
                f"Commit: <code>{escape_html(commit_sha[:8])}</code>",
                f"Files: {diff_stats.files_changed} | +{diff_stats.lines_added}/-{diff_stats.lines_removed}",
            ]
            if top_files:
                detail_lines.append("Top files:")
                for f in top_files[:3]:
                    detail_lines.append(f"  • {escape_html(safe_truncate(f, 40))}")
            await card.send_detail_reply("\n".join(detail_lines))
    
    async def notify_final_result(
        self,
        job: SupervisorJob,
        success: bool,
        message: str = "",
    ) -> None:
        """Update card with final result."""
        card = self.get_card(job)
        card.current_phase = "DONE" if success else "NEEDS_HUMAN"
        card.loop_info = ""
        
        if not success and message:
            card.last_error = message
        
        await card.update_card()
    
    async def notify_error(self, job: SupervisorJob, error: str) -> None:
        """Update card with error."""
        card = self.get_card(job)
        card.current_phase = "ERROR"
        card.last_error = error
        await card.update_card()
        
        escaped_error = escape_html(safe_truncate(error, 500))
        await card.send_detail_reply(
            f"<b>Error:</b>\n<pre>{escaped_error}</pre>"
        )
