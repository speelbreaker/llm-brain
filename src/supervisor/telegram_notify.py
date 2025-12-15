"""Enhanced Telegram notification with status card UX for PR Supervisor."""

import asyncio
from datetime import datetime
from typing import Optional

import httpx

from .config import SupervisorSettings
from .models import ArbiterDecision, CheckResult, DiffStats, JobStatus, SupervisorJob


def safe_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Safely truncate text to max_chars, adding suffix if truncated."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


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
        """Get HTTP client for Telegram API."""
        return httpx.AsyncClient(timeout=10.0)
    
    def _build_card_text(self) -> str:
        """Build the status card message text."""
        lines = []
        
        lines.append(f"<b>PR #{self.pr_number}</b> — {safe_truncate(self.pr_title, 50)}")
        if self.pr_url:
            lines.append(f"<a href=\"{self.pr_url}\">View PR</a>")
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
        
        phase_text = self.current_phase
        if self.loop_info:
            phase_text += f" {self.loop_info}"
        lines.append(f"<b>Phase:</b> {phase_emoji} {phase_text}")
        
        if self.checks:
            lines.append("")
            lines.append("<b>Checks:</b>")
            for check in self.checks:
                cmd = check.get("command", "unknown")
                cmd_short = cmd.split()[0].split("/")[-1] if cmd else "?"
                passed = check.get("passed", False)
                emoji = "✅" if passed else "❌"
                lines.append(f"  {emoji} {cmd_short}")
        
        if self.arbiter_verdict:
            lines.append("")
            lines.append(f"<b>Arbiter:</b> {self.arbiter_verdict}")
        
        if self.commit_sha:
            lines.append("")
            lines.append(f"<b>Commit:</b> <code>{self.commit_sha[:8]}</code>")
        
        lines.append("")
        lines.append(f"<i>Updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC</i>")
        
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
        
        text = self._build_card_text()
        key = (self.repo, self.pr_number)
        existing_msg_id = self.registry.get_message_id(key)
        
        async with await self._get_client() as client:
            if existing_msg_id:
                return await self._edit_message(client, existing_msg_id, text)
            else:
                return await self._send_new_message(client, text, key)
    
    async def _send_new_message(
        self,
        client: httpx.AsyncClient,
        text: str,
        key: tuple,
    ) -> bool:
        """Send a new status card message."""
        try:
            response = await client.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("ok") and data.get("result", {}).get("message_id"):
                    msg_id = data["result"]["message_id"]
                    self.registry.set_message_id(key, msg_id)
                    return True
            return False
        except Exception:
            return False
    
    async def _edit_message(
        self,
        client: httpx.AsyncClient,
        message_id: int,
        text: str,
    ) -> bool:
        """Edit an existing status card message."""
        try:
            response = await client.post(
                f"{self.base_url}/editMessageText",
                json={
                    "chat_id": self.settings.telegram_chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def send_detail_reply(self, text: str) -> bool:
        """Send a detail reply to the status card (for failures, arbiter, etc)."""
        if not self.enabled:
            return False
        
        key = (self.repo, self.pr_number)
        parent_msg_id = self.registry.get_message_id(key)
        
        truncated_text = safe_truncate(text, self.settings.telegram_max_chars)
        
        async with await self._get_client() as client:
            try:
                payload = {
                    "chat_id": self.settings.telegram_chat_id,
                    "text": truncated_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                
                if parent_msg_id:
                    payload["reply_to_message_id"] = parent_msg_id
                
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                )
                return response.status_code == 200
            except Exception:
                return False


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
            detail = f"<b>Check Failure Excerpt:</b>\n<pre>{safe_truncate(last_lines, 2500)}</pre>"
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
            f"• Risk: {decision.risk_level}",
        ]
        if decision.fix_objectives:
            objectives_str = ", ".join(decision.fix_objectives[:3])
            detail_lines.append(f"• Objectives: {safe_truncate(objectives_str, 100)}")
        
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
                f"Commit: <code>{commit_sha[:8]}</code>",
                f"Files: {diff_stats.files_changed} | +{diff_stats.lines_added}/-{diff_stats.lines_removed}",
            ]
            if top_files:
                detail_lines.append("Top files:")
                for f in top_files[:3]:
                    detail_lines.append(f"  • {safe_truncate(f, 40)}")
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
        
        await card.send_detail_reply(
            f"<b>Error:</b>\n<pre>{safe_truncate(error, 500)}</pre>"
        )
