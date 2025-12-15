"""Telegram notification helpers for PR Supervisor."""

from typing import Optional

import httpx

from .config import SupervisorSettings
from .models import ArbiterDecision, SupervisorJob


class TelegramNotifier:
    """Sends status updates to Telegram."""
    
    def __init__(self, settings: SupervisorSettings):
        self.settings = settings
        self.enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self.base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.settings.telegram_chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    }
                )
                return response.status_code == 200
        except Exception:
            return False
    
    async def notify_job_start(self, job: SupervisorJob) -> None:
        """Notify about job start."""
        text = (
            f"🔄 <b>Supervisor Job Started</b>\n"
            f"PR #{job.pr_number} on <code>{job.repo_full_name}</code>\n"
            f"Branch: <code>{job.head_ref}</code>\n"
            f"Commit: <code>{job.head_sha[:8]}</code>\n"
            f"<a href=\"{job.pr_url}\">View PR</a>"
        )
        await self.send_message(text)
    
    async def notify_checks_result(
        self,
        job: SupervisorJob,
        passed: bool,
        summary: str = "",
    ) -> None:
        """Notify about check results."""
        status = "✅ Passed" if passed else "❌ Failed"
        text = (
            f"<b>Checks {status}</b>\n"
            f"PR #{job.pr_number} | <code>{job.head_sha[:8]}</code>\n"
        )
        if summary and not passed:
            text += f"\n{summary[:200]}"
        
        await self.send_message(text)
    
    async def notify_arbiter_decision(
        self,
        job: SupervisorJob,
        decision: ArbiterDecision,
    ) -> None:
        """Notify about Arbiter decision."""
        status = "🟢 Auto-fix approved" if decision.auto_fix_allowed else "🔴 Auto-fix denied"
        text = (
            f"<b>Arbiter Decision</b>\n"
            f"PR #{job.pr_number} | {status}\n"
            f"Risk: {decision.risk_level}\n"
        )
        
        if decision.stop_reason:
            text += f"Reason: {decision.stop_reason[:100]}\n"
        
        if decision.fix_objectives:
            text += "\nObjectives:\n"
            for obj in decision.fix_objectives[:3]:
                text += f"• {obj[:50]}\n"
        
        await self.send_message(text)
    
    async def notify_fix_started(self, job: SupervisorJob, loop_num: int) -> None:
        """Notify that Codex fix loop started."""
        text = (
            f"🔧 <b>Codex Fix Loop #{loop_num}</b>\n"
            f"PR #{job.pr_number} | <code>{job.head_sha[:8]}</code>"
        )
        await self.send_message(text)
    
    async def notify_fix_pushed(
        self,
        job: SupervisorJob,
        commit_sha: str,
    ) -> None:
        """Notify that a fix was pushed."""
        text = (
            f"📤 <b>Fix Pushed</b>\n"
            f"PR #{job.pr_number}\n"
            f"Commit: <code>{commit_sha[:8]}</code>\n"
            f"<a href=\"{job.pr_url}\">View PR</a>"
        )
        await self.send_message(text)
    
    async def notify_final_result(
        self,
        job: SupervisorJob,
        success: bool,
        message: str = "",
    ) -> None:
        """Notify final job result."""
        if success:
            text = (
                f"✅ <b>Ready to Merge</b>\n"
                f"PR #{job.pr_number} on <code>{job.repo_full_name}</code>\n"
                f"<a href=\"{job.pr_url}\">View PR</a>"
            )
        else:
            text = (
                f"🛑 <b>Needs Human Review</b>\n"
                f"PR #{job.pr_number} on <code>{job.repo_full_name}</code>\n"
            )
            if message:
                text += f"\n{message[:200]}"
            text += f"\n<a href=\"{job.pr_url}\">View PR</a>"
        
        await self.send_message(text)
