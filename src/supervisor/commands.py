"""Telegram command handlers for PR Supervisor."""

from dataclasses import dataclass
from typing import Optional
import logging

from .config import SupervisorSettings
from .store import JobStore

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    message: str
    data: Optional[dict] = None


class SupervisorCommands:
    """Telegram command handlers for supervisor operations."""

    def __init__(self, settings: SupervisorSettings, store: JobStore):
        self.settings = settings
        self.store = store
        self._repo: Optional[str] = None

    def set_default_repo(self, repo: str) -> None:
        """Set default repository for commands."""
        self._repo = repo

    def is_user_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use supervisor commands."""
        allowed = self.settings.get_allowed_user_ids()
        if not allowed:
            return True
        return user_id in allowed

    async def handle_command(
        self,
        command: str,
        args: list[str],
        user_id: int,
    ) -> CommandResult:
        """Route and execute a supervisor command."""
        if not self.is_user_authorized(user_id):
            return CommandResult(
                success=False,
                message="Unauthorized. Your user ID is not in TELEGRAM_ALLOWED_USER_IDS.",
            )

        cmd_map = {
            "last": self._cmd_last,
            "pr": self._cmd_pr,
            "help": self._cmd_help,
        }

        handler = cmd_map.get(command)
        if handler:
            return await handler(args)

        return CommandResult(
            success=False, message=f"Unknown supervisor command: {command}"
        )

    async def cmd_autofix(
        self, pr_number: int, user_id: int, repo: Optional[str] = None
    ) -> CommandResult:
        """Approve autofix for a PR."""
        if not self.is_user_authorized(user_id):
            return CommandResult(success=False, message="Unauthorized")

        repo = repo or self._repo
        if not repo:
            return CommandResult(success=False, message="No repository specified")

        approval = self.store.set_pr_approval(
            repo, pr_number, approved=True, user_id=user_id
        )

        policy_msg = ""
        if self.settings.autofix_policy == "both":
            policy_msg = f"\n(Also requires label '{self.settings.autofix_label}')"
        elif self.settings.autofix_policy == "label":
            policy_msg = (
                "\n(Note: Policy is 'label', Telegram approval stored but not required)"
            )

        return CommandResult(
            success=True,
            message=f"Autofix approved for PR #{pr_number}{policy_msg}",
            data={"approval": approval.__dict__},
        )

    async def cmd_revoke(
        self, pr_number: int, user_id: int, repo: Optional[str] = None
    ) -> CommandResult:
        """Revoke autofix approval for a PR."""
        if not self.is_user_authorized(user_id):
            return CommandResult(success=False, message="Unauthorized")

        repo = repo or self._repo
        if not repo:
            return CommandResult(success=False, message="No repository specified")

        approval = self.store.set_pr_approval(repo, pr_number, approved=False)

        return CommandResult(
            success=True,
            message=f"Autofix approval revoked for PR #{pr_number}",
            data={"approval": approval.__dict__},
        )

    async def cmd_pause(
        self, pr_number: int, user_id: int, repo: Optional[str] = None
    ) -> CommandResult:
        """Pause supervisor for a PR."""
        if not self.is_user_authorized(user_id):
            return CommandResult(success=False, message="Unauthorized")

        repo = repo or self._repo
        if not repo:
            return CommandResult(success=False, message="No repository specified")

        approval = self.store.set_pr_paused(
            repo, pr_number, paused=True, user_id=user_id
        )

        return CommandResult(
            success=True,
            message=f"Supervisor paused for PR #{pr_number}. No new jobs will be queued.",
            data={"approval": approval.__dict__},
        )

    async def cmd_resume(
        self, pr_number: int, user_id: int, repo: Optional[str] = None
    ) -> CommandResult:
        """Resume supervisor for a PR."""
        if not self.is_user_authorized(user_id):
            return CommandResult(success=False, message="Unauthorized")

        repo = repo or self._repo
        if not repo:
            return CommandResult(success=False, message="No repository specified")

        approval = self.store.set_pr_paused(repo, pr_number, paused=False)

        return CommandResult(
            success=True,
            message=f"Supervisor resumed for PR #{pr_number}",
            data={"approval": approval.__dict__},
        )

    async def cmd_rerun(
        self, pr_number: int, user_id: int, repo: Optional[str] = None
    ) -> CommandResult:
        """Request a rerun for a PR (returns info, actual queueing done by caller)."""
        if not self.is_user_authorized(user_id):
            return CommandResult(success=False, message="Unauthorized")

        repo = repo or self._repo
        if not repo:
            return CommandResult(success=False, message="No repository specified")

        approval = self.store.get_pr_approval(repo, pr_number)
        if approval.paused:
            return CommandResult(
                success=False,
                message=f"Cannot rerun: PR #{pr_number} is paused. Use /resume first.",
            )

        return CommandResult(
            success=True,
            message=f"Rerun requested for PR #{pr_number}",
            data={"repo": repo, "pr_number": pr_number},
        )

    async def _cmd_last(self, args: list[str]) -> CommandResult:
        """Get last N jobs summary."""
        limit = 5
        if args and args[0].isdigit():
            limit = min(int(args[0]), 20)

        jobs = self.store.list_recent(limit)

        if not jobs:
            return CommandResult(success=True, message="No jobs found")

        lines = ["📋 *Recent Jobs*\n"]
        for job in jobs:
            status_emoji = {
                "pending": "⏳",
                "running": "🔄",
                "checks_passed": "✅",
                "checks_failed": "❌",
                "debating": "🗣",
                "fixing": "🔧",
                "fixed": "✅",
                "needs_human": "👤",
                "error": "❗",
            }.get(job.status, "❓")

            lines.append(
                f"{status_emoji} PR #{job.pr_number} | {job.status} | {job.job_id[:12]}"
            )

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data={"jobs": [j.model_dump() for j in jobs]},
        )

    async def _cmd_pr(self, args: list[str]) -> CommandResult:
        """Get status for a specific PR."""
        if not args or not args[0].isdigit():
            return CommandResult(
                success=False, message="Usage: /supervisor pr <number>"
            )

        pr_number = int(args[0])
        repo = self._repo

        if not repo:
            return CommandResult(
                success=False, message="No default repository configured"
            )

        approval = self.store.get_pr_approval(repo, pr_number)
        job = self.store.get_latest_job_for_pr(repo, pr_number)

        lines = [f"📋 *PR #{pr_number} Status*\n"]

        if approval.paused:
            lines.append("⏸ *PAUSED*")

        lines.append(
            f"Autofix approved: {'✅' if approval.approved_by_telegram else '❌'}"
        )

        if job:
            lines.append(f"Last job: {job.status}")
            lines.append(f"SHA: {job.head_sha[:8]}")
            lines.append(f"ID: {job.job_id[:16]}")
        else:
            lines.append("No jobs found for this PR")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data={
                "approval": approval.__dict__,
                "job": job.model_dump() if job else None,
            },
        )

    async def _cmd_help(self, args: list[str]) -> CommandResult:
        """Show help message."""
        help_text = """*Supervisor Commands*

/supervisor last [N] - Show last N jobs (default 5)
/supervisor pr <number> - Show PR status
/rerun <pr_number> - Queue rerun for PR
/autofix <pr_number> - Approve autofix
/pause <pr_number> - Pause PR processing
/resume <pr_number> - Resume PR processing
/revoke <pr_number> - Revoke autofix approval
/help - Show this help

*Current Policy*
Autofix policy: {policy}
Codex enabled: {codex}
""".format(
            policy=self.settings.autofix_policy,
            codex="Yes" if self.settings.enable_codex else "No",
        )

        return CommandResult(success=True, message=help_text)
