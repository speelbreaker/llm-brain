"""Tests for Telegram command handlers."""

import pytest

from src.supervisor.commands import SupervisorCommands
from src.supervisor.store import PRApprovalState


class MockSettings:
    """Mock settings for testing."""

    def __init__(self):
        self.enable_codex = True
        self.autofix_policy = "label"
        self.autofix_label = "autofix-ok"
        self.telegram_allowed_user_ids = "123,456"

    def get_allowed_user_ids(self):
        return {123, 456}


class MockStore:
    """Mock store for testing."""

    def __init__(self):
        self._approvals = {}
        self._jobs = []

    def get_pr_approval(self, repo: str, pr_number: int) -> PRApprovalState:
        key = f"{repo}:{pr_number}"
        if key in self._approvals:
            return self._approvals[key]
        return PRApprovalState(repo=repo, pr_number=pr_number)

    def set_pr_approval(
        self, repo: str, pr_number: int, approved: bool, user_id=None
    ) -> PRApprovalState:
        key = f"{repo}:{pr_number}"
        state = PRApprovalState(
            repo=repo,
            pr_number=pr_number,
            approved_by_telegram=approved,
            approved_by_user_id=user_id,
        )
        self._approvals[key] = state
        return state

    def set_pr_paused(
        self, repo: str, pr_number: int, paused: bool, user_id=None
    ) -> PRApprovalState:
        key = f"{repo}:{pr_number}"
        state = PRApprovalState(
            repo=repo,
            pr_number=pr_number,
            paused=paused,
            paused_by_user_id=user_id,
        )
        self._approvals[key] = state
        return state

    def list_recent(self, limit: int = 50):
        return self._jobs[:limit]

    def get_latest_job_for_pr(self, repo: str, pr_number: int):
        return None


class TestSupervisorCommands:
    """Tests for SupervisorCommands class."""

    def test_unauthorized_user_rejected(self):
        """Unauthorized users are rejected."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)

        assert commands.is_user_authorized(123)
        assert commands.is_user_authorized(456)
        assert not commands.is_user_authorized(999)

    @pytest.mark.asyncio
    async def test_autofix_approval(self):
        """Test autofix command approves PR."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        result = await commands.cmd_autofix(pr_number=42, user_id=123)

        assert result.success
        assert "approved" in result.message.lower()

        approval = store.get_pr_approval("owner/repo", 42)
        assert approval.approved_by_telegram

    @pytest.mark.asyncio
    async def test_autofix_unauthorized(self):
        """Unauthorized user cannot approve autofix."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        result = await commands.cmd_autofix(pr_number=42, user_id=999)

        assert not result.success
        assert "unauthorized" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pause_command(self):
        """Test pause command pauses PR."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        result = await commands.cmd_pause(pr_number=42, user_id=123)

        assert result.success
        assert "paused" in result.message.lower()

        approval = store.get_pr_approval("owner/repo", 42)
        assert approval.paused

    @pytest.mark.asyncio
    async def test_resume_command(self):
        """Test resume command unpauses PR."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        await commands.cmd_pause(pr_number=42, user_id=123)
        result = await commands.cmd_resume(pr_number=42, user_id=123)

        assert result.success
        assert "resumed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_revoke_command(self):
        """Test revoke command removes approval."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        await commands.cmd_autofix(pr_number=42, user_id=123)
        result = await commands.cmd_revoke(pr_number=42, user_id=123)

        assert result.success
        assert "revoked" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rerun_blocked_when_paused(self):
        """Rerun is blocked when PR is paused."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)
        commands.set_default_repo("owner/repo")

        await commands.cmd_pause(pr_number=42, user_id=123)
        result = await commands.cmd_rerun(pr_number=42, user_id=123)

        assert not result.success
        assert "paused" in result.message.lower()

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test help command returns help text."""
        settings = MockSettings()
        store = MockStore()
        commands = SupervisorCommands(settings, store)

        result = await commands._cmd_help([])

        assert result.success
        assert "/supervisor" in result.message
        assert "/autofix" in result.message
