"""Configuration settings for PR Supervisor."""

from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class SupervisorSettings(BaseSettings):
    """Environment-based configuration for the PR Supervisor."""
    
    enabled: bool = Field(default=False, alias="SUPERVISOR_ENABLED")
    debug: bool = Field(default=False, alias="SUPERVISOR_DEBUG")
    debug_token: Optional[str] = Field(default=None, alias="SUPERVISOR_DEBUG_TOKEN")
    
    enable_codex: bool = Field(default=False, alias="SUPERVISOR_ENABLE_CODEX")
    autofix_policy: str = Field(default="label", alias="SUPERVISOR_AUTOFIX_POLICY")
    autofix_label: str = Field(default="autofix-ok", alias="SUPERVISOR_AUTOFIX_LABEL")
    require_human_for_high_risk: bool = Field(default=True, alias="SUPERVISOR_REQUIRE_HUMAN_FOR_HIGH_RISK")
    base_jobs_dir: str = Field(default="/tmp/pr_supervisor_jobs", alias="SUPERVISOR_BASE_JOBS_DIR")
    max_loops: int = Field(default=3, alias="SUPERVISOR_MAX_LOOPS")
    max_files_changed: int = Field(default=10, alias="SUPERVISOR_MAX_FILES_CHANGED")
    max_loc_changed: int = Field(default=300, alias="SUPERVISOR_MAX_LOC_CHANGED")
    allow_forks: bool = Field(default=False, alias="SUPERVISOR_ALLOW_FORKS")
    
    github_webhook_secret: Optional[str] = Field(default=None, alias="GITHUB_WEBHOOK_SECRET")
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    
    telegram_enabled: bool = Field(default=False, alias="SUPERVISOR_TELEGRAM_ENABLED")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    telegram_admin_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_ADMIN_CHAT_ID")
    telegram_allowed_user_ids: str = Field(default="", alias="TELEGRAM_ALLOWED_USER_IDS")
    telegram_status_mode: str = Field(default="card", alias="TELEGRAM_STATUS_MODE")
    telegram_max_chars: int = Field(default=3500, alias="TELEGRAM_MAX_CHARS")
    telegram_debounce_seconds: int = Field(default=3, alias="TELEGRAM_DEBOUNCE_SECONDS")
    
    workspace_ttl_hours: int = Field(default=24, alias="SUPERVISOR_WORKSPACE_TTL_HOURS")
    store_path: Optional[str] = Field(default=None, alias="SUPERVISOR_STORE_PATH")
    
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_BASE_URL"
    )
    
    optimist_provider: str = Field(default="openai", alias="OPTIMIST_PROVIDER")
    skeptic_provider: str = Field(default="openai", alias="SKEPTIC_PROVIDER")
    arbiter_provider: str = Field(default="openai", alias="ARBITER_PROVIDER")
    
    model_optimist: str = Field(default="gpt-4o-mini", alias="MODEL_OPTIMIST")
    model_skeptic: str = Field(default="gpt-4o", alias="MODEL_SKEPTIC")
    model_arbiter: str = Field(default="gpt-4o-mini", alias="MODEL_ARBITER")
    codex_model: str = Field(default="gpt-4o", alias="CODEX_MODEL")
    codex_bin: str = Field(default="codex", alias="CODEX_BIN")
    
    check_cmd_1: str = Field(default="python -m pytest -q", alias="CHECK_CMD_1")
    check_cmd_2: str = Field(default="python -m ruff check .", alias="CHECK_CMD_2")
    check_cmd_3: Optional[str] = Field(default=None, alias="CHECK_CMD_3")
    
    command_timeout: int = Field(default=600, alias="SUPERVISOR_COMMAND_TIMEOUT")
    
    supervisor_api_url: Optional[str] = Field(default=None, alias="SUPERVISOR_API_URL")
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    def get_check_commands(self) -> list[str]:
        """Return list of configured check commands."""
        cmds = [self.check_cmd_1]
        if self.check_cmd_2:
            cmds.append(self.check_cmd_2)
        if self.check_cmd_3:
            cmds.append(self.check_cmd_3)
        return cmds
    
    def get_allowed_user_ids(self) -> set[int]:
        """Parse TELEGRAM_ALLOWED_USER_IDS into a set of integers."""
        if not self.telegram_allowed_user_ids:
            return set()
        ids = set()
        for part in self.telegram_allowed_user_ids.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        return ids
    
    def is_autofix_policy_valid(self) -> bool:
        """Check if autofix policy is valid."""
        return self.autofix_policy in ("label", "telegram", "both")
    
    def get_store_path(self) -> str:
        """Get the store file path."""
        if self.store_path:
            return self.store_path
        return f"{self.base_jobs_dir}/job_history.jsonl"


def get_settings() -> SupervisorSettings:
    """Get cached settings instance."""
    return SupervisorSettings()
