"""Configuration settings for PR Supervisor."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class SupervisorSettings(BaseSettings):
    """Environment-based configuration for the PR Supervisor."""
    
    enabled: bool = Field(default=False, alias="SUPERVISOR_ENABLED")
    enable_codex: bool = Field(default=True, alias="SUPERVISOR_ENABLE_CODEX")
    base_jobs_dir: str = Field(default="/tmp/pr_supervisor_jobs", alias="SUPERVISOR_BASE_JOBS_DIR")
    max_loops: int = Field(default=3, alias="SUPERVISOR_MAX_LOOPS")
    max_files_changed: int = Field(default=10, alias="SUPERVISOR_MAX_FILES_CHANGED")
    max_loc_changed: int = Field(default=300, alias="SUPERVISOR_MAX_LOC_CHANGED")
    allow_forks: bool = Field(default=False, alias="SUPERVISOR_ALLOW_FORKS")
    
    github_webhook_secret: Optional[str] = Field(default=None, alias="GITHUB_WEBHOOK_SECRET")
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    
    model_optimist: str = Field(default="gpt-4o-mini", alias="MODEL_OPTIMIST")
    model_skeptic: str = Field(default="gpt-4o", alias="MODEL_SKEPTIC")
    model_arbiter: str = Field(default="gpt-4o-mini", alias="MODEL_ARBITER")
    codex_model: str = Field(default="gpt-4o", alias="CODEX_MODEL")
    codex_bin: str = Field(default="codex", alias="CODEX_BIN")
    
    check_cmd_1: str = Field(default="python -m pytest -q", alias="CHECK_CMD_1")
    check_cmd_2: str = Field(default="python -m ruff check .", alias="CHECK_CMD_2")
    check_cmd_3: Optional[str] = Field(default=None, alias="CHECK_CMD_3")
    
    command_timeout: int = Field(default=600, alias="SUPERVISOR_COMMAND_TIMEOUT")
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    def get_check_commands(self) -> list[str]:
        """Return list of configured check commands."""
        cmds = [self.check_cmd_1]
        if self.check_cmd_2:
            cmds.append(self.check_cmd_2)
        if self.check_cmd_3:
            cmds.append(self.check_cmd_3)
        return cmds


def get_settings() -> SupervisorSettings:
    """Get cached settings instance."""
    return SupervisorSettings()
