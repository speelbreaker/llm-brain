"""
Configuration module using Pydantic BaseSettings.
Reads from environment variables and .env file.
"""
from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Main configuration for the options trading agent.
    All values can be overridden via environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deribit_base_url: str = Field(
        default="https://test.deribit.com",
        description="Deribit API base URL (testnet by default)",
    )
    deribit_client_id: str = Field(
        default="",
        description="Deribit API client ID",
    )
    deribit_client_secret: str = Field(
        default="",
        description="Deribit API client secret",
    )

    max_margin_used_pct: float = Field(
        default=80.0,
        description="Maximum margin usage percentage allowed",
    )
    max_net_delta_abs: float = Field(
        default=5.0,
        description="Maximum absolute net delta exposure",
    )
    max_expiry_exposure: float = Field(
        default=10.0,
        description="Maximum exposure per expiry (in contracts)",
    )

    ivrv_min: float = Field(
        default=1.0,
        description="Minimum IV/RV ratio to consider selling options",
    )
    delta_min: float = Field(
        default=0.10,
        description="Minimum delta for candidate options",
    )
    delta_max: float = Field(
        default=0.35,
        description="Maximum delta for candidate options",
    )
    dte_min: int = Field(
        default=1,
        description="Minimum days to expiry for candidate options",
    )
    dte_max: int = Field(
        default=14,
        description="Maximum days to expiry for candidate options",
    )
    premium_min_usd: float = Field(
        default=50.0,
        description="Minimum premium in USD for candidate options",
    )

    default_order_size: float = Field(
        default=0.1,
        description="Default order size in BTC/ETH",
    )

    underlyings: list[str] = Field(
        default=["BTC", "ETH"],
        description="List of underlyings to trade",
    )

    llm_enabled: bool = Field(
        default=False,
        description="Enable LLM-based decision making",
    )
    llm_model_name: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model name for LLM decisions",
    )
    llm_chat_model_name: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model name for chat_with_agent (can be same or larger)",
    )
    llm_max_decision_tokens: int = Field(
        default=1024,
        description="Maximum tokens for LLM decision output",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for LLM API calls in seconds",
    )

    dry_run: bool = Field(
        default=True,
        description="If True, no real orders are placed (simulation mode)",
    )
    loop_interval_sec: int = Field(
        default=300,
        description="Sleep interval between agent loop iterations (seconds)",
    )

    log_dir: str = Field(
        default="logs",
        description="Directory for JSON decision logs",
    )


settings = Settings()
