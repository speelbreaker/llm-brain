"""
Configuration module using Pydantic BaseSettings.
Reads from environment variables and .env file.
"""
from __future__ import annotations

from typing import Literal
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

    mode: Literal["production", "research"] = Field(
        default="research",
        description="Operating mode: 'production' for mainnet, 'research' for testnet exploration",
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
        default=0.3,
        description="Maximum exposure per expiry (in BTC/ETH notionals) for production",
    )

    ivrv_min: float = Field(
        default=1.2,
        description="Minimum IV/RV ratio to consider selling options (production)",
    )
    delta_min: float = Field(
        default=0.20,
        description="Minimum delta for candidate options (production)",
    )
    delta_max: float = Field(
        default=0.30,
        description="Maximum delta for candidate options (production)",
    )
    dte_min: int = Field(
        default=5,
        description="Minimum days to expiry for candidate options (production)",
    )
    dte_max: int = Field(
        default=10,
        description="Maximum days to expiry for candidate options (production)",
    )
    premium_min_usd: float = Field(
        default=50.0,
        description="Minimum premium in USD for candidate options",
    )

    research_ivrv_min: float = Field(
        default=1.0,
        description="Minimum IV/RV ratio in research mode (looser)",
    )
    research_delta_min: float = Field(
        default=0.10,
        description="Minimum delta in research mode (wider range)",
    )
    research_delta_max: float = Field(
        default=0.40,
        description="Maximum delta in research mode (wider range)",
    )
    research_dte_min: int = Field(
        default=1,
        description="Minimum DTE in research mode (allow shorter)",
    )
    research_dte_max: int = Field(
        default=21,
        description="Maximum DTE in research mode (allow longer)",
    )
    research_max_expiry_exposure: float = Field(
        default=1.0,
        description="Maximum exposure per expiry in research mode (higher for testnet)",
    )

    explore_prob: float = Field(
        default=0.25,
        description="Probability of exploring non-best candidate (research mode only)",
    )
    explore_top_k: int = Field(
        default=3,
        description="Number of top candidates to consider for exploration",
    )
    policy_version: str = Field(
        default="rb_v1_explore",
        description="Policy version identifier for logging",
    )

    training_mode: bool = Field(
        default=False,
        description="Enable training mode for multi-strategy experimentation (research only)",
    )
    max_calls_per_underlying_live: int = Field(
        default=1,
        description="Maximum covered calls per underlying in live mode",
    )
    max_calls_per_underlying_training: int = Field(
        default=5,
        description="Maximum covered calls per underlying in training mode",
    )
    training_strategies: list[str] = Field(
        default=["conservative", "moderate", "aggressive"],
        description="Strategy profiles to test in training mode",
    )
    
    save_training_data: bool = Field(
        default=False,
        description="If True, export training data CSV/JSONL after each backtest",
    )
    training_data_dir: str = Field(
        default="data",
        description="Directory to save training data files",
    )

    default_order_size: float = Field(
        default=0.1,
        description="Default order size in BTC/ETH",
    )

    underlyings: list[str] = Field(
        default=["BTC", "ETH"],
        description="List of underlyings to trade",
    )
    
    option_margin_type: Literal["linear", "inverse"] = Field(
        default="linear",
        description="Option margin type: 'linear' for USDC-settled, 'inverse' for coin-margined",
    )
    option_settlement_ccy: str = Field(
        default="USDC",
        description="Settlement currency for options (USDC for linear)",
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
        default=False,
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

    @property
    def is_research(self) -> bool:
        """Check if running in research mode."""
        return self.mode == "research"

    @property
    def is_training_enabled(self) -> bool:
        """Check if training mode is active (research + training_mode)."""
        return self.is_research and self.training_mode and self.dry_run

    @property
    def max_calls_per_underlying(self) -> int:
        """Get the maximum calls per underlying based on mode."""
        if self.is_training_enabled:
            return self.max_calls_per_underlying_training
        return self.max_calls_per_underlying_live

    @property
    def effective_ivrv_min(self) -> float:
        """Get mode-appropriate IVRV minimum."""
        return self.research_ivrv_min if self.is_research else self.ivrv_min

    @property
    def effective_delta_min(self) -> float:
        """Get mode-appropriate delta minimum."""
        return self.research_delta_min if self.is_research else self.delta_min

    @property
    def effective_delta_max(self) -> float:
        """Get mode-appropriate delta maximum."""
        return self.research_delta_max if self.is_research else self.delta_max

    @property
    def effective_dte_min(self) -> int:
        """Get mode-appropriate DTE minimum."""
        return self.research_dte_min if self.is_research else self.dte_min

    @property
    def effective_dte_max(self) -> int:
        """Get mode-appropriate DTE maximum."""
        return self.research_dte_max if self.is_research else self.dte_max

    @property
    def effective_max_expiry_exposure(self) -> float:
        """Get mode-appropriate per-expiry exposure limit."""
        return self.research_max_expiry_exposure if self.is_research else self.max_expiry_exposure


settings = Settings()
