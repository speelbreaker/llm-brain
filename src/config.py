"""
Configuration module using Pydantic BaseSettings.
Reads from environment variables and .env file.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GregTradingMode(str, Enum):
    """Trading mode for Greg strategies."""
    ADVICE_ONLY = "advice_only"
    PAPER = "paper"
    LIVE = "live"


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

    deribit_env: Literal["testnet", "mainnet"] = Field(
        default="testnet",
        description="Deribit environment: 'testnet' or 'mainnet'",
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
    daily_drawdown_limit_pct: float = Field(
        default=0.0,
        description=(
            "Maximum allowed daily peak-to-trough equity loss (percent). "
            "0.0 disables the daily drawdown guard."
        ),
    )
    kill_switch_enabled: bool = Field(
        default=False,
        description=(
            "Global kill switch for trading. When True, all non-DO_NOTHING "
            "actions are blocked at the risk layer."
        ),
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
    training_profile_mode: Literal["single", "ladder"] = Field(
        default="ladder",
        description="Training behavior: 'single' = one call logic, 'ladder' = multi-profile exploration with aggressive candidate selection",
    )
    max_calls_per_underlying_live: int = Field(
        default=1,
        description="Maximum covered calls per underlying in live mode",
    )
    max_calls_per_underlying_training: int = Field(
        default=6,
        description="Maximum covered calls per underlying in training mode",
    )
    training_max_calls_per_expiry: int = Field(
        default=3,
        description="Soft cap per expiry in training ladders (per underlying)",
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

    synthetic_iv_multiplier: float = Field(
        default=1.0,
        description="Multiplier applied to realized vol for synthetic IV",
    )
    synthetic_skew_enabled: bool = Field(
        default=True,
        description="If True, apply a static skew curve to synthetic IV based on live Deribit smile",
    )
    synthetic_skew_min_dte: float = Field(
        default=3.0,
        description="Minimum DTE (days) for options used to estimate skew",
    )
    synthetic_skew_max_dte: float = Field(
        default=14.0,
        description="Maximum DTE (days) for options used to estimate skew",
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
    decision_mode: Literal["rule_only", "llm_only", "hybrid_shadow"] = Field(
        default="rule_only",
        description=(
            "Decision mode for non-training runs: "
            "'rule_only' = only rule-based executes (LLM optional shadow); "
            "'llm_only' = LLM executes with fallback to rules on error/invalid; "
            "'hybrid_shadow' = rules execute, LLM runs in shadow for logging/comparison."
        ),
    )
    llm_shadow_enabled: bool = Field(
        default=True,
        description=(
            "If True and llm_enabled, compute an LLM proposal even when it won't be executed, "
            "so we can log and compare against the rule-based action."
        ),
    )
    llm_validation_strict: bool = Field(
        default=True,
        description=(
            "If True, apply strict validation to LLM decisions; on any validation failure, "
            "fall back to DO_NOTHING or rule-based action."
        ),
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

    greg_trading_mode: GregTradingMode = Field(
        default=GregTradingMode.ADVICE_ONLY,
        description=(
            "Trading mode for Greg strategies: "
            "'advice_only' = never sends orders (default); "
            "'paper' = DRY_RUN/testnet only; "
            "'live' = can send mainnet orders (requires additional flags)"
        ),
    )
    greg_enable_live_execution: bool = Field(
        default=False,
        description=(
            "Master switch for live execution. MUST be True + mode=LIVE to send real orders. "
            "Acts as a secondary safety gate."
        ),
    )
    greg_strategy_live_enabled: Dict[str, bool] = Field(
        default={
            "STRATEGY_A_STRADDLE": False,
            "STRATEGY_A_STRANGLE": False,
            "STRATEGY_B_CALENDAR": False,
            "STRATEGY_C_SHORT_PUT": False,
            "STRATEGY_D_IRON_BUTTERFLY": False,
            "STRATEGY_F_BULL_PUT_SPREAD": False,
            "STRATEGY_F_BEAR_CALL_SPREAD": False,
        },
        description=(
            "Per-strategy live execution flags. Each strategy must be explicitly enabled "
            "for live orders to be sent. Only applies when mode=LIVE and enable_live_execution=True."
        ),
    )
    greg_live_max_notional_usd_per_position: float = Field(
        default=500.0,
        description=(
            "Maximum notional USD per position for live Greg trades. "
            "Positions exceeding this limit will be rejected or downsized."
        ),
    )
    greg_live_max_notional_usd_per_underlying: float = Field(
        default=2000.0,
        description=(
            "Maximum total notional USD exposure per underlying (BTC/ETH) for live Greg trades. "
            "New trades exceeding this limit will be rejected."
        ),
    )

    log_dir: str = Field(
        default="logs",
        description="Directory for JSON decision logs",
    )

    position_reconcile_action: Literal["halt", "auto_heal"] = Field(
        default="halt",
        description="Action on position divergence: 'halt' stops trading, 'auto_heal' rebuilds local state from exchange",
    )
    position_reconcile_on_startup: bool = Field(
        default=True,
        description="Run position reconciliation on agent startup before entering main loop",
    )

    auto_kill_on_health_fail: bool = Field(
        default=True,
        description=(
            "If True, the agent loop will pause trading when healthcheck returns FAIL. "
            "Trading resumes when health returns to OK/WARN. On mainnet, agent startup is aborted on FAIL."
        ),
    )
    health_check_on_startup: bool = Field(
        default=True,
        description="Run healthcheck on agent startup before entering main loop",
    )
    health_recheck_interval_seconds: int = Field(
        default=600,
        description=(
            "Interval in seconds for runtime health re-checks during the agent loop. "
            "Set to 0 to disable runtime re-checks."
        ),
    )
    position_reconcile_on_each_loop: bool = Field(
        default=True,
        description="Run position reconciliation at the start of each loop iteration",
    )
    position_reconcile_tolerance_usd: float = Field(
        default=10.0,
        description="Tolerance in USD for size mismatches (to avoid panicking over tiny rounding differences)",
    )

    # Liquidity safeguards (per-strike)
    liquidity_max_spread_pct: float = Field(
        default=5.0,
        description="Maximum bid/ask spread as percentage of mid price (e.g. 5.0 = 5%)",
    )
    liquidity_min_open_interest: int = Field(
        default=50,
        description="Minimum open interest (contracts) per strike for liquidity checks",
    )

    # Delta hedging configuration
    hedge_enabled: bool = Field(
        default=False,
        description="Enable delta hedging for Greg strategies",
    )
    hedge_check_interval_sec: int = Field(
        default=60,
        description="Interval between hedge checks in seconds",
    )
    hedge_on_price_move_pct: float = Field(
        default=0.01,
        description="Trigger hedge check on price move greater than this percentage",
    )

    # Force test mode for smoke testing hedging
    force_test_strategy: str = Field(
        default="",
        description="Force a specific strategy for testing (e.g., STRATEGY_A_STRADDLE). Skips normal entry logic.",
    )
    force_test_underlying: str = Field(
        default="BTC",
        description="Underlying for force test (BTC or ETH)",
    )
    force_test_size: float = Field(
        default=0.01,
        description="Size for force test positions (small for safety)",
    )

    @property
    def is_research(self) -> bool:
        """Check if running in research mode."""
        return self.mode == "research"

    @property
    def is_training_enabled(self) -> bool:
        """Check if training mode is active (research + training_mode)."""
        return self.is_research and self.training_mode

    @property
    def is_testnet(self) -> bool:
        """Check if connected to Deribit testnet."""
        return self.deribit_env == "testnet"

    @property
    def is_training_on_testnet(self) -> bool:
        """Check if training mode is active AND on testnet (safe for bypassing risk)."""
        return self.is_training_enabled and self.is_testnet

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
