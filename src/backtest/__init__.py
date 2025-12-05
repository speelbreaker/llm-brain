"""
Backtesting and simulation module for covered call strategies.
Provides historical data analysis and ML training data generation.

NOTE: When option_margin_type="linear" and option_settlement_ccy="USDC" (defaults),
all prices (spot, option mark_price) and PnL are in USD/USDC.
"""
from .types import (
    CallSimulationConfig,
    SimulatedTrade,
    SimulationResult,
    TrainingExample,
    OptionSnapshot,
    ExitStyle,
)
from .data_source import MarketDataSource, Timeframe
from .deribit_client import DeribitPublicClient
from .deribit_data_source import DeribitDataSource
from .covered_call_simulator import (
    CoveredCallSimulator,
    always_trade_policy,
    never_trade_policy,
)
from .training_dataset import (
    generate_training_data,
    export_to_csv,
    export_to_jsonl,
    compute_dataset_stats,
    run_grid_search,
)
from .market_context_backtest import (
    compute_market_context_from_ds,
    market_context_to_dict,
)
from .state_builder import (
    build_historical_state,
    create_state_builder,
)

__all__ = [
    "CallSimulationConfig",
    "SimulatedTrade",
    "SimulationResult",
    "TrainingExample",
    "OptionSnapshot",
    "ExitStyle",
    "MarketDataSource",
    "Timeframe",
    "DeribitPublicClient",
    "DeribitDataSource",
    "CoveredCallSimulator",
    "always_trade_policy",
    "never_trade_policy",
    "generate_training_data",
    "export_to_csv",
    "export_to_jsonl",
    "compute_dataset_stats",
    "run_grid_search",
    "compute_market_context_from_ds",
    "market_context_to_dict",
    "build_historical_state",
    "create_state_builder",
]
