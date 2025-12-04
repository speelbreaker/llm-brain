"""
Backtesting and simulation module for covered call strategies.
Provides historical data analysis and ML training data generation.
"""
from .types import (
    CallSimulationConfig,
    SimulatedTrade,
    SimulationResult,
    TrainingExample,
    OptionSnapshot,
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

__all__ = [
    "CallSimulationConfig",
    "SimulatedTrade",
    "SimulationResult",
    "TrainingExample",
    "OptionSnapshot",
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
]
