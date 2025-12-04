"""
Backtesting environment simulator stub.
Provides a minimal RL-compatible interface for future training.

TODO: This module will be expanded to include:
- Historical price data loading for BTC/ETH
- Option pricing simulation using Black-Scholes or historical IVs
- Portfolio tracking with margin calculations
- Reward function based on PnL, Sharpe ratio, or other metrics
- Support for the same action space as the live agent:
  - DO_NOTHING
  - OPEN_COVERED_CALL
  - ROLL_COVERED_CALL
  - CLOSE_COVERED_CALL

The interface follows the OpenAI Gym convention for RL environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class SimulatedState:
    """Simulated state for backtesting."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    btc_spot: float = 100000.0
    eth_spot: float = 3500.0
    
    btc_iv: float = 60.0
    eth_iv: float = 65.0
    
    equity_usd: float = 10000.0
    margin_used_pct: float = 0.0
    net_delta: float = 0.0
    
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    
    candidate_options: list[dict[str, Any]] = field(default_factory=list)
    
    def to_observation(self) -> np.ndarray:
        """Convert state to observation array for RL."""
        return np.array([
            self.btc_spot / 100000,
            self.eth_spot / 10000,
            self.btc_iv / 100,
            self.eth_iv / 100,
            self.equity_usd / 10000,
            self.margin_used_pct / 100,
            self.net_delta,
            len(self.open_positions),
        ], dtype=np.float32)


class CoveredCallEnv:
    """
    Simulated environment for covered call strategy backtesting.
    
    Follows the Gym-style interface:
    - reset() -> state
    - step(action) -> (state, reward, done, truncated, info)
    
    TODO: Implement the following:
    - Load historical price data
    - Simulate option pricing
    - Track positions and PnL
    - Calculate realistic rewards
    - Handle episode boundaries
    """
    
    ACTIONS = {
        0: "DO_NOTHING",
        1: "OPEN_COVERED_CALL",
        2: "ROLL_COVERED_CALL",
        3: "CLOSE_COVERED_CALL",
    }
    
    def __init__(
        self,
        initial_equity: float = 10000.0,
        episode_length_days: int = 30,
        time_step_hours: int = 1,
    ):
        """
        Initialize the backtesting environment.
        
        Args:
            initial_equity: Starting equity in USD
            episode_length_days: Length of each episode in days
            time_step_hours: Time step size in hours
        """
        self.initial_equity = initial_equity
        self.episode_length_days = episode_length_days
        self.time_step_hours = time_step_hours
        
        self.state: SimulatedState | None = None
        self.step_count: int = 0
        self.max_steps: int = (episode_length_days * 24) // time_step_hours
        
        self.trade_history: list[dict[str, Any]] = []
        self.pnl_history: list[float] = []
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
        
        Returns:
            Tuple of (observation, info dict)
        
        TODO: Implement proper initialization with:
        - Random or specific historical date
        - Load price data for the episode period
        - Generate realistic option chain
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.state = SimulatedState(
            timestamp=datetime.utcnow(),
            btc_spot=100000.0 + np.random.normal(0, 5000),
            eth_spot=3500.0 + np.random.normal(0, 200),
            btc_iv=60.0 + np.random.normal(0, 10),
            eth_iv=65.0 + np.random.normal(0, 10),
            equity_usd=self.initial_equity,
            margin_used_pct=0.0,
            net_delta=0.0,
            open_positions=[],
            candidate_options=self._generate_mock_candidates(),
        )
        
        self.step_count = 0
        self.trade_history = []
        self.pnl_history = [self.initial_equity]
        
        info = {
            "episode_start": self.state.timestamp.isoformat(),
            "initial_equity": self.initial_equity,
        }
        
        return self.state.to_observation(), info
    
    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0-3)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        
        TODO: Implement proper simulation with:
        - Price evolution (GBM or historical replay)
        - Option value changes
        - Position PnL tracking
        - Margin calculations
        - Realistic reward function
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        action_name = self.ACTIONS.get(action, "DO_NOTHING")
        
        self.state.timestamp += timedelta(hours=self.time_step_hours)
        self.step_count += 1
        
        btc_return = np.random.normal(0, 0.02)
        eth_return = np.random.normal(0, 0.025)
        
        self.state.btc_spot *= (1 + btc_return)
        self.state.eth_spot *= (1 + eth_return)
        
        self.state.btc_iv = max(20, min(150, self.state.btc_iv + np.random.normal(0, 2)))
        self.state.eth_iv = max(20, min(150, self.state.eth_iv + np.random.normal(0, 2.5)))
        
        reward = 0.0
        
        if action_name == "OPEN_COVERED_CALL":
            if not self.state.open_positions:
                self.state.margin_used_pct = 20.0
                self.state.net_delta = -0.3
                self.state.open_positions.append({
                    "symbol": "MOCK-OPTION",
                    "premium_collected": 50.0,
                })
                reward = 0.1
        
        elif action_name == "CLOSE_COVERED_CALL":
            if self.state.open_positions:
                self.state.margin_used_pct = 0.0
                self.state.net_delta = 0.0
                self.state.open_positions = []
                reward = 0.05
        
        elif action_name == "ROLL_COVERED_CALL":
            if self.state.open_positions:
                reward = 0.02
        
        else:
            reward = -0.001
        
        self.state.candidate_options = self._generate_mock_candidates()
        
        equity_change = np.random.normal(0.001, 0.005) * self.state.equity_usd
        self.state.equity_usd += equity_change
        self.pnl_history.append(self.state.equity_usd)
        
        terminated = self.state.equity_usd <= 0
        truncated = self.step_count >= self.max_steps
        
        info = {
            "action_taken": action_name,
            "btc_spot": self.state.btc_spot,
            "eth_spot": self.state.eth_spot,
            "equity_usd": self.state.equity_usd,
            "step": self.step_count,
            "positions": len(self.state.open_positions),
        }
        
        return self.state.to_observation(), reward, terminated, truncated, info
    
    def _generate_mock_candidates(self) -> list[dict[str, Any]]:
        """Generate mock candidate options for simulation."""
        if self.state is None:
            return []
        
        candidates = []
        for i in range(3):
            strike_pct = 1.05 + i * 0.05
            candidates.append({
                "symbol": f"BTC-MOCK-{int(self.state.btc_spot * strike_pct)}-C",
                "strike": self.state.btc_spot * strike_pct,
                "dte": 7 + i * 7,
                "delta": 0.25 - i * 0.05,
                "premium_usd": 100 - i * 20,
            })
        
        return candidates
    
    def render(self) -> None:
        """Render the current state (for debugging)."""
        if self.state is None:
            print("Environment not initialized")
            return
        
        print(f"\n{'='*40}")
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"Time: {self.state.timestamp}")
        print(f"BTC: ${self.state.btc_spot:,.0f}")
        print(f"ETH: ${self.state.eth_spot:,.0f}")
        print(f"Equity: ${self.state.equity_usd:,.2f}")
        print(f"Margin: {self.state.margin_used_pct:.1f}%")
        print(f"Positions: {len(self.state.open_positions)}")
        print(f"{'='*40}\n")
    
    def close(self) -> None:
        """Clean up environment resources."""
        self.state = None
        self.trade_history = []
        self.pnl_history = []


if __name__ == "__main__":
    print("Testing CoveredCallEnv...")
    
    env = CoveredCallEnv(initial_equity=10000.0, episode_length_days=7)
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for step in range(24):
        action = np.random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 6 == 0:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nFinal equity: ${info['equity_usd']:,.2f}")
    print(f"Total steps: {info['step']}")
    
    env.close()
    print("\nTest complete!")
