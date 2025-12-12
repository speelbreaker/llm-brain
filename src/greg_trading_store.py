"""
Mutable runtime store for Greg trading mode settings.

Provides thread-safe access to trading mode configuration that can be
modified at runtime via API endpoints without restarting the server.

IMPORTANT: The singleton `greg_trading_store` is initialized from settings
at import time to maintain consistency with static configuration.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from src.config import GregTradingMode


@dataclass
class GregTradingState:
    """Current Greg trading state."""
    mode: GregTradingMode = GregTradingMode.ADVICE_ONLY
    enable_live_execution: bool = False
    strategy_live_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "STRATEGY_A_STRADDLE": False,
        "STRATEGY_A_STRANGLE": False,
        "STRATEGY_B_CALENDAR": False,
        "STRATEGY_C_SHORT_PUT": False,
        "STRATEGY_D_IRON_BUTTERFLY": False,
        "STRATEGY_F_BULL_PUT_SPREAD": False,
        "STRATEGY_F_BEAR_CALL_SPREAD": False,
    })
    last_mode_change: Optional[datetime] = None
    last_change_reason: Optional[str] = None


class GregTradingStore:
    """
    Thread-safe store for Greg trading mode settings.
    
    This provides runtime-mutable configuration separate from the static
    Pydantic settings, ensuring mode changes take effect immediately.
    """
    
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = GregTradingState()
    
    def get_mode(self) -> GregTradingMode:
        """Get current trading mode."""
        with self._lock:
            return self._state.mode
    
    def get_enable_live(self) -> bool:
        """Get master live execution switch."""
        with self._lock:
            return self._state.enable_live_execution
    
    def get_strategy_enabled(self, strategy: str) -> bool:
        """Check if a specific strategy is enabled for live execution."""
        with self._lock:
            return self._state.strategy_live_enabled.get(strategy, False)
    
    def get_all_strategy_flags(self) -> Dict[str, bool]:
        """Get all strategy live execution flags."""
        with self._lock:
            return self._state.strategy_live_enabled.copy()
    
    def get_state(self) -> Dict:
        """Get full current state as dict."""
        with self._lock:
            return {
                "mode": self._state.mode.value,
                "enable_live_execution": self._state.enable_live_execution,
                "strategy_live_enabled": self._state.strategy_live_enabled.copy(),
                "last_mode_change": self._state.last_mode_change.isoformat() if self._state.last_mode_change else None,
                "last_change_reason": self._state.last_change_reason,
            }
    
    def set_mode(self, mode: GregTradingMode, reason: str = "") -> None:
        """Set trading mode."""
        with self._lock:
            self._state.mode = mode
            self._state.last_mode_change = datetime.now(timezone.utc)
            self._state.last_change_reason = reason or f"Mode changed to {mode.value}"
    
    def set_enable_live(self, enabled: bool) -> None:
        """Set master live execution switch."""
        with self._lock:
            self._state.enable_live_execution = enabled
    
    def set_strategy_enabled(self, strategy: str, enabled: bool) -> None:
        """Set live execution flag for a specific strategy."""
        with self._lock:
            if strategy in self._state.strategy_live_enabled:
                self._state.strategy_live_enabled[strategy] = enabled
    
    def set_all_strategy_flags(self, flags: Dict[str, bool]) -> None:
        """Update multiple strategy flags at once."""
        with self._lock:
            for strategy, enabled in flags.items():
                if strategy in self._state.strategy_live_enabled:
                    self._state.strategy_live_enabled[strategy] = enabled
    
    def can_execute(self, strategy: str) -> Tuple[bool, str]:
        """
        Check if execution is allowed for a given strategy.
        
        Returns:
            (allowed, reason) tuple
        """
        with self._lock:
            mode = self._state.mode
            
            if mode == GregTradingMode.ADVICE_ONLY:
                return False, "Advice-only mode - no execution allowed"
            
            if mode == GregTradingMode.PAPER:
                return True, "Paper mode - DRY_RUN execution allowed"
            
            if mode == GregTradingMode.LIVE:
                if not self._state.enable_live_execution:
                    return False, "Live execution master switch is disabled"
                
                if not self._state.strategy_live_enabled.get(strategy, False):
                    return False, f"Strategy {strategy} is not enabled for live execution"
                
                return True, "Live execution allowed"
            
            return False, f"Unknown mode: {mode}"
    
    def atomic_execute_check(self, strategy: str, deribit_env: str) -> Tuple[bool, str, bool]:
        """
        Atomic check for execution permission with environment validation.
        
        Returns:
            (allowed, reason, is_dry_run) tuple
        
        This combines all safety checks in a single locked operation to prevent
        TOCTOU race conditions.
        """
        with self._lock:
            mode = self._state.mode
            
            if mode == GregTradingMode.ADVICE_ONLY:
                return False, "Advice-only mode - no execution allowed", True
            
            if mode == GregTradingMode.PAPER:
                if deribit_env != "testnet":
                    return False, f"PAPER mode requires testnet (current: {deribit_env})", True
                return True, "Paper mode - DRY_RUN execution", True
            
            if mode == GregTradingMode.LIVE:
                if deribit_env != "mainnet":
                    return False, f"LIVE mode requires mainnet (current: {deribit_env})", False
                
                if not self._state.enable_live_execution:
                    return False, "Live execution master switch is disabled", False
                
                if not self._state.strategy_live_enabled.get(strategy, False):
                    return False, f"Strategy {strategy} is not enabled for live execution", False
                
                return True, "Live execution allowed", False
            
            return False, f"Unknown mode: {mode}", True
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to safe defaults."""
        with self._lock:
            self._state = GregTradingState()
            self._state.last_mode_change = datetime.now(timezone.utc)
            self._state.last_change_reason = "Reset to defaults"
    
    def init_from_settings(self) -> None:
        """Initialize store from static settings at startup."""
        from src.config import settings
        
        with self._lock:
            self._state.mode = settings.greg_trading_mode
            self._state.enable_live_execution = settings.greg_enable_live_execution
            
            for strategy, enabled in settings.greg_strategy_live_enabled.items():
                if strategy in self._state.strategy_live_enabled:
                    self._state.strategy_live_enabled[strategy] = enabled
            
            self._state.last_mode_change = datetime.now(timezone.utc)
            self._state.last_change_reason = "Initialized from settings"


greg_trading_store = GregTradingStore()
greg_trading_store.init_from_settings()
