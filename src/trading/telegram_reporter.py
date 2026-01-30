"""
Telegram Reporter for Trading Loop.

Posts significant trading events to the Telegram trading topic.
Designed for minimal overhead - only posts on significant events.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    ParseMode = None
    TelegramError = Exception

from src.config import settings

logger = logging.getLogger(__name__)


class TradingTelegramReporter:
    """Reports significant events to Telegram.

    Primary use is the trading loop, but we also reuse it for:
    - supervisor status changes
    - agent health changes
    - paper portfolio OPEN/CLOSE/ROLL events

    Features:
    - Rate limiting (max 1 message per minute unless critical)
    - Significant event filtering
    - Graceful degradation if Telegram unavailable
    """

    def __init__(self):
        self.bot: Optional[Bot] = None
        self.last_message_time: Optional[datetime] = None
        self.min_interval = timedelta(minutes=1)
        self._initialized = False

        # Change detectors
        self._last_health_state: Optional[str] = None

        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed, Telegram disabled")
    
    def _initialize(self) -> bool:
        """Lazy initialization of bot."""
        if self._initialized:
            return self.bot is not None
        
        self._initialized = True
        
        if not TELEGRAM_AVAILABLE:
            return False

        if not settings.trading_telegram_enabled:
            logger.info("Trading Telegram disabled in settings")
            return False
        
        if not settings.telegram_bot_token:
            logger.warning("No telegram_bot_token configured")
            return False
        
        if not settings.telegram_supergroup_id:
            logger.warning("No telegram_supergroup_id configured")
            return False
        
        try:
            self.bot = Bot(token=settings.telegram_bot_token)
            logger.info("Trading Telegram reporter initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False
    
    def _should_send(self, is_critical: bool = False) -> bool:
        """Check if we should send based on rate limiting."""
        if is_critical:
            return True
        
        if self.last_message_time is None:
            return True
        
        elapsed = datetime.utcnow() - self.last_message_time
        return elapsed >= self.min_interval
    
    def _is_significant(self, snapshot: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if snapshot represents a significant event.
        
        Returns (is_significant, reason)
        """
        execution = snapshot.get("execution", {})
        risk_check = snapshot.get("risk_check", {})
        config = snapshot.get("config_snapshot", {})
        reconciliation = snapshot.get("reconciliation", {})
        
        # Trade executed
        if execution.get("status") == "executed":
            return True, "trade_executed"
        
        # Execution error
        if execution.get("status") == "error":
            return True, "error"
            
        # Risk check blocked (optional: maybe too noisy? skipping for now unless critical)
        # if not risk_check.get("allowed", True):
        #     return True, "risk_blocked"
        
        # Kill switch activated
        if config.get("kill_switch_enabled"):
            return True, "kill_switch"
            
        # Reconciliation failed/halted
        if reconciliation.get("divergent") and settings.position_reconcile_action == "halt":
            return True, "reconciliation_halt"
        
        return False, ""
    
    def _format_status_card(self, snapshot: Dict[str, Any]) -> str:
        """Format snapshot as Telegram message."""
        execution = snapshot.get("execution", {})
        risk_check = snapshot.get("risk_check", {})
        final_action = snapshot.get("final_action", {})
        state = snapshot.get("state", {})
        portfolio = state.get("portfolio", {})
        
        # Determine status emoji
        status_emoji = "🟢"
        if execution.get("status") == "error":
            status_emoji = "🔴"
        elif not risk_check.get("allowed", True):
            status_emoji = "🟡"
            
        timestamp = snapshot.get("log_timestamp", datetime.utcnow().isoformat())
        try:
            # Try to parse ISO format to cleaner time
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M UTC")
        except ValueError:
            time_str = timestamp
        
        # Header
        lines = [
            f"{status_emoji} <b>STATUS UPDATE</b> {time_str}",
            "",
        ]
        
        # Action
        action_type = final_action.get("action", "UNKNOWN")
        lines.append(f"<b>Action:</b> {action_type}")
        
        if final_action.get("reasoning"):
            lines.append(f"<b>Reason:</b> {final_action['reasoning']}")

        # Execution Result
        exec_status = execution.get("status")
        if exec_status == "executed":
            lines.append(f"<b>Execution:</b> ✅ {execution.get('message', 'Done')}")
        elif exec_status == "error":
            lines.append(f"<b>Execution:</b> ❌ {execution.get('message', 'Failed')}")
        
        # Portfolio Stats
        equity = portfolio.get("equity_usd", 0)
        margin = portfolio.get("margin_used_pct", 0)
        lines.append(f"<b>Equity:</b> ${equity:,.2f} (Margin: {margin:.1f}%)")
        
        # Risk Rejection
        if not risk_check.get("allowed", True):
            reasons = risk_check.get("reasons", [])
            if reasons:
                lines.append("")
                lines.append("<b>Risk Block:</b>")
                for r in reasons:
                    lines.append(f"  • {r}")
        
        return "\n".join(lines)
    
    async def send_message(self, text: str, is_critical: bool = False) -> bool:
        """
        Send a message to the trading topic.
        
        Args:
            text: Message text (HTML format)
            is_critical: If True, bypass rate limiting
        
        Returns:
            True if sent successfully
        """
        if not self._initialize():
            return False
        
        if not self._should_send(is_critical):
            logger.debug("Rate limited, skipping message")
            return False
        
        try:
            kwargs = {
                "chat_id": settings.telegram_supergroup_id,
                "text": text,
                "parse_mode": ParseMode.HTML,
            }
            
            # Add topic ID if configured
            if settings.telegram_topic_trading:
                kwargs["message_thread_id"] = settings.telegram_topic_trading
            
            await self.bot.send_message(**kwargs)
            self.last_message_time = datetime.utcnow()
            logger.debug("Telegram message sent")
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram: {e}")
            return False
    
    async def on_status_update(self, snapshot: Dict[str, Any]) -> None:
        """Called by agent_loop on each status update."""
        # 1) Health change notifications (best-effort)
        try:
            from src.healthcheck import get_cached_health_status

            cached = get_cached_health_status()
            if cached is not None:
                state = str(cached.overall_status or "").upper()
                if self._last_health_state is None:
                    self._last_health_state = state
                elif state != self._last_health_state:
                    self._last_health_state = state
                    await self._send_health_change(cached)
        except Exception:
            pass

        # 2) Paper trade notifications (if present)
        try:
            events = snapshot.get("paper_events") or []
            if isinstance(events, list) and events:
                for ev in events:
                    await self._send_paper_event(ev)
        except Exception:
            pass

        # 3) Existing significant-event status card
        is_significant, reason = self._is_significant(snapshot)
        if not is_significant:
            return

        is_critical = reason in ("error", "kill_switch", "trade_executed")
        message = self._format_status_card(snapshot)
        await self.send_message(message, is_critical=is_critical)
    
    async def send_startup_message(self) -> None:
        """Send a message when the trading loop starts."""
        message = (
            "🚀 <b>TRADING LOOP STARTED</b>\n\n"
            f"<b>Mode:</b> {settings.mode}\n"
            f"<b>Deribit:</b> {settings.deribit_env}\n"
            f"<b>Trade Mode:</b> {settings.trade_mode.value}\n"
            f"<b>Interval:</b> {settings.loop_interval_sec}s\n"
            f"<b>Kill Switch:</b> {'🔴 ON' if settings.kill_switch_enabled else '🟢 OFF'}\n\n"
            f"<i>Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        await self.send_message(message, is_critical=True)
    
    async def send_shutdown_message(self, reason: str = "normal") -> None:
        """Send a message when the trading loop stops."""
        message = (
            "🛑 <b>TRADING LOOP STOPPED</b>\n\n"
            f"<b>Reason:</b> {reason}\n"
            f"<i>Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        await self.send_message(message, is_critical=True)
    
    async def send_trade_alert(
        self,
        action: str,
        instrument: str,
        size: float,
        premium: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Send alert for trade execution."""
        emoji = "🔔" if action.upper() != "CLOSE" else "✅"

        lines = [
            f"{emoji} <b>TRADE EXECUTED</b>",
            "",
            f"<b>Action:</b> {action.upper()}",
            f"<b>Instrument:</b> {instrument}",
            f"<b>Size:</b> {size:+.4f}",
        ]

        if premium is not None:
            lines.append(f"<b>Premium:</b> ${premium:,.2f}")

        if reason:
            lines.append(f"<b>Reason:</b> {reason}")

        lines.append("")
        lines.append(f"<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>")

        await self.send_message("\n".join(lines), is_critical=True)

    async def send_supervisor_status(
        self,
        *,
        status: str,
        pr_url: Optional[str] = None,
        job_id: Optional[str] = None,
        comment_url: Optional[str] = None,
        message: Optional[str] = None,
        is_error: bool = False,
    ) -> None:
        """Send PR supervisor status changes (queued/running/done/error)."""
        emoji = "🧰"
        if is_error:
            emoji = "🛑"
        elif status.lower() in ("queued", "running"):
            emoji = "🟡" if status.lower() == "queued" else "🔵"
        elif status.lower() in ("done", "fixed", "checks_passed"):
            emoji = "🟢"

        lines = [f"{emoji} <b>PR SUPERVISOR</b>", "", f"<b>Status:</b> {status}"]
        if job_id:
            lines.append(f"<b>Job:</b> {job_id}")
        if pr_url:
            lines.append(f"<b>PR:</b> {pr_url}")
        if comment_url:
            lines.append(f"<b>Comment:</b> {comment_url}")
        if message:
            lines.append(f"<b>Info:</b> {message}")

        lines.append("")
        lines.append(f"<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>")
        await self.send_message("\n".join(lines), is_critical=True)

    async def _send_health_change(self, cached: Any) -> None:
        """Send agent health state changes (OK/WARN/FAIL) and failing checks."""
        overall = str(getattr(cached, "overall_status", "UNKNOWN") or "UNKNOWN").upper()
        emoji = {"OK": "🟢", "WARN": "🟡", "FAIL": "🔴"}.get(overall, "⚪")
        summary = getattr(cached, "summary", "") or ""

        lines = [f"{emoji} <b>AGENT HEALTH</b>", "", f"<b>Status:</b> {overall}"]
        if summary:
            lines.append(f"<b>Summary:</b> {summary}")

        details = getattr(cached, "details", None) or {}
        checks = details.get("checks") or details.get("results") or []
        failing = [c for c in checks if str(c.get("status") or "").upper() in ("FAIL", "WARN")]
        if failing:
            lines.append("")
            lines.append("<b>Failing/Warn checks:</b>")
            for c in failing[:12]:
                name = c.get("name") or "check"
                st = str(c.get("status") or "?")
                detail = c.get("detail") or ""
                lines.append(f"  • {name}: {st} — {detail}"[:350])

        lines.append("")
        lines.append(f"<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>")
        await self.send_message("\n".join(lines), is_critical=True)

    async def _send_paper_event(self, ev: Dict[str, Any]) -> None:
        """Send a paper lane OPEN/CLOSE/ROLL event."""
        et = str(ev.get("type") or "").upper()
        lane = str(ev.get("lane") or "?")
        emoji = {"OPEN": "📝", "CLOSE": "✅", "ROLL": "🔄"}.get(et, "🧾")

        lines = [f"{emoji} <b>PAPER {lane.upper()}</b>", "", f"<b>Event:</b> {et}"]
        if et == "ROLL":
            lines.append(f"<b>From:</b> {ev.get('from_symbol')}")
            lines.append(f"<b>To:</b> {ev.get('to_symbol')}")
            lines.append(f"<b>Size:</b> {float(ev.get('size') or 0):.4f}")
            lines.append(f"<b>Close:</b> {float(ev.get('close_price') or 0):.6f}")
            lines.append(f"<b>Open:</b> {float(ev.get('open_price') or 0):.6f}")
        else:
            lines.append(f"<b>Symbol:</b> {ev.get('symbol')}")
            if ev.get("underlying"):
                lines.append(f"<b>Underlying:</b> {ev.get('underlying')}")
            lines.append(f"<b>Size:</b> {float(ev.get('size') or 0):.4f}")
            lines.append(f"<b>Price:</b> {float(ev.get('price') or 0):.6f}")

        lines.append("")
        lines.append(f"<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>")
        await self.send_message("\n".join(lines), is_critical=True)


# Singleton instance
_reporter: Optional[TradingTelegramReporter] = None


def get_trading_telegram_reporter() -> TradingTelegramReporter:
    """Get or create the singleton reporter instance."""
    global _reporter
    if _reporter is None:
        _reporter = TradingTelegramReporter()
    return _reporter


async def trading_status_callback(snapshot: Dict[str, Any]) -> None:
    """
    Convenience callback for agent_loop status_callback.
    
    Usage in agent_loop.py:
        from src.trading.telegram_reporter import trading_status_callback
        
        await run_agent_loop_forever(
            status_callback=trading_status_callback
        )
    """
    reporter = get_trading_telegram_reporter()
    await reporter.on_status_update(snapshot)