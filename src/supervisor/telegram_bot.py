"Telegram bot interface for PR Supervisor."

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any

try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
    _TELEGRAM_AVAILABLE = True
except Exception:  # pragma: no cover
    # Optional dependency in many deployments/tests.
    Update = object  # type: ignore
    ApplicationBuilder = None  # type: ignore
    CommandHandler = None  # type: ignore
    MessageHandler = None  # type: ignore
    class _DummyContextTypes:  # pragma: no cover
        DEFAULT_TYPE = object

    ContextTypes = _DummyContextTypes  # type: ignore
    filters = None  # type: ignore
    _TELEGRAM_AVAILABLE = False

import httpx
from src.config import settings as trading_settings
from src.healthcheck import get_cached_health_status
from src.status_store import status_store
from src.telegram.store import TelegramConversationStore

from .config import SupervisorSettings
from .store import JobStore
from .redact import redact_secrets
from .models import JobStatus

logger = logging.getLogger(__name__)


class BotStoreAdapter:
    """Adapter for bot to access job history and vault data safely."""
    
    def __init__(self, settings: SupervisorSettings, store: JobStore):
        self.settings = settings
        self.store = store
        self.vault_path = Path(os.environ.get("SUPERVISOR_VAULT_REPO_DIR") or ".").resolve() / "docs" / "obsidian"

    def get_job_summary(self, job_id: str) -> str:
        job = self.store.get(job_id)
        if not job:
            return f"❌ Job {job_id} not found."
        
        lines = [
            f"🆔 *Job:* `{job.job_id}`",
            f"📂 *Repo:* {job.repo_full_name}",
            f"🔢 *PR:* #{job.pr_number}",
            f"📊 *Status:* {job.status.value}",
            f"🏁 *Stage:* {job.stage.value}",
        ]
        
        if job.reason_code:
            lines.append(f"❓ *Reason:* {job.reason_code}")
        
        if job.final_message:
            lines.append(f"📝 *Message:* {job.final_message}")
            
        return "\n".join(lines)

    def get_latest_job_for_pr(self, pr_number: int) -> str:
        # Simple scan of history
        jobs = self.store.list_recent(limit=100)
        pr_jobs = [j for j in jobs if j.pr_number == pr_number]
        
        if not pr_jobs:
            return f"❌ No recent jobs found for PR #{pr_number}."
        
        latest = pr_jobs[0]
        summary = self.get_job_summary(latest.job_id)
        
        if latest.arbiter_decision:
            summary += f"\n\n⚖️ *Arbiter Reasoning:*\n{latest.arbiter_decision.arbiter_reasoning}"
            
        return summary

    def get_queue_status(self) -> str:
        queue_file = self.vault_path / "02_QUEUE" / "QUEUE.md"
        if not queue_file.exists():
            return "❌ Queue file not found in vault."
            
        try:
            content = queue_file.read_text(encoding="utf-8")
            sections = {"READY": [], "IN_PROGRESS": [], "IN_REVIEW": []}
            current = None
            
            for line in content.splitlines():
                if "## READY" in line: current = "READY"
                elif "## IN_PROGRESS" in line: current = "IN_PROGRESS"
                elif "## IN_REVIEW" in line: current = "IN_REVIEW"
                elif line.startswith("## "): current = None
                
                if current and line.strip().startswith("- "):
                    sections[current].append(line.strip("- "))
            
            lines = ["📋 *Current Queue Status*"]
            for sec, items in sections.items():
                lines.append(f"\n*{sec}*")
                if not items:
                    lines.append("_None_")
                else:
                    for item in items[:5]: # Top 5
                        lines.append(f"• {item}")
                    if len(items) > 5:
                        lines.append(f"_... and {len(items)-5} more_")
            
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error reading queue: {str(e)}"

    def log_interaction(self, user_id: int, command: str, response: str):
        """Append interaction to 04_OPS/Telegram_Log.md."""
        log_dir = self.vault_path / "04_OPS"
        log_file = log_dir / "Telegram_Log.md"
        
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        # Sanitize response for logging
        safe_response = redact_secrets(response, self.settings).replace("\n", " ")
        if len(safe_response) > 200:
            safe_response = safe_response[:197] + "..."
            
        entry = f"| {timestamp} | {user_id} | {command} | {safe_response} |\n"
        
        if not log_file.exists():
            log_file.write_text("| Timestamp | User ID | Command | Response Summary |\n|---|---|---|---|" + "\n", encoding="utf-8")
            
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)


class TelegramBotManager:
    """Manages the Telegram bot lifecycle and command routing."""

    def __init__(self, settings: SupervisorSettings, store: JobStore):
        self.settings = settings
        self.adapter = BotStoreAdapter(settings, store)
        self.app = None
        self._is_running = False

        if settings.telegram_enabled and not _TELEGRAM_AVAILABLE:
            logger.warning(
                "python-telegram-bot is not installed; Supervisor Telegram bot is disabled. "
                "Install python-telegram-bot to enable it."
            )

    async def start(self):
        if not self.settings.telegram_enabled or not self.settings.telegram_bot_token:
            logger.warning("Telegram bot disabled (missing token or disabled in settings)")
            return
        if not _TELEGRAM_AVAILABLE:
            logger.warning("Telegram bot cannot start: python-telegram-bot not installed")
            return

        token = self.settings.telegram_bot_token
        self.app = ApplicationBuilder().token(token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("diag", self.cmd_diag))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("job", self.cmd_job))
        self.app.add_handler(CommandHandler("why", self.cmd_why))
        self.app.add_handler(CommandHandler("queue", self.cmd_queue))
        
        # Add handler for non-command messages (OpenAI chat)
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_chat))

        logger.info("Starting Telegram bot (polling mode)...")
        self._is_running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def stop(self):
        if self.app:
            logger.info("Stopping Telegram bot...")
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        self._is_running = False

    def _check_access(self, update: Update) -> bool:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        allowed_users = self.settings.get_allowed_user_ids()
        allowed_chats = self.settings.get_allowed_chat_ids()
        
        print(f"[Telegram Debug] Check Access: user={user_id} chat={chat_id}")
        print(f"[Telegram Debug] Allowed: users={allowed_users} chats={allowed_chats}")
        
        # Admin bypass
        if self.settings.telegram_admin_chat_id and str(chat_id) == self.settings.telegram_admin_chat_id:
            return True
            
        # Grant access if user is allowed OR if chat is allowed
        user_is_allowed = not allowed_users or user_id in allowed_users
        chat_is_allowed = not allowed_chats or chat_id in allowed_chats
        
        if user_is_allowed and chat_is_allowed:
            return True
            
        logger.warning(f"Unauthorized access attempt: user={user_id}, chat={chat_id}")
        return False

    async def send_redacted_reply(self, update: Update, text: str):
        safe_text = redact_secrets(text, self.settings)
        await update.message.reply_text(safe_text, parse_mode="Markdown")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        await update.message.reply_text("🤖 *PR Supervisor Bot* active.\nCommands: /diag, /status, /job <id>, /why <pr>, /queue")

    async def cmd_diag(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        
        # 1. Supervisor Diagnostics
        sup_lines = [
            "🔍 *Supervisor Health*",
            f"• Enabled: {self.settings.enabled}",
            f"• Dry Run: {self.settings.autofix_dry_run}",
            f"• LLM Available: {self.settings.is_llm_available()}",
            f"• Ready: {getattr(self, 'ready', False)}",
        ]
        
        # 2. Trading Agent Diagnostics
        trading_health = get_cached_health_status()
        if trading_health:
            trading_lines = [
                "",
                "📈 *Trading Health*",
                f"• Status: {trading_health.overall_status}",
                f"• Can Trade: {trading_health.can_trade}",
                f"• Summary: {trading_health.summary}",
                f"• Last Check: {trading_health.last_run_at.strftime('%H:%M:%S UTC')}",
            ]
        else:
            trading_lines = ["", "📈 *Trading Health*: _No data (Initial check may be in progress)_"]

        response = "\n".join(sup_lines + trading_lines)
        self.adapter.log_interaction(update.effective_user.id, "/diag", response)
        await self.send_redacted_reply(update, response)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        
        snapshot = status_store.get()
        if not snapshot or snapshot.get("status") == "starting":
            response = "⏳ *Trading Status*: Starting up or no snapshot available."
        else:
            # Extract key info from snapshot (structure from agent_loop.py)
            execution = snapshot.get("execution", {})
            status = execution.get("status", "unknown").upper()
            msg = execution.get("message", "")
            
            lines = [
                f"📊 *Trading Status: {status}*",
                f"📝 {msg}",
            ]
            
            # Optional portfolio summary
            state = snapshot.get("state", {})
            portfolio = state.get("portfolio", {})
            if portfolio:
                equity = portfolio.get("equity_usd", 0.0)
                margin = portfolio.get("margin_used_pct", 0.0)
                lines.append(f"💰 Equity: ${equity:,.2f} | Margin: {margin:.1f}%")
                
            # Latest decision
            final_action = snapshot.get("final_action", {})
            if final_action:
                action = final_action.get("action", "WAIT")
                lines.append(f"🤖 Latest Action: *{action}*")

            response = "\n".join(lines)

        self.adapter.log_interaction(update.effective_user.id, "/status", response)
        await self.send_redacted_reply(update, response)

    async def cmd_job(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        if not context.args:
            await update.message.reply_text("Usage: /job <job_id>")
            return
            
        job_id = context.args[0]
        response = self.adapter.get_job_summary(job_id)
        self.adapter.log_interaction(update.effective_user.id, f"/job {job_id}", response)
        await self.send_redacted_reply(update, response)

    async def cmd_why(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text("Usage: /why <pr_number>")
            return
            
        pr_number = int(context.args[0])
        response = self.adapter.get_latest_job_for_pr(pr_number)
        self.adapter.log_interaction(update.effective_user.id, f"/why {pr_number}", response)
        await self.send_redacted_reply(update, response)

    async def cmd_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_access(update): return
        response = self.adapter.get_queue_status()
        self.adapter.log_interaction(update.effective_user.id, "/queue", response)
        await self.send_redacted_reply(update, response)

    async def handle_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-command messages via OpenAI conversation logic."""
        if not self._check_access(update): return
        
        chat_id = update.effective_chat.id
        text = update.message.text
        
        print(f"[Telegram Debug] Chat message received: '{text}' from {chat_id}")
        
        if not trading_settings.openai_api_key:
            print("[Telegram Debug] OPENAI_API_KEY is missing in trading_settings")
            logger.warning("OpenAI chat attempted but OPENAI_API_KEY missing")
            return

        async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as http_client:
            try:
                # Use standard ChatCompletion API
                messages = []
                if trading_settings.telegram_bootstrap_context:
                    messages.append({"role": "system", "content": trading_settings.telegram_bootstrap_context})
                
                messages.append({"role": "user", "content": text})

                response = await http_client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {trading_settings.openai_api_key}"},
                    json={
                        "model": trading_settings.openai_model,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                
                response_text = ""
                choices = payload.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")

                if not response_text:
                    response_text = "Sorry, I could not generate a response."

                self.adapter.log_interaction(update.effective_user.id, f"chat: {text[:20]}", response_text)
                await self.send_redacted_reply(update, response_text)
            except Exception as exc:
                print(f"[Telegram Debug] OpenAI error: {exc}")
                logger.error(f"Error in OpenAI chat: {exc}")
                await update.message.reply_text("❌ Sorry, I encountered an error processing your request.")
