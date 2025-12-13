"""
Telegram bot for the Code Review Agent.

Handles commands and delegates to ReviewService.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from agent.config import settings
from agent.review_service import ReviewService
from agent.storage import init_db

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram Markdown."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def _is_authorized(update: Update) -> bool:
    """Check if the user is authorized."""
    if not settings:
        return False
    user = update.effective_user
    if not user:
        return False
    return settings.is_user_allowed(user.id)


def _unauthorized_response() -> str:
    """Response for unauthorized users."""
    return "⛔ Unauthorized. This bot is private."


class TelegramBot:
    """Telegram bot for code review."""
    
    def __init__(self):
        if not settings or not settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        
        self.review_service = ReviewService()
        self.application: Optional[Application] = None
    
    def build_application(self) -> Application:
        """Build the Telegram application with handlers."""
        self.application = (
            Application.builder()
            .token(settings.telegram_bot_token)
            .build()
        )
        
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("review", self.cmd_review))
        self.application.add_handler(CommandHandler("diff", self.cmd_diff))
        self.application.add_handler(CommandHandler("risks", self.cmd_risks))
        self.application.add_handler(CommandHandler("next", self.cmd_next))
        
        return self.application
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        text = """👋 *Code Review Agent*

I review code changes and report correctness, risks, and next steps.

*Commands:*
/status — Show current status
/review — Review latest changes
/diff — Show diff summary for last review
/risks — Show risks from last review
/next — Show recommended actions

Start with /review to analyze recent changes!"""
        
        await update.message.reply_text(text, parse_mode="Markdown")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        text = """*Code Review Agent Commands*

/start — Welcome message
/help — This help message
/status — System status and last review info
/review — Analyze changes since last review
/diff — Summary of changed files
/risks — Detailed issues from last review
/next — Recommended next actions

*How it works:*
1. I detect changes using git
2. I analyze the diff for patterns
3. I use AI to review the code
4. I report issues with severity levels

*Severity Levels:*
🔴 CRITICAL — Must fix before deploy
🟠 HIGH — Strongly recommended
🟡 MEDIUM — Fix soon
🟢 LOW — Nice to have
ℹ️ INFO — Observations"""
        
        await update.message.reply_text(text, parse_mode="Markdown")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        try:
            status = self.review_service.get_status()
            
            parts = ["*Status*", ""]
            
            last_review = status.get("last_review")
            if last_review and last_review.get("id"):
                severity = last_review.get("severity", "INFO")
                created = last_review.get("created_at", "unknown")[:16]
                target = last_review.get("target_ref", "unknown")
                parts.append(f"• Last review: {created} on `{target}` ({severity})")
            else:
                parts.append("• Last review: None yet")
            
            mode = status.get("change_detection_mode", "unknown")
            git_ok = "✅" if status.get("git_available") else "❌"
            parts.append(f"• Change detection: {mode} (git: {git_ok})")
            
            parts.append(f"• Reviews stored: {status.get('review_count', 0)}")
            
            llm_ok = "✅" if status.get("llm_available") else "❌"
            parts.append(f"• LLM backend: {llm_ok}")
            
            if status.get("current_head"):
                parts.append(f"• Current HEAD: `{status['current_head']}`")
            
            if status.get("last_reviewed_commit"):
                parts.append(f"• Last reviewed: `{status['last_reviewed_commit']}`")
            
            await update.message.reply_text("\n".join(parts), parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /status: {e}")
            await update.message.reply_text(f"Error getting status: {str(e)[:200]}")
    
    async def cmd_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /review command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        await update.message.reply_text("🔍 Analyzing changes...")
        
        try:
            user_id = update.effective_user.id
            result = self.review_service.review_latest_changes(user_id)
            
            if not result.has_changes:
                await update.message.reply_text("✅ No new changes since last review.")
                return
            
            await update.message.reply_text(result.summary_md, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /review: {e}")
            await update.message.reply_text(f"Error during review: {str(e)[:200]}")
    
    async def cmd_diff(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /diff command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        try:
            summary = self.review_service.get_diff_summary_for_last_review()
            
            if not summary:
                await update.message.reply_text("No reviews yet. Run /review first.")
                return
            
            await update.message.reply_text(summary, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /diff: {e}")
            await update.message.reply_text(f"Error getting diff: {str(e)[:200]}")
    
    async def cmd_risks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risks command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        try:
            risks = self.review_service.get_risks_for_last_review()
            
            if not risks:
                await update.message.reply_text("No reviews yet. Run /review first.")
                return
            
            if len(risks) > 4000:
                risks = risks[:4000] + "\n...(truncated)"
            
            await update.message.reply_text(risks, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /risks: {e}")
            await update.message.reply_text(f"Error getting risks: {str(e)[:200]}")
    
    async def cmd_next(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /next command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        try:
            actions = self.review_service.get_next_actions_for_last_review()
            
            if not actions:
                await update.message.reply_text("No reviews yet. Run /review first.")
                return
            
            await update.message.reply_text(actions, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /next: {e}")
            await update.message.reply_text(f"Error getting actions: {str(e)[:200]}")
    
    def run_polling(self) -> None:
        """Run the bot with polling (blocking)."""
        init_db()
        app = self.build_application()
        logger.info("Starting Telegram bot with polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def start_polling_async(self) -> None:
        """Start polling in async context."""
        init_db()
        app = self.build_application()
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram bot polling started")
    
    async def stop_async(self) -> None:
        """Stop the bot."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()


def run_bot() -> None:
    """Entry point to run the bot standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not settings:
        print("ERROR: TELEGRAM_BOT_TOKEN not configured")
        return
    
    bot = TelegramBot()
    bot.run_polling()


if __name__ == "__main__":
    run_bot()
