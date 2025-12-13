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
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from agent.chat_controller import ChatController
from agent.chat_tools import open_file, search_repo
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
        self.chat_controller = ChatController()
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
        self.application.add_handler(CommandHandler("clear", self.cmd_clear))
        self.application.add_handler(CommandHandler("ask", self.cmd_ask))
        self.application.add_handler(CommandHandler("search", self.cmd_search))
        self.application.add_handler(CommandHandler("open", self.cmd_open))
        
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))
        
        return self.application
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        text = """👋 *Code Review Agent + Repo Q&A*

I review code changes and answer questions about the codebase.

*Code Review:*
/review — Review latest changes
/diff — Show diff summary
/risks — Show issues from last review
/next — Recommended actions
/status — System status

*Repo Q&A:*
/ask <question> — Ask about the codebase
/search <query> — Search for code
/open <path>:<lines> — View file excerpt

Or just type any question naturally!"""
        
        await update.message.reply_text(text, parse_mode="Markdown")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        text = """*Code Review Agent + Repo Q&A*

*Code Review Commands:*
/review — Analyze changes since last review
/diff — Summary of changed files
/risks — Detailed issues from last review
/next — Recommended next actions
/status — System status and last review info

*Repo Q&A Commands:*
/ask <question> — Ask questions about the codebase
/search <query> — Search for code patterns
/open <path>:<start>-<end> — View file excerpt
/clear — Clear chat session

*Natural Language:*
Just type any question! Examples:
• "Where is the Telegram bot created?"
• "How does search_repo work?"
• "Show me the config file"

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
    
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command - clears chat session."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        try:
            chat_id = str(update.effective_chat.id)
            self.chat_controller.clear_session(chat_id)
            await update.message.reply_text("🧹 Chat session cleared. Starting fresh!")
        except Exception as e:
            logger.error(f"Error in /clear: {e}")
            await update.message.reply_text(f"Error: {str(e)[:100]}")
    
    async def cmd_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ask <question> - forces Repo Q&A flow."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        question = " ".join(context.args) if context.args else ""
        if not question:
            await update.message.reply_text(
                "Usage: /ask <question>\n\n"
                "Examples:\n"
                "• /ask where is the Telegram bot created?\n"
                "• /ask how does search_repo work?\n"
                "• /ask what files handle authentication?"
            )
            return
        
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("🔍 Searching the codebase...")
        
        try:
            response = self.chat_controller.process_message(chat_id, question)
            
            if response.tools_used:
                tools_info = f"🔧 Used: {', '.join(response.tools_used)}\n\n"
            else:
                tools_info = ""
            
            reply_text = tools_info + response.text
            
            if len(reply_text) > 4000:
                reply_text = reply_text[:4000] + "\n\n... (truncated)"
            
            await update.message.reply_text(reply_text)
            
        except Exception as e:
            logger.error(f"Error in /ask: {e}")
            await update.message.reply_text(f"Error: {str(e)[:200]}")
    
    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search <query> - direct search without LLM routing."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text(
                "Usage: /search <query>\n\n"
                "Examples:\n"
                "• /search TelegramBot\n"
                "• /search def process_message\n"
                "• /search async def cmd_"
            )
            return
        
        await update.message.reply_text(f"🔍 Searching for: {query}...")
        
        try:
            result = search_repo(query, limit=15)
            
            if not result.success:
                await update.message.reply_text(f"Search failed: {result.output}")
                return
            
            output = f"🔎 Search results for '{query}':\n\n{result.output}"
            
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await update.message.reply_text(output)
            
        except Exception as e:
            logger.error(f"Error in /search: {e}")
            await update.message.reply_text(f"Error: {str(e)[:200]}")
    
    async def cmd_open(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /open <path>:<start>-<end> - open file excerpt."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        args = " ".join(context.args) if context.args else ""
        if not args:
            await update.message.reply_text(
                "Usage: /open <path>:<start>-<end>\n\n"
                "Examples:\n"
                "• /open agent/telegram_bot.py:1-50\n"
                "• /open src/config.py:100-150\n"
                "• /open agent/chat_tools.py (shows lines 1-50)"
            )
            return
        
        path = args
        start_line = 1
        end_line = 50
        
        if ":" in args:
            parts = args.rsplit(":", 1)
            path = parts[0]
            line_spec = parts[1]
            
            if "-" in line_spec:
                try:
                    start_str, end_str = line_spec.split("-", 1)
                    start_line = int(start_str) if start_str else 1
                    end_line = int(end_str) if end_str else start_line + 50
                except ValueError:
                    pass
            else:
                try:
                    start_line = int(line_spec)
                    end_line = start_line + 50
                except ValueError:
                    pass
        
        await update.message.reply_text(f"📄 Opening {path}...")
        
        try:
            result = open_file(path, start_line, end_line)
            
            if not result.success:
                await update.message.reply_text(f"Error: {result.output}")
                return
            
            output = result.output
            
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await update.message.reply_text(output)
            
        except Exception as e:
            logger.error(f"Error in /open: {e}")
            await update.message.reply_text(f"Error: {str(e)[:200]}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-command text messages via chat controller."""
        if not _is_authorized(update):
            await update.message.reply_text(_unauthorized_response())
            return
        
        user_message = update.message.text
        if not user_message:
            return
        
        chat_id = str(update.effective_chat.id)
        
        await update.message.reply_text("💭 Thinking...")
        
        try:
            response = self.chat_controller.process_message(chat_id, user_message)
            
            if response.tools_used:
                tools_info = f"🔧 Used: {', '.join(response.tools_used)}\n\n"
            else:
                tools_info = ""
            
            reply_text = tools_info + response.text
            
            if len(reply_text) > 4000:
                reply_text = reply_text[:4000] + "\n\n... (truncated)"
            
            await update.message.reply_text(reply_text)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(f"Error: {str(e)[:200]}")
    
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
