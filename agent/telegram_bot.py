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
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from agent.chat_controller import ChatController
from agent.chat_tools import open_file, search_repo, run_pytest, run_health_checks, run_enhanced_security_scans
from agent.codex_cli_runner import get_codex_status, codex_exec, codex_via_api
from agent.config import settings
from agent.review_service import ReviewService
from agent.storage import init_db, get_recent_check_runs, save_check_run

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram Markdown."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


async def reply_safe(
    update: Update,
    text: str,
    context: ContextTypes.DEFAULT_TYPE | None = None,
    parse_mode: str | None = None,
):
    """Send a reply with automatic fallback for robustness.
    
    Handles:
    - Markdown parse errors -> falls back to plain text
    - Missing message object -> uses context.bot.send_message
    - Any BadRequest -> logs warning and retries without parse_mode
    
    Args:
        update: Telegram Update object
        text: Message text to send
        context: Optional context for fallback send_message
        parse_mode: Optional parse mode (Markdown, HTML, etc.)
    """
    chat_id = update.effective_chat.id if update.effective_chat else None
    message = update.effective_message
    
    async def _send_plain(chat_id_: int, text_: str) -> bool:
        """Last resort: send via context.bot.send_message without parse_mode."""
        if context and chat_id_:
            try:
                await context.bot.send_message(chat_id=chat_id_, text=text_)
                return True
            except Exception as e2:
                logger.error(f"reply_safe: context.bot.send_message failed: {e2}")
        return False
    
    try:
        if message:
            await message.reply_text(text, parse_mode=parse_mode)
        elif chat_id and context:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        else:
            logger.error("reply_safe: No message or chat_id available")
    except BadRequest as e:
        logger.warning(f"reply_safe: BadRequest with parse_mode={parse_mode}: {e}")
        try:
            if message:
                await message.reply_text(text, parse_mode=None)
            elif chat_id:
                await _send_plain(chat_id, text)
        except Exception as e2:
            logger.error(f"reply_safe: plain text fallback failed: {e2}")
            if chat_id:
                await _send_plain(chat_id, text[:500] + "... [truncated due to error]")
    except Exception as e:
        logger.error(f"reply_safe: Unexpected error: {e}")
        if chat_id:
            await _send_plain(chat_id, text[:500] + "... [error occurred]")


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
        self.application.add_handler(CommandHandler("smoke", self.cmd_smoke))
        self.application.add_handler(CommandHandler("security", self.cmd_security))
        self.application.add_handler(CommandHandler("health", self.cmd_health))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("codex", self.cmd_codex))
        self.application.add_handler(CommandHandler("codex_status", self.cmd_codex_status))
        
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))
        
        return self.application
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        text = """Code Review Agent + Auditor

I review code, run tests, scan for security issues, and answer questions.

Code Review:
/review - Review latest changes
/diff - Show diff summary
/risks - Show issues from last review
/next - Recommended actions

Auditor:
/smoke - Run pytest tests
/security - Security scans
/health - App health checks
/history - Check run history

Repo Q&A:
/ask <question> - Ask about code
/search <query> - Search code
/open <path> - View file

Codex CLI:
/codex <task> - Run Codex AI task
/codex_status - Check Codex CLI status

Or just type any question!"""
        
        await reply_safe(update, text, context)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        text = """Code Review Agent + Repo Q&A

Code Review Commands:
/review - Analyze changes since last review
/diff - Summary of changed files
/risks - Detailed issues from last review
/next - Recommended next actions
/status - System status and last review info

Auditor Commands:
/smoke [path] - Run pytest tests
/security - Run security scans (pip-audit, bandit, ruff)
/health - Run app health checks
/history [type] - Show recent check runs

Repo Q&A Commands:
/ask <question> - Ask questions about the codebase
/search <query> - Search for code patterns
/open <path>:<start>-<end> - View file excerpt
/clear - Clear chat session

Natural Language:
Just type any question! Examples:
- Where is the Telegram bot created?
- How does search_repo work?
- Show me the config file

Codex CLI Commands:
/codex <task> - Run Codex AI for coding tasks
/codex_status - Check Codex CLI installation status

Severity Levels:
CRITICAL - Must fix before deploy
HIGH - Strongly recommended
MEDIUM - Fix soon
LOW - Nice to have
INFO - Observations"""
        
        await reply_safe(update, text, context)
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            status = self.review_service.get_status()
            
            parts = ["Status", ""]
            
            last_review = status.get("last_review")
            if last_review and last_review.get("id"):
                severity = last_review.get("severity", "INFO")
                created = last_review.get("created_at", "unknown")[:16]
                target = last_review.get("target_ref", "unknown")
                parts.append(f"- Last review: {created} on {target} ({severity})")
            else:
                parts.append("- Last review: None yet")
            
            mode = status.get("change_detection_mode", "unknown")
            git_ok = "OK" if status.get("git_available") else "N/A"
            parts.append(f"- Change detection: {mode} (git: {git_ok})")
            
            parts.append(f"- Reviews stored: {status.get('review_count', 0)}")
            
            llm_ok = "OK" if status.get("llm_available") else "N/A"
            parts.append(f"- LLM backend: {llm_ok}")
            
            if status.get("current_head"):
                parts.append(f"- Current HEAD: {status['current_head']}")
            
            if status.get("last_reviewed_commit"):
                parts.append(f"- Last reviewed: {status['last_reviewed_commit']}")
            
            await reply_safe(update, "\n".join(parts), context)
            
        except Exception as e:
            logger.error(f"Error in /status: {e}")
            await reply_safe(update, f"Error getting status: {str(e)[:200]}", context)
    
    async def cmd_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /review command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        await reply_safe(update, "Analyzing changes...", context)
        
        try:
            user_id = update.effective_user.id
            result = self.review_service.review_latest_changes(user_id)
            
            if not result.has_changes:
                await reply_safe(update, "No new changes since last review.", context)
                return
            
            await reply_safe(update, result.summary_md, context)
            
        except Exception as e:
            logger.error(f"Error in /review: {e}")
            await reply_safe(update, f"Error during review: {str(e)[:200]}", context)
    
    async def cmd_diff(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /diff command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            summary = self.review_service.get_diff_summary_for_last_review()
            
            if not summary:
                await reply_safe(update, "No reviews yet. Run /review first.", context)
                return
            
            await reply_safe(update, summary, context)
            
        except Exception as e:
            logger.error(f"Error in /diff: {e}")
            await reply_safe(update, f"Error getting diff: {str(e)[:200]}", context)
    
    async def cmd_risks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risks command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            risks = self.review_service.get_risks_for_last_review()
            
            if not risks:
                await reply_safe(update, "No reviews yet. Run /review first.", context)
                return
            
            if len(risks) > 4000:
                risks = risks[:4000] + "\n...(truncated)"
            
            await reply_safe(update, risks, context)
            
        except Exception as e:
            logger.error(f"Error in /risks: {e}")
            await reply_safe(update, f"Error getting risks: {str(e)[:200]}", context)
    
    async def cmd_next(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /next command."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            actions = self.review_service.get_next_actions_for_last_review()
            
            if not actions:
                await reply_safe(update, "No reviews yet. Run /review first.", context)
                return
            
            await reply_safe(update, actions, context)
            
        except Exception as e:
            logger.error(f"Error in /next: {e}")
            await reply_safe(update, f"Error getting actions: {str(e)[:200]}", context)
    
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command - clears chat session."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            chat_id = str(update.effective_chat.id)
            self.chat_controller.clear_session(chat_id)
            await reply_safe(update, "Chat session cleared. Starting fresh!", context)
        except Exception as e:
            logger.error(f"Error in /clear: {e}")
            await reply_safe(update, f"Error: {str(e)[:100]}", context)
    
    async def cmd_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ask <question> - forces Repo Q&A flow."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        question = " ".join(context.args) if context.args else ""
        if not question:
            await reply_safe(update,
                "Usage: /ask <question>\n\n"
                "Examples:\n"
                "- /ask where is the Telegram bot created?\n"
                "- /ask how does search_repo work?\n"
                "- /ask what files handle authentication?",
                context
            )
            return
        
        chat_id = str(update.effective_chat.id)
        await reply_safe(update, "Searching the codebase...", context)
        
        try:
            response = self.chat_controller.process_message(chat_id, question)
            
            if response.tools_used:
                tools_info = f"Used: {', '.join(response.tools_used)}\n\n"
            else:
                tools_info = ""
            
            reply_text = tools_info + response.text
            
            if len(reply_text) > 4000:
                reply_text = reply_text[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, reply_text, context)
            
        except Exception as e:
            logger.error(f"Error in /ask: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search <query> - direct search without LLM routing."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        query = " ".join(context.args) if context.args else ""
        if not query:
            await reply_safe(update,
                "Usage: /search <query>\n\n"
                "Examples:\n"
                "- /search TelegramBot\n"
                "- /search def process_message\n"
                "- /search async def cmd_",
                context
            )
            return
        
        await reply_safe(update, f"Searching for: {query}...", context)
        
        try:
            result = search_repo(query, limit=15)
            
            if not result.success:
                await reply_safe(update, f"Search failed: {result.output}", context)
                return
            
            output = f"Search results for '{query}':\n\n{result.output}"
            
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, output, context)
            
        except Exception as e:
            logger.error(f"Error in /search: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def cmd_open(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /open <path>:<start>-<end> - open file excerpt."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        args = " ".join(context.args) if context.args else ""
        if not args:
            await reply_safe(update,
                "Usage: /open <path>:<start>-<end>\n\n"
                "Examples:\n"
                "- /open agent/telegram_bot.py:1-50\n"
                "- /open src/config.py:100-150\n"
                "- /open agent/chat_tools.py (shows lines 1-50)",
                context
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
        
        await reply_safe(update, f"Opening {path}...", context)
        
        try:
            result = open_file(path, start_line, end_line)
            
            if not result.success:
                await reply_safe(update, f"Error: {result.output}", context)
                return
            
            output = result.output
            
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, output, context)
            
        except Exception as e:
            logger.error(f"Error in /open: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def cmd_smoke(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /smoke command - run pytest tests."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        await reply_safe(update, "Running pytest tests...", context)
        
        try:
            import time
            start_time = time.time()
            
            test_path = " ".join(context.args) if context.args else None
            result = run_pytest(test_path=test_path)
            
            duration = time.time() - start_time
            
            save_check_run(
                check_type="pytest",
                status="passed" if result.success else "failed",
                duration_seconds=duration,
                summary=result.output[:500] if result.output else "No output",
            )
            
            output = result.output
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, output, context)
            
        except Exception as e:
            logger.error(f"Error in /smoke: {e}")
            await reply_safe(update, f"Error running tests: {str(e)[:200]}", context)
    
    async def cmd_security(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /security command - run comprehensive security scans."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        await reply_safe(update, "Running security scans (pip-audit, bandit, ruff)...", context)
        
        try:
            import time
            start_time = time.time()
            
            result = run_enhanced_security_scans()
            
            duration = time.time() - start_time
            
            save_check_run(
                check_type="security",
                status="passed" if result.success else "issues_found",
                duration_seconds=duration,
                summary=result.output[:500] if result.output else "No output",
            )
            
            output = result.output
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, output, context)
            
        except Exception as e:
            logger.error(f"Error in /security: {e}")
            await reply_safe(update, f"Error running security scan: {str(e)[:200]}", context)
    
    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /health command - run in-process app health checks."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        await reply_safe(update, "Running health checks...", context)
        
        try:
            import time
            start_time = time.time()
            
            result = run_health_checks()
            
            duration = time.time() - start_time
            
            save_check_run(
                check_type="health",
                status="passed" if result.success else "failed",
                duration_seconds=duration,
                summary=result.output[:500] if result.output else "No output",
            )
            
            output = result.output
            if len(output) > 4000:
                output = output[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, output, context)
            
        except Exception as e:
            logger.error(f"Error in /health: {e}")
            await reply_safe(update, f"Error running health checks: {str(e)[:200]}", context)
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command - show recent check runs."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            check_type = context.args[0] if context.args else None
            runs = get_recent_check_runs(limit=10, check_type=check_type)
            
            if not runs:
                await reply_safe(update, "No check runs recorded yet. Try /smoke, /security, or /health first.", context)
                return
            
            lines = ["Recent Check Runs", ""]
            
            for run in runs:
                status_icon = "[OK]" if run.status in ("passed", "success") else "[FAIL]" if run.status == "failed" else "[WARN]"
                created = run.created_at[:16].replace("T", " ")
                duration = f"{run.duration_seconds:.1f}s" if run.duration_seconds else "N/A"
                lines.append(f"{status_icon} {run.check_type} - {created} ({duration})")
            
            if check_type:
                lines.append(f"\nFiltered by: {check_type}")
            else:
                lines.append("\nUse /history <type> to filter (pytest, security, health)")
            
            await reply_safe(update, "\n".join(lines), context)
            
        except Exception as e:
            logger.error(f"Error in /history: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def cmd_codex_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /codex_status command - show Codex CLI installation status."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        try:
            status = get_codex_status()
            await reply_safe(update, status, context)
        except Exception as e:
            logger.error(f"Error in /codex_status: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def cmd_codex(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /codex command - run Codex AI task."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        if not context.args:
            await reply_safe(update,
                "Usage: /codex <task>\n\n"
                "Examples:\n"
                "- /codex explain this function\n"
                "- /codex write a test for UserService\n"
                "- /codex how do I add authentication?",
                context
            )
            return
        
        task = " ".join(context.args)
        await reply_safe(update, f"Running Codex: {task[:50]}...", context)
        
        try:
            result = await codex_via_api(task)
            
            if len(result) > 4000:
                result = result[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, result, context)
            
        except Exception as e:
            logger.error(f"Error in /codex: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-command text messages via chat controller."""
        if not _is_authorized(update):
            await reply_safe(update, _unauthorized_response(), context)
            return
        
        message = update.effective_message
        user_message = message.text if message else None
        if not user_message:
            return
        
        chat_id = str(update.effective_chat.id)
        
        await reply_safe(update, "Thinking...", context)
        
        try:
            response = self.chat_controller.process_message(chat_id, user_message)
            
            if response.tools_used:
                tools_info = f"Used: {', '.join(response.tools_used)}\n\n"
            else:
                tools_info = ""
            
            reply_text = tools_info + response.text
            
            if len(reply_text) > 4000:
                reply_text = reply_text[:4000] + "\n\n... (truncated)"
            
            await reply_safe(update, reply_text, context)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await reply_safe(update, f"Error: {str(e)[:200]}", context)
    
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
