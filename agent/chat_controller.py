"""
Chat controller for the Telegram Code Review Agent.

Implements LLM-driven intent routing and tool execution for natural language chat.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.chat_tools import AVAILABLE_TOOLS, execute_tool, get_tools_for_openai
from agent.config import settings
from agent.storage import (
    clear_chat_session,
    get_chat_session,
    save_chat_session,
)

logger = logging.getLogger(__name__)


CHAT_SYSTEM_PROMPT = """You are a Repo Q&A assistant for a Python trading application. You help developers understand the codebase, find code, explain implementations, and answer architecture questions—all grounded in real code.

## Core Approach
1. **Search first, then open**: Use search_repo to find relevant code, then open_file to show details
2. **Quote minimal snippets**: Show only 5-30 relevant lines with file:line references
3. **Explain clearly**: Describe what the code does in plain English
4. **Offer follow-ups**: "Want me to show the next function?" instead of dumping entire files

## Repo Q&A Routing
- "Where is X implemented?" → search_repo for X, then open_file for best hit
- "Show me the code for Y" → search_repo + open_file
- "Why does Z happen?" → search_repo for related code, explain the flow
- "How does A work?" → search_repo + open_file for key parts
- "Find B" or "Search for B" → search_repo
- "What files are in C?" → list_files

## Available Tools
- **search_repo**: Find code patterns (case-insensitive). Use first for most questions.
- **open_file**: Read file sections (max 200 lines). Use after search to show details.
- **list_files**: Explore directory structure. Use before searching if unsure where to look.
- **get_project_map**: High-level project structure overview.
- **get_status**: Git state, review count, system health.
- **get_latest_diff**: Recent code changes.
- **run_security_scans**: Check for security issues.
- **run_smoke_tests**: Run basic tests.
- **tail_logs**: View recent application logs.

## Response Format
When showing code:
```
📄 path/to/file.py (lines 45-60)
──────────────────────────
[code snippet with line numbers]
──────────────────────────
This function does X by calling Y. The key part is line 52 where...
```

## Guardrails
- Never expose secrets (outputs are auto-redacted)
- Max 200 lines per open_file call
- Keep responses under 2000 chars for Telegram
- If no results, suggest alternative search terms
- Never execute code-modifying actions"""


@dataclass
class ChatResponse:
    """Response from chat processing."""
    text: str
    tools_used: List[str]
    needs_confirmation: bool = False
    confirmation_action: Optional[str] = None


class ChatController:
    """Controller for natural language chat interactions."""
    
    def __init__(self):
        self.api_key = settings.openai_api_key if settings else None
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client
    
    def is_available(self) -> bool:
        """Check if chat is available."""
        return bool(self.api_key)
    
    def process_message(self, chat_id: str, user_message: str) -> ChatResponse:
        """Process a natural language message and return a response."""
        if not self.api_key:
            return ChatResponse(
                text="Chat mode requires an OpenAI API key. Use /status to check configuration.",
                tools_used=[],
            )
        
        session = get_chat_session(chat_id)
        messages = session.messages if session else []
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._call_with_tools(messages)
            
            messages.append({"role": "assistant", "content": response.text})
            save_chat_session(chat_id, messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return ChatResponse(
                text=f"Sorry, I encountered an error: {str(e)[:100]}",
                tools_used=[],
            )
    
    def _call_with_tools(self, messages: List[Dict[str, str]]) -> ChatResponse:
        """Call OpenAI with tool support and execute any tool calls."""
        client = self._get_client()
        tools = get_tools_for_openai()
        tools_used = []
        
        full_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + messages[-10:]
        
        model = settings.openai_model_fast if settings else "gpt-4o"
        
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=1500,
            temperature=0.7,
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tools_used.append(tool_name)
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                result = execute_tool(tool_name, arguments)
                
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": result.output[:3000],
                })
            
            followup_messages = full_messages + [
                {"role": "assistant", "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]},
            ] + tool_results
            
            followup_response = client.chat.completions.create(
                model=model,
                messages=followup_messages,
                max_tokens=1500,
                temperature=0.7,
            )
            
            final_text = followup_response.choices[0].message.content or ""
        else:
            final_text = message.content or ""
        
        if len(final_text) > 3500:
            final_text = final_text[:3500] + "\n\n... (truncated, ask for more details)"
        
        return ChatResponse(
            text=final_text,
            tools_used=tools_used,
        )
    
    def clear_session(self, chat_id: str) -> None:
        """Clear chat session history."""
        clear_chat_session(chat_id)
    
    def get_session_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a chat session."""
        session = get_chat_session(chat_id)
        if not session:
            return {"exists": False, "message_count": 0}
        
        return {
            "exists": True,
            "message_count": len(session.messages),
            "updated_at": session.updated_at,
        }
