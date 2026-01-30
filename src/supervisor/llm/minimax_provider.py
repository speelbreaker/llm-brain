"""MiniMax LLM provider implementation (OpenAI-compatible)."""

from .openai_provider import OpenAIProvider


class MinimaxProvider(OpenAIProvider):
    """LLM provider using MiniMax API via OpenAI compatibility."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.minimax.chat/v1"):
        super().__init__(api_key=api_key, base_url=base_url)
