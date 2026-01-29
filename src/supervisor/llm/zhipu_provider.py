"""Zhipu (GLM) LLM provider implementation (OpenAI-compatible)."""

from .openai_provider import OpenAIProvider


class ZhipuProvider(OpenAIProvider):
    """LLM provider using Zhipu (GLM) API via OpenAI compatibility."""
    
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        super().__init__(api_key=api_key, base_url=base_url)
