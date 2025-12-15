"""OpenAI LLM provider implementation."""

import json
from typing import Any

from openai import AsyncOpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API."""
    
    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
    
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from OpenAI."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    
    async def generate_json(
        self,
        prompt: str,
        model: str,
        schema_hint: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON response from OpenAI."""
        full_prompt = f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{schema_hint}"
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content or "{}"
        return self._parse_json(content)
    
    def _parse_json(self, content: str) -> dict:
        """Parse JSON from response, handling markdown code blocks."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        return json.loads(content)
    
    async def close(self) -> None:
        """Close the OpenAI client."""
        await self.client.close()
