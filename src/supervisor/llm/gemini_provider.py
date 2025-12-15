"""Gemini LLM provider implementation."""

import json
from typing import Any

import httpx

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini API."""
    
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from Gemini."""
        client = await self._get_client()
        
        url = f"{self.base_url}/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return self._extract_text(data)
    
    async def generate_json(
        self,
        prompt: str,
        model: str,
        schema_hint: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON response from Gemini."""
        full_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond with ONLY valid JSON matching this schema:\n{schema_hint}\n"
            f"Do not include any text before or after the JSON object."
        )
        
        client = await self._get_client()
        url = f"{self.base_url}/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "responseMimeType": "application/json",
            }
        }
        
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        text = self._extract_text(data)
        return self._parse_json(text)
    
    def _extract_text(self, data: dict) -> str:
        """Extract text from Gemini response."""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "")
        except (KeyError, IndexError):
            return ""
    
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
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
