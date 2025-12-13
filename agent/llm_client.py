"""
LLM client module for the Telegram Code Review Agent.

Builds prompts and calls the OpenAI API for code review.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.analyzers import AnalysisContext
from agent.config import settings

logger = logging.getLogger(__name__)


REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer for a Python/FastAPI project. Your job is to review code changes and identify issues, risks, and improvements.

## Severity Levels
- CRITICAL: Must fix before deploy (security vulnerabilities, crashes, secrets in code)
- HIGH: Strongly recommended to fix (correctness issues, unhandled exceptions)
- MEDIUM: Fix soon (edge cases, performance issues, fragile assumptions)
- LOW: Nice to have (code smells, naming issues, minor duplication)
- INFO: Observations and suggestions

## Review Checklist
1. Summary: What changed in 3-8 bullet points
2. Intent vs Implementation: Does the change match its apparent purpose?
3. Correctness: Missing error handling, null checks, boundary issues
4. Security: Hardcoded credentials, weak auth, SQL injection, exposed logs
5. Reliability: Unhandled exceptions, missing retries/timeouts
6. Performance: Hot paths, unnecessary queries, loops over large data
7. Test coverage: What tests should be added

## Response Format
Return valid JSON with this structure:
{
  "summary": ["bullet point 1", "bullet point 2", ...],
  "overall_severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "issues": [
    {
      "id": "ISSUE-1",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "category": "security|correctness|reliability|performance|maintainability",
      "title": "Short title",
      "description": "Detailed description",
      "file": "path/to/file.py",
      "suggested_fix": "How to fix this"
    }
  ],
  "next_steps": ["Action 1", "Action 2", ...],
  "reasoning_summary": "A concise 2-3 sentence summary of your reasoning process and key insights."
}

Be specific and actionable. Reference exact files and line numbers when possible."""


MODEL_FALLBACK_CHAIN = [
    "gpt-5.2-pro",
    "gpt-5.2",
    "gpt-5.2-thinking",
    "o3",
    "o1",
    "gpt-4.1",
    "gpt-4o",
]


@dataclass
class LLMReviewResult:
    """Structured result from LLM review."""
    summary: List[str] = field(default_factory=list)
    overall_severity: str = "INFO"
    issues: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    model_used: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    
    @classmethod
    def from_json(cls, data: Dict[str, Any], model: str = "") -> "LLMReviewResult":
        return cls(
            summary=data.get("summary", []),
            overall_severity=data.get("overall_severity", "INFO"),
            issues=data.get("issues", []),
            next_steps=data.get("next_steps", []),
            reasoning_summary=data.get("reasoning_summary", ""),
            model_used=model,
        )
    
    @classmethod
    def error_result(cls, error: str) -> "LLMReviewResult":
        return cls(
            summary=["Error during review"],
            overall_severity="INFO",
            error=error,
        )


def _truncate_diff(diff_text: str, max_chars: int = 50000) -> str:
    """Truncate diff text to fit within limits."""
    if len(diff_text) <= max_chars:
        return diff_text
    
    half = max_chars // 2
    return (
        diff_text[:half] +
        "\n\n... [DIFF TRUNCATED FOR LENGTH] ...\n\n" +
        diff_text[-half:]
    )


def _build_review_prompt(
    analysis: AnalysisContext,
    diff_text: str,
    commit_message: Optional[str] = None,
) -> str:
    """Build the user prompt for code review."""
    parts = []
    
    parts.append("## Files Changed")
    for fs in analysis.file_summaries:
        parts.append(f"- {fs.path} ({fs.category}): {fs.description}")
    
    if analysis.config_changes:
        parts.append("\n## Config Changes")
        for cc in analysis.config_changes:
            parts.append(f"- {cc['path']}: {cc['status']}")
    
    if analysis.suspicious_patterns:
        parts.append("\n## Suspicious Patterns Detected")
        for sp in analysis.suspicious_patterns:
            parts.append(f"- [{sp['severity']}] {sp['pattern']}")
    
    if commit_message:
        parts.append(f"\n## Commit Message\n{commit_message}")
    
    if diff_text:
        truncated = _truncate_diff(diff_text)
        parts.append(f"\n## Diff\n```diff\n{truncated}\n```")
    
    parts.append("\n## Task\nReview these changes and return a JSON response following the format specified.")
    
    return "\n".join(parts)


class LLMClient:
    """Client for LLM-powered code review."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or (settings.openai_api_key if settings else None)
        self._client = None
        self._model_review = settings.openai_model_review if settings else "gpt-5.2-pro"
        self._model_fast = settings.openai_model_fast if settings else "gpt-5.2"
        self._reasoning_effort = settings.openai_reasoning_effort if settings else "high"
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import os
                from openai import OpenAI
                base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=base_url,
                )
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client
    
    def _get_fallback_models(self) -> List[str]:
        """Get ordered list of models to try."""
        models = [self._model_review]
        for m in MODEL_FALLBACK_CHAIN:
            if m not in models:
                models.append(m)
        return models
    
    def is_available(self) -> bool:
        """Check if LLM backend is reachable."""
        if not self.api_key:
            return False
        
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False
    
    def _call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000,
    ) -> tuple[str, str]:
        """Call API with model fallback chain. Returns (content, model_used)."""
        client = self._get_client()
        models = self._get_fallback_models()
        last_error = None
        
        for model in models:
            try:
                logger.info(f"Trying model: {model}")
                
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                
                if model.startswith(("o1", "o3", "gpt-5")):
                    kwargs["reasoning_effort"] = self._reasoning_effort
                else:
                    kwargs["temperature"] = 0.3
                
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.info(f"Success with model: {model}")
                return content, model
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
                
                if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg or "invalid" in error_msg):
                    continue
                elif "rate" in error_msg or "quota" in error_msg:
                    continue
                else:
                    raise
        
        raise last_error or RuntimeError("All models failed")
    
    def review_changes(
        self,
        analysis: AnalysisContext,
        diff_text: str = "",
        commit_message: Optional[str] = None,
    ) -> LLMReviewResult:
        """Review code changes using LLM with smartest model and max reasoning."""
        if not self.api_key:
            return self._fallback_review(analysis)
        
        try:
            user_prompt = _build_review_prompt(analysis, diff_text, commit_message)
            
            messages = [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            content, model_used = self._call_with_fallback(messages)
            
            json_match = None
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    json_match = content[start:end].strip()
            elif content.strip().startswith("{"):
                json_match = content.strip()
            
            if json_match:
                data = json.loads(json_match)
                result = LLMReviewResult.from_json(data, model=model_used)
                result.raw_response = content
                return result
            else:
                return LLMReviewResult(
                    summary=["Review completed but response format was unexpected"],
                    overall_severity="INFO",
                    model_used=model_used,
                    raw_response=content,
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_review(analysis)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return LLMReviewResult.error_result(str(e))
    
    def _fallback_review(self, analysis: AnalysisContext) -> LLMReviewResult:
        """Generate a basic review without LLM."""
        summary = []
        issues = []
        next_steps = []
        overall_severity = "INFO"
        
        file_count = len(analysis.file_summaries)
        summary.append(f"Changed {file_count} file(s)")
        
        for category, files in analysis.categories.items():
            summary.append(f"- {category}: {len(files)} file(s)")
        
        for sp in analysis.suspicious_patterns:
            issues.append({
                "id": f"AUTO-{len(issues)+1}",
                "severity": sp["severity"],
                "category": "security",
                "title": sp["pattern"],
                "description": f"Detected: {sp['match']}",
                "suggested_fix": "Review and ensure this is intentional",
            })
            if sp["severity"] == "CRITICAL":
                overall_severity = "CRITICAL"
            elif sp["severity"] == "HIGH" and overall_severity not in ["CRITICAL"]:
                overall_severity = "HIGH"
        
        if analysis.config_changes:
            next_steps.append("Review configuration changes for correctness")
        
        if "tests" not in analysis.categories and any(
            cat in analysis.categories for cat in ["api", "models", "database"]
        ):
            next_steps.append("Consider adding tests for the new/modified code")
            issues.append({
                "id": f"AUTO-{len(issues)+1}",
                "severity": "MEDIUM",
                "category": "maintainability",
                "title": "No test changes detected",
                "description": "Code changes were made but no test files were modified",
                "suggested_fix": "Add or update tests to cover the changes",
            })
            if overall_severity == "INFO":
                overall_severity = "MEDIUM"
        
        return LLMReviewResult(
            summary=summary,
            overall_severity=overall_severity,
            issues=issues,
            next_steps=next_steps,
        )
