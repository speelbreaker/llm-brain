"""
LLM client module for the Telegram Code Review Agent.

Builds prompts and calls the OpenAI API for code review.
Supports both Chat Completions and Responses API (for gpt-5.2-pro).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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

## CRITICAL OUTPUT RULES
1. Return ONLY a single JSON object - no markdown, no code fences, no extra text
2. The response MUST start with { and end with }
3. Do NOT wrap in ```json or any code blocks
4. ALL fields below are REQUIRED

## Required JSON Structure
{
  "overall_severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "summary": ["bullet point 1", "bullet point 2"],
  "risks": [
    {
      "id": "RISK-1",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "category": "security|correctness|reliability|performance|maintainability",
      "title": "Short title",
      "description": "Detailed description",
      "file": "path/to/file.py",
      "suggested_fix": "How to fix this"
    }
  ],
  "next_actions": ["Action 1", "Action 2"],
  "reasoning_summary": "A concise 2-3 sentence summary of your reasoning process."
}

Be specific and actionable. Reference exact files and line numbers when possible.
The summary array MUST have at least 1 item. risks and next_actions can be empty arrays."""


MODEL_FALLBACK_CHAIN = [
    "gpt-5.2",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
]

RESPONSES_API_MODELS = {"gpt-5.2-pro", "o3", "o3-pro"}

ARTIFACTS_DIR = Path(".auditor/artifacts")


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
        summary = data.get("summary", [])
        if not summary:
            summary = ["No summary provided by model"]
        
        issues = data.get("issues", []) or data.get("risks", [])
        next_steps = data.get("next_steps", []) or data.get("next_actions", [])
        
        return cls(
            summary=summary,
            overall_severity=data.get("overall_severity", "INFO"),
            issues=issues,
            next_steps=next_steps,
            reasoning_summary=data.get("reasoning_summary", ""),
            model_used=model,
        )
    
    @classmethod
    def error_result(cls, error: str, raw_response: str = "") -> "LLMReviewResult":
        return cls(
            summary=[f"Review error: {error[:200]}"] if error else ["Error during review"],
            overall_severity="INFO",
            error=error,
            raw_response=raw_response,
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
    
    parts.append("\n## Task\nReview these changes. Return ONLY a valid JSON object (no code fences, no markdown). Start with { and end with }.")
    
    return "\n".join(parts)


def _save_artifact(run_id: str, filename: str, content: str) -> Path:
    """Save raw output artifact for debugging."""
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = artifact_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"Saved artifact: {filepath}")
    return filepath


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response using multiple strategies."""
    if not content or not content.strip():
        return None
    
    content = content.strip()
    
    if content.startswith("{") and content.endswith("}"):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            try:
                return json.loads(content[start:end].strip())
            except json.JSONDecodeError:
                pass
    
    if "```" in content:
        import re
        blocks = re.findall(r'```\s*\n?(.*?)```', content, re.DOTALL)
        for block in blocks:
            block = block.strip()
            if block.startswith("{") and block.endswith("}"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return None


class LLMClient:
    """Client for LLM-powered code review."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or (settings.openai_api_key if settings else None)
        self._client = None
        self._model_review = os.environ.get("OPENAI_MODEL_REVIEW") or (
            settings.openai_model_review if settings else "gpt-5.2-pro"
        )
        self._model_fast = os.environ.get("OPENAI_MODEL_FAST") or (
            settings.openai_model_fast if settings else "gpt-4.1"
        )
        self._reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT") or (
            settings.openai_reasoning_effort if settings else "high"
        )
        self._fallback_models = self._parse_fallbacks()
    
    def _parse_fallbacks(self) -> List[str]:
        """Parse fallback models from env or use defaults."""
        fallbacks_env = os.environ.get("OPENAI_MODEL_FALLBACKS", "")
        if fallbacks_env:
            return [m.strip() for m in fallbacks_env.split(",") if m.strip()]
        return MODEL_FALLBACK_CHAIN
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
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
        for m in self._fallback_models:
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
    
    def _call_responses_api(
        self,
        client,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
    ) -> str:
        """Call OpenAI Responses API for models like gpt-5.2-pro."""
        logger.info(f"Using Responses API for model: {model}")
        
        response = client.responses.create(
            model=model,
            input=messages,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_tokens,
        )
        
        content = response.output_text or ""
        return content
    
    def _call_chat_completions_api(
        self,
        client,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
    ) -> str:
        """Call OpenAI Chat Completions API."""
        logger.info(f"Using Chat Completions API for model: {model}")
        
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        
        if model.startswith(("o1", "o3", "gpt-5")):
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = 0.3
        
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content
    
    def _call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        retry_on_empty: bool = True,
    ) -> tuple[str, str]:
        """Call API with model fallback chain. Returns (content, model_used)."""
        client = self._get_client()
        models = self._get_fallback_models()
        last_error = None
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        
        for model in models:
            try:
                logger.info(f"Trying model: {model}")
                
                use_responses_api = model in RESPONSES_API_MODELS
                
                if use_responses_api and base_url:
                    logger.warning(
                        f"Custom base_url set ({base_url}), Responses API may not be supported. "
                        f"Falling back to Chat Completions for {model}."
                    )
                    use_responses_api = False
                
                if use_responses_api:
                    content = self._call_responses_api(
                        client, model, messages, max_tokens, self._reasoning_effort
                    )
                else:
                    content = self._call_chat_completions_api(
                        client, model, messages, max_tokens, self._reasoning_effort
                    )
                
                if not content.strip() and retry_on_empty:
                    logger.warning(f"Empty response from {model}, retrying with higher tokens...")
                    retry_tokens = min(max_tokens * 2, 20000)
                    retry_effort = "high" if self._reasoning_effort == "xhigh" else self._reasoning_effort
                    
                    if use_responses_api:
                        content = self._call_responses_api(
                            client, model, messages, retry_tokens, retry_effort
                        )
                    else:
                        content = self._call_chat_completions_api(
                            client, model, messages, retry_tokens, retry_effort
                        )
                    
                    if not content.strip():
                        logger.warning(f"Still empty after retry, trying next model...")
                        continue
                
                logger.info(f"Success with model: {model} (length={len(content)})")
                return content, model
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
                
                if any(kw in error_msg for kw in [
                    "model", "not found", "does not exist", "invalid", 
                    "unknown", "unsupported", "rate", "quota", "unavailable",
                    "responses", "not supported"
                ]):
                    continue
                else:
                    continue
        
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
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        
        try:
            user_prompt = _build_review_prompt(analysis, diff_text, commit_message)
            
            messages = [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            content, model_used = self._call_with_fallback(messages, max_tokens=12000)
            
            if not content.strip():
                _save_artifact(run_id, "llm_empty.txt", f"Model: {model_used}\nResponse was empty")
                return LLMReviewResult(
                    summary=["Model returned empty response - no issues detected or review failed"],
                    overall_severity="INFO",
                    model_used=model_used,
                    raw_response="",
                    error="Empty response from model after retry",
                )
            
            data = _extract_json(content)
            
            if data:
                result = LLMReviewResult.from_json(data, model=model_used)
                result.raw_response = content
                
                if not result.summary:
                    result.summary = ["Review completed (no summary provided)"]
                
                return result
            
            _save_artifact(run_id, "llm_raw.txt", content)
            logger.warning(f"Could not extract JSON from LLM response (length={len(content)})")
            
            preview = content[:500] if len(content) > 500 else content
            return LLMReviewResult(
                summary=[f"Review output (non-JSON): {preview}"],
                overall_severity="INFO",
                model_used=model_used,
                raw_response=content,
                error="Failed to parse JSON from model response",
            )
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_review(analysis)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            _save_artifact(run_id, "llm_error.txt", str(e))
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
        
        if not summary:
            summary = ["No changes detected"]
        
        return LLMReviewResult(
            summary=summary,
            overall_severity=overall_severity,
            issues=issues,
            next_steps=next_steps,
        )
