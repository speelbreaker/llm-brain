"""3-agent debate system with multi-provider support: Optimist, Skeptic, Arbiter."""

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError

from .config import SupervisorSettings
from .llm import get_provider_for_role, DebateResponse
from .models import ArbiterDecision, VerificationReport

logger = logging.getLogger(__name__)


DEBATE_SCHEMA = """{
  "role": "optimist|skeptic|arbiter",
  "summary": "string (max 300 chars)",
  "bullets": ["string", "string", "string"] (max 3 items),
  "auto_fix_allowed": true|false (arbiter only),
  "objectives": ["string", "string"] (arbiter only, max 5 items),
  "risk_level": "low|med|high" (arbiter only),
  "stop_reason": "string|null" (arbiter only, required if auto_fix_allowed=false)
}"""


class DebateSystem:
    """Runs Optimist/Skeptic/Arbiter debate with multi-provider support."""
    
    def __init__(self, settings: SupervisorSettings):
        self.settings = settings
        self.max_context_chars = 4000
    
    async def run_debate(
        self,
        verification: VerificationReport,
        changed_files: list[str],
        pr_title: str = "",
        pr_body: str = "",
    ) -> ArbiterDecision:
        """Run the 3-agent debate and return Arbiter's decision."""
        context = self._build_context(verification, changed_files, pr_title, pr_body)
        
        optimist_response = await self._call_agent("optimist", context)
        skeptic_response = await self._call_agent(
            "skeptic", 
            context, 
            optimist_view=optimist_response
        )
        arbiter_decision = await self._call_arbiter(
            context, 
            optimist_response, 
            skeptic_response
        )
        
        arbiter_decision.optimist_summary = optimist_response.get("summary", "")[:500]
        arbiter_decision.skeptic_summary = skeptic_response.get("summary", "")[:500]
        
        return arbiter_decision
    
    def _build_context(
        self,
        verification: VerificationReport,
        changed_files: list[str],
        pr_title: str,
        pr_body: str,
    ) -> str:
        """Build context string for debate agents."""
        parts = [
            f"PR Title: {pr_title}",
            f"PR Description: {pr_body[:500]}",
            "",
            f"Changed files ({len(changed_files)}):",
        ]
        
        for f in changed_files[:20]:
            parts.append(f"  - {f}")
        
        if len(changed_files) > 20:
            parts.append(f"  ... and {len(changed_files) - 20} more")
        
        parts.extend([
            "",
            "Verification Results:",
            f"  All passed: {verification.all_passed}",
            f"  Failing tests: {', '.join(verification.failing_tests[:5])}",
            "",
            "Failure Summary:",
            verification.failure_summary[:1500] if verification.failure_summary else "None",
        ])
        
        return "\n".join(parts)[:self.max_context_chars]
    
    async def _call_agent(
        self,
        role: Literal["optimist", "skeptic"],
        context: str,
        optimist_view: Optional[dict] = None,
    ) -> dict:
        """Call an agent (Optimist or Skeptic) with retry for JSON validation."""
        provider, model = get_provider_for_role(role, self.settings)
        
        if role == "optimist":
            prompt = self._build_optimist_prompt(context)
        else:
            prompt = self._build_skeptic_prompt(context, optimist_view or {})
        
        for attempt in range(2):
            try:
                result = await provider.generate_json(
                    prompt=prompt,
                    model=model,
                    schema_hint=DEBATE_SCHEMA,
                    max_tokens=600,
                    temperature=0.7 if attempt == 0 else 0.3,
                )

                # Providers may include "role" in the payload; avoid double-passing
                payload = dict(result or {})
                agent_role = payload.pop("role", role)
                response = DebateResponse(role=agent_role, **payload)
                return response.model_dump()
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"JSON validation failed for {role} (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON matching the schema."
                continue
        
        return {
            "role": role,
            "summary": "Failed to generate valid response",
            "bullets": [],
        }
    
    async def _call_arbiter(
        self,
        context: str,
        optimist_view: dict,
        skeptic_view: dict,
    ) -> ArbiterDecision:
        """Call the Arbiter agent to make final decision with retry."""
        provider, model = get_provider_for_role("arbiter", self.settings)
        prompt = self._build_arbiter_prompt(context, optimist_view, skeptic_view)
        
        for attempt in range(2):
            try:
                result = await provider.generate_json(
                    prompt=prompt,
                    model=model,
                    schema_hint=DEBATE_SCHEMA,
                    max_tokens=500,
                    temperature=0.3,
                )

                payload = dict(result or {})
                payload_role = payload.pop("role", "arbiter")
                response = DebateResponse(role=payload_role, **payload)

                return ArbiterDecision(
                    auto_fix_allowed=response.auto_fix_allowed or False,
                    fix_objectives=response.objectives or [],
                    risk_level=response.risk_level or "unknown",
                    stop_reason=response.stop_reason,
                    arbiter_reasoning=response.summary,
                )
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"JSON validation failed for arbiter (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON matching the schema."
                continue
        
        return ArbiterDecision(
            auto_fix_allowed=False,
            risk_level="unknown",
            stop_reason="debate_output_invalid",
            arbiter_reasoning="Failed to parse arbiter response",
        )
    
    def _build_optimist_prompt(self, context: str) -> str:
        """Build the Optimist agent prompt."""
        return f"""You are the OPTIMIST in a code review debate. Your role is to:
1. Argue why the code changes are likely correct
2. Propose minimal fixes for any failing tests
3. Be constructive and solution-oriented

Context:
{context}

Respond with JSON matching this schema:
{{
  "role": "optimist",
  "summary": "Brief analysis of why changes are correct (max 300 chars)",
  "bullets": ["point 1", "point 2", "point 3"]
}}

Be concise. Max 3 bullet points. Focus on solutions."""

    def _build_skeptic_prompt(self, context: str, optimist_view: dict) -> str:
        """Build the Skeptic agent prompt."""
        optimist_summary = optimist_view.get("summary", "")
        optimist_bullets = optimist_view.get("bullets", [])
        
        return f"""You are the SKEPTIC in a code review debate. Your role is to:
1. Find hidden risks and edge cases
2. Challenge the Optimist's assumptions
3. Identify why tests might be failing for good reasons

Context:
{context}

Optimist's view:
Summary: {optimist_summary}
Points: {', '.join(optimist_bullets[:3])}

Respond with JSON matching this schema:
{{
  "role": "skeptic",
  "summary": "Counter-analysis of risks (max 300 chars)",
  "bullets": ["risk 1", "risk 2", "risk 3"]
}}

Be thorough but concise. Max 3 bullet points."""

    def _build_arbiter_prompt(
        self,
        context: str,
        optimist_view: dict,
        skeptic_view: dict,
    ) -> str:
        """Build the Arbiter agent prompt."""
        return f"""You are the ARBITER in a code review debate. Make a decision based on both perspectives.

Context:
{context}

Optimist's view:
{optimist_view.get('summary', '')}
{', '.join(optimist_view.get('bullets', [])[:3])}

Skeptic's view:
{skeptic_view.get('summary', '')}
{', '.join(skeptic_view.get('bullets', [])[:3])}

Respond with JSON matching this schema:
{{
  "role": "arbiter",
  "summary": "Your decision reasoning (max 300 chars)",
  "bullets": ["key point 1", "key point 2"],
  "auto_fix_allowed": true or false,
  "objectives": ["fix objective 1", "fix objective 2"] (if auto_fix_allowed),
  "risk_level": "low" or "med" or "high",
  "stop_reason": "reason for denial" (required if auto_fix_allowed is false)
}}

Rules:
- Allow auto-fix only for low/medium risk with clear fix objectives
- Reject if: security concerns, complex refactoring, unclear failures
- Keep objectives specific and actionable"""
