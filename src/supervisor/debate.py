"""3-agent debate system: Optimist, Skeptic, Arbiter."""

import json
from typing import Any, Optional

from openai import AsyncOpenAI

from .config import SupervisorSettings
from .models import ArbiterDecision, VerificationReport


class DebateSystem:
    """Runs Optimist/Skeptic/Arbiter debate to decide on auto-fix."""
    
    def __init__(self, settings: SupervisorSettings, client: Optional[AsyncOpenAI] = None):
        self.settings = settings
        self.client = client or AsyncOpenAI()
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
        
        optimist_response = await self._call_optimist(context)
        skeptic_response = await self._call_skeptic(context, optimist_response)
        arbiter_decision = await self._call_arbiter(context, optimist_response, skeptic_response)
        
        arbiter_decision.optimist_summary = optimist_response[:500]
        arbiter_decision.skeptic_summary = skeptic_response[:500]
        
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
            verification.failure_summary[:1500],
        ])
        
        return "\n".join(parts)[:self.max_context_chars]
    
    async def _call_optimist(self, context: str) -> str:
        """Call the Optimist agent."""
        prompt = f"""You are the OPTIMIST in a code review debate. Your role is to:
1. Argue why the code changes are likely correct
2. Propose minimal fixes for any failing tests
3. Be constructive and solution-oriented

Context:
{context}

Provide a brief analysis (max 300 words):
1. Why the changes are probably correct
2. Likely causes of test failures (if any)
3. Suggested minimal fixes

Be concise and actionable."""

        response = await self.client.chat.completions.create(
            model=self.settings.model_optimist,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        
        return response.choices[0].message.content or ""
    
    async def _call_skeptic(self, context: str, optimist_view: str) -> str:
        """Call the Skeptic agent."""
        prompt = f"""You are the SKEPTIC in a code review debate. Your role is to:
1. Find hidden risks and edge cases
2. Challenge the Optimist's assumptions
3. Identify why tests might be failing for good reasons

Context:
{context}

Optimist's view:
{optimist_view[:800]}

Provide a brief counter-analysis (max 300 words):
1. Potential risks the Optimist missed
2. Edge cases that could cause issues
3. Reasons why auto-fix might be dangerous

Be constructive but thorough in finding risks."""

        response = await self.client.chat.completions.create(
            model=self.settings.model_skeptic,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        
        return response.choices[0].message.content or ""
    
    async def _call_arbiter(
        self,
        context: str,
        optimist_view: str,
        skeptic_view: str,
    ) -> ArbiterDecision:
        """Call the Arbiter agent to make final decision."""
        prompt = f"""You are the ARBITER in a code review debate. Based on the Optimist and Skeptic views, make a decision.

Context:
{context}

Optimist's view:
{optimist_view[:600]}

Skeptic's view:
{skeptic_view[:600]}

Make a decision in JSON format:
{{
    "auto_fix_allowed": true/false,
    "fix_objectives": ["objective 1", "objective 2"],
    "risk_level": "low" | "medium" | "high",
    "stop_reason": "reason if auto_fix_allowed is false, else null",
    "reasoning": "brief explanation of your decision"
}}

Rules:
- Allow auto-fix only for low/medium risk with clear fix objectives
- Reject if: security concerns, complex refactoring needed, unclear failures
- Keep fix_objectives specific and actionable

Respond with only the JSON object."""

        response = await self.client.chat.completions.create(
            model=self.settings.model_arbiter,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        
        content = response.choices[0].message.content or "{}"
        
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            
            return ArbiterDecision(
                auto_fix_allowed=data.get("auto_fix_allowed", False),
                fix_objectives=data.get("fix_objectives", []),
                risk_level=data.get("risk_level", "unknown"),
                stop_reason=data.get("stop_reason"),
                arbiter_reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return ArbiterDecision(
                auto_fix_allowed=False,
                risk_level="unknown",
                stop_reason="Failed to parse Arbiter decision",
                arbiter_reasoning=content[:500],
            )
