"""
Review orchestrator for the Telegram Code Review Agent.

Coordinates change detection, analysis, LLM review, and storage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.analyzers import AnalysisContext, FileSummary, analyze_changes
from agent.change_detector import ChangeDetector, ChangeResult
from agent.llm_client import LLMClient, LLMReviewResult
from agent.storage import (
    ReviewRecord,
    get_last_review,
    get_meta,
    get_review_count,
    save_review,
    set_meta,
)


def _escape_markdown(text: str) -> str:
    """Escape special characters for Telegram Markdown."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


@dataclass
class ReviewResult:
    """Complete result of a code review."""
    review_id: int
    summary_md: str
    overall_severity: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    diff_summary: List[Dict[str, Any]] = field(default_factory=list)
    change_result: Optional[ChangeResult] = None
    has_changes: bool = True
    error: Optional[str] = None


class ReviewService:
    """Orchestrates the code review process."""
    
    def __init__(self):
        self.change_detector = ChangeDetector()
        self.llm_client = LLMClient()
    
    def review_latest_changes(self, initiator_id: int) -> ReviewResult:
        """Review changes since the last reviewed commit."""
        change_result = self.change_detector.get_recent_changes_since_last_review()
        
        if change_result.error and not change_result.has_changes:
            return ReviewResult(
                review_id=0,
                summary_md=f"Error detecting changes: {change_result.error}",
                overall_severity="INFO",
                has_changes=False,
                error=change_result.error,
            )
        
        if not change_result.has_changes:
            return ReviewResult(
                review_id=0,
                summary_md="No new changes since the last review.",
                overall_severity="INFO",
                has_changes=False,
            )
        
        analysis = analyze_changes(
            change_result.changed_files,
            change_result.diff_text,
        )
        
        llm_result = self.llm_client.review_changes(
            analysis=analysis,
            diff_text=change_result.diff_text,
        )
        
        summary_md = self._build_summary_md(llm_result, change_result)
        diff_summary = [fs.to_dict() for fs in analysis.file_summaries]
        
        review_id = save_review(
            initiator_id=initiator_id,
            target_type="latest",
            target_ref=change_result.to_ref or "HEAD",
            git_head=self.change_detector.get_current_head(),
            change_detector_mode=change_result.mode,
            overall_severity=llm_result.overall_severity,
            summary_md=summary_md,
            issues=llm_result.issues,
            next_steps=llm_result.next_steps,
            diff_summary=diff_summary,
        )
        
        self.change_detector.mark_reviewed(change_result.to_ref)
        
        return ReviewResult(
            review_id=review_id,
            summary_md=summary_md,
            overall_severity=llm_result.overall_severity,
            issues=llm_result.issues,
            next_steps=llm_result.next_steps,
            diff_summary=diff_summary,
            change_result=change_result,
            has_changes=True,
        )
    
    def _build_summary_md(
        self,
        llm_result: LLMReviewResult,
        change_result: ChangeResult,
    ) -> str:
        """Build a Markdown summary for Telegram."""
        parts = []
        
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "INFO": "ℹ️",
        }
        
        emoji = severity_emoji.get(llm_result.overall_severity, "ℹ️")
        parts.append(f"*Review Result: {emoji} {llm_result.overall_severity}*")
        
        if llm_result.error:
            escaped_error = _escape_markdown(llm_result.error[:100])
            parts.append(f"⚠️ AI review failed: {escaped_error}")
        
        if llm_result.model_used:
            parts.append(f"🤖 _Model: {llm_result.model_used}_")
        
        if change_result.from_ref and change_result.to_ref:
            parts.append(f"`{change_result.from_ref}..{change_result.to_ref}`")
        elif change_result.to_ref:
            parts.append(f"Commit: `{change_result.to_ref}`")
        
        parts.append("")
        
        if llm_result.summary:
            parts.append("*Summary:*")
            for point in llm_result.summary[:5]:
                parts.append(f"• {point}")
            parts.append("")
        
        if llm_result.issues:
            issue_count = len(llm_result.issues)
            critical = sum(1 for i in llm_result.issues if i.get("severity") == "CRITICAL")
            high = sum(1 for i in llm_result.issues if i.get("severity") == "HIGH")
            
            parts.append(f"*Issues:* {issue_count} found")
            if critical > 0:
                parts.append(f"  🔴 {critical} critical")
            if high > 0:
                parts.append(f"  🟠 {high} high")
            parts.append("")
        
        if llm_result.reasoning_summary:
            parts.append(f"💭 _{llm_result.reasoning_summary}_")
            parts.append("")
        
        parts.append("Use /risks for details, /diff for changes, /next for actions.")
        
        return "\n".join(parts)
    
    def get_diff_summary_for_last_review(self) -> Optional[str]:
        """Get formatted diff summary for the last review."""
        review = get_last_review()
        if not review:
            return None
        
        parts = [f"*Diff Summary* (review #{review.id})"]
        
        if review.git_head:
            parts.append(f"Commit: `{review.git_head}`")
        parts.append("")
        
        diff_summary = review.diff_summary
        if not diff_summary:
            parts.append("No diff summary available.")
            return "\n".join(parts)
        
        categories: Dict[str, List[Dict]] = {}
        for item in diff_summary:
            cat = item.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        category_labels = {
            "api": "📡 API",
            "tests": "🧪 Tests",
            "config": "⚙️ Config",
            "infra": "🏗️ Infra",
            "models": "📊 Models",
            "database": "🗄️ Database",
            "docs": "📚 Docs",
            "frontend": "🖥️ Frontend",
            "other": "📁 Other",
        }
        
        for cat, items in categories.items():
            label = category_labels.get(cat, cat.title())
            parts.append(f"*{label}*")
            for item in items:
                adds = item.get("additions", 0)
                dels = item.get("deletions", 0)
                path = item.get("path", "unknown")
                parts.append(f"• `{path}` — +{adds}/-{dels}")
            parts.append("")
        
        return "\n".join(parts)
    
    def get_risks_for_last_review(self) -> Optional[str]:
        """Get formatted risks summary for the last review."""
        review = get_last_review()
        if not review:
            return None
        
        parts = [f"*Risks* (review #{review.id})"]
        parts.append("")
        
        issues = review.issues
        if not issues:
            parts.append("✅ No issues detected.")
            return "\n".join(parts)
        
        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "INFO": "ℹ️",
        }
        
        by_severity: Dict[str, List[Dict]] = {}
        for issue in issues:
            sev = issue.get("severity", "INFO")
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(issue)
        
        for sev in severity_order:
            if sev not in by_severity:
                continue
            
            emoji = severity_emoji.get(sev, "•")
            for issue in by_severity[sev]:
                title = issue.get("title", "Untitled issue")
                parts.append(f"{emoji} *{sev}* — {title}")
                
                if issue.get("file"):
                    parts.append(f"  📄 `{issue['file']}`")
                
                if issue.get("description"):
                    desc = issue["description"][:200]
                    parts.append(f"  {desc}")
                
                if issue.get("suggested_fix"):
                    fix = issue["suggested_fix"][:150]
                    parts.append(f"  💡 Fix: {fix}")
                
                parts.append("")
        
        return "\n".join(parts)
    
    def get_next_actions_for_last_review(self) -> Optional[str]:
        """Get formatted next actions for the last review."""
        review = get_last_review()
        if not review:
            return None
        
        parts = [f"*Next Actions* (review #{review.id})"]
        parts.append("")
        
        next_steps = review.next_steps
        if not next_steps:
            parts.append("✅ No specific actions recommended.")
            return "\n".join(parts)
        
        for i, step in enumerate(next_steps, 1):
            parts.append(f"{i}. {step}")
        
        return "\n".join(parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for /status command."""
        review = get_last_review()
        review_count = get_review_count()
        git_available = self.change_detector.is_git_available()
        current_head = self.change_detector.get_current_head() if git_available else None
        llm_available = self.llm_client.is_available()
        last_reviewed = get_meta("last_reviewed_commit")
        
        return {
            "last_review": {
                "id": review.id if review else None,
                "created_at": review.created_at if review else None,
                "target_ref": review.target_ref if review else None,
                "severity": review.overall_severity if review else None,
            } if review else None,
            "review_count": review_count,
            "git_available": git_available,
            "current_head": current_head,
            "last_reviewed_commit": last_reviewed,
            "llm_available": llm_available,
            "change_detection_mode": "git" if git_available else "snapshot",
        }
