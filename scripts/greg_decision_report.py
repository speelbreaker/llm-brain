#!/usr/bin/env python3
"""
Greg Decision Report Script

Generates a "what if I followed Greg" analysis from the decision log.
Shows counts of suggestions vs executions, and estimates impact.

Usage:
    python scripts/greg_decision_report.py --from 2025-01-01 --to 2025-12-31 --underlying BTC
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from io import StringIO
from typing import Optional

from src.db.models_greg_decision import get_decision_history, get_decision_stats


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def format_date(dt: Optional[datetime]) -> str:
    """Format datetime to string."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_report(
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    underlying: Optional[str] = None,
    output_format: str = "text",
) -> str:
    """
    Generate the decision report.
    
    Args:
        from_date: Start date for filtering
        to_date: End date for filtering
        underlying: Filter by underlying (BTC/ETH)
        output_format: "text" or "csv"
        
    Returns:
        Report content as string
    """
    history = get_decision_history(
        underlying=underlying,
        from_date=from_date,
        to_date=to_date,
        limit=1000,
    )
    
    stats = get_decision_stats(
        underlying=underlying,
        from_date=from_date,
        to_date=to_date,
    )
    
    if output_format == "csv":
        return generate_csv_report(history, stats)
    else:
        return generate_text_report(history, stats, from_date, to_date, underlying)


def generate_text_report(
    history: list,
    stats: dict,
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    underlying: Optional[str],
) -> str:
    """Generate text format report."""
    lines = []
    lines.append("=" * 70)
    lines.append("GREG DECISION REPORT - 'What If I Followed Greg?'")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append(f"Date Range: {format_date(from_date)} to {format_date(to_date)}")
    lines.append(f"Underlying Filter: {underlying or 'ALL'}")
    lines.append(f"Total Records: {len(history)}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 70)
    lines.append(f"Total Suggestions: {stats.get('total_suggestions', 0)}")
    lines.append(f"Total Executed: {stats.get('total_executed', 0)}")
    
    total_sugg = stats.get('total_suggestions', 0)
    total_exec = stats.get('total_executed', 0)
    follow_rate = (total_exec / total_sugg * 100) if total_sugg > 0 else 0
    lines.append(f"Follow Rate: {follow_rate:.1f}%")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("BY STRATEGY")
    lines.append("-" * 70)
    
    by_strategy = stats.get('by_strategy', {})
    if by_strategy:
        for strat, data in sorted(by_strategy.items()):
            sugg = data.get('suggestions', 0)
            exec_ = data.get('executions', 0)
            avg_pnl = data.get('avg_pnl_pct')
            pnl_str = f"{avg_pnl*100:.1f}%" if avg_pnl is not None else "N/A"
            rate = (exec_ / sugg * 100) if sugg > 0 else 0
            
            strat_name = strat.replace('STRATEGY_', '').replace('_', ' ')
            lines.append(f"  {strat_name}:")
            lines.append(f"    Suggestions: {sugg}, Executed: {exec_} ({rate:.0f}%)")
            lines.append(f"    Avg PnL at suggestion: {pnl_str}")
    else:
        lines.append("  No strategy data available.")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("BY ACTION TYPE")
    lines.append("-" * 70)
    
    by_action = stats.get('by_action', {})
    if by_action:
        for action, data in sorted(by_action.items()):
            sugg = data.get('suggestions', 0)
            exec_ = data.get('executions', 0)
            rate = (exec_ / sugg * 100) if sugg > 0 else 0
            lines.append(f"  {action}: {sugg} suggested, {exec_} executed ({rate:.0f}%)")
    else:
        lines.append("  No action data available.")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("FOLLOW VS IGNORE ANALYSIS")
    lines.append("-" * 70)
    
    followed = [h for h in history if h.get('executed')]
    ignored = [h for h in history if h.get('suggested') and not h.get('executed')]
    
    def avg_pnl(entries):
        pnls = [e.get('pnl_pct') for e in entries if e.get('pnl_pct') is not None]
        return sum(pnls) / len(pnls) if pnls else None
    
    followed_avg = avg_pnl(followed)
    ignored_avg = avg_pnl(ignored)
    
    lines.append(f"  Suggestions Followed: {len(followed)}")
    lines.append(f"  Suggestions Ignored: {len(ignored)}")
    lines.append(f"  Avg PnL when Followed: {followed_avg*100:.1f}%" if followed_avg else "  Avg PnL when Followed: N/A")
    lines.append(f"  Avg PnL when Ignored: {ignored_avg*100:.1f}%" if ignored_avg else "  Avg PnL when Ignored: N/A")
    lines.append("")
    
    if followed_avg is not None and ignored_avg is not None:
        diff = followed_avg - ignored_avg
        direction = "better" if diff > 0 else "worse"
        lines.append(f"  Impact: Following suggestions was {abs(diff)*100:.1f}% {direction} on average.")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("RECENT DECISIONS (Last 20)")
    lines.append("-" * 70)
    
    for entry in history[:20]:
        ts = entry.get('timestamp', 'N/A')
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]
        underlying = entry.get('underlying', '?')
        action = entry.get('action_type', '?')
        executed = "EXEC" if entry.get('executed') else "SKIP"
        mode = entry.get('mode', '?')
        pnl = entry.get('pnl_pct')
        pnl_str = f"{pnl*100:.1f}%" if pnl is not None else "N/A"
        
        lines.append(f"  [{ts}] {underlying} {action:12} {executed} ({mode}) PnL={pnl_str}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_csv_report(history: list, stats: dict) -> str:
    """Generate CSV format report."""
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "timestamp", "underlying", "strategy_type", "position_id",
        "action_type", "mode", "suggested", "executed", "reason",
        "pnl_pct", "pnl_usd", "net_delta", "vrp_30d"
    ])
    
    for entry in history:
        writer.writerow([
            entry.get('timestamp', ''),
            entry.get('underlying', ''),
            entry.get('strategy_type', ''),
            entry.get('position_id', ''),
            entry.get('action_type', ''),
            entry.get('mode', ''),
            entry.get('suggested', ''),
            entry.get('executed', ''),
            entry.get('reason', ''),
            entry.get('pnl_pct', ''),
            entry.get('pnl_usd', ''),
            entry.get('net_delta', ''),
            entry.get('vrp_30d', ''),
        ])
    
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Greg decision analysis report"
    )
    parser.add_argument(
        "--from", dest="from_date",
        help="Start date (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--to", dest="to_date",
        help="End date (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--underlying",
        help="Filter by underlying (BTC, ETH)",
        default=None,
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv"],
        default="text",
        help="Output format (text or csv)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
        default=None,
    )
    
    args = parser.parse_args()
    
    from_date = parse_date(args.from_date) if args.from_date else None
    to_date = parse_date(args.to_date) if args.to_date else None
    
    report = generate_report(
        from_date=from_date,
        to_date=to_date,
        underlying=args.underlying,
        output_format=args.format,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
