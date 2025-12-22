"""Reusable diff logic for comparing backtest runs.

This module is intentionally DB-free and reads backtest metrics from
src/backtest/run_store.py, matching the Backtest Lab's file-based storage.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.backtest.run_store import load_result


METRICS_FIELDS: List[Tuple[str, str]] = [
    ("net_profit_pct", "pct"),
    ("net_profit_usd", "usd"),
    ("final_pnl_vs_hodl", "usd"),
    ("max_drawdown_pct", "pct"),
    ("max_drawdown_usd", "usd"),
    ("num_trades", "int"),
    ("win_rate", "pct"),
    ("profit_factor", "float"),
    ("avg_trade_usd", "usd"),
    ("sharpe_ratio", "float"),
    ("sortino_ratio", "float"),
]


def get_metric_value(metrics: Dict[str, Any], field: str) -> float:
    """Extract a metric value from a run_store metrics dict."""
    try:
        val = metrics.get(field)
        if val is None:
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def format_value(val: float, fmt_type: str) -> str:
    """Format a value based on its type."""
    if fmt_type == "pct":
        return f"{val:.1f} %"
    elif fmt_type == "usd":
        return f"{val:,.2f}"
    elif fmt_type == "int":
        return f"{int(val)}"
    else:
        return f"{val:.2f}"


def format_diff(diff: float, fmt_type: str) -> str:
    """Format a diff value with +/- sign."""
    sign = "+" if diff > 0 else ""
    if fmt_type == "pct":
        return f"{sign}{diff:.1f} pp"
    elif fmt_type == "usd":
        return f"{sign}{diff:,.2f}"
    elif fmt_type == "int":
        return f"{sign}{int(diff)}"
    else:
        return f"{sign}{diff:.2f}"


def compute_diff_for_runs(
    run_id_a: str,
    run_id_b: str,
    exit_style: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute diff metrics between two backtest runs.
    
    Args:
        run_id_a: Run ID of first backtest (typically synthetic)
        run_id_b: Run ID of second backtest (typically live_deribit)
        exit_style: Exit style to compare (defaults to primary_exit_style)
        
    Returns:
        Dict with structure:
        {
            "run_a": {run metadata},
            "run_b": {run metadata},
            "exit_style": str,
            "metrics": {
                "net_profit_pct": {"a": float, "b": float, "diff": float},
                ...
            }
        }
        
    Raises:
        ValueError if runs or metrics not found
    """
    run_a = load_result(run_id_a)
    if not run_a:
        raise ValueError(f"Run A not found: {run_id_a}")

    run_b = load_result(run_id_b)
    if not run_b:
        raise ValueError(f"Run B not found: {run_id_b}")

    if exit_style:
        effective_exit_style = exit_style
    else:
        effective_exit_style = run_a.config.get("exit_style") or run_b.config.get("exit_style")
        if not effective_exit_style:
            raise ValueError("exit_style not provided and not present in run configs")

    metrics_a = (run_a.metrics or {}).get(effective_exit_style)
    if not metrics_a:
        raise ValueError(f"Metrics not found for run A ({run_id_a}) with exit_style={effective_exit_style}")

    metrics_b = (run_b.metrics or {}).get(effective_exit_style)
    if not metrics_b:
        raise ValueError(f"Metrics not found for run B ({run_id_b}) with exit_style={effective_exit_style}")

    run_a_metadata = {
        "run_id": run_a.run_id,
        "underlying": run_a.config.get("underlying"),
        "data_source": run_a.config.get("data_source"),
        "start_ts": run_a.config.get("start_date") or run_a.config.get("start"),
        "end_ts": run_a.config.get("end_date") or run_a.config.get("end"),
        "decision_interval_minutes": run_a.config.get("decision_interval_minutes"),
    }

    run_b_metadata = {
        "run_id": run_b.run_id,
        "underlying": run_b.config.get("underlying"),
        "data_source": run_b.config.get("data_source"),
        "start_ts": run_b.config.get("start_date") or run_b.config.get("start"),
        "end_ts": run_b.config.get("end_date") or run_b.config.get("end"),
        "decision_interval_minutes": run_b.config.get("decision_interval_minutes"),
    }

    metrics_dict: Dict[str, Any] = {}
    for field, fmt_type in METRICS_FIELDS:
        val_a = get_metric_value(metrics_a, field)
        val_b = get_metric_value(metrics_b, field)
        d = val_b - val_a
        metrics_dict[field] = {"a": val_a, "b": val_b, "diff": d, "fmt_type": fmt_type}

    return {
        "run_a": run_a_metadata,
        "run_b": run_b_metadata,
        "exit_style": effective_exit_style,
        "metrics": metrics_dict,
    }


def print_diff_report_from_data(diff_data: Dict[str, Any]) -> None:
    """
    Print a formatted diff report from computed diff data.
    
    Args:
        diff_data: Output from compute_diff_for_runs()
    """
    run_a = diff_data["run_a"]
    run_b = diff_data["run_b"]
    exit_style = diff_data["exit_style"]
    metrics = diff_data["metrics"]
    
    print()
    print("=" * 80)
    print("BACKTEST DIFF REPORT")
    print("=" * 80)
    print()
    print("Comparing runs:")
    print(f"  A: {run_a['run_id']}  (data_source={run_a['data_source']})")
    print(f"  B: {run_b['run_id']}  (data_source={run_b['data_source']})")
    print(f"  Underlying: {run_a['underlying']}")
    print(f"  Exit style: {exit_style}")
    print(f"  Period A: {run_a['start_ts']} -> {run_a['end_ts']}")
    print(f"  Period B: {run_b['start_ts']} -> {run_b['end_ts']}")
    print(f"  Decision interval: {run_a['decision_interval_minutes']} minutes")
    print()
    
    col_metric = 24
    col_val = 20
    
    header = f"{'Metric':<{col_metric}} {'A (' + run_a['data_source'] + ')':<{col_val}} {'B (' + run_b['data_source'] + ')':<{col_val}} {'Diff (B - A)':<{col_val}}"
    print(header)
    print("-" * len(header))
    
    for field, _ in METRICS_FIELDS:
        m = metrics[field]
        val_a = m["a"]
        val_b = m["b"]
        diff = m["diff"]
        fmt_type = m["fmt_type"]
        
        str_a = format_value(val_a, fmt_type)
        str_b = format_value(val_b, fmt_type)
        str_diff = format_diff(diff, fmt_type)
        
        print(f"{field:<{col_metric}} {str_a:<{col_val}} {str_b:<{col_val}} {str_diff:<{col_val}}")
    
    print("=" * 80)
    print()
