"""
Rules summary builder for UI display.
Transforms ResolvedBacktestConfig into a UI-friendly summary structure.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.backtest.config_schema import (
    ResolvedBacktestConfig,
    BacktestPreset,
    BacktestMode,
)


def _on_off(value: bool | None) -> str:
    """Convert boolean to ON/OFF string."""
    if value is None:
        return "N/A"
    return "ON" if value else "OFF"


def build_rules_summary(cfg: ResolvedBacktestConfig) -> Dict[str, Any]:
    """
    Turn a ResolvedBacktestConfig into a UI-friendly summary structure
    that templates can render as a box with sections and badges.
    """

    t = cfg.thresholds
    r = cfg.rule_toggles

    mode_label = "Training mode" if cfg.mode == BacktestMode.TRAINING else "Live mode"

    preset_map = {
        BacktestPreset.ULTRA_SAFE: "Ultra Safe",
        BacktestPreset.BALANCED: "Balanced",
        BacktestPreset.AGGRESSIVE: "Aggressive",
        BacktestPreset.CUSTOM: "Custom (manual overrides)",
    }
    preset_label = preset_map.get(cfg.preset, cfg.preset.value)

    headline = f"{mode_label} - {preset_label}"

    subtitle_parts: List[str] = []

    if t.max_margin_used_pct is not None:
        subtitle_parts.append(f"Max margin: {t.max_margin_used_pct:.0f}%")
    if t.max_net_delta_abs is not None:
        subtitle_parts.append(f"Max |net delta|: {t.max_net_delta_abs:.2f}")
    if t.delta_range is not None:
        subtitle_parts.append(
            f"Delta range: {t.delta_range[0]:.2f} - {t.delta_range[1]:.2f}"
        )
    if t.dte_range is not None:
        subtitle_parts.append(
            f"DTE range: {t.dte_range[0]}-{t.dte_range[1]} days"
        )

    subtitle = " | ".join(subtitle_parts)

    badges: List[Dict[str, str]] = []

    badges.append({"label": "Mode", "value": cfg.mode.value})
    badges.append({"label": "Preset", "value": cfg.preset.value})

    if r.enforce_margin_cap is not None:
        badges.append({"label": "Margin cap", "value": _on_off(r.enforce_margin_cap)})
    if r.enforce_net_delta_cap is not None:
        badges.append({"label": "Delta cap", "value": _on_off(r.enforce_net_delta_cap)})
    if r.require_ivrv_filter is not None:
        badges.append({"label": "IV/RV filter", "value": _on_off(r.require_ivrv_filter)})

    sections: List[Dict[str, Any]] = []

    risk_items: List[Dict[str, str]] = []
    if r.enforce_margin_cap is not None:
        risk_items.append({
            "label": "Enforce margin usage cap",
            "value": _on_off(r.enforce_margin_cap),
            "detail": (
                f"Max margin used: {t.max_margin_used_pct:.0f}%"
                if t.max_margin_used_pct is not None
                else ""
            ),
        })
    if r.enforce_net_delta_cap is not None:
        risk_items.append({
            "label": "Enforce net delta cap",
            "value": _on_off(r.enforce_net_delta_cap),
            "detail": (
                f"Max |net delta|: {t.max_net_delta_abs:.2f}"
                if t.max_net_delta_abs is not None
                else ""
            ),
        })
    if r.enforce_per_expiry_exposure is not None:
        risk_items.append({
            "label": "Per-expiry exposure cap",
            "value": _on_off(r.enforce_per_expiry_exposure),
            "detail": (
                f"Cap: {t.per_expiry_exposure_cap:.2f}"
                if t.per_expiry_exposure_cap is not None
                else ""
            ),
        })

    if risk_items:
        sections.append({"title": "Risk limits", "items": risk_items})

    filter_items: List[Dict[str, str]] = []

    if r.require_ivrv_filter is not None:
        filter_items.append({
            "label": "Require IV/RV filter",
            "value": _on_off(r.require_ivrv_filter),
            "detail": (
                f"Min IV/RV: {t.min_ivrv:.2f}"
                if t.min_ivrv is not None
                else ""
            ),
        })

    if t.delta_range is not None:
        filter_items.append({
            "label": "Delta band",
            "value": f"{t.delta_range[0]:.2f} - {t.delta_range[1]:.2f}",
            "detail": "Target short-call deltas",
        })

    if t.dte_range is not None:
        filter_items.append({
            "label": "DTE band",
            "value": f"{t.dte_range[0]}-{t.dte_range[1]} days",
            "detail": "Target time to expiry",
        })

    if r.respect_min_premium_filter is not None:
        filter_items.append({
            "label": "Min premium filter",
            "value": _on_off(r.respect_min_premium_filter),
            "detail": (
                f"Min premium: ${t.min_premium_usd:.0f}"
                if t.min_premium_usd is not None
                else ""
            ),
        })

    if filter_items:
        sections.append({"title": "Trade selection", "items": filter_items})

    structure_items: List[Dict[str, str]] = []

    if r.restrict_single_primary_call_per_expiry is not None:
        structure_items.append({
            "label": "Single primary call per expiry",
            "value": _on_off(r.restrict_single_primary_call_per_expiry),
            "detail": "Prevents multiple overlapping main calls on same expiry.",
        })

    if r.allow_multi_profile_laddering is not None:
        structure_items.append({
            "label": "Multi-profile laddering",
            "value": _on_off(r.allow_multi_profile_laddering),
            "detail": "Allow conservative / moderate / aggressive legs together.",
        })

    if r.use_synthetic_iv_and_skew is not None:
        structure_items.append({
            "label": "Synthetic IV & skew engine",
            "value": _on_off(r.use_synthetic_iv_and_skew),
            "detail": "Use synthetic universe for IV/skew instead of raw quotes.",
        })

    if structure_items:
        sections.append({"title": "Structure & laddering", "items": structure_items})

    if cfg.mode == BacktestMode.TRAINING:
        safety_tagline = "Training mode: some safeguards may be relaxed for exploration."
    else:
        safety_tagline = "Live mode: safeguards enforced according to preset."

    return {
        "headline": headline,
        "subtitle": subtitle,
        "mode": cfg.mode.value,
        "preset": cfg.preset.value,
        "badges": badges,
        "sections": sections,
        "safety_tagline": safety_tagline,
    }


def build_rules_summary_from_settings() -> Dict[str, Any]:
    """Build a rules summary from current application settings."""
    from src.config import settings
    from src.backtest.config_schema import (
        BacktestRuleToggles,
        BacktestThresholds,
        BacktestMode,
    )
    
    mode = BacktestMode.TRAINING if settings.is_training_enabled else BacktestMode.LIVE
    
    if settings.is_research:
        preset = BacktestPreset.AGGRESSIVE
    else:
        preset = BacktestPreset.BALANCED
    
    cfg = ResolvedBacktestConfig(
        preset=preset,
        mode=mode,
        rule_toggles=BacktestRuleToggles(
            enforce_per_expiry_exposure=not settings.is_training_on_testnet,
            enforce_margin_cap=True,
            enforce_net_delta_cap=True,
            restrict_single_primary_call_per_expiry=not settings.is_training_enabled,
            require_ivrv_filter=True,
            use_synthetic_iv_and_skew=settings.synthetic_skew_enabled,
            allow_multi_profile_laddering=settings.training_profile_mode == "ladder",
            respect_min_premium_filter=True,
        ),
        thresholds=BacktestThresholds(
            max_margin_used_pct=settings.max_margin_used_pct,
            max_net_delta_abs=settings.max_net_delta_abs,
            per_expiry_exposure_cap=settings.effective_max_expiry_exposure,
            min_ivrv=settings.effective_ivrv_min,
            delta_range=(settings.effective_delta_min, settings.effective_delta_max),
            dte_range=(settings.effective_dte_min, settings.effective_dte_max),
            min_premium_usd=settings.premium_min_usd,
        ),
    )
    
    return build_rules_summary(cfg)
