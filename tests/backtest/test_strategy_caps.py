"""Tests for strategy capabilities and config override logic."""

from src.backtest.strategy_caps import (
    get_strategy_caps,
    apply_strategy_overrides,
    list_available_strategies,
    GREGBOT_DEFAULTS,
)


class TestStrategyCapabilities:
    """Tests for strategy capability metadata."""

    def test_get_generic_covered_call_caps(self):
        """Test that generic_covered_call returns full user control."""
        caps = get_strategy_caps("generic_covered_call")

        assert caps is not None
        assert caps.selector_name == "generic_covered_call"
        assert caps.supports_exit_style is True
        assert caps.supports_dte_range is True
        assert caps.supports_delta_range is True
        assert caps.supports_targets is True
        assert "exit_style" in caps.user_fields
        assert len(caps.config_owner_fields) == 0

    def test_get_gregbot_caps(self):
        """Test that GregBot caps show managed fields."""
        caps = get_strategy_caps("gregbot")

        assert caps is not None
        assert caps.selector_name == "gregbot"
        assert caps.supports_exit_style is False
        assert caps.supports_dte_range is False
        assert caps.supports_delta_range is False
        assert caps.supports_targets is False
        assert "exit_style" in caps.config_owner_fields
        assert "target_dte" in caps.config_owner_fields
        assert len(caps.strategy_defaults_summary) > 0

    def test_unknown_selector_returns_none(self):
        """Test that unknown selector returns None."""
        caps = get_strategy_caps("unknown_strategy")
        assert caps is None

    def test_list_available_strategies(self):
        """Test listing all available strategies."""
        strategies = list_available_strategies()

        assert len(strategies) >= 2
        names = [s["selector_name"] for s in strategies]
        assert "generic_covered_call" in names
        assert "gregbot" in names


class TestApplyStrategyOverrides:
    """Tests for config override logic."""

    def test_generic_no_overrides(self):
        """Test that generic_covered_call applies no overrides."""
        user_config = {
            "exit_style": "tp_and_roll",
            "target_dte": 14,
            "target_delta": 0.30,
            "min_dte": 5,
            "max_dte": 25,
            "delta_min": 0.10,
            "delta_max": 0.40,
        }

        result = apply_strategy_overrides("generic_covered_call", user_config)

        assert len(result.warnings) == 0
        assert result.effective_config["exit_style"] == "tp_and_roll"
        assert result.effective_config["target_dte"] == 14
        assert result.effective_config["target_delta"] == 0.30

    def test_gregbot_ignores_exit_style(self):
        """Test that GregBot ignores user's exit_style and returns warning."""
        user_config = {
            "exit_style": "hold_to_expiry",
            "target_dte": 7,
            "target_delta": 0.25,
        }

        result = apply_strategy_overrides("gregbot", user_config)

        assert len(result.warnings) > 0
        assert any("exit_style" in w for w in result.warnings)
        assert result.effective_config["exit_style"] == "gregbot_managed"

    def test_gregbot_ignores_dte_delta_targets(self):
        """Test that GregBot ignores user's DTE/delta targets."""
        user_config = {
            "exit_style": "hold_to_expiry",
            "target_dte": 3,
            "target_delta": 0.50,
            "min_dte": 1,
            "max_dte": 7,
            "delta_min": 0.05,
            "delta_max": 0.60,
        }

        result = apply_strategy_overrides("gregbot", user_config)

        assert len(result.warnings) >= 2
        assert result.effective_config["target_dte"] == GREGBOT_DEFAULTS["target_dte"]
        assert (
            result.effective_config["target_delta"] == GREGBOT_DEFAULTS["target_delta"]
        )
        assert result.effective_config["min_dte"] == GREGBOT_DEFAULTS["min_dte"]
        assert result.effective_config["max_dte"] == GREGBOT_DEFAULTS["max_dte"]

    def test_gregbot_effective_config_returned(self):
        """Test that effective_config contains final values."""
        user_config = {"exit_style": "tp_and_roll"}

        result = apply_strategy_overrides("gregbot", user_config)

        assert "exit_style" in result.effective_config
        assert "target_dte" in result.effective_config
        assert "target_delta" in result.effective_config
        assert "min_dte" in result.effective_config
        assert "max_dte" in result.effective_config
        assert "delta_min" in result.effective_config
        assert "delta_max" in result.effective_config

    def test_gregbot_no_warning_for_matching_defaults(self):
        """Test that matching defaults don't generate warnings."""
        user_config = {
            "exit_style": "gregbot_managed",
            "target_dte": GREGBOT_DEFAULTS["target_dte"],
        }

        result = apply_strategy_overrides("gregbot", user_config)

        assert not any("exit_style" in w for w in result.warnings)

    def test_unknown_selector_warning(self):
        """Test that unknown selector generates warning."""
        result = apply_strategy_overrides(
            "unknown_strategy", {"exit_style": "hold_to_expiry"}
        )

        assert len(result.warnings) == 1
        assert "Unknown selector" in result.warnings[0]
        assert result.effective_config["exit_style"] == "hold_to_expiry"

    def test_overrides_applied_tracking(self):
        """Test that overrides_applied tracks what was changed."""
        user_config = {
            "exit_style": "hold_to_expiry",
            "target_dte": 3,
        }

        result = apply_strategy_overrides("gregbot", user_config)

        assert "exit_style" in result.overrides_applied
        assert result.overrides_applied["exit_style"]["user_value"] == "hold_to_expiry"
        assert (
            result.overrides_applied["exit_style"]["effective_value"]
            == "gregbot_managed"
        )
