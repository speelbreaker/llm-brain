from __future__ import annotations

from datetime import datetime, timedelta


def test_covered_call_simulator_passes_regime_state_into_sigma(monkeypatch):
    from src.backtest.covered_call_simulator import CoveredCallSimulator
    from src.backtest.types import CallSimulationConfig

    calls = []

    def fake_get_sigma_for_option(*, regime_state=None, **kwargs):
        calls.append(regime_state)
        return 0.5

    monkeypatch.setattr(
        "src.backtest.covered_call_simulator.get_sigma_for_option",
        fake_get_sigma_for_option,
    )

    cfg = CallSimulationConfig(
        underlying="BTC",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        timeframe="1d",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0,
        pricing_mode="synthetic_bs",
    )

    sim = CoveredCallSimulator(data_source=object(), config=cfg)
    sim._spot_history_cache = [(cfg.start, 100.0), (cfg.start + timedelta(days=1), 101.0)]

    regime = object()
    price, delta = sim._compute_synthetic_option_price(
        spot=100.0,
        strike=110.0,
        expiry=cfg.start + timedelta(days=7),
        as_of=cfg.start,
        regime_state=regime,
    )

    assert price >= 0.0
    assert -1.0 <= float(delta) <= 1.0
    assert calls, "expected get_sigma_for_option to be called"
    assert all(c is regime for c in calls)
