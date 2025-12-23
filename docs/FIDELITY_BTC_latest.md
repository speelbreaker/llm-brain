# Synthetic Fidelity Report

- Run ID: 20251219_234717
- Timestamp (UTC): 2025-12-19T23:47:17.593360+00:00
- Gate: **TRUSTED**

## Scores

- Overall: **100.0**
- strategy_pnl_parity: 100.0
- underlying_returns: 100.0

## Market Meta

### Live
```json
{
  "ds_class": "LiveDeribitDataSource",
  "margin_type": "linear",
  "settlement_ccy": "USDC",
  "type": "live_replay",
  "underlying": "BTC"
}
```

### Synthetic
```json
{
  "cfg_class": "CallSimulationConfig",
  "type": "synthetic_replay",
  "underlying": "BTC"
}
```

## Strategy Parity (P0 placeholder)

```json
{
  "decision_times": [
    "2025-12-07T00:00:00+00:00",
    "2025-12-08T00:00:00+00:00",
    "2025-12-09T00:00:00+00:00",
    "2025-12-10T00:00:00+00:00",
    "2025-12-11T00:00:00+00:00",
    "2025-12-12T00:00:00+00:00",
    "2025-12-13T00:00:00+00:00",
    "2025-12-14T00:00:00+00:00",
    "2025-12-15T00:00:00+00:00",
    "2025-12-16T00:00:00+00:00",
    "2025-12-17T00:00:00+00:00",
    "2025-12-18T00:00:00+00:00",
    "2025-12-19T00:00:00+00:00"
  ],
  "strategies": [
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "covered_call",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    },
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "cash_secured_put",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    },
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "short_strangle",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    },
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "put_spread_credit",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    },
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "call_spread_debit",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    },
    {
      "live": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 0.0
      },
      "name": "calendar",
      "synthetic": {
        "notes": "P0 placeholder (execution not implemented)",
        "num_trades": 0,
        "spot_first": 0.0,
        "spot_last": 85758.41
      }
    }
  ]
}
```
