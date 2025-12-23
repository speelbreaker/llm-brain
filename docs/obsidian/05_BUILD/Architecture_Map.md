# Architecture Map

## Layers
- **Data layer**: Deribit feeds, calibrators, and the repository of historical snapshots.
- **Decision layer**: Agent loop, policy gates (`can_trade`, `close_only_mode`), and risk checks.
- **Execution layer**: Hedger, backtest connectors, and live order placement with Deribit and other venues.

## Notes
- Maintain diagrams off-site but link them here when they change.
- Each release must verify the map before shipping to ensure the layers still align.
