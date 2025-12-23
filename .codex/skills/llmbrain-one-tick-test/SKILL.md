---
name: llmbrain-one-tick-test
description: Add an integration test that runs one agent loop tick with a mocked Deribit client.
metadata:
  short-description: One-tick agent loop integration test
---

## Goal
- Build a deterministic integration-style test that executes exactly one agent loop tick (or the closest orchestration layer) with a fake Deribit client and validates the behavior.

## Requirements
- Do not use the network; mock the Deribit client and related dependencies.
- Keep timestamps deterministic so the test does not depend on real-time progress.
- Assert both gate scenarios:
  1. When `can_trade`/permission gates block trading, no order placement calls happen.
  2. When allowed to trade, the expected order placement call(s) are made in that single tick.
