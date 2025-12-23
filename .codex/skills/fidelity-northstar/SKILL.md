---
name: fidelity-northstar
description: "Fidelity North Star guide: use when demonstrating that Truth → Trust → Trade gating is respected end-to-end."
---

# Fidelity North Star Skill

## Use when
- A change touches fidelity facts, data readiness, or calibration outputs that feed into gate decisions.
- Tests or artifacts make claims about `most_recent`, `fact_resolvers`, or `gate_overall`.

## Instructions
1. Reference `docs/obsidian/03_NORTHSTAR/Fidelity_NorthStar.md` and `Truth_Trust_Trade.md` before modifying calibration or gate code.
2. Ensure the change produces a deterministic artifact (context pack, fidelity docs) that can be published to Drive alongside `OPS_HEALTH_latest.json`.
3. Capture the gating story in a prompt under `docs/obsidian/06_PROMPTS/` and mark the queue/changelog to show Truth+Trust+Trade alignment.
4. Run any relevant fidelity tests (e.g., `tests/test_fidelity_*`) and mention them in the prompt's "Tests required" section.
