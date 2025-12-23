# Release Checklist

1. Review the queue and confirm every prompt that touches production is in `DONE`.
2. Confirm `System Health` notes show green or documented mitigation for any red.
3. Run the necessary tests (unit, integration, `one-tick` integration where applicable).
4. Verify changelog entry is complete (`date`, `what changed`, `why`, `tests`, `links`).
5. Publish the vault snapshots (`OBSIDEAN_QUEUE_latest.md`, `OBSIDEAN_NOW_latest.md`, optionally `OBSIDEAN_PROMPT_latest.md`).
