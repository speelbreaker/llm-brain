# codexw — CLI wrapper for Codex

`scripts/codexw.sh` builds the full prompt by concatenating:

1. `docs/CLI_AGENT_CONTEXT.md`
2. `docs/CODEX_PROMPT_WRAPPER.txt`
3. A `TASK: …` section derived from the CLI arguments.

It then attempts to run `npx @openai/codex exec --dangerously-bypass-approvals-and-sandbox --json` with that prompt via stdin. If the CLI refuses stdin or fails, the script prints the prompt path and contents so you can paste them manually.

## Usage

From the repo root:

```sh
scripts/codexw.sh "Fix the new runtime config endpoint so it validates trade_mode."
```

The Makefile exposes a thin wrapper that passes a `TASK` value:

```sh
make codex TASK="Inspect the latest PR for deadlocks."
```

### Setting an alias

Add the following line to your `~/.zshrc` or `~/.bashrc` and reload your shell:

```sh
alias codexw='/opt/llm-brain/llm-brain/scripts/codexw.sh'
```

If you prefer a custom Codex binary (for example, a local `codex` release), set `CODER_BIN` before invoking the script:

```sh
CODER_BIN=codex scripts/codexw.sh "Inspect the latest PR."
```

## Examples

- `scripts/codexw.sh "Diagnose why the supervisor health endpoint is flaky."`
- `make codex TASK="Propose a guardrail for new endpoints."`
- `CODER_BIN=codex scripts/codexw.sh "Summarize repo safety rules."`
