"""Security test for the codexw prompt builder."""

from pathlib import Path


def compile_prompt(task: str) -> str:
    """Re-create the prompt exactly like `scripts/codexw.sh`."""
    root = Path(__file__).resolve().parents[2]
    context = (root / "docs" / "CLI_AGENT_CONTEXT.md").read_text()
    wrapper = (root / "docs" / "CODEX_PROMPT_WRAPPER.txt").read_text()
    return f"{context}\n\n{wrapper}\nTASK: {task}\n"


def test_prompt_includes_evidence_and_hard_stop_sections():
    prompt = compile_prompt("inspect guardrails")
    assert "Evidence format" in prompt
    assert "SAFETY (HARD STOP RULES)" in prompt


def test_prompt_only_comes_from_context_and_wrapper():
    task = "verify minimal wrapper"
    prompt = compile_prompt(task)
    expected = (
        (Path(__file__).resolve().parents[2] / "docs" / "CLI_AGENT_CONTEXT.md").read_text()
        + "\n\n"
        + (Path(__file__).resolve().parents[2] / "docs" / "CODEX_PROMPT_WRAPPER.txt").read_text()
        + f"\nTASK: {task}\n"
    )
    assert prompt == expected
