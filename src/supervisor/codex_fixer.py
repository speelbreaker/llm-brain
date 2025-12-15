"""Codex CLI invocation for auto-fixes."""

import asyncio
import os
from typing import Optional

from .config import SupervisorSettings
from .models import ArbiterDecision, VerificationReport


class CodexFixer:
    """Invokes Codex CLI to apply minimal fixes."""
    
    def __init__(self, settings: SupervisorSettings):
        self.settings = settings
        self.max_prompt_chars = 6000
    
    def build_fix_prompt(
        self,
        arbiter_decision: ArbiterDecision,
        verification: VerificationReport,
        changed_files: list[str],
    ) -> str:
        """Build a constrained prompt for Codex."""
        objectives = "\n".join(f"- {obj}" for obj in arbiter_decision.fix_objectives)
        
        failure_excerpt = verification.failure_summary[:2000]
        
        files_list = "\n".join(f"- {f}" for f in changed_files[:15])
        
        prompt = f"""Fix the following test/lint failures with MINIMAL changes.

## Fix Objectives
{objectives}

## Failing Output
```
{failure_excerpt}
```

## Changed Files in PR
{files_list}

## CONSTRAINTS (CRITICAL)
1. Make MINIMAL changes - only what's needed to fix the failures
2. Do NOT refactor or improve unrelated code
3. Do NOT change existing behavior
4. Add tests only if explicitly needed
5. Keep the same coding style as existing code
6. Do not modify any files outside the changed files list unless absolutely necessary

Focus only on fixing the specific failures. Be surgical and precise."""

        return prompt[:self.max_prompt_chars]
    
    async def run_codex(
        self,
        workspace_path: str,
        prompt: str,
    ) -> tuple[bool, str]:
        """Run Codex CLI in the workspace."""
        codex_bin = self.settings.codex_bin
        model = self.settings.codex_model
        
        env = os.environ.copy()
        env["CODEX_WORKDIR"] = workspace_path
        
        cmd = [
            codex_bin,
            "--model", model,
            "--approval-mode", "full-auto",
            "--quiet",
            prompt,
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, "Codex timed out after 5 minutes"
            
            output = stdout.decode(errors="replace")
            if stderr:
                output += "\n" + stderr.decode(errors="replace")
            
            success = process.returncode == 0
            return success, output[:5000]
        
        except FileNotFoundError:
            return False, f"Codex binary not found: {codex_bin}"
        except Exception as e:
            return False, f"Error running Codex: {str(e)}"
    
    async def apply_fix(
        self,
        workspace_path: str,
        arbiter_decision: ArbiterDecision,
        verification: VerificationReport,
        changed_files: list[str],
    ) -> tuple[bool, str]:
        """Build prompt and run Codex to apply fixes."""
        prompt = self.build_fix_prompt(arbiter_decision, verification, changed_files)
        return await self.run_codex(workspace_path, prompt)
