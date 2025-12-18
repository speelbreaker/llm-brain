"""
Deploy webhook routes for GitHub Actions auto-sync.

Provides a /deploy-hook endpoint that GitHub Actions can call to trigger
a git pull and application restart when changes are pushed to GitHub.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel


router = APIRouter(tags=["deploy"])


class DeployResponse(BaseModel):
    status: str
    message: str
    git_output: Optional[str] = None


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature (HMAC-SHA256)."""
    if not signature.startswith("sha256="):
        return False
    
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected}", signature)


@router.post("/deploy-hook", response_model=DeployResponse)
async def deploy_hook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    x_deploy_token: Optional[str] = Header(None, alias="X-Deploy-Token"),
):
    """
    Webhook endpoint for GitHub Actions to trigger git pull and restart.
    
    Security: Requires either:
    - GitHub webhook signature (X-Hub-Signature-256 header)
    - Simple deploy token (X-Deploy-Token header)
    
    Returns status of git pull operation.
    """
    secret = os.environ.get("DEPLOY_WEBHOOK_SECRET", "")
    
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="Deploy webhook not configured (missing DEPLOY_WEBHOOK_SECRET)"
        )
    
    body = await request.body()
    
    if x_hub_signature_256:
        if not _verify_signature(body, x_hub_signature_256, secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
    elif x_deploy_token:
        if not hmac.compare_digest(x_deploy_token, secret):
            raise HTTPException(status_code=401, detail="Invalid deploy token")
    else:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication (X-Hub-Signature-256 or X-Deploy-Token)"
        )
    
    try:
        result = subprocess.run(
            ["git", "pull", "--rebase", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/home/runner/workspace"
        )
        
        git_output = result.stdout + result.stderr
        
        if result.returncode != 0:
            return DeployResponse(
                status="error",
                message="Git pull failed",
                git_output=git_output[:500]
            )
        
        return DeployResponse(
            status="success",
            message="Code synced from GitHub. Restart may be needed for changes to take effect.",
            git_output=git_output[:500]
        )
        
    except subprocess.TimeoutExpired:
        return DeployResponse(
            status="error",
            message="Git pull timed out"
        )
    except Exception as e:
        return DeployResponse(
            status="error",
            message=f"Git pull error: {str(e)[:200]}"
        )


@router.get("/deploy-status")
async def deploy_status():
    """Check if deploy webhook is configured and ready."""
    secret = os.environ.get("DEPLOY_WEBHOOK_SECRET", "")
    
    return {
        "configured": bool(secret),
        "message": "Deploy webhook ready" if secret else "Missing DEPLOY_WEBHOOK_SECRET"
    }
