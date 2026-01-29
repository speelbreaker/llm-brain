"""
Supervisor routes adapted for the main application.
Wraps the functionality from src/supervisor/app.py into an APIRouter.
"""
from fastapi import APIRouter, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List

from src.supervisor.app import (
    health as supervisor_health,
    diag as supervisor_diag,
    github_webhook,
    list_jobs,
    get_job,
    HealthResponse,
    JobResponse
)
from src.supervisor.config import get_settings as get_supervisor_settings
from src.supervisor.store import JobStore
import asyncio

router = APIRouter(prefix="/api/supervisor", tags=["Supervisor"])

# We need to ensure the main app's state has the supervisor components initialized.
# This logic will be handled in the main app's startup event, but we define
# the dependency here or expect request.app.state to be populated.

@router.get("/health", response_model=HealthResponse)
async def health_endpoint(request: Request):
    """Health check for supervisor module."""
    return await supervisor_health(request)

@router.get("/diag")
async def diag_endpoint(request: Request):
    """Diagnostic info for supervisor module."""
    return await supervisor_diag(request)

@router.post("/github/webhook")
async def webhook_endpoint(
    request: Request,
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
):
    """Handle GitHub PR webhooks."""
    return await github_webhook(request, x_hub_signature_256, x_github_event)

@router.get("/jobs")
async def list_jobs_endpoint(request: Request, limit: int = 50):
    """List recent supervisor jobs."""
    return await list_jobs(request, limit)


@router.get("/jobs/{job_id}")
async def get_job_endpoint(request: Request, job_id: str):
    """Get a specific job details."""
    return await get_job(request, job_id)
