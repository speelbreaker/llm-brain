import asyncio
import pytest

from src.supervisor.app import app
from src.supervisor.config import get_settings
from src.supervisor.store import JobStore


@pytest.fixture(autouse=True)
def supervisor_app_state(tmp_path):
    settings = get_settings()
    app.state.settings = settings
    app.state.ready = True
    app.state.startup_errors = []
    app.state.job_queue = asyncio.Queue()
    app.state.store = JobStore(str(tmp_path / "jobs.jsonl"))
    yield
