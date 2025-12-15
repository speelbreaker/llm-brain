"""Tests for job queue worker functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestJobQueue:
    """Tests for in-process job queue."""
    
    @pytest.mark.asyncio
    async def test_job_enqueued_and_processed(self):
        """Test that enqueued jobs are processed by the worker."""
        from src.supervisor.models import SupervisorJob
        
        job_queue = asyncio.Queue()
        processed_jobs = []
        
        mock_job = SupervisorJob(
            job_id="test-job-123",
            repo_full_name="owner/repo",
            pr_number=42,
            head_sha="abc123def",
            head_ref="feature-branch",
            base_ref="main",
            pr_url="https://github.com/owner/repo/pull/42",
        )
        
        async def mock_run_job(job, app):
            processed_jobs.append(job.job_id)
        
        async def worker_task():
            while True:
                try:
                    job, app = await asyncio.wait_for(job_queue.get(), timeout=1.0)
                    await mock_run_job(job, app)
                    job_queue.task_done()
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break
        
        worker = asyncio.create_task(worker_task())
        
        await job_queue.put((mock_job, MagicMock()))
        
        await asyncio.sleep(0.1)
        
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        
        assert "test-job-123" in processed_jobs
    
    @pytest.mark.asyncio
    async def test_worker_handles_job_error(self):
        """Test that worker continues after job processing error."""
        job_queue = asyncio.Queue()
        processed_count = 0
        
        async def failing_job_handler(job, app):
            nonlocal processed_count
            processed_count += 1
            if processed_count == 1:
                raise ValueError("Simulated error")
        
        async def worker_task():
            while True:
                try:
                    job, app = await asyncio.wait_for(job_queue.get(), timeout=0.5)
                    try:
                        await failing_job_handler(job, app)
                    except Exception:
                        pass
                    finally:
                        job_queue.task_done()
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break
        
        worker = asyncio.create_task(worker_task())
        
        from src.supervisor.models import SupervisorJob
        
        job1 = SupervisorJob(
            job_id="job-1",
            repo_full_name="owner/repo",
            pr_number=1,
            head_sha="sha1",
            head_ref="branch1",
            base_ref="main",
            pr_url="https://github.com/owner/repo/pull/1",
        )
        job2 = SupervisorJob(
            job_id="job-2",
            repo_full_name="owner/repo",
            pr_number=2,
            head_sha="sha2",
            head_ref="branch2",
            base_ref="main",
            pr_url="https://github.com/owner/repo/pull/2",
        )
        
        await job_queue.put((job1, MagicMock()))
        await job_queue.put((job2, MagicMock()))
        
        await asyncio.sleep(0.3)
        
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        
        assert processed_count == 2
    
    @pytest.mark.asyncio
    async def test_worker_graceful_shutdown(self):
        """Test that worker shuts down gracefully on cancel."""
        job_queue = asyncio.Queue()
        shutdown_clean = False
        
        async def worker_task():
            nonlocal shutdown_clean
            try:
                while True:
                    job, app = await job_queue.get()
                    job_queue.task_done()
            except asyncio.CancelledError:
                shutdown_clean = True
                raise
        
        worker = asyncio.create_task(worker_task())
        
        await asyncio.sleep(0.1)
        
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        
        assert shutdown_clean
