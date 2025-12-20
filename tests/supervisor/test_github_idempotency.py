"""Tests for GitHub comment idempotency."""

from unittest.mock import AsyncMock

import pytest

from src.supervisor.github import GitHubClient


@pytest.mark.asyncio
async def test_post_pr_comment_once_skips_duplicate(monkeypatch):
    client = GitHubClient(token="dummy")
    marker = "<!-- supervisor:autofix:abc -->"

    monkeypatch.setattr(client, "get_pr_comments", AsyncMock(return_value=[{"body": marker}]))
    post_mock = AsyncMock()
    monkeypatch.setattr(client, "post_pr_comment", post_mock)

    posted = await client.post_pr_comment_once("owner/repo", 1, "body", marker)

    assert posted is False
    post_mock.assert_not_called()


@pytest.mark.asyncio
async def test_post_pr_comment_once_posts_when_missing(monkeypatch):
    client = GitHubClient(token="dummy")
    marker = "<!-- supervisor:autofix:def -->"

    monkeypatch.setattr(client, "get_pr_comments", AsyncMock(return_value=[{"body": "other"}]))
    post_mock = AsyncMock()
    monkeypatch.setattr(client, "post_pr_comment", post_mock)

    posted = await client.post_pr_comment_once("owner/repo", 1, "body", marker)

    assert posted is True
    post_mock.assert_called_once()
