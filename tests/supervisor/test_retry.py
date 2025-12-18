"""Tests for retry helper."""

from unittest.mock import MagicMock

import pytest
import httpx

from src.supervisor.retry import with_retry, RetryableError, RETRYABLE_STATUS_CODES


class TestRetryHelper:
    """Tests for the async retry helper."""

    def test_module_imports(self):
        """Test that the retry module imports correctly."""
        from src.supervisor import retry

        assert hasattr(retry, "with_retry")
        assert hasattr(retry, "RetryableError")
        assert hasattr(retry, "get_retry_client")
        assert hasattr(retry, "RETRYABLE_STATUS_CODES")

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        """Test that successful function returns immediately."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await with_retry(success_func, operation_name="test")

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        """Test that transient errors trigger retry."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Transient failure")
            return "success"

        result = await with_retry(
            flaky_func,
            operation_name="test",
            max_retries=3,
            base_delay=0.01,
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_stops_after_max_retries(self):
        """Test that retries stop after max attempts."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Always fails")

        with pytest.raises(RetryableError):
            await with_retry(
                always_fails,
                operation_name="test",
                max_retries=3,
                base_delay=0.01,
            )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx_except_429(self):
        """Test that 4xx errors (except 429) don't trigger retry."""
        call_count = 0

        mock_response = MagicMock()
        mock_response.status_code = 404

        async def not_found_func():
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=mock_response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(
                not_found_func,
                operation_name="test",
                max_retries=3,
                base_delay=0.01,
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        """Test that 429 (rate limit) triggers retry."""
        call_count = 0

        mock_response = MagicMock()
        mock_response.status_code = 429

        async def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=mock_response,
                )
            return "success"

        result = await with_retry(
            rate_limited_func,
            operation_name="test",
            max_retries=3,
            base_delay=0.01,
        )

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_5xx(self):
        """Test that 5xx errors trigger retry."""
        for status_code in [500, 502, 503, 504]:
            call_count = 0

            mock_response = MagicMock()
            mock_response.status_code = status_code

            async def server_error_func():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise httpx.HTTPStatusError(
                        f"Server error {status_code}",
                        request=MagicMock(),
                        response=mock_response,
                    )
                return "success"

            result = await with_retry(
                server_error_func,
                operation_name="test",
                max_retries=3,
                base_delay=0.01,
            )

            assert result == "success"
            assert call_count == 2, f"Failed for status code {status_code}"

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """Test that timeout exceptions trigger retry."""
        call_count = 0

        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            return "success"

        result = await with_retry(
            timeout_func,
            operation_name="test",
            max_retries=3,
            base_delay=0.01,
        )

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Test that connection errors trigger retry."""
        call_count = 0

        async def connection_error_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            return "success"

        result = await with_retry(
            connection_error_func,
            operation_name="test",
            max_retries=3,
            base_delay=0.01,
        )

        assert result == "success"
        assert call_count == 2

    def test_retryable_status_codes(self):
        """Test that the expected status codes are retryable."""
        assert 429 in RETRYABLE_STATUS_CODES
        assert 500 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES

        assert 400 not in RETRYABLE_STATUS_CODES
        assert 401 not in RETRYABLE_STATUS_CODES
        assert 403 not in RETRYABLE_STATUS_CODES
        assert 404 not in RETRYABLE_STATUS_CODES
