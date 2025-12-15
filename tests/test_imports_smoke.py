"""
Import smoke tests - verify all critical modules can be imported.

These tests catch missing imports, syntax errors, and circular imports.
Run early in CI to fail fast on obvious issues.
"""
import pytest


class TestWebImports:
    """Smoke tests for web layer imports."""

    def test_web_app_imports(self):
        """src.web_app should import without errors."""
        from src.web_app import app
        assert app is not None

    def test_web_dashboard_imports(self):
        """src.web.dashboard should import without errors."""
        from src.web.dashboard import render_dashboard_html
        assert callable(render_dashboard_html)

    def test_web_routes_main_imports(self):
        """src.web.routes_main should import without errors."""
        from src.web.routes_main import router
        assert router is not None

    def test_web_routes_backtest_imports(self):
        """src.web.routes_backtest should import without errors."""
        from src.web.routes_backtest import router
        assert router is not None

    def test_web_routes_positions_imports(self):
        """src.web.routes_positions should import without errors."""
        from src.web.routes_positions import router
        assert router is not None

    def test_web_routes_bots_imports(self):
        """src.web.routes_bots should import without errors."""
        from src.web.routes_bots import router
        assert router is not None

    def test_web_routes_health_imports(self):
        """src.web.routes_health should import without errors."""
        from src.web.routes_health import router
        assert router is not None


class TestSupervisorImports:
    """Smoke tests for supervisor module imports."""

    def test_supervisor_app_imports(self):
        """src.supervisor.app should import without errors."""
        import src.supervisor.app
        assert hasattr(src.supervisor.app, 'app') or hasattr(src.supervisor.app, 'create_app')

    def test_supervisor_store_imports(self):
        """src.supervisor.store should import without errors."""
        import src.supervisor.store
        assert src.supervisor.store is not None

    def test_supervisor_telegram_notify_imports(self):
        """src.supervisor.telegram_notify should import without errors."""
        import src.supervisor.telegram_notify
        assert src.supervisor.telegram_notify is not None


class TestFastapiImports:
    """Verify critical FastAPI imports are available."""

    def test_header_import(self):
        """Header should be importable from fastapi."""
        from fastapi import Header
        assert Header is not None

    def test_request_import(self):
        """Request should be importable from fastapi."""
        from fastapi import Request
        assert Request is not None

    def test_apirouter_import(self):
        """APIRouter should be importable from fastapi."""
        from fastapi import APIRouter
        assert APIRouter is not None
