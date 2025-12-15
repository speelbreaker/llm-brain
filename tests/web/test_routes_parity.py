"""
Route parity test - ensures routes match the committed snapshot.

This test prevents silent path/method regressions during refactors.
If you intentionally add/remove/modify routes, update expected_routes.json.

Last updated: 2025-12-15 (web layer refactor to src/web/ package)
"""
import json
from pathlib import Path

import pytest


def get_current_routes():
    """Get current routes from the app, excluding OpenAPI/docs routes."""
    from src.web_app import app

    routes = []
    excluded_paths = {'/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc'}
    
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            if route.path in excluded_paths:
                continue
            methods = sorted([m for m in route.methods if m not in ['HEAD', 'OPTIONS']])
            if methods:
                routes.append({
                    'path': route.path,
                    'methods': methods,
                    'name': getattr(route, 'name', None)
                })
    
    return sorted(routes, key=lambda r: (r['path'], r['methods']))


def load_expected_routes():
    """Load the expected routes from the snapshot file."""
    snapshot_path = Path(__file__).parent / 'expected_routes.json'
    with open(snapshot_path) as f:
        return json.load(f)


class TestRouteParity:
    """Ensure routes match the committed snapshot."""

    def test_route_count_matches(self):
        """Route count should match snapshot."""
        current = get_current_routes()
        expected = load_expected_routes()
        assert len(current) == len(expected), (
            f"Route count mismatch: {len(current)} current vs {len(expected)} expected. "
            "If intentional, update tests/web/expected_routes.json"
        )

    def test_all_expected_routes_exist(self):
        """All expected routes should exist in current app."""
        current = get_current_routes()
        expected = load_expected_routes()
        
        current_paths = {(r['path'], tuple(r['methods'])) for r in current}
        
        missing = []
        for route in expected:
            key = (route['path'], tuple(route['methods']))
            if key not in current_paths:
                missing.append(f"{route['methods']} {route['path']}")
        
        assert not missing, f"Missing routes: {missing}"

    def test_no_unexpected_routes(self):
        """No unexpected routes should exist."""
        current = get_current_routes()
        expected = load_expected_routes()
        
        expected_paths = {(r['path'], tuple(r['methods'])) for r in expected}
        
        unexpected = []
        for route in current:
            key = (route['path'], tuple(route['methods']))
            if key not in expected_paths:
                unexpected.append(f"{route['methods']} {route['path']}")
        
        assert not unexpected, (
            f"Unexpected routes: {unexpected}. "
            "If intentional, update tests/web/expected_routes.json"
        )

    def test_route_names_match(self):
        """Route names should match snapshot."""
        current = get_current_routes()
        expected = load_expected_routes()
        
        current_by_path = {(r['path'], tuple(r['methods'])): r['name'] for r in current}
        
        mismatches = []
        for route in expected:
            key = (route['path'], tuple(route['methods']))
            if key in current_by_path:
                current_name = current_by_path[key]
                if current_name != route['name']:
                    mismatches.append(
                        f"{route['path']}: expected name '{route['name']}', got '{current_name}'"
                    )
        
        assert not mismatches, f"Route name mismatches: {mismatches}"
