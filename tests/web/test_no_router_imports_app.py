"""
Circular import guard - routers must not import app from src.web_app.

Routers should use Request injection (request.app.state) instead.
This prevents circular imports and maintains clean dependency flow.
"""
import ast
from pathlib import Path

import pytest


def get_router_files():
    """Get all router files in src/web/."""
    web_dir = Path('src/web')
    return list(web_dir.glob('routes_*.py'))


def check_file_for_forbidden_imports(filepath: Path) -> list[str]:
    """Check a file for forbidden imports of app from web_app."""
    violations = []
    
    with open(filepath) as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        violations.append(f"{filepath}: SyntaxError - cannot parse")
        return violations
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and 'web_app' in node.module:
                for alias in node.names:
                    if alias.name == 'app':
                        violations.append(
                            f"{filepath}:{node.lineno}: 'from {node.module} import app' forbidden"
                        )
        
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if 'web_app' in alias.name:
                    violations.append(
                        f"{filepath}:{node.lineno}: 'import {alias.name}' - accessing app is forbidden"
                    )
    
    return violations


class TestNoRouterImportsApp:
    """Ensure routers don't import app from src.web_app."""

    def test_router_files_exist(self):
        """At least one router file should exist."""
        routers = get_router_files()
        assert len(routers) > 0, "No router files found in src/web/"

    def test_no_circular_imports(self):
        """No router should import app from src.web_app."""
        routers = get_router_files()
        all_violations = []
        
        for router in routers:
            violations = check_file_for_forbidden_imports(router)
            all_violations.extend(violations)
        
        assert not all_violations, (
            "Circular import violations found:\n" + "\n".join(all_violations) +
            "\n\nUse 'request.app.state' instead of importing app directly."
        )

    def test_dashboard_no_app_import(self):
        """Dashboard should not import app from src.web_app."""
        dashboard = Path('src/web/dashboard.py')
        if dashboard.exists():
            violations = check_file_for_forbidden_imports(dashboard)
            assert not violations, (
                "Dashboard circular import violations:\n" + "\n".join(violations)
            )
