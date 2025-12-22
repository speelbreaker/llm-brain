import importlib.util
import os
import subprocess
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gen_repo_manifest_repo_root_matches_git():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "gen_repo_manifest.py"
    mod = load_module(script_path)

    resolved = mod.resolve_repo_root(repo_root)
    git_root = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    assert str(resolved) == git_root


def test_push_script_refuses_wrong_root():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "push_context_pack_to_drive.sh"

    env = os.environ.copy()
    env.update({
        "EXPECTED_REPO_ROOT": "/tmp/not-the-repo",
    })

    result = subprocess.run(
        [str(script_path)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout + result.stderr).strip()
    assert result.returncode == 2
    assert "Wrong repo root" in combined
