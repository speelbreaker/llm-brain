#!/usr/bin/env python3
"""Generate deterministic Fidelity "latest" artifacts under docs/.

Copies (byte-for-byte, no reformatting) from:
  data/fidelity_runs/<UNDERLYING>/latest/fidelity_report.json
  data/fidelity_runs/<UNDERLYING>/latest/fidelity_report.md

To:
  docs/FIDELITY_<UNDERLYING>_latest.json
  docs/FIDELITY_<UNDERLYING>_latest.md

Behavior:
- Missing source files do not fail the run (exit 0); a warning is printed.
- Existing source files are copied exactly; "Wrote ..." is printed.
- Designed to be safe for context-pack generation (no server dependency).

"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_UNDERLYINGS = ("BTC", "ETH")


def _repo_root_from_script() -> Path:
    # scripts/<this_file>
    return Path(__file__).resolve().parents[1]


def _validate_repo_root(root: Path) -> None:
    # Light guard: ensures we don't accidentally operate on '/' or random cwd.
    if not (root / "pyproject.toml").exists():
        raise SystemExit(
            f"Refusing to run: repo root does not look valid (missing pyproject.toml): {root}"
        )


def _copy_if_present(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"WARN: missing fidelity source: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"Wrote {dst}")
    return True


def generate(repo_root: Path, underlyings: tuple[str, ...] = DEFAULT_UNDERLYINGS) -> int:
    _validate_repo_root(repo_root)

    for underlying in underlyings:
        symbol = underlying.strip().upper()
        if not symbol:
            continue

        src_dir = repo_root / "data" / "fidelity_runs" / symbol / "latest"
        dst_json = repo_root / "docs" / f"FIDELITY_{symbol}_latest.json"
        dst_md = repo_root / "docs" / f"FIDELITY_{symbol}_latest.md"

        _copy_if_present(src_dir / "fidelity_report.json", dst_json)
        _copy_if_present(src_dir / "fidelity_report.md", dst_md)

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Copy latest Fidelity reports (BTC/ETH) into docs/ for context-pack publishing."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (defaults to script parent repo). Used by tests.",
    )
    parser.add_argument(
        "--underlyings",
        default=",".join(DEFAULT_UNDERLYINGS),
        help="Comma-separated list of underlyings (default: BTC,ETH)",
    )

    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_script()
    underlyings = tuple(u.strip() for u in str(args.underlyings).split(",") if u.strip())

    return generate(repo_root=repo_root, underlyings=underlyings or DEFAULT_UNDERLYINGS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
