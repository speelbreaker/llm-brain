# Repo Context Pack

This repo ships two generated artifacts to help LLMs stay repo-aware without bundling zips.

## Outputs

- `docs/REPO_MANIFEST.json`: repo tree, git metadata, hotspots, endpoints index, and important paths.
- `docs/RECENT_DIFF.md`: recent git history and diff from a base ref.

## Usage

Generate both artifacts:

```bash
make context-pack
```

Or run each generator directly:

```bash
python3 scripts/gen_repo_manifest.py
bash scripts/gen_recent_diff.sh
```

## Notes

- `docs/RECENT_DIFF.md` redacts lines containing common secret env keys.
- Large diffs are truncated after ~2000 lines and marked with `TRUNCATED`.
- The diff base is `origin/main` when available, otherwise `HEAD~10`.
