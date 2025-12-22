context-pack:
	python3 scripts/gen_repo_manifest.py
	bash scripts/gen_recent_diff.sh
	python3 scripts/gen_fidelity_latest_docs.py

extras:
	python3 scripts/gen_ops_health_latest.py

context-pack-push: context-pack extras
	@if [ ! -f docs/TEST_SUMMARY_latest.txt ]; then \
		printf "%s\n%s\n" "$$(date -u +%Y-%m-%dT%H:%MZ)" "pytest summary unavailable" > docs/TEST_SUMMARY_latest.txt; \
	fi
	cp ROADMAP_BACKLOG.md docs/ROADMAP_BACKLOG_latest.md
	@echo "Updated docs/ROADMAP_BACKLOG_latest.md (upload handled externally)"
