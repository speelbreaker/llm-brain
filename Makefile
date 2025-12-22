.PHONY: extras context-pack context-pack-extras context-pack-all context-pack-push

context-pack-extras:
	python3 scripts/gen_ops_health_latest.py
	./scripts/gen_roadmap_latest.sh
	./scripts/gen_test_summary_latest.sh
	@if [ ! -f docs/TEST_SUMMARY_latest.txt ]; then \
		printf "%s\n%s\n" "$$(date -u +%Y-%m-%dT%H:%MZ)" "pytest summary unavailable" > docs/TEST_SUMMARY_latest.txt; \
	fi
	cp ROADMAP_BACKLOG.md docs/ROADMAP_BACKLOG_latest.md
	@echo "Updated docs/ROADMAP_BACKLOG_latest.md (upload handled externally)"

extras: context-pack-extras

context-pack: context-pack-extras
	python3 scripts/gen_repo_manifest.py
	python3 scripts/gen_repo_manifest_md.py
	bash scripts/gen_recent_diff.sh
	python3 scripts/gen_fidelity_latest_docs.py

context-pack-all: context-pack

# Generation-only target used by uploader scripts.
context-pack-push: context-pack-all
