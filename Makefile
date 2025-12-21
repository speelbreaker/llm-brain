.PHONY: context-pack context-pack-extras context-pack-all context-pack-push

context-pack:
	./scripts/gen_repo_manifest.py
	./scripts/gen_repo_manifest_md.py
	./scripts/gen_recent_diff.sh

context-pack-extras:
	./scripts/gen_roadmap_latest.sh
	./scripts/gen_test_summary_latest.sh

context-pack-all: context-pack context-pack-extras

context-pack-push: context-pack-all
	CONTEXT_PACK_PUSH_DIRECT=1 ./scripts/push_context_pack_to_drive.sh
