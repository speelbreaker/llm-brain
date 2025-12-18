#!/usr/bin/env bash
set -euo pipefail

docker rm -f pr-supervisor >/dev/null 2>&1 || true

docker run -d \
  --name pr-supervisor \
  --restart unless-stopped \
  --network docker_default \
  -p 127.0.0.1:8080:8080 \
  --env-file /opt/llm-brain/llm-brain/docker/pr-supervisor.env \
  -e PYTHONPATH=/app \
  -v docker_supervisor_data:/var/lib/pr_supervisor \
  -v /opt/llm-brain/llm-brain/src:/app/src:ro \
  -v /opt/llm-brain/llm-brain/tests:/app/tests:ro \
  --health-cmd='python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8080/health\").read()"' \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=15s \
  docker-supervisor \
  uvicorn src.supervisor.app:app --host 0.0.0.0 --port 8080
