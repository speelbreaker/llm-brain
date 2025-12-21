#!/usr/bin/env bash
set -euo pipefail

IMAGE="$(docker inspect pr-supervisor --format '{{.Config.Image}}' 2>/dev/null || true)"
if [ -z "$IMAGE" ]; then
  IMAGE="docker-supervisor"
fi
echo "IMAGE=$IMAGE"

docker rm -f pr-supervisor >/dev/null 2>&1 || true

BUILD_ID="$(git rev-parse HEAD)"

docker run -d \
  --name pr-supervisor \
  --restart unless-stopped \
  --network docker_default \
  -p 127.0.0.1:8080:8080 \
  --env-file docker/pr-supervisor.env \
  -e BUILD_ID="${BUILD_ID}" \
  -e PYTHONPATH=/app \
  -v docker_supervisor_data:/var/lib/pr_supervisor \
  -v "$(pwd)":/app:rw \
  -w /app \
  --health-cmd='python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8080/health\").read()"' \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=15s \
  "$IMAGE" \
  uvicorn src.supervisor.app:app --host 0.0.0.0 --port 8080
