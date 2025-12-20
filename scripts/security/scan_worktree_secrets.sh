#!/bin/bash
set -euo pipefail

PATH_ORIG="${PATH:-}"
PATH_EMPTY=0
if [ -z "$PATH_ORIG" ]; then
  PATH="/usr/bin:/bin"
  PATH_EMPTY=1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
repo_path="$root_dir"

TOOL_MISSING_EXIT=2
TOOL_ERROR_EXIT=3
GITLEAKS_VERSION="8.21.0"
GITLEAKS_IMAGE="${GITLEAKS_IMAGE:-ghcr.io/gitleaks/gitleaks:latest}"

run_gitleaks() { gitleaks "$@"; }

tool_missing() {
  echo "WARN: gitleaks unavailable. Install gitleaks, ensure Docker can pull $GITLEAKS_IMAGE, or allow the temp download fallback." >&2
  exit "$TOOL_MISSING_EXIT"
}

tool_error() {
  echo "ERROR: gitleaks failed: $1" >&2
  exit "$TOOL_ERROR_EXIT"
}

ensure_gitleaks() {
  if [ "$PATH_EMPTY" -eq 1 ]; then
    tool_missing
  fi
  if command -v gitleaks >/dev/null 2>&1; then
    run_gitleaks() { gitleaks "$@"; }
    return
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "gitleaks not found locally; trying docker image $GITLEAKS_IMAGE..." >&2
    if docker run --rm "$GITLEAKS_IMAGE" version >/dev/null 2>&1; then
      repo_path="/repo"
      run_gitleaks() {
        docker run --rm -v "$root_dir":/repo:ro -w /repo "$GITLEAKS_IMAGE" "$@"
      }
      return
    fi
    if docker pull "$GITLEAKS_IMAGE" >/dev/null 2>&1 \
      && docker run --rm "$GITLEAKS_IMAGE" version >/dev/null 2>&1; then
      repo_path="/repo"
      run_gitleaks() {
        docker run --rm -v "$root_dir":/repo:ro -w /repo "$GITLEAKS_IMAGE" "$@"
      }
      return
    fi
    echo "docker image unavailable; falling back to temp binary..." >&2
  fi

  echo "gitleaks not found; downloading temp binary (no install required)..." >&2
  tmp_dir="$(mktemp -d)"
  cleanup() { rm -rf "$tmp_dir"; }
  trap cleanup EXIT

  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    linux) ;;
    *) tool_missing ;;
  esac
  case "$arch" in
    x86_64|amd64) arch="x64" ;;
    *) tool_missing ;;
  esac

  url="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${arch}.tar.gz"
  tarball="$tmp_dir/gitleaks.tar.gz"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$tarball" || tool_missing
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tarball" "$url" || tool_missing
  else
    tool_missing
  fi

  tar -xzf "$tarball" -C "$tmp_dir" || tool_missing
  "$tmp_dir/gitleaks" version >/dev/null 2>&1 || tool_missing
  run_gitleaks() { "$tmp_dir/gitleaks" "$@"; }
}

ensure_gitleaks

if run_gitleaks detect \
  --source "$repo_path" \
  --no-git \
  --redact \
  --config "$repo_path/.gitleaks.toml"; then
  exit 0
fi

status=$?
if [ "$status" -eq 1 ]; then
  echo "Leaks detected." >&2
  exit 1
fi

tool_error "exit code $status"
