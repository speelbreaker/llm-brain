#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

run_gitleaks() {
  gitleaks "$@"
}

ensure_gitleaks() {
  if command -v gitleaks >/dev/null 2>&1; then
    run_gitleaks() { gitleaks "$@"; }
    return
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "gitleaks not found locally; using docker image..." >&2
    run_gitleaks() {
      docker run --rm -v "$root_dir":/repo zricethezav/gitleaks:8.18.4 "$@"
    }
    return
  fi

  echo "gitleaks not found; downloading temp binary (no install required)..." >&2
  tmp_dir="$(mktemp -d)"
  cleanup() { rm -rf "$tmp_dir"; }
  trap cleanup EXIT

  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) arch="x64" ;;
    arm64|aarch64) arch="arm64" ;;
    *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
  esac

  case "$os" in
    linux|darwin) ;;
    *) echo "Unsupported OS: $os" >&2; exit 1 ;;
  esac

  url="https://github.com/gitleaks/gitleaks/releases/download/v8.18.4/gitleaks_8.18.4_${os}_${arch}.tar.gz"
  tarball="$tmp_dir/gitleaks.tar.gz"

  if command -v curl >/dev/null 2>&1; then
    curl -sSL "$url" -o "$tarball"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tarball" "$url"
  else
    echo "Install curl or wget to download gitleaks" >&2
    exit 1
  fi

  tar -xzf "$tarball" -C "$tmp_dir"
  run_gitleaks() { "$tmp_dir/gitleaks" "$@"; }
}

ensure_gitleaks

mode="${1:-shallow}"

if [[ "$mode" == "deep" ]]; then
  echo "Running deep history scan (full git history)..."
  run_gitleaks detect --redact --config "$root_dir/.gitleaks.toml" --source "$root_dir" --log-opts="--all"
else
  echo "Running lightweight history scan (current tree)..."
  run_gitleaks detect --redact --config "$root_dir/.gitleaks.toml" --source "$root_dir" --no-git
fi
