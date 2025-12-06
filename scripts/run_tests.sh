#!/usr/bin/env bash
set -e

echo "=== Running Unit Tests ==="
echo ""

pytest tests/ -v --tb=short

echo ""
echo "=== ALL TESTS PASSED ==="
