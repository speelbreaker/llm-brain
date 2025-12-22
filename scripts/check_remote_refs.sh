#!/bin/bash
# Script to validate remote git references
# Ensures that both HEAD and main branch references exist in the remote repository

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Validating remote git references..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Check remote HEAD reference
echo -n "Checking remote HEAD reference... "
if HEAD_REF=$(git ls-remote origin HEAD 2>&1) && [ -n "$HEAD_REF" ]; then
    HEAD_COMMIT=$(echo "$HEAD_REF" | awk '{print $1}')
    echo -e "${GREEN}✓ Found${NC}"
    echo "  HEAD -> $HEAD_COMMIT"
else
    echo -e "${RED}✗ Not found${NC}"
    exit 1
fi

# Check remote main branch reference
echo -n "Checking remote main branch reference... "
if MAIN_REF=$(git ls-remote origin refs/heads/main 2>&1) && [ -n "$MAIN_REF" ]; then
    MAIN_COMMIT=$(echo "$MAIN_REF" | awk '{print $1}')
    echo -e "${GREEN}✓ Found${NC}"
    echo "  refs/heads/main -> $MAIN_COMMIT"
else
    echo -e "${RED}✗ Not found${NC}"
    exit 1
fi

# Verify HEAD points to main
if [ "$HEAD_COMMIT" = "$MAIN_COMMIT" ]; then
    echo -e "${GREEN}✓ HEAD correctly points to main branch${NC}"
    echo -e "\n${GREEN}All remote reference checks passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Warning: HEAD ($HEAD_COMMIT) does not point to main ($MAIN_COMMIT)${NC}"
    echo -e "\n${YELLOW}Remote reference checks passed with warnings${NC}"
    exit 0
fi
