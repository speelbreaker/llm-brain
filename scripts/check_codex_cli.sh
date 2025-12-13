#!/bin/bash
# Verification script for Codex CLI installation in Replit

echo "=== Codex CLI Installation Check ==="
echo ""

echo "1. Node.js version:"
node -v 2>/dev/null || echo "   NOT FOUND"

echo ""
echo "2. npm version:"
npm -v 2>/dev/null || echo "   NOT FOUND"

echo ""
echo "3. Codex CLI package:"
if [ -d "node_modules/@openai/codex" ]; then
    VERSION=$(node -e "console.log(require('@openai/codex/package.json').version)" 2>/dev/null)
    echo "   Installed: v${VERSION:-unknown}"
else
    echo "   NOT INSTALLED"
    echo "   Run: npm install"
fi

echo ""
echo "4. OPENAI_API_KEY:"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "   Configured (${#OPENAI_API_KEY} chars)"
else
    echo "   NOT SET - Add to Replit Secrets"
fi

echo ""
echo "5. package.json devDependencies:"
if [ -f "package.json" ]; then
    grep -A 5 '"devDependencies"' package.json | head -6
else
    echo "   package.json not found"
fi

echo ""
echo "=== Notes ==="
echo "- Codex CLI uses Landlock sandbox which conflicts with Replit"
echo "- Use /codex command in Telegram bot (uses OpenAI API directly)"
echo "- For full CLI, run locally: npx @openai/codex"
