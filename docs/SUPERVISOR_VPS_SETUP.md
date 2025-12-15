# PR Supervisor VPS Setup Guide

This guide walks through deploying the PR Supervisor on a VPS using Docker.

## Prerequisites

- VPS with Docker and Docker Compose installed
- GitHub repository with webhook access
- OpenAI API key
- (Optional) Telegram bot for notifications

## Step 1: Create GitHub Webhook

1. Go to your repository Settings > Webhooks > Add webhook
2. Set the Payload URL to: `https://your-vps-domain:8080/webhook`
3. Set Content type to: `application/json`
4. Create a secret and save it for `GITHUB_WEBHOOK_SECRET`
5. Select events: "Pull requests"
6. Create the webhook

## Step 2: Create GitHub Token

1. Go to GitHub Settings > Developer settings > Personal access tokens > Fine-grained tokens
2. Create a new token with these permissions:
   - Repository access: Select your repo
   - Permissions:
     - Pull requests: Read and write
     - Contents: Read and write (for pushing fixes)
3. Save the token for `GITHUB_TOKEN`

## Step 3: Configure Environment

```bash
cd docker
cp .env.supervisor.example .env.supervisor
```

Edit `.env.supervisor` with your values:

```bash
# Required
SUPERVISOR_ENABLED=1
GITHUB_WEBHOOK_SECRET=your-webhook-secret
GITHUB_TOKEN=ghp_your-token

# OpenAI (required for debate system)
OPENAI_API_KEY=sk-your-key

# Optional: Telegram
SUPERVISOR_TELEGRAM_ENABLED=1
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
TELEGRAM_ALLOWED_USER_IDS=123456789,987654321

# Optional: Enable Codex auto-fix
SUPERVISOR_ENABLE_CODEX=1
SUPERVISOR_AUTOFIX_POLICY=label
SUPERVISOR_AUTOFIX_LABEL=autofix-ok
```

## Step 4: Start the Supervisor

```bash
cd docker
docker compose -f docker-compose.supervisor.yml up -d
```

## Step 5: Verify Health

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "ok": true,
  "enabled": true,
  "ready": true,
  "version": "0.2.0"
}
```

## Step 6: Test Telegram Commands

If Telegram is enabled, send `/help` to your bot to see available commands.

## Step 7: Test Webhook (Debug Mode)

If `SUPERVISOR_DEBUG=1`, you can test without GitHub:

```bash
curl -X POST http://localhost:8080/debug/simulate_pr_event \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo", "pr_number": 123}'
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/supervisor last` | Show last 5 jobs |
| `/supervisor pr <n>` | Show PR status |
| `/rerun <n>` | Queue rerun for PR |
| `/autofix <n>` | Approve autofix |
| `/pause <n>` | Pause PR processing |
| `/resume <n>` | Resume PR processing |
| `/revoke <n>` | Revoke autofix approval |
| `/help` | Show help |

## Autofix Policy

The `SUPERVISOR_AUTOFIX_POLICY` setting controls when Codex can auto-fix:

| Policy | Requirement |
|--------|-------------|
| `label` | PR must have the `autofix-ok` label |
| `telegram` | User must approve via `/autofix` command |
| `both` | Requires both label AND telegram approval |

## Troubleshooting

### Supervisor not ready

Check logs:
```bash
docker compose -f docker-compose.supervisor.yml logs supervisor
```

Common issues:
- Missing `GITHUB_WEBHOOK_SECRET`
- Missing `GITHUB_TOKEN`
- Invalid token permissions

### Webhook signature failed

- Verify `GITHUB_WEBHOOK_SECRET` matches GitHub webhook settings
- Check payload is being sent as JSON

### Telegram commands not working

- Verify `TELEGRAM_ALLOWED_USER_IDS` includes your Telegram user ID
- Check bot token is valid
- Ensure `SUPERVISOR_TELEGRAM_ENABLED=1`

## Maintenance

### View logs
```bash
docker compose -f docker-compose.supervisor.yml logs -f supervisor
```

### Restart
```bash
docker compose -f docker-compose.supervisor.yml restart
```

### Update
```bash
docker compose -f docker-compose.supervisor.yml pull
docker compose -f docker-compose.supervisor.yml up -d
```

### Cleanup old workspaces

Workspaces older than `SUPERVISOR_WORKSPACE_TTL_HOURS` are automatically cleaned up when new jobs start.
