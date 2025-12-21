# MVP Schedule Configuration

To schedule the Testnet Covered Call MVP to run periodically (e.g., every hour), add the following to your crontab on the VPS:

```bash
# Run MVP cycle every hour at minute 0
0 * * * * /opt/llm-brain/llm-brain/scripts/run_mvp_cycle.sh >> /var/log/mvp_cron.log 2>&1
```

Ensure `SUPERVISOR_API_URL` is set if running on a different port, though the script defaults to localhost:8080.
The script calls the local API endpoint, so no external auth is needed (localhost is allowlisted).
