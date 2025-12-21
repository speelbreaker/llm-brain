"""Entry point for running PR Supervisor as a module."""

import logging
import sys

import uvicorn

from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    settings = get_settings()

    if not settings.enabled:
        logger.warning(
            "PR Supervisor is DISABLED. Set SUPERVISOR_ENABLED=1 to enable.\n"
            "Starting in health-check only mode..."
        )
    else:
        logger.info("PR Supervisor is ENABLED")

        if not settings.github_webhook_secret:
            logger.error("GITHUB_WEBHOOK_SECRET is required when enabled")
            sys.exit(1)

        if not settings.github_token:
            logger.error("GITHUB_TOKEN is required when enabled")
            sys.exit(1)

    uvicorn.run(
        "src.supervisor.app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
