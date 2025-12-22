from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


class ApiErrorCode(str, Enum):
    INVALID_DATE_FORMAT = "INVALID_DATE_FORMAT"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    INVALID_REQUEST = "INVALID_REQUEST"

    BACKTEST_ALREADY_RUNNING = "BACKTEST_ALREADY_RUNNING"

    NO_HARVESTED_FILES = "NO_HARVESTED_FILES"
    HARVEST_STALE = "HARVEST_STALE"

    FIDELITY_UNTRUSTED = "FIDELITY_UNTRUSTED"


def api_error(
    *,
    status_code: int,
    code: ApiErrorCode | str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": {
            "code": str(code),
            "message": message,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)
