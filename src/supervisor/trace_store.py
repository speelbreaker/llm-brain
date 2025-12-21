import json
import logging
from pathlib import Path
from typing import List, Optional
from collections import deque

from .models_mvp import DecisionTrace
from .config import get_settings
from .redact import redact_job_for_api

logger = logging.getLogger(__name__)

class TraceStore:
    def __init__(self, path: Optional[str] = None):
        if path:
            self.path = Path(path)
        else:
            settings = get_settings()
            # Use base_jobs_dir from settings or default to /var/lib/pr_supervisor
            base = Path(getattr(settings, "base_jobs_dir", "/var/lib/pr_supervisor"))
            self.path = base / "mvp_traces.jsonl"
        
        # Ensure directory exists
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create trace store directory {self.path.parent}: {e}")

    def append_trace(self, trace: DecisionTrace) -> None:
        """Append a trace to the log file after redaction."""
        settings = get_settings()
        trace_dict = trace.model_dump()
        # Redact secrets
        safe_trace = redact_job_for_api(trace_dict, settings)
        
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_trace) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trace to {self.path}: {e}")

    def list_traces(self, limit: int = 50) -> List[DecisionTrace]:
        """List the N most recent traces."""
        traces = []
        if not self.path.exists():
            return []
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                # Use deque to keep only last N lines efficiently
                last_lines = deque(f, maxlen=limit)
                
            for line in reversed(last_lines): # Newest first
                try:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    traces.append(DecisionTrace(**data))
                except Exception:
                    continue # Skip corrupted lines
        except Exception as e:
            logger.error(f"Failed to list traces from {self.path}: {e}")
            
        return traces

    def get_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        """Get a specific trace by ID."""
        if not self.path.exists():
            return None
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                # Linear scan. For very large files, this should be optimized.
                # For MVP, it's acceptable.
                for line in f:
                    try:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("trace_id") == trace_id:
                            return DecisionTrace(**data)
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            
        return None
