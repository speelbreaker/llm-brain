"""
In-memory chat history store for multi-turn conversations.
Provides a thread-safe buffer of recent chat messages.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List

MAX_HISTORY = 20


class ChatStore:
    """Thread-safe store for multi-turn chat history."""
    
    def __init__(self, max_size: int = MAX_HISTORY):
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def append(self, role: str, content: str) -> None:
        """Append a chat message with role ('user' or 'assistant')."""
        with self._lock:
            self._history.append({"role": role, "content": content})
    
    def get_history(self) -> List[Dict[str, str]]:
        """Return chat history as a list of {role, content} dicts."""
        with self._lock:
            return list(self._history)
    
    def clear(self) -> None:
        """Clear all chat history."""
        with self._lock:
            self._history.clear()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._history)


chat_store = ChatStore()
