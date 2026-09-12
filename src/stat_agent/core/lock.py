"""Session-wide re-entrant lock shared by the agent runtime and GUI mutations."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class SessionBusy(RuntimeError):
    """Raised when the session lock cannot be acquired within the timeout."""

    def __init__(self, holder: str | None):
        self.holder = holder
        super().__init__(f"session is busy (held by {holder or 'unknown'})")


class SessionLock:
    """Re-entrant lock with a human-readable holder label.

    ``acquire(timeout)`` returns False instead of blocking forever; ``hold()`` is the
    context-manager form and raises :class:`SessionBusy` on timeout.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._meta = threading.Lock()
        self._holder: str | None = None
        self._depth = 0

    def acquire(self, timeout: float = -1, holder: str = "") -> bool:
        ok = self._lock.acquire(timeout=timeout)
        if ok:
            with self._meta:
                self._depth += 1
                if self._depth == 1:
                    self._holder = holder or "anonymous"
        return ok

    def release(self) -> None:
        with self._meta:
            self._depth -= 1
            if self._depth <= 0:
                self._depth = 0
                self._holder = None
        self._lock.release()

    @contextmanager
    def hold(self, holder: str = "", timeout: float = -1) -> Iterator[None]:
        if not self.acquire(timeout=timeout, holder=holder):
            raise SessionBusy(self.holder)
        try:
            yield
        finally:
            self.release()

    @property
    def holder(self) -> str | None:
        return self._holder

    @property
    def locked(self) -> bool:
        return self._depth > 0

    def __repr__(self) -> str:
        return f"SessionLock(locked={self.locked}, holder={self._holder!r})"


__all__ = ["SessionLock", "SessionBusy"]
