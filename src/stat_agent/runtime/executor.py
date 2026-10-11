"""Guarded in-process runtime for generated code.

The canvas and the agent must share one live ``session`` object, so code runs in this
process, in a persistent namespace. What makes it "guarded":

* an AST safety check before anything runs;
* the session lock is held for the whole run (GUI mutations wait or get 409);
* stdout/stderr are *routed*: only the run thread's writes are captured, every other
  thread keeps writing to the original streams (no global swap races);
* a timeout delivered as ``KeyboardInterrupt`` to the run thread; if the thread is stuck
  in a C call the runtime enters an ``orphaned`` state and refuses new runs;
* before/after snapshots give the ground truth of what the code changed;
* the namespace is committed only when the run succeeds.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import io
import logging
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stat_agent.core.lock import SessionBusy
from stat_agent.runtime.artifacts import CodeAccess, code_access
from stat_agent.runtime.plots import capture_figures, close_all_figures, ensure_headless_backend
from stat_agent.runtime.safety import SafetyViolation, Violation, check_code, clean_code
from stat_agent.runtime.snapshot import Snapshot, SnapshotDiff, diff_snapshots, take_snapshot

logger = logging.getLogger(__name__)

OnOutput = Callable[[str, str], None]  # (stream, line)


# ---- routed streams ----------------------------------------------------------------------------------


class _ThreadRoutedStream(io.TextIOBase):
    """Routes writes from registered threads to their handlers; others pass through."""

    def __init__(self, original, name: str) -> None:
        self._original = original
        self._name = name
        self._routes: dict[int, tuple[OnOutput, list]] = {}
        self._lock = threading.Lock()

    def register(self, ident: int, on_line: OnOutput) -> None:
        with self._lock:
            self._routes[ident] = (on_line, [""])

    def unregister(self, ident: int) -> None:
        with self._lock:
            route = self._routes.pop(ident, None)
        if route:
            on_line, buf = route
            if buf[0]:
                on_line(self._name, buf[0])

    def write(self, text: str) -> int:
        route = self._routes.get(threading.get_ident())
        if route is None:
            return self._original.write(text)
        on_line, buf = route
        text = text.replace("\r\n", "\n")
        if "\r" in text:
            head, _, tail = text.rpartition("\r")
            if "\n" in head:
                # complete lines before a progress-bar overwrite are kept
                lines = (buf[0] + head).split("\n")
                for ln in lines[:-1]:
                    on_line(self._name, ln)
            buf[0] = tail
            # a progress bar (tqdm) never emits a newline: forward its latest state at most every 2 s
            now = time.monotonic()
            if len(buf) < 2:
                buf.append(now)  # first progress line after 2 s, not immediately
            elif tail.strip() and now - buf[1] > 2.0:
                buf[1] = now
                on_line(self._name, tail.strip())
        else:
            buf[0] += text
        if "\n" in buf[0]:
            *lines, rest = buf[0].split("\n")
            for ln in lines:
                on_line(self._name, ln)
            buf[0] = rest
        return len(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._original, "encoding", "utf-8") or "utf-8"

    def fileno(self) -> int:
        return self._original.fileno()


# ---- results ------------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    success: bool
    code: str
    code_hash: str
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    figures: list[bytes] = field(default_factory=list)
    duration_s: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    refused: bool = False
    violations: list[Violation] = field(default_factory=list)
    snapshot_before: Snapshot | None = None
    snapshot_after: Snapshot | None = None
    diff: SnapshotDiff | None = None
    access: CodeAccess | None = None
    namespace_added: list[str] = field(default_factory=list)

    @property
    def stdout_tail(self) -> str:
        return self.stdout[-4000:]

    def display(self) -> str:
        if not self.success:
            return f"Error: {self.error}\n\n{self.stderr}".strip()
        out = self.stdout
        if self.stderr.strip():
            out += f"\n\nWarnings:\n{self.stderr}"
        return out.strip() or "Code executed successfully (no output)"


class RuntimeBusy(RuntimeError):
    pass


SEEDED_NAMES = ("session", "np", "pd")
KEEP_RECENT = 5  # newest variables always listed, whatever they hold
MAX_ERROR_CHARS = 3000  # an exception message (and each traceback line) longer than this is cut in the middle
MAX_TRACEBACK_CHARS = 12000


def value_rank(v: Any) -> int:
    """Listing priority of a namespace value: 0 data, 1 anything else, 2 plotting handles and
    callables (the first to be dropped when there are more variables than the listing holds)."""
    try:
        if inspect.isclass(v) or inspect.isroutine(v):
            return 2
        if (getattr(type(v), "__module__", "") or "").split(".")[0] == "matplotlib":
            return 2
        if isinstance(v, (dict, list, tuple, set)) or hasattr(v, "shape") or hasattr(v, "columns"):
            return 0
    except Exception:  # noqa: BLE001  (an exotic object must never break the listing)
        return 1
    return 1


def _carries_data(v: Any) -> bool:
    """Whether a namespace value holds data computed from the session (an array, a table, an AnnData, or a list,
    tuple, set or dict of them), as opposed to a list of labels, a path, a number, a module or a function."""
    try:
        if hasattr(v, "shape") or hasattr(v, "columns"):
            return True
        held = v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple, set)) else ()
        return any(hasattr(x, "shape") or hasattr(x, "columns") for x in list(held)[:50])
    except Exception:  # noqa: BLE001  (an exotic object must never break a run)
        return False


def describe_value(v: Any) -> str:
    """One short line about a namespace value (type, shape, the keys a later step needs)."""
    try:
        tname = type(v).__name__
        if inspect.ismodule(v):
            return "module"
        if inspect.isfunction(v) or inspect.isbuiltin(v):
            return "function"
        if inspect.isclass(v):
            return "class"
        if isinstance(v, (bool, int, float, complex)):
            return f"{tname} = {v!r}"[:80]
        if isinstance(v, str):
            r = repr(v)
            return f"str = {r[:70]}{'…' if len(r) > 70 else ''}"
        if isinstance(v, (list, tuple, set, frozenset)):
            return f"{tname} ({len(v)} items)"
        if isinstance(v, dict):
            keys = ", ".join(str(k) for k in list(v)[:8])
            return f"dict ({len(v)} keys: {keys}{', …' if len(v) > 8 else ''})"
        if tname == "AnnData":
            obs = ", ".join(str(c) for c in list(v.obs.columns)[:12]) + (", …" if v.obs.shape[1] > 12 else "")
            obsm = ", ".join(str(k) for k in list(v.obsm.keys())[:6])
            return f"AnnData ({v.n_obs:,} × {v.n_vars:,}; obs: {obs}" + (f"; obsm: {obsm}" if obsm else "") + ")"
        if tname == "DataFrame":
            cols = ", ".join(str(c) for c in list(v.columns)[:10]) + (", …" if v.shape[1] > 10 else "")
            return f"DataFrame ({v.shape[0]:,} × {v.shape[1]}; columns: {cols})"
        if tname == "Series":
            return f"Series ({len(v):,}, {v.dtype}" + (f", name {v.name!r}" if v.name is not None else "") + ")"
        if hasattr(v, "shape"):
            shape = tuple(v.shape)
            return f"{tname} (shape {shape}" + (f", {v.dtype}" if hasattr(v, "dtype") else "") + ")"
        if hasattr(v, "__len__"):
            return f"{tname} ({len(v)} items)"
        return tname
    except Exception:  # noqa: BLE001 - a description must never fail a turn
        return type(v).__name__


# ---- runtime --------------------------------------------------------------------------------------------------


class Runtime:
    _process_lock = threading.Lock()

    def __init__(
        self,
        session,
        *,
        timeout_s: float = 1800.0,
        safe_mode: bool = True,
        grace_s: float = 10.0,
        lock_timeout_s: float = 5.0,
        snapshot: bool = True,
        workdir: Path | None = None,
    ) -> None:
        import numpy as np
        import pandas as pd

        self.session = session
        self.timeout_s = timeout_s
        self.safe_mode = safe_mode
        self.grace_s = grace_s
        self.lock_timeout_s = lock_timeout_s
        self.snapshot_enabled = snapshot
        self.workdir = Path(workdir) if workdir else getattr(session, "workdir", None)
        self.namespace: dict[str, Any] = {"session": session, "np": np, "pd": pd}
        self._regions_of: dict[str, set[str]] = {}  # namespace variable -> the regions read by the blocks it was computed from
        self.orphaned: bool = False
        self._current: threading.Thread | None = None
        self._cancel = threading.Event()
        self._cancel_requested = False  # set by cancel(); distinguishes a user stop from a timeout
        self.history: list[RunResult] = []
        ensure_headless_backend()

    # ---- namespace ---------------------------------------------------------------------------------
    def reset_namespace(self) -> None:
        import numpy as np
        import pandas as pd

        self.namespace = {"session": self.session, "np": np, "pd": pd}
        self._regions_of = {}

    def user_variables(self) -> list[str]:
        """Names user code left in the namespace (oldest first): no seeds, no modules, no private names."""
        return [k for k, v in self.namespace.items() if not k.startswith("_") and k not in SEEDED_NAMES and not inspect.ismodule(v)]

    def _derived_names(self) -> set[str]:
        """Namespace variables holding an AnnData that is not a session slice (a pooled or subset object
        kept from an earlier block): a read on one of them is not a read of any slice."""
        import anndata as ad

        slice_adatas = {id(s.adata) for s in self.session.slices.values()}
        return {n for n, v in self.namespace.items() if n not in SEEDED_NAMES and isinstance(v, ad.AnnData) and id(v) not in slice_adatas}

    def _slice_names(self) -> set[str]:
        """Namespace variables holding a session slice or its AnnData (``adata = session.get_slice(0).adata``
        kept from an earlier block): a write on one of them is a write on that slice."""
        from stat_agent.core.data_slice import DataSlice

        slice_adatas = {id(s.adata) for s in self.session.slices.values()}
        return {n for n, v in self.namespace.items() if n not in SEEDED_NAMES and (isinstance(v, DataSlice) or id(v) in slice_adatas)}

    def _roi_vars(self) -> dict[str, str]:
        """Namespace variables holding a region (``roi = session.get_roi('tumor')`` kept from an earlier
        block), with the region's name: a block that uses one reads that region without asking for it."""
        from stat_agent.core.roi_manager import ROI

        return {n: v.name for n, v in self.namespace.items() if n not in SEEDED_NAMES and isinstance(v, ROI)}

    def _inherit_regions(self, access: CodeAccess) -> None:
        """A block that goes on from a data object an earlier block left (``cmp_ad`` built from two regions, then
        analysed in the next block, step or turn) reads the regions that object was computed from."""
        for name in access.taken:
            access.roi_names |= self._regions_of.get(name, set())

    def _remember_regions(self, access: CodeAccess) -> None:
        """After a successful block: the data objects it bound carry the regions it read (its own and the inherited
        ones; every region of the session when it read one the scan could not name). Slices and region objects are
        left out, their reads are seen directly; so are strings, numbers, modules and functions, which would hand a
        region on to every block that reuses a path or a helper."""
        regions = set(access.roi_names) | (set(self.session.roi_manager.rois) if access.roi_any else set())
        direct = self._slice_names() | set(self._roi_vars())
        for name in access.assigned & set(self.namespace):
            if regions and name not in direct and _carries_data(self.namespace[name]):
                self._regions_of[name] = regions
            else:
                self._regions_of.pop(name, None)

    def describe_namespace(self, limit: int = 40) -> list[tuple[str, str]]:
        """(name, description) for the user variables a later block is most likely to need; what the
        generator is shown. Over `limit` variables, data objects are kept before plotting handles and
        callables, so the locals of one figure cannot push every table and array out of the listing."""
        names = self.user_variables()
        if limit <= 0:
            return []
        if len(names) <= limit:
            keep = set(names)
        else:
            keep = set(names[-min(KEEP_RECENT, limit):])  # the newest few whatever they are: a continuation block may still be drawing
            for rank in (0, 1, 2):
                for n in reversed(names):
                    if len(keep) >= limit:
                        break
                    if value_rank(self.namespace[n]) == rank:
                        keep.add(n)
        return [(n, describe_value(self.namespace[n])) for n in names if n in keep]

    @property
    def busy(self) -> bool:
        return self.orphaned or (self._current is not None and self._current.is_alive())

    def cancel(self) -> None:
        self._cancel_requested = True
        self._cancel.set()
        t = self._current
        if t is not None and t.is_alive():
            _async_raise(t.ident, KeyboardInterrupt)

    # ---- run ---------------------------------------------------------------------------------------------
    async def arun(self, code: str, **kwargs: Any) -> RunResult:
        import asyncio

        return await asyncio.to_thread(self.run, code, **kwargs)

    def run(
        self,
        code: str,
        *,
        on_output: OnOutput | None = None,
        timeout_s: float | None = None,
        skills: Iterable[Any] = (),
        export_pythonpath: bool = False,
        capture_plots: bool = True,
        label: str = "runtime",
    ) -> RunResult:
        code = clean_code(code)
        run_id = uuid.uuid4().hex[:8]
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        result = RunResult(run_id=run_id, success=False, code=code, code_hash=code_hash)
        result.access = code_access(code, self._derived_names(), self._slice_names(), self._roi_vars())
        self._inherit_regions(result.access)
        if not code:
            result.error, result.error_type, result.refused = "Empty code provided", "EmptyCode", True
            return result
        if self.orphaned:
            result.error = "RuntimeBusy: a previous run is still executing inside a non-interruptible call; restart the server to recover"
            result.error_type, result.refused = "RuntimeBusy", True
            return result
        if self.safe_mode:
            allowed = [self.workdir] if self.workdir else []
            violations = check_code(code, allowed_write_dirs=allowed)
            if violations:
                result.violations = violations
                if all(v.code == "syntax_error" for v in violations):
                    # code that does not parse is a generation error the reflector can fix, not a refusal
                    result.error, result.error_type = "; ".join(str(v) for v in violations), "SyntaxError"
                    return result
                result.error = "SafetyViolation: " + "; ".join(str(v) for v in violations)
                result.error_type, result.refused = "SafetyViolation", True
                return result

        if not self.session.lock.acquire(timeout=self.lock_timeout_s, holder=label):
            raise SessionBusy(self.session.lock.holder)
        try:
            with Runtime._process_lock:
                return self._run_locked(code, result, on_output, timeout_s or self.timeout_s, list(skills), export_pythonpath, capture_plots)
        finally:
            self.session.lock.release()

    def _run_locked(self, code, result, on_output, timeout_s, skills, export_pythonpath, capture_plots) -> RunResult:
        from stat_agent.skills.loader import import_paths, scoped_sys_path

        self._cancel.clear()
        self._cancel_requested = False
        if self.snapshot_enabled:
            result.snapshot_before = take_snapshot(self.session)
        ns = dict(self.namespace)
        ns["__name__"] = "__stat_run__"
        ns["__builtins__"] = __builtins__
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def sink(stream: str, line: str) -> None:
            (stdout_lines if stream == "stdout" else stderr_lines).append(line)
            if on_output:
                try:
                    on_output(stream, line)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("on_output callback failed: %s", exc)

        out_router = _ThreadRoutedStream(sys.stdout, "stdout")
        err_router = _ThreadRoutedStream(sys.stderr, "stderr")
        box: dict[str, Any] = {}

        def worker() -> None:
            ident = threading.get_ident()
            out_router.register(ident, sink)
            err_router.register(ident, sink)
            try:
                compiled = compile(code, f"<stat-run-{result.run_id}>", "exec")
                with scoped_sys_path(import_paths(skills), export_pythonpath=export_pythonpath):
                    exec(compiled, ns)  # noqa: S102 - this is the analysis runtime
                box["ok"] = True
                if capture_plots:
                    box["figures"] = capture_figures()
            except KeyboardInterrupt:
                box["ok"] = False
                box["error_type"] = "Cancelled" if self._cancel_requested else "Timeout"
                box["error"] = "execution stopped by the user" if self._cancel_requested else "execution interrupted (timeout)"
                box["traceback"] = traceback.format_exc()
                close_all_figures()
            except BaseException as exc:  # noqa: BLE001
                box["ok"] = False
                if self._cancel_requested:
                    # some libraries catch KeyboardInterrupt and raise something else (lightning: ValueError)
                    box["error_type"], box["error"] = "Cancelled", "execution stopped by the user"
                else:
                    box["error_type"] = type(exc).__name__
                    box["error"] = clip_middle(f"{type(exc).__name__}: {exc}")
                box["traceback"] = _trim_traceback(traceback.format_exc(), result.run_id)
                close_all_figures()
            finally:
                out_router.unregister(ident)
                err_router.unregister(ident)

        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out_router, err_router
        start = time.perf_counter()
        thread = threading.Thread(target=worker, name=f"stat-run-{result.run_id}", daemon=True)
        self._current = thread
        try:
            thread.start()
            thread.join(timeout_s)
            if thread.is_alive():
                self._cancel.set()
                _async_raise(thread.ident, KeyboardInterrupt)
                thread.join(self.grace_s)
                result.timed_out = True
                if thread.is_alive():
                    self.orphaned = True
                    logger.error("run %s did not stop after timeout; runtime marked orphaned", result.run_id)
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
            self._current = None
        result.duration_s = time.perf_counter() - start
        result.stdout = "\n".join(stdout_lines) + ("\n" if stdout_lines else "")
        result.stderr = "\n".join(stderr_lines) + ("\n" if stderr_lines else "")

        if result.timed_out and thread.is_alive():
            result.error_type = "Timeout"
            result.error = f"Timeout: execution exceeded {timeout_s:.0f}s and could not be interrupted (runtime orphaned)"
        elif result.timed_out:
            result.error_type = "Timeout"
            result.error = f"Timeout: execution exceeded {timeout_s:.0f}s"
            result.traceback = box.get("traceback")
        elif box.get("ok"):
            result.success = True
            result.figures = box.get("figures", [])
            before_keys = set(self.namespace)
            for k, v in ns.items():
                if not k.startswith("__") and k not in SEEDED_NAMES:
                    self.namespace[k] = v
            result.namespace_added = sorted(set(self.namespace) - before_keys)
            self._remember_regions(result.access)
        else:
            result.error = box.get("error", "unknown error")
            result.error_type = box.get("error_type", "Error")
            result.traceback = box.get("traceback")
            result.cancelled = result.error_type == "Cancelled"

        if self.snapshot_enabled and not (result.timed_out and thread.is_alive()):
            result.snapshot_after = take_snapshot(self.session)
            result.diff = diff_snapshots(result.snapshot_before, result.snapshot_after)
        self.history.append(result)
        return result


def clip_middle(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    """``text`` cut to ``limit`` characters and a marker, keeping its start and its end. Some exceptions quote a whole
    array (matplotlib names every value of a bad ``c=``: megabytes for a slice), and the message goes to
    the chat, the logs and the reflector's prompt."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]} ... [{len(text) - limit:,} characters omitted] ... {text[head - limit:]}"


def _trim_traceback(tb: str, run_id: str) -> str:
    """Keep the frames from the generated code onward (drop runtime internals), with long lines (the
    message, repeated for every chained exception) and the whole cut like the message."""
    marker = f'<stat-run-{run_id}>'
    lines = [clip_middle(ln) for ln in tb.splitlines()]
    for i, ln in enumerate(lines):
        if marker in ln:
            lines = ["Traceback (most recent call last):"] + lines[i:]
            break
    return clip_middle("\n".join(lines), MAX_TRACEBACK_CHARS)


def _async_raise(tid: int | None, exc_type) -> None:
    if tid is None:
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(exc_type))
    if res > 1:  # pragma: no cover - undo on the (impossible) multi-thread match
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


__all__ = ["Runtime", "RunResult", "RuntimeBusy", "SafetyViolation"]
