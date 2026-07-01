"""Entry point — run with: python main.py"""

import asyncio
import os
import sys

# Enable UTF-8 mode for proper Unicode handling on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    # Re-configure stdout/stderr for UTF-8 if not already set
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # Python 3.12+ defaults to ProactorEventLoop on Windows, which interacts
    # with the Win32 console via I/O completion ports and can disrupt the
    # console input handle state that prompt_toolkit relies on.  The Selector-
    # based loop avoids IOCP entirely and is fully adequate for our HTTP-only
    # async workloads (LangChain/httpx).
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Rich 14.x _RefreshThread.run() has no exception handling around
    # self.live.refresh().  When LangChain/httpx async calls momentarily
    # invalidate the Win32 console output handle, the refresh raises an
    # OSError that escapes the thread and triggers a cascade of
    # "Exception in thread / Exception in threading.excepthook" messages.
    # Wrap the call in try/except so the loop survives temporary handle
    # failures and the spinner keeps running (or gracefully stops) without
    # polluting stderr.
    from rich.live import _RefreshThread as _RichRefreshThread

    def _safe_refresh_run(self) -> None:
        while not self.done.wait(1 / self.refresh_per_second):
            with self.live._lock:
                if not self.done.is_set():
                    try:
                        self.live.refresh()
                    except Exception:
                        pass

    _RichRefreshThread.run = _safe_refresh_run

    # When an LLM client closes its asyncio event loop, httpx async transports try to
    # close their connections and raise RuntimeError('Event loop is closed').
    # Python surfaces these as "Task exception was never retrieved" via the
    # asyncio logger — they are benign cleanup artifacts, not real failures.
    # Filter them out so they don't pollute the terminal output.
    import logging

    class _SuppressEventLoopClosed(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "Event loop is closed" not in msg and "Task exception was never retrieved" not in msg

    logging.getLogger("asyncio").addFilter(_SuppressEventLoopClosed())

    # Second defence: some Python versions surface the same error through
    # sys.unraisablehook (GC path) rather than the logging framework.
    _orig_unraisablehook = sys.unraisablehook

    def _unraisable_filter(unraisable) -> None:
        if isinstance(unraisable.exc_value, RuntimeError) and "Event loop is closed" in str(unraisable.exc_value):
            return
        _orig_unraisablehook(unraisable)

    sys.unraisablehook = _unraisable_filter

from src.app.cli import run

if __name__ == "__main__":
    run()
