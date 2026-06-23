from __future__ import annotations

import sys


def log(msg: str) -> None:
    """A flushed stdout line — stays real-time even when output is redirected to a file."""
    print(msg, flush=True)


def bar_disabled() -> bool:
    """Progress bars only make sense on a TTY; suppress them when piped / backgrounded."""
    return not sys.stderr.isatty()
