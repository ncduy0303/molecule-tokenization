"""
Timeout utilities for encoding functions.

Provides a context manager that raises EncodeTimeoutError after a given
number of seconds.  Uses SIGALRM on Unix, which can interrupt C-extension
code (e.g. RDKit BRICS/MMPA decomposition hangs).  Falls back silently to
a no-op on Windows or when called from a non-main thread.
"""

import signal
import threading
from contextlib import contextmanager


class EncodeTimeoutError(Exception):
    """Raised when an encoding call exceeds the allowed time budget."""


@contextmanager
def encoding_timeout(seconds: int):
    """Context manager that raises EncodeTimeoutError after *seconds*.

    Uses SIGALRM on Unix (can interrupt C-extension hangs).
    Falls back to a no-op when SIGALRM is unavailable (Windows) or when
    called from a non-main thread (where signal delivery is not allowed).

    Example::

        from utils.timeout_utils import encoding_timeout, EncodeTimeoutError

        try:
            with encoding_timeout(60):
                result = some_slow_encoding(smiles)
        except EncodeTimeoutError:
            result = smiles  # fallback
    """
    use_signal = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not use_signal:
        yield
        return

    def _handler(signum, frame):
        raise EncodeTimeoutError(f"Encoding timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # cancel any pending alarm
        signal.signal(signal.SIGALRM, old_handler)
