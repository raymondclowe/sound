"""
Utility helpers for the sound_recognition package.

Provides a deduplicating debug print function to avoid logging identical
consecutive debug messages.
"""
from typing import Optional

_last_debug_message: Optional[str] = None
_last_debug_count: int = 0

def debug_print(msg: str, debug: bool = True, dedup: bool = True) -> None:
    """
    Print a debug message if debug is True. Optionally suppress consecutive
    duplicate messages if dedup is True.

    Args:
        msg: Message string to print.
        debug: Whether to print (default True). If False, do nothing.
        dedup: If True, suppress consecutive identical messages.
    """
    global _last_debug_message, _last_debug_count
    if not debug:
        return
    if dedup and _last_debug_message is not None and msg == _last_debug_message:
        _last_debug_count += 1
        return
    # Message changed — print and reset counter
    _last_debug_message = msg
    _last_debug_count = 0
    print(msg)

def reset_debug_state() -> None:
    """Reset internal debug state (for tests or to clear suppression history)."""
    global _last_debug_message, _last_debug_count
    _last_debug_message = None
    _last_debug_count = 0
