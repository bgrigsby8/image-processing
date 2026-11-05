"""Retry a short call to a dependency camera once when the module's
connection to viam-server hiccups (CONSULT-2298 step 2.2).

Dependency cameras are reached over the module's one parent connection. Two
failures on it are transient:

  * ``StreamTerminatedError: Connection lost``: the connection dropped
    (CONSULT-2296). The next call reconnects.
  * ``RuntimeError: ... is bound to a different event loop``: grpclib's
    reconnect lock was claimed by another event loop (CONSULT-2298). grpclib
    raises it while reconnecting, before the request is sent.

Only short calls that are harmless to repeat are retried: reads and absolute
moves/settings. Long or one-shot calls (``upload``, ``capture``, ``develop``,
relative focus nudges, shutter triggers) are not: a "Connection lost" can
arrive after the far side already did the work, and a retry would do it again.

If the retry hits "different event loop" again, the connection is stuck: the
lock stays bound to the other loop, so every later call that has to reconnect
fails the same way. The Python SDK can't mark a module unhealthy, so we log
at error saying the module needs a restart. (We deliberately don't call
``RestartModule`` on ourselves: that path is untested on a real viam-server.)
"""

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Optional, Set, TypeVar

T = TypeVar("T")

DIFFERENT_LOOP = "bound to a different event loop"
CONNECTION_LOST = "Connection lost"
RETRY_DELAY_S = 0.5

# Forwarded DoCommands that are short and do the same thing when run twice.
# Names are sony-remote's (and color-correction's) commands; anything not
# listed is forwarded once, as before.
RETRY_SAFE_COMMANDS = frozenset(
    {
        "get_status",
        "get_settings",
        "set_settings",
        "get_zoom",
        "set_zoom",
        "zoom_stop",
        "zoom_preset_load",
        "get_focus_position",
        "set_focus_position",
        "home_focus",
        "autofocus_once",
        "capture_count",
        "summary",
        "list_files",
        "focus_sweep_status",
    }
)


def command_names(command: Mapping[str, Any]) -> Set[str]:
    """The commands a DoCommand mapping asks for, in either style sony-remote
    accepts: ``{"set_zoom": {...}}`` or ``{"command": "set_zoom", ...}``."""
    named = command.get("command")
    if isinstance(named, str):
        return {named}
    return set(command)


def is_retry_safe(command: Mapping[str, Any]) -> bool:
    names = command_names(command)
    return bool(names) and names <= RETRY_SAFE_COMMANDS


def connection_error_kind(exc: BaseException) -> Optional[str]:
    """"different_loop", "connection_lost", or None for any other error."""
    text = str(exc)
    if DIFFERENT_LOOP in text:
        return "different_loop"
    try:
        from grpclib.exceptions import StreamTerminatedError
    except ImportError:  # pragma: no cover - grpclib ships with viam-sdk
        StreamTerminatedError = ()  # type: ignore[assignment,misc]
    if isinstance(exc, StreamTerminatedError) or CONNECTION_LOST in text:
        return "connection_lost"
    return None


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    what: str,
    logger: Any,
    retry: bool = True,
    delay: Optional[float] = None,
) -> T:
    """Run ``call()``; on a parent-connection error, wait ``delay`` (default
    ``RETRY_DELAY_S``) and run it
    once more. ``retry=False`` runs it exactly once (for calls that mustn't
    repeat), so call sites can route everything through here."""
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
        kind = connection_error_kind(exc)
        if not retry or kind is None:
            raise
        logger.warning(f"{what} failed ({type(exc).__name__}: {exc}); retrying once")
    await asyncio.sleep(RETRY_DELAY_S if delay is None else delay)
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001
        # Only our own RuntimeError means OUR connection is stuck; the same
        # text inside an error relayed from another module is that module's.
        if isinstance(exc, RuntimeError) and DIFFERENT_LOOP in str(exc):
            logger.error(
                f"{what} failed twice with '{DIFFERENT_LOOP}': this module's "
                f"connection to viam-server is stuck and calls through it will "
                f"keep failing until the module restarts (CONSULT-2298). "
                f"Restart this module."
            )
        raise
