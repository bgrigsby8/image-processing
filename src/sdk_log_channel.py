"""Send this module's logs to viam-server over their own connection.

Why (CONSULT-2296, CONSULT-2298): viam-sdk (checked 0.77.0, 0.80.0 and
0.85.2) ships module logs from a background thread that runs its own asyncio
loop, over the SAME grpclib channel the module uses for its normal calls on
the main loop (``viam/logging.py`` ``_ModuleHandler``). grpclib is not thread
safe. Under load the two loops interleave HTTP/2 frames on that one socket,
viam-server answers with GOAWAY PROTOCOL_ERROR ("illegal stream id"), and
every call in flight fails with ``StreamTerminatedError: Connection lost``.
On the reconnect both loops race for grpclib's ``Channel._connect_lock``,
which gets tied to one of them: "Lock ... is bound to a different event
loop".

The workaround: swap the SDK's handler class for one that keeps a background
thread (so logging never waits on a busy main loop) but gives that thread its
OWN grpclib channel to viam-server, created on the thread's own loop. The
module's main channel is then only ever touched by the main loop.

Call ``install()`` once before the module connects to its parent, i.e. before
``Module.run_from_registry()``. If the SDK internals it relies on are not
there, it warns and leaves the SDK's own handler in place.

Rules this file keeps:
  * nothing on the send path logs through ``logging`` (that would recurse);
    send failures are counted, and a state change (failing / recovered) is
    written straight to stderr, which viam-server also captures;
  * the queue is capped, so a dead parent can't grow memory without bound
    (the oldest entries are dropped first);
  * the channel is closed when ``logging.shutdown()`` closes the handler,
    which ``Module.stop()`` does on the way out.
"""

import asyncio
import collections
import logging
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Optional

# Biggest number of log entries waiting to be sent. Entries are a few hundred
# bytes each, so this bounds the backlog to a few MB when the parent is gone.
MAX_QUEUE = 5000
# Log RPCs in flight at once. viam-server rejects a LogRequest holding more
# than one entry ("batching not yet supported"), so entries go one per RPC.
MAX_IN_FLIGHT = 32
SEND_TIMEOUT_S = 5.0
# Pause after a round in which every send failed, so a dead parent doesn't
# turn the sender into a busy loop.
FAILURE_BACKOFF_S = 1.0
# How long close() waits for queued entries to go out before giving up.
CLOSE_FLUSH_S = 1.0

_LOGGER_NAME = "sdk_log_channel"

# The SDK's own handler class, saved by install() for the fallback path.
_SDK_HANDLER: Optional[type] = None
_SENDER: Optional["_LogSender"] = None
_SENDER_LOCK = threading.Lock()


def _warn(message: str) -> None:
    try:
        from viam import logging as vlog

        vlog.getLogger(_LOGGER_NAME).warning(message)
    except Exception:  # noqa: BLE001 - no SDK logging: stderr will do
        logging.getLogger(_LOGGER_NAME).warning(message)


def _stderr(message: str) -> None:
    """The send path's only way to say something: never through `logging`."""
    try:
        sys.stderr.write(f"{_LOGGER_NAME}: {message}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def install() -> bool:
    """Route module logs over their own connection. Returns True if patched.

    Safe to call more than once. A no-op (with a warning) when the SDK doesn't
    look like the one this was written against.
    """
    global _SDK_HANDLER
    try:
        import viam.logging as vlog
        from grpclib.client import Channel  # noqa: F401
        from viam.proto.common import LogEntry  # noqa: F401
        from viam.proto.robot import LogRequest, RobotServiceStub  # noqa: F401
        from viam.utils import datetime_to_timestamp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _warn(
            f"not installed, the SDK's module log handler stays in use "
            f"(missing import: {exc})"
        )
        return False

    current = getattr(vlog, "_ModuleHandler", None)
    if current is OwnChannelModuleHandler:
        return True
    add_handlers = getattr(vlog, "_addHandlers", None)
    # _addHandlers must still build the handler by looking up the module
    # global `_ModuleHandler`, or replacing that global does nothing.
    if (
        not isinstance(current, type)
        or not issubclass(current, logging.Handler)
        or add_handlers is None
        or "_ModuleHandler" not in getattr(getattr(add_handlers, "__code__", None), "co_names", ())
        or not callable(getattr(vlog, "setParent", None))
    ):
        _warn(
            "not installed, the SDK's module log handler stays in use "
            "(viam.logging no longer has the _ModuleHandler/_addHandlers "
            "this patches; check whether the SDK fixed the shared-channel bug)"
        )
        return False
    _SDK_HANDLER = current
    vlog._ModuleHandler = OwnChannelModuleHandler
    return True


def _channel_kwargs(parent: Any) -> Optional[Dict[str, Any]]:
    """How to dial the same viam-server the parent RobotClient talks to."""
    channel = getattr(parent, "_channel", None)
    path = getattr(channel, "_path", None)
    if path:
        return {"path": path}
    host = getattr(channel, "_host", None)
    port = getattr(channel, "_port", None)
    if host and port:
        return {"host": host, "port": port, "ssl": getattr(channel, "_ssl", None)}
    return None


def _entry(record: logging.LogRecord) -> Any:
    """The same LogEntry the SDK's handler builds."""
    from viam.proto.common import LogEntry
    from viam.utils import datetime_to_timestamp

    try:
        text = record.getMessage()
    except Exception:  # noqa: BLE001 - bad format args: keep the raw message
        text = str(record.msg)
    return LogEntry(
        level=record.levelname,
        time=datetime_to_timestamp(datetime.fromtimestamp(record.created)),
        logger_name=record.name.replace(".", "/"),
        message=f"{record.filename}:{record.lineno}\t{text}",
        stack=(
            f"exc_info: {record.exc_info}, exc_text: {record.exc_text}, "
            f"stack_info: {record.stack_info}"
        ),
    )


class _LogSender:
    """One thread, one event loop, one grpclib channel: all owned here."""

    def __init__(
        self,
        channel_kwargs: Dict[str, Any],
        *,
        max_queue: int = MAX_QUEUE,
        max_in_flight: int = MAX_IN_FLIGHT,
        send_timeout: float = SEND_TIMEOUT_S,
        failure_backoff: float = FAILURE_BACKOFF_S,
        stub_factory: Optional[Callable[[Any], Any]] = None,
    ):
        self.channel_kwargs = dict(channel_kwargs)
        self._max_in_flight = max_in_flight
        self._send_timeout = send_timeout
        self._failure_backoff = failure_backoff
        self._stub_factory = stub_factory
        self._queue: Deque[Any] = collections.deque()
        self._max_queue = max_queue
        self._lock = threading.Lock()
        self._draining = False
        self._failing = False
        self.closed = False
        self.stats = {"sent": 0, "failed": 0, "dropped": 0}
        self.channel: Any = None
        self._stub: Any = None
        self._drain_task: Optional["asyncio.Task[None]"] = None
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="module-log-sender", daemon=True
        )
        self._thread.start()
        ready.wait()

    def _run(self, ready: threading.Event) -> None:
        from grpclib.client import Channel
        from viam.proto.robot import RobotServiceStub

        asyncio.set_event_loop(self._loop)
        try:
            # Built on this thread's loop, so the channel's loop, its
            # _connect_lock and its transport all belong to this loop alone.
            self.channel = Channel(**self.channel_kwargs)
            factory = self._stub_factory or RobotServiceStub
            self._stub = factory(self.channel)
        finally:
            ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    # -- any thread ---------------------------------------------------------

    def submit(self, entry: Any) -> None:
        with self._lock:
            if self.closed:
                self.stats["dropped"] += 1
                return
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self.stats["dropped"] += 1
            self._queue.append(entry)
            if self._draining:
                return
            self._draining = True
        try:
            self._loop.call_soon_threadsafe(self._start_drain)
        except RuntimeError:  # loop already stopped
            with self._lock:
                self._draining = False

    def close(self, flush_s: float = CLOSE_FLUSH_S) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
        if not self._thread.is_alive():
            return
        try:
            done = asyncio.run_coroutine_threadsafe(self._shutdown(flush_s), self._loop)
            done.result(timeout=flush_s + 2.0)
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        self._thread.join(timeout=2.0)

    # -- sender loop only ---------------------------------------------------

    def _start_drain(self) -> None:
        self._drain_task = self._loop.create_task(self._drain())

    async def _send(self, entry: Any) -> None:
        from viam.proto.robot import LogRequest

        await self._stub.Log(LogRequest(logs=[entry]), timeout=self._send_timeout)

    async def _drain(self) -> None:
        while True:
            with self._lock:
                batch = [
                    self._queue.popleft()
                    for _ in range(min(self._max_in_flight, len(self._queue)))
                ]
                if not batch:
                    self._draining = False
                    return
            # Every failure (including a stub that never got built) lands in
            # `results`; nothing here may raise or log.
            results = await asyncio.gather(
                *(self._send(e) for e in batch), return_exceptions=True
            )
            errors = [r for r in results if isinstance(r, BaseException)]
            self.stats["sent"] += len(results) - len(errors)
            self.stats["failed"] += len(errors)
            if errors and len(errors) == len(results):
                if not self._failing:
                    self._failing = True
                    _stderr(
                        f"sending module logs to viam-server is failing "
                        f"({type(errors[0]).__name__}: {errors[0]}); logs are "
                        f"dropped until it recovers"
                    )
                await asyncio.sleep(self._failure_backoff)
            elif self._failing and len(errors) < len(results):
                self._failing = False
                _stderr(
                    f"sending module logs recovered "
                    f"(failed so far: {self.stats['failed']}, dropped: {self.stats['dropped']})"
                )

    async def _shutdown(self, flush_s: float) -> None:
        deadline = time.monotonic() + flush_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._draining:
                    break
            await asyncio.sleep(0.02)
        task = self._drain_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - cancelled on purpose
                pass
        if self.channel is not None:
            self.channel.close()


def _shared_sender(channel_kwargs: Dict[str, Any]) -> "_LogSender":
    """One sender for the whole process: the SDK builds a new handler every
    time the log level changes, and they must not each open a connection."""
    global _SENDER
    with _SENDER_LOCK:
        old = _SENDER
        if old is not None and not old.closed and old.channel_kwargs == channel_kwargs:
            return old
        _SENDER = _LogSender(channel_kwargs)
        sender = _SENDER
    if old is not None and not old.closed:
        old.close()
    return sender


def _close_shared() -> None:
    global _SENDER
    with _SENDER_LOCK:
        sender, _SENDER = _SENDER, None
    if sender is not None:
        sender.close()


class OwnChannelModuleHandler(logging.Handler):
    """Stands in for ``viam.logging._ModuleHandler`` once install() ran.

    Built by the SDK as ``_ModuleHandler(parent)``. If the parent's address
    can't be read, construction hands back the SDK's own handler instead.
    """

    def __new__(cls, parent: Any = None, *args: Any, **kwargs: Any) -> Any:
        if _channel_kwargs(parent) is None and _SDK_HANDLER is not None:
            _warn(
                "can't read the viam-server address from the module's parent "
                "connection; using the SDK's module log handler"
            )
            return _SDK_HANDLER(parent)
        return super().__new__(cls)

    def __init__(self, parent: Any):
        super().__init__()
        kwargs = _channel_kwargs(parent)
        if kwargs is None:
            raise ValueError("parent connection has no address to dial")
        self._sender = _shared_sender(kwargs)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = _entry(record)
        except Exception:  # noqa: BLE001 - never raise (or log) from emit
            self._sender.stats["dropped"] += 1
            return
        self._sender.submit(entry)

    def close(self) -> None:
        _close_shared()
        super().close()
