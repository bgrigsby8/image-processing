"""sdk_log_channel: module logs go to viam-server over their own connection
(CONSULT-2296 / CONSULT-2298)."""

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time
import types

import grpclib.client
import pytest
import viam.logging as vlog
from grpclib.const import Cardinality, Handler
from grpclib.exceptions import StreamTerminatedError
from grpclib.server import Server
from viam.proto.common import LogEntry
from viam.proto.robot import LogRequest, LogResponse

import sdk_log_channel


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture(autouse=True)
def _restore_sdk():
    """Every test leaves viam.logging and the shared sender as it found them."""
    handler = vlog._ModuleHandler
    add_handlers = vlog._addHandlers
    yield
    sdk_log_channel._close_shared()
    vlog._ModuleHandler = handler
    vlog._addHandlers = add_handlers
    if vlog._MODULE_PARENT is not None:
        vlog.setParent(None)  # back to stdout/stderr handlers


class _LogServer:
    """A stand-in viam-server that only answers RobotService.Log, on a unix
    socket, from its own thread."""

    def __init__(self):
        self._dir = tempfile.mkdtemp(prefix="slc", dir="/tmp")  # short: AF_UNIX path limit
        self.path = os.path.join(self._dir, "parent.sock")
        self.entries = []

    def __mapping__(self):
        return {
            "/viam.robot.v1.RobotService/Log": Handler(
                self._log, Cardinality.UNARY_UNARY, LogRequest, LogResponse
            )
        }

    async def _log(self, stream):
        request = await stream.recv_message()
        self.entries.extend(request.logs)
        await stream.send_message(LogResponse())

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self._loop)
            self._server = Server([self])
            self._loop.run_until_complete(self._server.start(path=self.path))
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(5)
        return self

    def __exit__(self, *exc):
        async def stop():
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 2)
            except asyncio.TimeoutError:
                pass

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        self._loop.close()
        shutil.rmtree(self._dir, ignore_errors=True)


class _FakeParent:
    """The module's RobotClient as far as the handler can see it: an address on
    `_channel`, plus a `log` that must never be called (that's the SDK's path
    over the shared channel)."""

    def __init__(self, path):
        self._channel = types.SimpleNamespace(_path=path, _host=None, _port=None)
        self.log_calls = []

    async def log(self, *args):
        self.log_calls.append(args)


def test_install_patches_the_sdk_handler_and_is_idempotent():
    assert sdk_log_channel.install() is True
    assert vlog._ModuleHandler is sdk_log_channel.OwnChannelModuleHandler
    assert sdk_log_channel.install() is True
    assert vlog._ModuleHandler is sdk_log_channel.OwnChannelModuleHandler


def test_module_logs_go_over_their_own_channel_not_the_parents():
    with _LogServer() as server:
        assert sdk_log_channel.install()
        parent = _FakeParent(server.path)
        vlog.setParent(parent)  # what Module._connect_to_parent does
        logger = vlog.getLogger("rdk:component:camera/slc-test")
        handler = logger.handlers[0]
        assert isinstance(handler, sdk_log_channel.OwnChannelModuleHandler)

        logger.info("hello %s", "world")
        assert _wait_for(lambda: len(server.entries) == 1)

        sender = sdk_log_channel._SENDER
        # Sent on a channel of its own, made on the sender thread's loop...
        assert isinstance(sender.channel, grpclib.client.Channel)
        assert sender.channel is not parent._channel
        assert sender.channel._loop is sender._loop
        assert sender._thread is not threading.current_thread()
        # ...and never through the parent RobotClient.
        assert parent.log_calls == []
        entry = server.entries[0]
        assert entry.logger_name == "rdk:component:camera/slc-test"
        assert entry.level == "INFO"
        assert entry.message.endswith("\thello world")
        assert sender.stats["sent"] == 1
        sdk_log_channel._close_shared()


def test_handlers_share_one_sender():
    with _LogServer() as server:
        parent = _FakeParent(server.path)
        a = sdk_log_channel.OwnChannelModuleHandler(parent)
        b = sdk_log_channel.OwnChannelModuleHandler(parent)
        assert a._sender is b._sender


def test_install_falls_back_when_sdk_handler_is_missing(monkeypatch, caplog):
    monkeypatch.delattr(vlog, "_ModuleHandler")
    with caplog.at_level(logging.WARNING):
        assert sdk_log_channel.install() is False
    assert not hasattr(vlog, "_ModuleHandler")
    assert "not installed" in caplog.text


def test_install_falls_back_when_add_handlers_no_longer_uses_it(monkeypatch, caplog):
    sdk_handler = vlog._ModuleHandler
    monkeypatch.setattr(vlog, "_addHandlers", lambda loggers, use_default_handlers=False: None)
    with caplog.at_level(logging.WARNING):
        assert sdk_log_channel.install() is False
    assert vlog._ModuleHandler is sdk_handler
    assert "not installed" in caplog.text


def test_handler_falls_back_to_sdk_handler_without_a_parent_address(monkeypatch, caplog):
    class _SdkHandler(logging.Handler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

    monkeypatch.setattr(sdk_log_channel, "_SDK_HANDLER", _SdkHandler)
    parent = types.SimpleNamespace(_channel=types.SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        handler = sdk_log_channel.OwnChannelModuleHandler(parent)
    assert type(handler) is _SdkHandler
    assert handler.parent is parent
    assert sdk_log_channel._SENDER is None
    assert "using the SDK's module log handler" in caplog.text


class _RecordingChannel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        _RecordingChannel.instances.append(self)

    def close(self):
        self.closed = True


class _StuckStub:
    """A parent that accepted the connection but never answers."""

    def __init__(self, channel):
        self.calls = 0

    async def Log(self, request, timeout=None):
        self.calls += 1
        await asyncio.Event().wait()


def test_queue_is_capped_when_the_parent_stops_answering(monkeypatch):
    monkeypatch.setattr(grpclib.client, "Channel", _RecordingChannel)
    sender = sdk_log_channel._LogSender(
        {"path": "/nonexistent"}, max_queue=5, max_in_flight=2, stub_factory=_StuckStub
    )
    try:
        for i in range(50):
            sender.submit(LogEntry(message=str(i)))
        assert _wait_for(lambda: sender._stub.calls == 2)
        assert len(sender._queue) <= 5
        # 50 submitted = at most 5 queued + 2 stuck in flight + the rest dropped.
        assert sender.stats["dropped"] >= 43
        # The newest entries are the ones kept.
        assert [e.message for e in sender._queue][-1] == "49"
    finally:
        started = time.monotonic()
        sender.close(flush_s=0.2)
    assert time.monotonic() - started < 3
    assert not sender._thread.is_alive()
    assert sender.channel.closed
    sender.submit(LogEntry(message="after close"))  # dropped, no error
    assert sender.stats["dropped"] >= 44


class _FailingStub:
    def __init__(self, channel):
        pass

    async def Log(self, request, timeout=None):
        raise StreamTerminatedError("Connection lost")


def test_send_failures_never_log(monkeypatch, caplog, capsys):
    monkeypatch.setattr(grpclib.client, "Channel", _RecordingChannel)
    sender = sdk_log_channel._LogSender(
        {"path": "/nonexistent"}, failure_backoff=0.01, stub_factory=_FailingStub
    )
    with caplog.at_level(logging.DEBUG):
        for i in range(10):
            sender.submit(LogEntry(message=str(i)))
        assert _wait_for(lambda: sender.stats["failed"] == 10)
        sender.close()
    # (asyncio's own debug logging, in -X dev runs, isn't ours.)
    assert [r for r in caplog.records if r.name != "asyncio"] == []
    err = capsys.readouterr().err
    assert err.count("sending module logs to viam-server is failing") == 1
    assert sender.channel.closed


def test_close_flushes_queued_logs_and_closes_the_channel():
    with _LogServer() as server:
        parent = _FakeParent(server.path)
        handler = sdk_log_channel.OwnChannelModuleHandler(parent)
        logger = logging.getLogger("slc-close-test")
        logger.propagate = False
        logger.addHandler(handler)
        try:
            for i in range(20):
                logger.warning("entry %d", i)
            sender = handler._sender
            handler.close()  # what logging.shutdown() does in Module.stop()
        finally:
            logger.removeHandler(handler)
        assert len(server.entries) == 20
        assert sender.closed
        assert not sender._thread.is_alive()
        assert sender.channel._protocol is None
