"""dependency_retry: one retry for short dependency calls when the module's
parent connection hiccups (CONSULT-2298 step 2.2)."""

import asyncio
import logging

import pytest
from grpclib.exceptions import StreamTerminatedError

from models import dependency_retry

DIFFERENT_LOOP = RuntimeError(
    "<asyncio.locks.Lock object at 0x71a749bbaba0 [locked]> is bound to a different event loop"
)


class _Calls:
    """An async call that raises the queued errors, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.count = 0

    async def __call__(self):
        self.count += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _run(call, **kwargs):
    kwargs.setdefault("logger", logging.getLogger("dependency-retry-test"))
    kwargs.setdefault("what", "source camera set_zoom")
    kwargs.setdefault("delay", 0)
    return asyncio.run(dependency_retry.call_with_retry(call, **kwargs))


@pytest.mark.parametrize(
    "error",
    [
        DIFFERENT_LOOP,
        StreamTerminatedError("Connection lost"),
        # The same failure relayed back by another module through viam-server.
        Exception("rpc error: code = Unknown desc = StreamTerminatedError - Connection lost"),
    ],
)
def test_parent_connection_error_is_retried_once(error, caplog):
    call = _Calls(error)
    with caplog.at_level(logging.WARNING):
        assert _run(call) == "ok"
    assert call.count == 2
    assert "retrying once" in caplog.text


def test_only_one_retry():
    call = _Calls(StreamTerminatedError("Connection lost"), StreamTerminatedError("Connection lost"))
    with pytest.raises(StreamTerminatedError):
        _run(call)
    assert call.count == 2


def test_other_errors_are_not_retried():
    call = _Calls(ValueError("`set_zoom` needs `focal_length_mm`"))
    with pytest.raises(ValueError):
        _run(call)
    assert call.count == 1


def test_retry_false_runs_once():
    call = _Calls(StreamTerminatedError("Connection lost"))
    with pytest.raises(StreamTerminatedError):
        _run(call, retry=False)
    assert call.count == 1


def test_repeated_different_loop_logs_error(caplog):
    async def twice():
        for _ in range(2):
            call = _Calls(DIFFERENT_LOOP, DIFFERENT_LOOP)
            with pytest.raises(RuntimeError):
                await dependency_retry.call_with_retry(
                    call, what="source camera set_zoom",
                    logger=logging.getLogger("dependency-retry-test"), delay=0,
                )

    with caplog.at_level(logging.ERROR):
        asyncio.run(twice())
    assert "connection to viam-server is stuck" in caplog.text


def test_relayed_different_loop_is_retried_once():
    relayed = Exception("rpc error: code = Unknown desc = RuntimeError - <Lock> is bound to a different event loop")
    call = _Calls(relayed, relayed)
    with pytest.raises(Exception):
        _run(call)
    assert call.count == 2


@pytest.mark.parametrize(
    "command, safe",
    [
        ({"set_zoom": {"focal_length_mm": 50}}, True),
        ({"command": "set_zoom", "focal_length_mm": 50}, True),
        ({"get_status": {}}, True),
        ({"set_focus_position": {"position": 10}, "get_zoom": {}}, True),
        ({"upload": {"paths": []}}, False),
        ({"capture": {}}, False),
        ({"trigger": {}}, False),
        ({"focus_near_far": {"step": 1}}, False),
        ({"command": "focus_near_far", "step": 1}, False),
        ({"set_zoom": {}, "trigger": {}}, False),
        ({}, False),
    ],
)
def test_is_retry_safe(command, safe):
    assert dependency_retry.is_retry_safe(command) is safe
