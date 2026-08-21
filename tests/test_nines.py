"""Tests for the Nines client's failure handling and the retry queue.

Unit-level counterparts to the delivery tests in test_color_correction.py,
which drive the same machinery through the component's `upload` command. These
touch only models/nines.py, so they need no numpy/rawpy/viam - and no network:
the two timeout tests talk to a local socket that accepts and then stalls,
which is the only honest way to prove what urllib actually raises.
"""

import asyncio
import json
import logging
import os
import socket
import threading
import urllib.parse
import time

import pytest

from models.nines import (
    NINES_RETRY_FIRST_DELAY_SEC,
    NINES_USER_AGENT,
    NinesAPIError,
    NinesClient,
    NinesDeliveryQueue,
)

LOGGER = logging.getLogger("test-nines")


# ---------------------------------------------------------------------------
# Failure classification: what is worth retrying, and what may already have
# been committed before we lost the answer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,retryable,ambiguous", [
    (None, True, True),    # unreachable / timed out: unknown either way
    (408, True, False),    # the API said it timed out - it did not act
    (429, True, False),    # rate limited: rejected before processing
    (500, True, True),
    (502, True, True),
    (503, True, True),
    (504, True, True),
    (401, False, False),   # revoked key - identical forever
    (403, False, False),   # wrong org / missing scope
    (404, False, False),
    (409, False, False),
    (422, False, False),   # the API rejected the image itself
])
def test_status_decides_retryable_and_ambiguous(status, retryable, ambiguous):
    exc = NinesAPIError("x", status=status)
    assert exc.retryable is retryable
    assert exc.ambiguous is ambiguous
    assert exc.status == status


def test_non_transport_failures_override_the_status_derived_flags():
    """A missing local file and a malformed 2xx body both carry no status but
    are nothing like an unreachable API - neither may be retried."""
    exc = NinesAPIError("no such file", retryable=False, ambiguous=False)
    assert not exc.retryable and not exc.ambiguous


def _stalling_server():
    """A socket that accepts a connection and then never answers."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    threading.Thread(
        target=lambda: (srv.accept(), time.sleep(3)), daemon=True
    ).start()
    return srv.getsockname()[1]


def _client(base_url, timeout=0.3):
    return NinesClient(
        "nines_live_test", "viam-org", base_url, logger=LOGGER,
        request_timeout_s=timeout, upload_timeout_s=timeout,
    )


def test_read_timeout_arrives_as_a_nines_error():
    """urlopen wraps a *connect* timeout in URLError but lets a *read* timeout
    out bare, so without the explicit branch this escapes past every caller
    that catches NinesAPIError - and past the classification above."""
    client = _client(f"http://127.0.0.1:{_stalling_server()}")
    with pytest.raises(NinesAPIError) as caught:
        client.request("GET", "/api/v1/reference_items", None, 0.3)
    assert caught.value.status is None
    assert caught.value.retryable and caught.value.ambiguous
    assert "timed out" in str(caught.value)


def test_unreachable_api_arrives_as_a_retryable_nines_error():
    client = _client("http://127.0.0.1:1")
    with pytest.raises(NinesAPIError) as caught:
        client.request("GET", "/api/v1/reference_items", None, 0.3)
    assert caught.value.status is None and caught.value.retryable


def test_requests_identify_the_integration(monkeypatch):
    """Every call names this integration in its User-Agent (urllib's default
    is an anonymous "Python-urllib/x.y" Nines cannot attribute) and asks for
    the JSON the API speaks."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _client("http://nines.test")
    client.request("GET", "/api/v1/reference_items", None, 1.0)
    request = captured["request"]
    assert request.get_header("User-agent") == NINES_USER_AGENT
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") == "Bearer nines_live_test"


# ---------------------------------------------------------------------------
# Did a lost append actually land? The evidence, without the HTTP.
# ---------------------------------------------------------------------------

def _remote(*tags):
    return {"id": "rimg", "position": 0, "tags": list(tags)}


@pytest.mark.parametrize("before,remote,batch,verdict", [
    # With a baseline the answer is arithmetic.
    (2, [_remote("a"), _remote("b")], [["s1"]], False),
    (2, [_remote("a"), _remote("b"), _remote("s1")], [["s1"]], True),
    (1, [_remote("a"), _remote("x"), _remote("y")], [["x"], ["y"]], True),
    # Grew by something other than our batch: stop guessing.
    (2, [_remote("a"), _remote("b"), _remote("c"), _remote("d")], [["s1"]], None),
    # No baseline: only a negative is trustworthy.
    (None, [], [["IMG_0042"]], False),
    (None, [_remote("front")], [["IMG_0042"]], False),
    # A tag match proves nothing - `nines_upload` reuses tags like "front",
    # so claiming True here would silently drop a re-shoot.
    (None, [_remote("IMG_0042")], [["IMG_0042"]], None),
    # The API lowercases tags on ingest, so a landed batch comes back
    # lowercased. It must still count as a match (None) - calling it absent
    # (False) would re-append it, the duplicate this check exists to prevent.
    (None, [_remote("img_0042")], [["IMG_0042"]], None),
    # Nothing to match on at all.
    (None, [_remote("front")], [[]], None),
])
def test_appended_verdict(before, remote, batch, verdict):
    assert NinesClient._appended_verdict(before, remote, batch) is verdict


def test_already_appended_is_none_when_the_check_itself_fails():
    """If we cannot reach the API to ask, that is 'undecidable', not a failure
    the caller has to handle."""
    client = _client("http://127.0.0.1:1")
    verdict = asyncio.run(
        client.already_appended("ritem_1", "viam-org", [("/a.jpg", "a.jpg", ["t"])])
    )
    assert verdict is None


class _FakeAPI:
    """Stands in for NinesClient.request: lookup, show, create, and append.

    ``existing`` pre-loads products a lookup can resolve (as a real catalog
    would); with none, a lookup misses and delivery falls to the create path.
    """

    def __init__(self, images=(), append_error=None, existing=()):
        self.images = list(images)
        self.append_error = append_error
        self.existing = list(existing)
        self.calls = []

    def __call__(self, method, path, body, timeout_s):
        self.calls.append((method, path))
        base = path.split("?", 1)[0]
        if method == "GET" and base.startswith("/api/v1/reference_items/"):
            return {"id": base.rsplit("/", 1)[-1], "images": list(self.images)}
        if method == "GET" and base == "/api/v1/reference_items":
            query = urllib.parse.parse_qs(path.split("?", 1)[1])
            matches = [
                item for item in self.existing
                if ("external_id" in query
                    and str(item.get("external_id", "")).lower()
                    == query["external_id"][0].lower())
                or ("upc" in query
                    and str((item.get("product_details") or {}).get("upc", ""))
                    == query["upc"][0])
            ]
            return {"reference_items": matches, "total": len(matches)}
        if base == "/api/v1/reference_items":
            return {"id": "ritem_1", "external_id": body["external_id"],
                    "created": True, "images_count": 0}
        if base.endswith("/images"):
            if self.append_error is not None:
                raise self.append_error
            return {"id": "ritem_1", "added_count": len(body["images"]),
                    "images_count": len(self.images) + len(body["images"])}
        raise AssertionError(f"unexpected path {path}")


def _delivering_client(fake):
    client = _client("http://nines.test")
    client.request = fake
    client.item_ids[("viam-org", "SKU")] = "ritem_1"
    return client


def test_verify_first_skips_an_append_that_already_landed(tmp_path):
    """The retry must not deliver a shot the API had already accepted. The
    file is never even read - the check happens before the base64."""
    fake = _FakeAPI(images=[_remote("a"), _remote("stem")])
    client = _delivering_client(fake)
    client.item_image_counts["ritem_1"] = 1     # it grew by our one image

    result = asyncio.run(client.deliver(
        "SKU", [("/no/such/file.jpg", "stem.jpg", ["stem"])], verify_first=True,
    ))
    assert result["deduplicated"] is True
    assert result["added_count"] == 0
    assert not any(path.endswith("/images") for _, path in fake.calls)


def test_verify_first_re_appends_when_the_earlier_attempt_missed(tmp_path):
    fake = _FakeAPI(images=[_remote("a")])
    client = _delivering_client(fake)
    client.item_image_counts["ritem_1"] = 1     # unchanged: it never landed
    image = tmp_path / "stem.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    result = asyncio.run(client.deliver(
        "SKU", [(str(image), "stem.jpg", ["stem"])], verify_first=True,
    ))
    assert result["added_count"] == 1
    assert "deduplicated" not in result
    # The successful append refreshes the baseline for the next question.
    assert client.item_image_counts["ritem_1"] == 2


def test_unreadable_file_is_a_terminal_failure(tmp_path):
    """A local filesystem problem is not a network one - it will be just as
    missing next time, so it must never enter a retry schedule."""
    client = _delivering_client(_FakeAPI())
    with pytest.raises(NinesAPIError) as caught:
        asyncio.run(client.deliver("SKU", [("/no/such/file.jpg", "a.jpg", ["t"])]))
    assert not caught.value.retryable and not caught.value.ambiguous


# ---------------------------------------------------------------------------
# The retry queue: quick first re-attempt, back of the queue, widening backoff.
# ---------------------------------------------------------------------------

class _ScriptedClient:
    """A NinesClient stand-in whose deliver() replays a scripted outcome list."""

    def __init__(self, script):
        self.script = {sku: list(outcomes) for sku, outcomes in script.items()}
        self.calls = []
        # The caches the queue snapshots when it takes a job on, so a restored
        # attempt can be told what the product looked like beforehand.
        self.item_ids = {}
        self.item_image_counts = {}

    async def deliver(self, sku, images, product_name=None, org_slug=None,
                      upc=None, verify_first=False):
        self.calls.append({"sku": sku, "verify_first": verify_first,
                           "at": time.monotonic()})
        outcome = self.script[sku].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


IMAGES = [("/photos/a.jpg", "a.jpg", ["stem"])]


def _queue(client, **kwargs):
    kwargs.setdefault("first_delay_s", 0.02)
    kwargs.setdefault("max_delay_s", 1.0)
    kwargs.setdefault("jitter", 0.0)
    return NinesDeliveryQueue(client, logger=LOGGER, **kwargs)


async def _drain(queue, timeout=5.0):
    # Mirrors do_command: restore() may have run without a loop to start the
    # worker on, so the first async entry point picks the queue up.
    queue.ensure_running()
    deadline = time.monotonic() + timeout
    while queue._jobs or (queue._worker and not queue._worker.done()):
        if time.monotonic() > deadline:
            raise AssertionError("the queue did not drain")
        await asyncio.sleep(0.005)


def test_first_retry_is_three_seconds_by_default():
    """The pacing the operator was promised, before any test-only shortening."""
    queue = NinesDeliveryQueue(None, logger=LOGGER, jitter=0.0)
    assert queue._delay(1) == NINES_RETRY_FIRST_DELAY_SEC == 3.0
    assert [queue._delay(n) for n in range(1, 6)] == [3.0, 6.0, 12.0, 24.0, 48.0]


def test_delay_is_capped():
    queue = NinesDeliveryQueue(
        None, logger=LOGGER, first_delay_s=1.0, max_delay_s=4.0, jitter=0.0
    )
    assert [queue._delay(n) for n in range(1, 7)] == [1, 2, 4, 4, 4, 4]


def test_jitter_spreads_reattempts_without_changing_the_scale():
    """A whole fleet recovering from one outage must not re-hit the API in
    lockstep, so the delay is spread - but it stays the delay it claims."""
    queue = NinesDeliveryQueue(None, logger=LOGGER, first_delay_s=10.0)
    delays = {queue._delay(1) for _ in range(50)}
    assert len(delays) > 1
    assert all(8.0 <= d <= 12.0 for d in delays)


def test_a_queued_delivery_is_retried_and_reported():
    client = _ScriptedClient({"S": [{"reference_item_id": "ritem_1"}]})
    queue = _queue(client)
    delivered = []

    async def scenario():
        info = queue.enqueue("S", IMAGES, org="viam-org", ambiguous=True,
                             on_success=delivered.append)
        await _drain(queue)
        return info

    info = asyncio.run(scenario())
    assert info["job_id"] == "nines-1"
    assert info["attempt"] == 1
    assert info["queued"] == 1
    assert len(delivered) == 1
    # The previous failure lost the answer, so this attempt checks first.
    assert client.calls[0]["verify_first"] is True


def test_a_clean_failure_does_not_pay_for_the_duplicate_check():
    client = _ScriptedClient({"S": [{"reference_item_id": "ritem_1"}]})
    queue = _queue(client)

    async def scenario():
        queue.enqueue("S", IMAGES, ambiguous=False)
        await _drain(queue)

    asyncio.run(scenario())
    assert client.calls[0]["verify_first"] is False


def test_each_failure_widens_the_gap():
    down = lambda: NinesAPIError("down", status=503)  # noqa: E731
    client = _ScriptedClient({"S": [down(), down(), down(), {"ok": True}]})
    queue = _queue(client, first_delay_s=0.05)

    async def scenario():
        start = time.monotonic()
        queue.enqueue("S", IMAGES)
        await _drain(queue)
        return start

    start = asyncio.run(scenario())
    starts = [start] + [call["at"] for call in client.calls[:-1]]
    gaps = [call["at"] - t for call, t in zip(client.calls, starts)]
    assert len(gaps) == 4
    for gap, expected in zip(gaps, [0.05, 0.10, 0.20, 0.40]):
        assert gap >= expected * 0.9, gaps


def test_a_failed_delivery_goes_behind_what_is_already_waiting():
    """One product Nines refuses must not stall the rest of the shoot."""
    client = _ScriptedClient({
        "A": [NinesAPIError("down", status=503), {"ok": True}],
        "B": [{"ok": True}],
    })
    queue = _queue(client)

    async def scenario():
        queue.enqueue("A", IMAGES)
        queue.enqueue("B", IMAGES)
        await _drain(queue)

    asyncio.run(scenario())
    assert [call["sku"] for call in client.calls] == ["A", "B", "A"]


def test_new_work_does_not_wait_out_another_job_s_backoff():
    client = _ScriptedClient({
        "A": [NinesAPIError("down", status=503), {"ok": True}],
        "B": [{"ok": True}],
    })
    # A's second delay is enormous; B must not be stuck behind it.
    queue = _queue(client, first_delay_s=0.02, max_delay_s=30.0, factor=1000.0)

    async def scenario():
        queue.enqueue("A", IMAGES)
        while not client.calls:
            await asyncio.sleep(0.005)
        start = time.monotonic()
        queue.enqueue("B", IMAGES)
        while not any(call["sku"] == "B" for call in client.calls):
            await asyncio.sleep(0.005)
            assert time.monotonic() - start < 2.0, "B waited on A's backoff"
        await queue.close()

    asyncio.run(scenario())


def test_a_permanent_failure_is_abandoned_without_a_schedule():
    client = _ScriptedClient({"S": [NinesAPIError("wrong org", status=403)]})
    queue = _queue(client)
    gave_up = []

    async def scenario():
        queue.enqueue("S", IMAGES, on_abandon=lambda job, exc: gave_up.append(exc))
        await _drain(queue)

    asyncio.run(scenario())
    assert len(client.calls) == 1
    assert len(gave_up) == 1


def test_a_delivery_is_abandoned_once_the_attempts_run_out():
    down = NinesAPIError("down", status=503)
    client = _ScriptedClient({"S": [down] * 9})
    queue = _queue(client, max_attempts=4)
    gave_up = []

    async def scenario():
        queue.enqueue("S", IMAGES, on_abandon=lambda job, exc: gave_up.append(job))
        await _drain(queue)

    asyncio.run(scenario())
    # One inline failure was already counted, so three more are made here.
    assert len(client.calls) == 3
    assert gave_up[0].attempt == 4
    report = queue.snapshot()
    assert report["pending_count"] == 0
    assert report["abandoned"][0]["files"] == ["/photos/a.jpg"]


def test_max_attempts_of_one_means_the_inline_attempt_only():
    """The budget counts the caller's failed attempt, so there is nothing left
    to schedule - the job must not be queued for one more try."""
    client = _ScriptedClient({"S": [{"ok": True}]})
    queue = _queue(client, max_attempts=1)
    assert queue.enqueue("S", IMAGES) is None
    assert queue.snapshot()["pending_count"] == 0
    assert client.calls == []


def test_max_attempts_of_two_allows_exactly_one_retry():
    client = _ScriptedClient({"S": [NinesAPIError("down", status=503)]})
    queue = _queue(client, max_attempts=2)
    gave_up = []

    async def scenario():
        assert queue.enqueue(
            "S", IMAGES, on_abandon=lambda job, exc: gave_up.append(job)
        ) is not None
        await _drain(queue)

    asyncio.run(scenario())
    assert len(client.calls) == 1
    assert gave_up[0].attempt == 2


def test_an_unexpected_exception_is_abandoned_rather_than_looped_on():
    client = _ScriptedClient({"S": [RuntimeError("bug")]})
    queue = _queue(client)
    gave_up = []

    async def scenario():
        queue.enqueue("S", IMAGES, on_abandon=lambda job, exc: gave_up.append(exc))
        await _drain(queue)

    asyncio.run(scenario())
    assert len(client.calls) == 1 and len(gave_up) == 1


def test_a_callback_that_raises_does_not_wedge_the_queue():
    """Cleanup is policy layered on delivery, not part of it."""
    client = _ScriptedClient({"S": [{"ok": True}], "T": [{"ok": True}]})
    queue = _queue(client)

    def explode(_result):
        raise ValueError("callback exploded")

    async def scenario():
        queue.enqueue("S", IMAGES, on_success=explode)
        queue.enqueue("T", IMAGES)
        await _drain(queue)

    asyncio.run(scenario())
    assert [call["sku"] for call in client.calls] == ["S", "T"]


def test_closing_keeps_pending_work_and_the_next_enqueue_resumes_it():
    """A reconfigure must not throw away deliveries that are still waiting."""
    client = _ScriptedClient({"S": [NinesAPIError("down", status=503), {"ok": True}]})
    queue = _queue(client, first_delay_s=0.02, max_delay_s=30.0, factor=1000.0)

    async def scenario():
        queue.enqueue("S", IMAGES)
        while not client.calls:
            await asyncio.sleep(0.005)
        await queue.close()
        assert len(queue._jobs) == 1
        report = queue.snapshot()
        assert report["pending_count"] == 1
        assert report["pending"][0]["attempt"] == 2
        assert "down" in report["pending"][0]["error"]
        queue.factor = 1.0
        queue.enqueue("T", [("/photos/b.jpg", "b.jpg", ["s"])])
        assert queue._worker is not None
        await queue.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# The image-count baseline: what makes a lost append answerable by arithmetic
# rather than by matching tags.
# ---------------------------------------------------------------------------

def test_a_found_product_learns_its_image_count_once():
    """A pre-loaded catalog is the production case, and the lookup's list
    response carries no images - so resolving one costs a single show request,
    and only the first time."""
    fake = _FakeAPI(images=[_remote("front"), _remote("back")],
                    existing=[{"id": "ritem_1", "external_id": "SKU"}])
    client = _client("http://nines.test")
    client.request = fake

    item_id = asyncio.run(client._resolve_item("SKU", None, "viam-org"))
    assert item_id == "ritem_1"
    assert client.item_image_counts["ritem_1"] == 2
    shows = [p for _, p in fake.calls if p.startswith("/api/v1/reference_items/")]
    assert len(shows) == 1

    # Cached from here on: resolving again asks the API nothing.
    fake.calls.clear()
    asyncio.run(client._resolve_item("SKU", None, "viam-org"))
    assert fake.calls == []


def test_a_list_hit_carrying_the_count_costs_no_extra_request():
    """If the list endpoint does report images_count, take it for free."""
    fake = _FakeAPI(
        existing=[{"id": "ritem_1", "external_id": "SKU", "images_count": 7}])
    client = _client("http://nines.test")
    client.request = fake

    asyncio.run(client._resolve_item("SKU", None, "viam-org"))
    assert client.item_image_counts["ritem_1"] == 7
    assert not any(p.startswith("/api/v1/reference_items/ritem_1")
                   for _, p in fake.calls)


def test_exact_lookups_ask_for_a_single_row():
    """The upc/external_id filters are exact matches, so find_item requests
    limit=1 - the webapp client's convention - rather than accepting the list
    endpoint's default 50-item page."""
    fake = _FakeAPI(existing=[{"id": "ritem_1", "external_id": "SKU",
                               "images_count": 0}])
    client = _client("http://nines.test")
    client.request = fake

    assert asyncio.run(
        client.find_item("SKU", "viam-org", upc="012345678905")
    ) == "ritem_1"
    lookups = [path for method, path in fake.calls
               if method == "GET"
               and path.split("?")[0] == "/api/v1/reference_items"]
    assert len(lookups) == 2  # the upc leg missed, the external_id leg hit
    for path in lookups:
        query = urllib.parse.parse_qs(path.split("?", 1)[1])
        assert query["limit"] == ["1"]


def test_a_created_product_starts_from_the_count_its_upsert_reported():
    fake = _FakeAPI()
    client = _client("http://nines.test")
    client.request = fake
    asyncio.run(client._resolve_item("SKU", None, "viam-org"))
    assert client.item_image_counts["ritem_1"] == 0


def test_a_failed_baseline_read_does_not_fail_the_resolution():
    """Losing the baseline downgrades a later duplicate check; it must not
    take the delivery down with it."""
    class _NoShow(_FakeAPI):
        def __call__(self, method, path, body, timeout_s):
            if method == "GET" and path.startswith("/api/v1/reference_items/"):
                raise NinesAPIError("boom", status=503)
            return super().__call__(method, path, body, timeout_s)

    fake = _NoShow(existing=[{"id": "ritem_1", "external_id": "SKU"}])
    client = _client("http://nines.test")
    client.request = fake
    assert asyncio.run(client._resolve_item("SKU", None, "viam-org")) == "ritem_1"
    assert "ritem_1" not in client.item_image_counts


# ---------------------------------------------------------------------------
# The journal: pending retries survive a restart.
# ---------------------------------------------------------------------------

def test_pending_jobs_are_journalled_and_come_back(tmp_path):
    journal = str(tmp_path / "queue.json")
    client = _ScriptedClient({"S": [NinesAPIError("down", status=503)]})
    queue = _queue(client, journal_path=journal, first_delay_s=0.02,
                   max_delay_s=30.0, factor=1000.0)

    async def before_the_restart():
        queue.enqueue("S", IMAGES, org="viam-org", upc="012345678905",
                      context={"delete_after": True})
        while not client.calls:
            await asyncio.sleep(0.005)
        await queue.close()

    asyncio.run(before_the_restart())
    assert os.path.exists(journal)

    # A fresh process: a new queue over the same journal.
    revived = _ScriptedClient({"S": [{"reference_item_id": "ritem_1"}]})
    delivered = []
    reborn = _queue(revived, journal_path=journal)
    assert reborn.restore(lambda job: (delivered.append, None)) == 1

    job = reborn._jobs[0]
    assert job.sku == "S"
    assert job.org == "viam-org"
    assert job.upc == "012345678905"
    assert job.attempt == 2
    assert job.context == {"delete_after": True}
    assert job.images == [("/photos/a.jpg", "a.jpg", ["stem"])]

    asyncio.run(_drain(reborn))
    assert len(delivered) == 1
    # A restored job cannot know whether its in-flight attempt landed.
    assert revived.calls[0]["verify_first"] is True
    # Drained, so nothing stale is left to restore next time.
    assert not os.path.exists(journal)


def test_a_delivered_job_leaves_no_journal_entry(tmp_path):
    journal = str(tmp_path / "queue.json")
    client = _ScriptedClient({"S": [{"ok": True}], "T": [{"ok": True}]})
    queue = _queue(client, journal_path=journal)

    async def scenario():
        queue.enqueue("S", IMAGES)
        queue.enqueue("T", [("/photos/b.jpg", "b.jpg", ["s2"])])
        await _drain(queue)

    asyncio.run(scenario())
    assert not os.path.exists(journal)


def test_an_abandoned_job_leaves_no_journal_entry(tmp_path):
    journal = str(tmp_path / "queue.json")
    client = _ScriptedClient({"S": [NinesAPIError("wrong org", status=403)]})
    queue = _queue(client, journal_path=journal)

    async def scenario():
        queue.enqueue("S", IMAGES)
        await _drain(queue)

    asyncio.run(scenario())
    assert not os.path.exists(journal)
    assert queue.snapshot()["abandoned"][0]["sku"] == "S"


def test_restored_ids_do_not_collide_with_new_ones(tmp_path):
    journal = str(tmp_path / "queue.json")
    open(journal, "w").write(json.dumps([{
        "job_id": "nines-7", "sku": "S",
        "images": [["/photos/a.jpg", "a.jpg", ["stem"]]],
        "org": "viam-org", "product_name": None, "upc": None,
        "attempt": 2, "ambiguous": False, "error": "down", "context": {},
    }]))
    queue = _queue(_ScriptedClient({}), journal_path=journal)
    assert queue.restore() == 1
    assert queue.enqueue("T", IMAGES)["job_id"] == "nines-8"


def test_a_corrupt_journal_is_reported_and_ignored(tmp_path):
    """Refusing to configure over a damaged scratch file would be a far worse
    failure than losing the schedule - the images are still on disk."""
    journal = str(tmp_path / "queue.json")
    open(journal, "w").write("{not json at all")
    queue = _queue(_ScriptedClient({}), journal_path=journal)
    assert queue.restore() == 0
    assert queue.snapshot()["pending_count"] == 0

    open(journal, "w").write(json.dumps([{"sku": "S"}]))  # missing fields
    assert queue.restore() == 0


def test_a_half_written_journal_never_replaces_a_good_one(tmp_path):
    """The write goes to a temp file and is renamed into place, so a crash
    mid-write cannot leave a truncated journal behind."""
    journal = str(tmp_path / "queue.json")
    queue = _queue(_ScriptedClient({}), journal_path=journal)
    queue.enqueue("S", IMAGES)
    assert json.loads(open(journal).read())[0]["sku"] == "S"
    assert not os.path.exists(journal + ".tmp")


def test_persistence_is_optional(tmp_path):
    """Without a journal path the queue still works - it just forgets."""
    queue = _queue(_ScriptedClient({"S": [{"ok": True}]}))
    assert queue.journal_path is None
    assert queue.restore() == 0

    async def scenario():
        queue.enqueue("S", IMAGES)
        await _drain(queue)

    asyncio.run(scenario())
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_journal_does_not_stop_a_retry(tmp_path):
    queue = _queue(_ScriptedClient({"S": [{"ok": True}]}),
                   journal_path=str(tmp_path / "no" / "such" / "dir.json"))

    async def scenario():
        assert queue.enqueue("S", IMAGES) is not None
        await _drain(queue)

    asyncio.run(scenario())
