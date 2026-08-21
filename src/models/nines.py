"""
nines.py
--------
Client for the Nines partner API (the REST API documented in the nines-webapp
repo's partner-api-guide.md). Products ("reference items") are pre-loaded on
the Nines platform ahead of a shoot; delivery *looks one up* - by UPC
(``product_details.upc``), then by exact ``external_id`` (our SKU) - and
appends imagery to it. A product is created only when the lookup finds nothing
and we have a real SKU (never a placeholder), with the UPC carried in
``product_details`` rather than the name.

:class:`NinesClient` holds the credentials and the per-``(org, SKU)``
reference-item cache; it knows nothing about Viam. The component in
color_correction.py builds one per reconfigure and layers the upload-flow
policy (which file of a capture set to deliver, never failing the Viam half
of a submit) on top.

Failures are *classified* here rather than at the call site:
:class:`NinesAPIError` carries ``retryable`` (could a later attempt plausibly
succeed?) and ``ambiguous`` (might the request have been committed before we
lost the answer?). The append endpoint is non-idempotent, so ``ambiguous``
is what stops a retry from double-delivering an image - see
:meth:`NinesClient.already_appended` and ``deliver(..., verify_first=True)``.

:class:`NinesDeliveryQueue` turns that classification into an actual retry: a
delivery the caller could not complete inline is handed over and re-attempted
on a widening backoff, behind whatever else is already waiting.
"""

import asyncio
import base64
import inspect
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

NINES_DEFAULT_BASE_URL = "https://review-app.ninesstyle.com"

# Sent on every request so Nines can attribute this traffic to the Viam
# integration; urllib's default "Python-urllib/x.y" is anonymous, which makes
# our calls impossible to pick out on their side when debugging an incident.
NINES_USER_AGENT = "viam-image-processing/color-correction"

# Content types the Nines image-append endpoint accepts, by file extension.
# The RAW master, TIFFs, and JSON sidecar of a capture set are Viam-archival
# only - Nines rejects them.
NINES_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Which partner-API failures a later attempt could plausibly survive. The
# guide's error table documents 401 (unknown/revoked key), 403 (wrong org or
# missing scope), 404 (no such record) and 422 (validation) as *client-side*
# problems: they fail identically forever, so retrying only hammers the API -
# a key scoped to one org pointed at another's slug 403s on every attempt.
# What is left is congestion (408, 429) and the server breaking on its own
# side (5xx), plus an unreachable API (no status at all).
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Which failures leave the outcome *unknown* - the request may already have
# been committed on the Nines side before we lost the answer. Appending images
# is not idempotent (`POST /reference_items/:id/images` adds to what is there),
# so a blind re-send after one of these can duplicate a shot on the product.
# 429 is deliberately absent: rate limiting rejects a request before it is
# processed, so a 429 is a clean miss.
_AMBIGUOUS_STATUSES = frozenset({500, 502, 503, 504})

# Retry pacing for NinesDeliveryQueue. The first re-attempt is quick - a studio
# submit that tripped over a momentary blip should land while the operator is
# still on the same product - and then doubles, so a Nines outage or a dead
# uplink is not hammered. Six attempts spans 3+6+12+24+48s of waiting, about a
# minute and a half, which outlasts a router reboot without holding a shot
# hostage for the rest of a shoot.
NINES_RETRY_FIRST_DELAY_SEC = 3.0
NINES_RETRY_MAX_DELAY_SEC = 300.0
NINES_RETRY_MAX_ATTEMPTS = 6
# Not configurable: doubling is the point of a backoff, and a factor an
# operator could set to 1.0 would turn the queue into a fixed-rate hammer.
_RETRY_FACTOR = 2.0
# Spread re-attempts by +/-20%. ptp.py's backoff needs no jitter because it
# talks to one local camera; this talks to a shared API that a whole fleet of
# studio machines hits, and an outage would otherwise have every machine
# reconnect in lockstep.
_RETRY_JITTER = 0.2


class NinesAPIError(RuntimeError):
    """
    A failed Nines partner-API call. ``status`` is the HTTP status code, or
    ``None`` when the API was unreachable.

    ``retryable`` says whether a later attempt could plausibly succeed, and
    ``ambiguous`` whether the request may already have been committed before
    we lost the answer - the flag that decides whether re-sending a
    non-idempotent append risks a duplicate image. Both are derived from
    ``status`` unless a caller overrides them, which the non-transport raise
    sites do: an unreadable local file and a malformed 2xx body both carry no
    status but are nothing like an unreachable API.
    """

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        *,
        retryable: Optional[bool] = None,
        ambiguous: Optional[bool] = None,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = (
            (status is None or status in _RETRYABLE_STATUSES)
            if retryable is None
            else retryable
        )
        self.ambiguous = (
            (status is None or status in _AMBIGUOUS_STATUSES)
            if ambiguous is None
            else ambiguous
        )


def _read_base64(path: str) -> str:
    """Whole-file base64 for the Nines inline image form (run in a thread)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class NinesClient:
    """
    Nines partner-API delivery for one configuration (key / default org /
    base URL). Rebuilt on every component reconfigure, which also resets the
    reference-item cache - the ids are scoped to the base URL, which may just
    have changed.
    """

    def __init__(
        self,
        api_key: Optional[str],
        org_slug: Optional[str],
        base_url: str,
        *,
        logger: Any,
        request_timeout_s: float,
        upload_timeout_s: float,
    ):
        self.api_key = api_key
        self.org_slug = org_slug
        self.base_url = str(base_url or NINES_DEFAULT_BASE_URL).rstrip("/")
        self.logger = logger
        # Small JSON calls (upsert) vs. the inline-base64 image append, which
        # scales with file size; mirrors the component's upload timeouts.
        self.request_timeout_s = request_timeout_s
        self.upload_timeout_s = upload_timeout_s
        # Upserted reference-item ids keyed by (org slug, SKU), so a multi-shot
        # submit upserts each product once. The org is part of the key because
        # one machine can serve multiple orgs (the webapp may pass a per-request
        # `shots_organization_slug`) and the same external_id/SKU can exist in
        # more than one org - a SKU-only key would deliver one org's shot to
        # another org's product.
        self.item_ids: Dict[Tuple[str, str], str] = {}
        # Last known image count per reference-item id, learned from every
        # upsert/append response. It is the baseline
        # :meth:`already_appended` compares against to decide whether an
        # append whose answer we lost actually landed. Keyed by item id alone
        # (ids are already unique across orgs) and, like item_ids, scoped to
        # this client's base URL. A product that was pre-loaded and never
        # appended to in this process has no entry - that case falls back to
        # the weaker tag check.
        self.item_image_counts: Dict[str, int] = {}
        # One in-flight resolution per (org, SKU). Two deliveries for the same
        # product that both miss the cache would otherwise both find nothing
        # and both create it - and a re-POST overwrites product_details on the
        # loser. Concurrent submits already made this possible; the retry
        # queue, which delivers alongside whatever the operator is doing now,
        # makes it ordinary.
        self._resolve_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    def ready(self, org_slug: Optional[str]) -> bool:
        """Whether Nines delivery can proceed for the given effective org slug.
        Needs an API key plus an org slug from *somewhere* - the per-request
        ``org_slug`` when the webapp supplies one, else the configured slug. A
        machine configured with only a key can still serve any org the webapp
        names; that's what lets one machine deliver to multiple orgs."""
        return bool(self.api_key and (org_slug or self.org_slug))

    @staticmethod
    def pick_image(paths: Sequence[str]) -> Optional[str]:
        """
        Choose the one file of a capture set to deliver to Nines. The partner
        API wants exactly one full-resolution original per view and accepts
        only jpeg/png/webp/gif - so out of a set that also carries the RAW
        master, TIFFs, and the sidecar, prefer the full-res JPEG, then the
        8-bit PNG, then the 16-bit PNG, then webp/gif. Returns ``None`` when
        nothing in the set is eligible.
        """
        def rank(path: str) -> Optional[Tuple[int, str]]:
            ext = os.path.splitext(path)[1].lower()
            if ext not in NINES_CONTENT_TYPES:
                return None
            if ext in (".jpg", ".jpeg"):
                order = 0
            elif ext == ".png":
                order = 2 if path.lower().endswith("_16.png") else 1
            elif ext == ".webp":
                order = 3
            else:  # .gif
                order = 4
            return order, path

        ranked = sorted(r for r in map(rank, paths) if r is not None)
        return ranked[0][1] if ranked else None

    def request(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]],
        timeout_s: float,
    ) -> Dict[str, Any]:
        """
        One JSON request to the Nines partner API. Synchronous (urllib) - call
        it via ``asyncio.to_thread``. Pass ``body=None`` for a GET (a bodyless
        request); a mapping is JSON-encoded for POST/PATCH. Raises
        :class:`NinesAPIError` carrying the HTTP status and the API's ``error``
        description on a non-2xx response, or without a status when the API was
        unreachable or timed out. Every transport failure arrives as a
        ``NinesAPIError`` so callers never have to catch urllib's or socket's
        types themselves.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": NINES_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode()).get("error", "")
            except Exception as parse_exc:  # noqa: BLE001 - error bodies aren't guaranteed JSON
                self.logger.debug(
                    f"Nines API {method} {path}: error body ({exc.code}) was not "
                    f"parseable JSON ({parse_exc})"
                )
            message = (
                f"Nines API {method} {path} failed with {exc.code}"
                + (f": {detail}" if detail else "")
            )
            self.logger.error(message)
            raise NinesAPIError(message, status=exc.code) from exc
        except urllib.error.URLError as exc:
            message = f"Nines API {method} {path} unreachable: {exc.reason}"
            self.logger.error(message)
            raise NinesAPIError(message) from exc
        except TimeoutError as exc:
            # urlopen wraps a *connect* timeout in URLError (caught above) but
            # lets a *read* timeout out bare, so without this branch the most
            # likely real failure on a studio machine - the API accepting the
            # connection and then stalling on a multi-megabyte append - would
            # escape past every caller that catches NinesAPIError, and past
            # the retry classification entirely.
            message = f"Nines API {method} {path} timed out after {timeout_s:.0f}s"
            self.logger.error(message)
            raise NinesAPIError(message) from exc
        except Exception:
            # Anything else (e.g. a malformed 2xx body) - never silently lost.
            self.logger.exception(f"Nines API {method} {path} raised unexpectedly")
            raise

    async def find_item(
        self, sku: str, org: Optional[str], upc: Optional[str] = None
    ) -> Optional[str]:
        """
        Look up an already-loaded reference item's id in ``org`` without
        creating anything, via ``GET /api/v1/reference_items``. Tries the
        ``upc`` filter first (the operator's primary identifier once catalogs
        are pre-loaded; matches ``product_details->>'upc'``), then the exact
        ``external_id``/SKU filter. Returns the id, or ``None`` when the product
        isn't loaded in the org yet. Does not cache - the caller does.
        """
        # UPC before SKU: a pre-loaded catalog is keyed to the client's UPC, and
        # the SKU we were handed may just be its fallback label.
        attempts: List[Dict[str, str]] = []
        if upc:
            attempts.append({"upc": str(upc)})
        attempts.append({"external_id": sku})
        for filt in attempts:
            # limit=1 mirrors the webapp's Go client: both filters are exact
            # matches, so one row decides the answer - no point accepting the
            # list endpoint's default 50-item page.
            query = urllib.parse.urlencode(
                {"shots_organization_slug": org, "limit": 1, **filt}
            )
            response = await asyncio.to_thread(
                self.request,
                "GET",
                f"/api/v1/reference_items?{query}",
                None,
                self.request_timeout_s,
            )
            for item in response.get("reference_items") or []:
                # The server filter is trusted for UPC, but re-check exact
                # (case-insensitive) SKU: an external_id lookup must not match a
                # near-neighbour product.
                if "external_id" in filt and (
                    str(item.get("external_id") or "").lower() != sku.lower()
                ):
                    continue
                item_id = str(item.get("id") or "")
                if item_id:
                    # Free baseline when the list endpoint carries one. The
                    # partner-API guide documents images_count on the upsert
                    # and append responses but does not say whether the list
                    # includes it, so this is opportunistic - _resolve_item
                    # pays for a show request only when it is missing.
                    count = item.get("images_count")
                    if isinstance(count, int):
                        # setdefault, not assignment: a baseline seeded from a
                        # restored job is a *pre-append* count and must win
                        # over anything read back afterwards.
                        self.item_image_counts.setdefault(item_id, count)
                    return item_id
        return None

    async def get_item(self, item_id: str, org: Optional[str]) -> Dict[str, Any]:
        """
        Fetch one reference item via ``GET /api/v1/reference_items/:id``. The
        show response carries the ``images`` array (id, position, tags, url)
        that the list endpoint used by :meth:`find_item` does not. Needs only
        the ``reference_items:read`` scope the lookup already requires.
        """
        query = urllib.parse.urlencode({"shots_organization_slug": org})
        return await asyncio.to_thread(
            self.request,
            "GET",
            f"/api/v1/reference_items/{item_id}?{query}",
            None,
            self.request_timeout_s,
        )

    @staticmethod
    def _appended_verdict(
        before_count: Optional[int],
        remote_images: Sequence[Mapping[str, Any]],
        batch_tags: Sequence[Sequence[str]],
    ) -> Optional[bool]:
        """
        Did a batch of ``len(batch_tags)`` images land? ``True`` yes, ``False``
        no, ``None`` can't tell. Pure function of the evidence, so the policy
        is testable without HTTP.

        With a ``before_count`` baseline the answer is arithmetic: unchanged
        means the append never happened, grown by exactly the batch size means
        it did, anything else means something changed concurrently and we
        stop guessing.

        Without a baseline (a pre-loaded product this process has never
        appended to) only a *negative* is trustworthy: if none of the remote
        images carries a tag set from our batch, the batch definitely is not
        there. A match proves nothing - the ``upload`` path tags each image
        with its filename stem, which is unique per shot, but ``nines_upload``
        passes operator tags like ``["front"]`` that an earlier session may
        already have used. Claiming ``True`` off that would silently drop a
        re-shoot, so it degrades to ``None``. Tags are compared
        case-insensitively: the API lowercases them on ingest, so the
        mixed-case stems the upload path sends (``IMG_0042``) come back
        lowercased, and an exact comparison would call a landed batch absent -
        and re-append it, the very duplicate this check exists to prevent.
        """
        after = len(remote_images)
        if before_count is not None:
            if after == before_count:
                return False
            if after == before_count + len(batch_tags):
                return True
            return None
        if any(not tags for tags in batch_tags):
            return None
        wanted = {frozenset(str(t).lower() for t in tags) for tags in batch_tags}
        for image in remote_images:
            if frozenset(
                str(t).lower() for t in (image.get("tags") or [])
            ) in wanted:
                return None
        return False

    async def already_appended(
        self,
        item_id: str,
        org: Optional[str],
        images: Sequence[Tuple[str, str, List[str]]],
    ) -> Optional[bool]:
        """
        Whether ``images`` are already on the reference item - the guard a
        retry needs before re-sending a non-idempotent append whose answer was
        lost. Returns ``True`` (present, don't re-send), ``False`` (absent,
        safe to re-send) or ``None`` (undecidable). Never raises: if the check
        itself can't reach the API, that is a ``None``, not a failure.

        Refreshes the recorded baseline from what it sees, whatever the
        verdict.
        """
        try:
            item = await self.get_item(item_id, org)
        except NinesAPIError as exc:
            self.logger.warning(
                f"could not check whether {len(images)} image(s) already "
                f"reached Nines item {item_id}: {exc}"
            )
            return None
        remote = item.get("images") or []
        verdict = self._appended_verdict(
            self.item_image_counts.get(item_id),
            remote,
            [tags for _, _, tags in images],
        )
        self.item_image_counts[item_id] = len(remote)
        return verdict

    async def _resolve_item(
        self,
        sku: str,
        product_name: Optional[str],
        org: Optional[str],
        upc: Optional[str] = None,
    ) -> str:
        """
        Resolve the reference item id for ``(org, sku)``: look the pre-loaded
        product up first, and create it only when the lookup finds nothing.
        Caches and returns the id.

        Serialized per ``(org, sku)``: only one caller looks a given product up
        at a time, and the others take the id it cached rather than racing to
        create a second copy.
        """
        key = (org, sku)
        lock = self._resolve_locks.get(key)
        if lock is None:
            lock = self._resolve_locks[key] = asyncio.Lock()
        async with lock:
            # Whoever held the lock may have just resolved this product - the
            # stale-404 path deliberately clears the cache before calling here,
            # so re-reading it inside the lock is what makes a concurrent pair
            # of retries share one lookup instead of racing to create.
            cached = self.item_ids.get(key)
            if cached is not None:
                return cached
            item_id = await self.find_item(sku, org, upc=upc)
            if item_id is not None:
                self.item_ids[key] = item_id
                self.logger.info(
                    f"Nines reference item {item_id} found for SKU {sku!r} "
                    f"in org {org!r}"
                )
                await self._learn_image_count(item_id, org)
                return item_id
            return await self.upsert_item(sku, product_name, org_slug=org, upc=upc)

    async def _learn_image_count(self, item_id: str, org: Optional[str]) -> None:
        """
        Record how many images a product already has, so a later append whose
        answer is lost can be settled by arithmetic instead of guesswork.

        Only for a product we *found*: one we created starts at the count its
        upsert reported, and one we have appended to since is tracked by the
        append response. That leaves exactly the production case - a catalog
        pre-loaded on the Nines platform - paying one small request, once per
        (org, SKU) for this client's lifetime, and only when the lookup did not
        already carry the number.

        Best-effort: without the baseline :meth:`already_appended` falls back
        to the weaker tag check, which is a worse answer but not a failure, so
        this must never be the thing that breaks a delivery.
        """
        if item_id in self.item_image_counts:
            return
        try:
            item = await self.get_item(item_id, org)
        except Exception as exc:  # noqa: BLE001 - a baseline is a nicety
            self.logger.warning(
                f"could not read the image count of Nines item {item_id}: "
                f"{exc}. A retry after a lost append will have to fall back to "
                "matching tags."
            )
            return
        self.item_image_counts[item_id] = len(item.get("images") or [])

    async def upsert_item(
        self,
        sku: str,
        product_name: Optional[str],
        org_slug: Optional[str] = None,
        upc: Optional[str] = None,
    ) -> str:
        """
        Create (or, for an existing SKU, update) the Nines reference item whose
        ``external_id`` is ``sku`` in the effective org (``org_slug`` when
        given, else the configured slug) and cache its id under ``(org, sku)``.
        A ``upc`` is carried in flat ``product_details`` - never in the name.
        Deliberately sends no ``images`` field: on an existing product that
        would *replace* all of its imagery; appending happens through the
        non-destructive images endpoint only. Reserved for the create path in
        :meth:`_resolve_item` - a re-POST also overwrites ``product_details``,
        so callers must confirm the product is absent first.
        """
        org = org_slug or self.org_slug
        body: Dict[str, Any] = {
            "shots_organization_slug": org,
            "name": product_name or sku,
            "external_id": sku,
        }
        if upc:
            body["product_details"] = {"upc": str(upc)}
        response = await asyncio.to_thread(
            self.request,
            "POST",
            "/api/v1/reference_items",
            body,
            self.request_timeout_s,
        )
        item_id = str(response.get("id") or "")
        if not item_id:
            self.logger.error(
                f"Nines upsert for SKU {sku!r} in org {org!r} returned no "
                f"reference item id (response: {response!r})"
            )
            # No status, but nothing a retry fixes: the call *succeeded* and
            # the body was malformed. Retrying would re-POST an upsert, which
            # overwrites product_details on a product that may now exist.
            raise NinesAPIError(
                "Nines upsert returned no reference item id",
                retryable=False,
                ambiguous=False,
            )
        self.item_ids[(org, sku)] = item_id
        # A create we issue carries no `images`, so this is 0 - the baseline
        # that lets a later ambiguous append be answered arithmetically.
        count = response.get("images_count")
        if isinstance(count, int):
            self.item_image_counts[item_id] = count
        self.logger.info(
            f"Nines reference item {item_id} "
            f"({'created' if response.get('created') else 'updated'}) "
            f"for SKU {sku!r} in org {org!r}"
        )
        return item_id

    async def deliver(
        self,
        sku: str,
        images: Sequence[Tuple[str, str, List[str]]],
        product_name: Optional[str] = None,
        org_slug: Optional[str] = None,
        upc: Optional[str] = None,
        verify_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Deliver on-disk image files to the Nines product identified by ``sku``
        (and, when known, ``upc``) in the effective org (``org_slug`` when
        given, else the configured slug): resolve the reference item once per
        ``(org, SKU)`` for this client's lifetime - looking the pre-loaded
        product up and creating it only if absent - then append every image
        non-destructively as inline base64. ``images`` is
        ``[(path, upload_filename, tags)]``; every file must carry a
        jpeg/png/webp/gif extension. Raises on any API failure - callers decide
        whether that fails their operation.

        Set ``verify_first`` when re-delivering after a failure that left the
        outcome unknown (``NinesAPIError.ambiguous``): the images already on
        the product are checked first, and a batch that is confirmed present
        is reported as delivered with ``added_count`` 0 and
        ``deduplicated`` true rather than appended a second time. An
        undecidable check re-appends and says so - losing a shot is worse than
        a duplicate a human can delete, but it should never be silent.
        """
        org = org_slug or self.org_slug
        cached = (org, sku) in self.item_ids
        item_id = self.item_ids.get((org, sku)) or await self._resolve_item(
            sku, product_name, org, upc=upc
        )

        if verify_first:
            verdict = await self.already_appended(item_id, org, images)
            if verdict:
                self.logger.info(
                    f"{len(images)} image(s) for SKU {sku!r} in org {org!r} "
                    f"already reached Nines item {item_id} on the earlier "
                    "attempt; not appending them again"
                )
                return {
                    "reference_item_id": item_id,
                    "external_id": sku,
                    "added_count": 0,
                    "images_count": self.item_image_counts.get(item_id),
                    "deduplicated": True,
                }
            if verdict is None:
                self.logger.warning(
                    f"could not confirm whether the earlier attempt for SKU "
                    f"{sku!r} in org {org!r} reached Nines item {item_id}; "
                    "re-appending, which may leave a duplicate image"
                )

        payload: List[Dict[str, Any]] = []
        for path, filename, tags in images:
            try:
                data = await asyncio.to_thread(_read_base64, path)
            except OSError as exc:
                self.logger.error(
                    f"Nines delivery for SKU {sku!r} in org {org!r}: could not "
                    f"read {path!r}: {exc}"
                )
                # A local filesystem problem, not a network one: the file is
                # missing or unreadable and will be just as missing next time.
                raise NinesAPIError(
                    f"could not read {path!r} for Nines delivery: {exc}",
                    retryable=False,
                    ambiguous=False,
                ) from exc
            image: Dict[str, Any] = {
                "data": data,
                "filename": filename,
                "content_type": NINES_CONTENT_TYPES[os.path.splitext(path)[1].lower()],
            }
            if tags:
                image["tags"] = tags
            payload.append(image)

        async def append(rid: str) -> Dict[str, Any]:
            return await asyncio.to_thread(
                self.request,
                "POST",
                f"/api/v1/reference_items/{rid}/images",
                {"shots_organization_slug": org, "images": payload},
                self.upload_timeout_s,
            )

        try:
            response = await append(item_id)
        except NinesAPIError as exc:
            # A cached id can go stale (product deleted on the Nines side);
            # re-upsert once and retry rather than wedging every later shot of
            # the session on the dead id.
            if not (cached and exc.status == 404):
                raise
            self.logger.warning(
                f"cached Nines item {item_id} for SKU {sku!r} in org {org!r} is "
                "gone (404); re-resolving and retrying"
            )
            self.item_ids.pop((org, sku), None)
            item_id = await self._resolve_item(sku, product_name, org, upc=upc)
            response = await append(item_id)

        count = response.get("images_count")
        if isinstance(count, int):
            self.item_image_counts[item_id] = count
        return {
            "reference_item_id": item_id,
            "external_id": sku,
            "added_count": response.get("added_count"),
            "images_count": response.get("images_count"),
        }


@dataclass
class NinesDeliveryJob:
    """One delivery waiting to be re-attempted.

    ``attempt`` counts failures *so far*, including the caller's inline one, so
    a job handed over after a single failed submit arrives with ``attempt`` 1
    and its first queued try is the second overall. ``ambiguous`` carries
    forward whether the last failure left the outcome unknown; it becomes the
    next attempt's ``verify_first``, which is what keeps a retry from
    double-appending a shot the API had already accepted.
    """

    job_id: str
    sku: str
    images: List[Tuple[str, str, List[str]]]
    org: Optional[str] = None
    product_name: Optional[str] = None
    upc: Optional[str] = None
    attempt: int = 1
    ambiguous: bool = False
    next_at: float = 0.0
    error: Optional[str] = None
    # JSON-safe, opaque to the queue, and written to the journal: whatever the
    # owner needs to rebuild this job's callbacks after a restart (for the
    # component, whether the delivery image should be deleted once it lands).
    # The product this batch is being appended to, and how many images it had
    # before the attempt that failed. Journalled because a restart empties the
    # client's caches: without the pre-append count, a restored job would
    # learn the count *after* its possibly-committed append, conclude nothing
    # landed, and deliver the shot twice.
    item_id: Optional[str] = None
    images_count_before: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    on_success: Optional[Callable[..., Any]] = None
    on_abandon: Optional[Callable[..., Any]] = None

    def record(self) -> Dict[str, Any]:
        """The job as JSON for the journal. Deliberately without ``next_at``
        (a monotonic clock means nothing across a restart - a restored job is
        rescheduled from now) and without the callbacks, which ``context``
        exists to rebuild."""
        return {
            "job_id": self.job_id,
            "sku": self.sku,
            "images": [[path, filename, list(tags)]
                       for path, filename, tags in self.images],
            "org": self.org,
            "product_name": self.product_name,
            "upc": self.upc,
            "attempt": self.attempt,
            "ambiguous": self.ambiguous,
            "error": self.error,
            "item_id": self.item_id,
            "images_count_before": self.images_count_before,
            "context": dict(self.context),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "NinesDeliveryJob":
        """Rebuild a job from :meth:`record`. Raises on anything malformed -
        the caller drops the journal rather than half-restoring it."""
        images = [
            (str(path), str(filename), [str(t) for t in (tags or [])])
            for path, filename, tags in record["images"]
        ]
        if not images:
            raise ValueError("a delivery job with no images")
        return cls(
            job_id=str(record["job_id"]),
            sku=str(record["sku"]),
            images=images,
            org=record.get("org"),
            product_name=record.get("product_name"),
            upc=record.get("upc"),
            attempt=max(1, int(record.get("attempt", 1))),
            ambiguous=bool(record.get("ambiguous", False)),
            error=record.get("error"),
            item_id=record.get("item_id"),
            images_count_before=record.get("images_count_before"),
            context=dict(record.get("context") or {}),
        )

    def summary(self, now: float) -> Dict[str, Any]:
        """JSON-safe view for a status command."""
        return {
            "job_id": self.job_id,
            "sku": self.sku,
            "org": self.org,
            "attempt": self.attempt,
            "next_attempt_in_s": round(max(0.0, self.next_at - now), 1),
            "files": [path for path, _, _ in self.images],
            "error": self.error,
        }


class NinesDeliveryQueue:
    """
    Re-attempts Nines deliveries that failed, so a transient network problem
    does not cost a shot. Knows nothing about Viam - it holds a
    :class:`NinesClient`, a logger, and per-job callbacks; the component
    supplies the policy those callbacks encode (when the local file may
    finally be deleted, how an abandoned delivery is surfaced).

    The shape follows the operator's mental model. A failed delivery is
    re-tried quickly, because the usual cause is momentary. If other
    deliveries are already waiting it goes behind them rather than ahead:
    one product that Nines refuses must never stall the rest of a shoot. Each
    further failure widens the gap, so the first re-attempt is roughly three
    seconds away but a persistently broken uplink is left alone. That makes
    the first delay a *floor*, not a deadline - with a queue in front of a
    job, its turn comes when the jobs ahead of it have had theirs.

    A single worker task drains the queue and exits when it empties; the next
    :meth:`enqueue` starts a fresh one. Failures that cannot improve
    (``NinesAPIError.retryable`` false - a bad key, the wrong org, a rejected
    image) are abandoned immediately rather than re-tried on a schedule.
    """

    def __init__(
        self,
        client: NinesClient,
        *,
        logger: Any,
        journal_path: Optional[str] = None,
        first_delay_s: float = NINES_RETRY_FIRST_DELAY_SEC,
        max_delay_s: float = NINES_RETRY_MAX_DELAY_SEC,
        max_attempts: int = NINES_RETRY_MAX_ATTEMPTS,
        factor: float = _RETRY_FACTOR,
        jitter: float = _RETRY_JITTER,
    ):
        # Reassignable: a reconfigure rebuilds the client (possibly with a
        # corrected key or base URL) while the queue and its pending jobs
        # survive, so the fix reaches work that is already waiting.
        self.client = client
        self.logger = logger
        # Where pending jobs are written so a module restart does not lose
        # them. None disables persistence: the queue still works, it just
        # forgets on restart (the files stay on disk either way).
        self.journal_path = journal_path
        self.first_delay_s = float(first_delay_s)
        self.max_delay_s = float(max_delay_s)
        self.max_attempts = int(max_attempts)
        self.factor = float(factor)
        self.jitter = float(jitter)

        self._jobs: Deque[NinesDeliveryJob] = deque()
        # Recent give-ups, for the status command. Bounded: an unattended
        # machine with a dead key must not grow this without limit.
        self._abandoned: Deque[Dict[str, Any]] = deque(maxlen=32)
        self._worker: Optional["asyncio.Task[None]"] = None
        # Lets a fresh enqueue interrupt a worker that is sleeping out some
        # other job's backoff, so new work is never stuck behind a timer.
        self._wake = asyncio.Event()
        self._seq = 0

    # -- public API ---------------------------------------------------------

    def enqueue(
        self,
        sku: str,
        images: Sequence[Tuple[str, str, List[str]]],
        *,
        org: Optional[str] = None,
        product_name: Optional[str] = None,
        upc: Optional[str] = None,
        attempt: int = 1,
        ambiguous: bool = False,
        error: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        on_success: Optional[Callable[..., Any]] = None,
        on_abandon: Optional[Callable[..., Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Hand a failed delivery over for re-attempt and return
        ``{"job_id", "attempt", "next_attempt_in_s", "queued"}`` - enough for
        the caller to tell an operator what will happen next. ``queued`` is
        how many jobs are waiting including this one; anything above 1 means
        the delay is a floor rather than the actual wait. Returns ``None``
        when the failures already reported use up ``max_attempts``, which is
        what makes ``max_attempts`` 1 mean "the inline attempt only" rather
        than "one inline attempt and one retry".

        Callers pass the failure they already saw as ``attempt`` (1 after one
        inline try) and ``ambiguous`` (from ``NinesAPIError.ambiguous``), and
        are expected to have checked ``retryable`` first - a queue is the wrong
        place to discover that a key is dead. Later failures are classified
        here.

        ``on_success(result)`` and ``on_abandon(job, error)`` may be sync or
        async; they run outside the delivery's own error handling, so a
        callback that raises is logged and never re-queues the job.
        """
        # Snapshot what the client knows now, while it still knows it.
        item_id = (self.client.item_ids.get((org, sku))
                   if self.client is not None else None)
        images_count_before = (
            self.client.item_image_counts.get(item_id)
            if item_id is not None else None
        )
        attempt = max(1, int(attempt))
        if attempt >= self.max_attempts:
            self.logger.error(
                f"not queueing a Nines retry for SKU {sku!r} in org {org!r}: "
                f"{attempt} attempt(s) have failed and max_attempts is "
                f"{self.max_attempts}. The file(s) are still on disk: "
                f"{[path for path, _, _ in images]}"
            )
            return None
        self._seq += 1
        job = NinesDeliveryJob(
            job_id=f"nines-{self._seq}",
            sku=sku,
            images=[(p, f, list(t)) for p, f, t in images],
            org=org,
            product_name=product_name,
            upc=upc,
            attempt=attempt,
            ambiguous=bool(ambiguous),
            error=error,
            item_id=item_id,
            images_count_before=images_count_before,
            context=dict(context or {}),
            on_success=on_success,
            on_abandon=on_abandon,
        )
        delay = self._delay(job.attempt)
        job.next_at = time.monotonic() + delay
        self._jobs.append(job)
        self._persist()
        self.logger.warning(
            f"queued Nines delivery {job.job_id} for SKU {sku!r} in org "
            f"{org!r} after attempt {job.attempt} failed; next try in "
            f"{delay:.1f}s ({len(self._jobs)} queued)"
        )
        self._ensure_worker()
        return {
            "job_id": job.job_id,
            "attempt": job.attempt,
            "next_attempt_in_s": round(delay, 1),
            "queued": len(self._jobs),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Pending and recently abandoned jobs, for a status command."""
        now = time.monotonic()
        return {
            "pending": [job.summary(now) for job in self._jobs],
            "pending_count": len(self._jobs),
            "abandoned": list(self._abandoned),
        }

    def restore(
        self, rebuild: Optional[Callable[[NinesDeliveryJob], Any]] = None
    ) -> int:
        """
        Reload pending jobs from the journal and return how many came back.

        Each restored job is rescheduled from now using the delay its attempt
        count earns - a machine that was off for an hour retries immediately
        rather than sitting out a backoff it already served. ``rebuild(job)``
        may return an ``(on_success, on_abandon)`` pair to reattach the
        callbacks, which the journal cannot carry.

        A journal that cannot be read or parsed is reported and ignored rather
        than retried: the delivery files are still on disk and nameable by
        hand, and refusing to configure over a corrupt scratch file would be a
        far worse failure than losing the schedule.
        """
        if not self.journal_path or not os.path.exists(self.journal_path):
            return 0
        try:
            with open(self.journal_path, encoding="utf-8") as handle:
                records = json.load(handle)
            jobs = [NinesDeliveryJob.from_record(r) for r in records]
        except Exception as exc:  # noqa: BLE001 - a bad journal is not fatal
            self.logger.error(
                f"could not read the Nines retry journal at "
                f"{self.journal_path!r} ({exc}); pending retries are lost. Any "
                "undelivered images are still on disk and can be sent with the "
                "`nines_upload` command."
            )
            return 0

        now = time.monotonic()
        for job in jobs:
            job.next_at = now + self._delay(job.attempt)
            # We cannot know whether the attempt that was in flight when the
            # process stopped reached Nines, so every restored job checks the
            # product before appending. One small request is a cheap price for
            # never delivering a shot twice across a restart.
            job.ambiguous = True
            if rebuild is not None:
                callbacks = rebuild(job)
                if callbacks:
                    job.on_success, job.on_abandon = callbacks
            self._jobs.append(job)
            # Keep issuing ids past the restored ones so a new job cannot
            # collide with one that came back.
            suffix = job.job_id.rsplit("-", 1)[-1]
            if suffix.isdigit():
                self._seq = max(self._seq, int(suffix))
        if jobs:
            self.logger.info(
                f"restored {len(jobs)} pending Nines deliver(ies) from "
                f"{self.journal_path!r}"
            )
            self._ensure_worker()
        return len(jobs)

    def ensure_running(self) -> None:
        """Start the worker if there is work and nothing draining it. Restore
        runs during reconfigure, which may not have an event loop to create a
        task on; calling this from any later async entry point picks the queue
        up without waiting for the next failed delivery."""
        if self._jobs:
            self._ensure_worker()

    def retarget_journal(self, journal_path: Optional[str]) -> None:
        """Point the journal somewhere else, moving the pending jobs with it
        and clearing the old file so nothing can be restored from it twice."""
        previous, self.journal_path = self.journal_path, journal_path
        self._persist()
        if previous and previous != journal_path:
            try:
                os.remove(previous)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.logger.warning(
                    f"could not remove the old Nines retry journal at "
                    f"{previous!r}: {exc}"
                )

    def _persist(self) -> None:
        """Write the pending jobs out, atomically. Never raises: losing the
        journal degrades a restart, but failing a delivery over it would be
        the larger harm."""
        if not self.journal_path:
            return
        try:
            if not self._jobs:
                # Nothing outstanding - leave no stale file to restore from.
                if os.path.exists(self.journal_path):
                    os.remove(self.journal_path)
                return
            tmp = f"{self.journal_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump([job.record() for job in self._jobs], handle)
            # Rename over the old file so a crash mid-write can never leave a
            # half-written journal in its place.
            os.replace(tmp, self.journal_path)
        except OSError as exc:
            self.logger.warning(
                f"could not write the Nines retry journal at "
                f"{self.journal_path!r}: {exc}"
            )

    async def close(self) -> None:
        """Stop the worker. Pending jobs are left in place, so a queue that is
        closed and re-opened (a reconfigure) resumes rather than forgets."""
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    # -- internals ----------------------------------------------------------

    def _delay(self, attempt: int) -> float:
        """Seconds to wait after the ``attempt``-th failure: the first delay
        doubled once per failure, capped, then jittered."""
        delay = min(
            self.first_delay_s * (self.factor ** max(0, attempt - 1)),
            self.max_delay_s,
        )
        if self.jitter:
            delay *= random.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        return max(0.0, delay)

    def _ensure_worker(self) -> None:
        self._wake.set()
        if self._worker is not None and not self._worker.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # restore() runs from a synchronous reconfigure, which may have no
            # loop to attach a task to. Not fatal: the jobs are in the deque,
            # and ensure_running() from the next async entry point picks them
            # up. Checked before building the coroutine, so a failure here
            # leaves no un-awaited coroutine behind.
            self.logger.debug(
                "no running event loop for the Nines retry worker yet; it "
                "starts on the next command"
            )
            return
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                now = time.monotonic()
                # Deque order, not soonest-due order: a job that failed went to
                # the back, and taking the first *due* one from the front is
                # what gives everything else its turn ahead of it.
                job = next((j for j in self._jobs if j.next_at <= now), None)
                if job is None:
                    if not self._jobs:
                        return  # drained; the next enqueue starts a new worker
                    wait = max(0.0, min(j.next_at for j in self._jobs) - now)
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=wait)
                    except asyncio.TimeoutError:
                        pass
                    continue
                self._jobs.remove(job)
                await self._attempt(job)
        except asyncio.CancelledError:
            raise
        finally:
            # Only deregister if this task is still the registered worker: a
            # cancelled worker's teardown runs after close() has awaited it,
            # by which point an enqueue may already have started a successor,
            # and clearing that would let a later enqueue start a second
            # worker racing this deque.
            if self._worker is asyncio.current_task():
                self._worker = None

    async def _attempt(self, job: NinesDeliveryJob) -> None:
        # Restore what the client knew when the job was queued, so the
        # duplicate check has a genuine *pre-append* baseline to compare
        # against. setdefault: a client that has since learned something newer
        # about this product is the better authority.
        if job.item_id is not None and job.images_count_before is not None:
            self.client.item_image_counts.setdefault(
                job.item_id, job.images_count_before
            )
        try:
            result = await self.client.deliver(
                job.sku,
                job.images,
                product_name=job.product_name,
                org_slug=job.org,
                upc=job.upc,
                # Only an attempt following an ambiguous failure pays for the
                # duplicate check; a clean 503 needs no verification.
                verify_first=job.ambiguous,
            )
        except NinesAPIError as exc:
            await self._failed(job, exc, exc.retryable, exc.ambiguous)
            return
        except asyncio.CancelledError:
            # Shutdown mid-flight: put it back so a reconfigure resumes it.
            self._jobs.appendleft(job)
            raise
        except Exception as exc:  # noqa: BLE001 - a bug here must not loop
            self.logger.exception(
                f"Nines delivery {job.job_id} for SKU {job.sku!r} raised "
                "unexpectedly; giving up on it"
            )
            await self._failed(job, exc, False, True)
            return

        self.logger.info(
            f"Nines delivery {job.job_id} for SKU {job.sku!r} succeeded on "
            f"attempt {job.attempt + 1}"
            + (" (already present, not re-appended)"
               if result.get("deduplicated") else "")
        )
        await self._invoke(job.on_success, result)
        # After the callback, so a crash in between leaves the job in the
        # journal: the restored attempt finds the images already there and
        # deduplicates rather than delivering twice.
        self._persist()

    async def _failed(
        self,
        job: NinesDeliveryJob,
        exc: BaseException,
        retryable: bool,
        ambiguous: bool,
    ) -> None:
        job.attempt += 1
        job.ambiguous = bool(ambiguous)
        job.error = str(exc)
        if not retryable or job.attempt >= self.max_attempts:
            reason = (
                "the failure cannot be retried"
                if not retryable
                else f"{job.attempt} attempts failed"
            )
            self.logger.error(
                f"giving up on Nines delivery {job.job_id} for SKU "
                f"{job.sku!r} in org {job.org!r}: {reason} ({exc}). The "
                f"file(s) are still on disk: {[p for p, _, _ in job.images]}"
            )
            self._abandoned.append(
                {
                    "job_id": job.job_id,
                    "sku": job.sku,
                    "org": job.org,
                    "attempts": job.attempt,
                    "error": job.error,
                    "files": [path for path, _, _ in job.images],
                }
            )
            await self._invoke(job.on_abandon, job, exc)
            self._persist()
            return

        delay = self._delay(job.attempt)
        job.next_at = time.monotonic() + delay
        self._jobs.append(job)  # the back of the queue
        self._persist()
        self.logger.warning(
            f"Nines delivery {job.job_id} for SKU {job.sku!r} failed on "
            f"attempt {job.attempt} ({exc}); next try in {delay:.1f}s "
            f"({len(self._jobs)} queued)"
        )

    async def _invoke(self, callback: Optional[Callable[..., Any]], *args: Any) -> None:
        """Run a job callback, sync or async. A callback that raises is logged
        and swallowed: it is cleanup policy, not part of the delivery."""
        if callback is None:
            return
        try:
            outcome = callback(*args)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:  # noqa: BLE001 - never let cleanup wedge the queue
            self.logger.exception("Nines delivery callback failed")
