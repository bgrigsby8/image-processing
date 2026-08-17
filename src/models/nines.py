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
"""

import asyncio
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

NINES_DEFAULT_BASE_URL = "https://review-app.ninesstyle.com"

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


class NinesAPIError(RuntimeError):
    """A failed Nines partner-API call; ``status`` is the HTTP status code, or
    ``None`` when the API was unreachable."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


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
        unreachable.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode()).get("error", "")
            except Exception:  # noqa: BLE001 - error bodies aren't guaranteed JSON
                pass
            raise NinesAPIError(
                f"Nines API {method} {path} failed with {exc.code}"
                + (f": {detail}" if detail else ""),
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise NinesAPIError(
                f"Nines API {method} {path} unreachable: {exc.reason}"
            ) from exc

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
            query = urllib.parse.urlencode({"shots_organization_slug": org, **filt})
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
                    return item_id
        return None

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
        """
        item_id = await self.find_item(sku, org, upc=upc)
        if item_id is not None:
            self.item_ids[(org, sku)] = item_id
            self.logger.info(
                f"Nines reference item {item_id} found for SKU {sku!r} "
                f"in org {org!r}"
            )
            return item_id
        return await self.upsert_item(sku, product_name, org_slug=org, upc=upc)

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
            raise NinesAPIError("Nines upsert returned no reference item id")
        self.item_ids[(org, sku)] = item_id
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
        """
        org = org_slug or self.org_slug
        cached = (org, sku) in self.item_ids
        item_id = self.item_ids.get((org, sku)) or await self._resolve_item(
            sku, product_name, org, upc=upc
        )

        payload: List[Dict[str, Any]] = []
        for path, filename, tags in images:
            image: Dict[str, Any] = {
                "data": await asyncio.to_thread(_read_base64, path),
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

        return {
            "reference_item_id": item_id,
            "external_id": sku,
            "added_count": response.get("added_count"),
            "images_count": response.get("images_count"),
        }
