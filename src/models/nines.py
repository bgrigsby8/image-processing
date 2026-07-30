"""
nines.py
--------
Client for the Nines partner API (the REST API documented in the nines-webapp
repo's partner-api-guide.md): products ("reference items") are upserted by
``external_id`` - our SKU - and imagery is appended to them.

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
        self, method: str, path: str, body: Mapping[str, Any], timeout_s: float
    ) -> Dict[str, Any]:
        """
        One JSON request to the Nines partner API. Synchronous (urllib) - call
        it via ``asyncio.to_thread``. Raises :class:`NinesAPIError` carrying
        the HTTP status and the API's ``error`` description on a non-2xx
        response, or without a status when the API was unreachable.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
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
        except Exception:
            # Anything else (e.g. a malformed 2xx body) - never silently lost.
            self.logger.exception(f"Nines API {method} {path} raised unexpectedly")
            raise

    async def upsert_item(
        self, sku: str, product_name: Optional[str], org_slug: Optional[str] = None
    ) -> str:
        """
        Upsert the Nines reference item whose ``external_id`` is ``sku`` in the
        effective org (``org_slug`` when given, else the configured slug) and
        cache its id under ``(org, sku)``. Deliberately sends no ``images``
        field - on an existing product that would *replace* all of its imagery;
        appending happens through the non-destructive images endpoint only.
        """
        org = org_slug or self.org_slug
        response = await asyncio.to_thread(
            self.request,
            "POST",
            "/api/v1/reference_items",
            {
                "shots_organization_slug": org,
                "name": product_name or sku,
                "external_id": sku,
            },
            self.request_timeout_s,
        )
        item_id = str(response.get("id") or "")
        if not item_id:
            self.logger.error(
                f"Nines upsert for SKU {sku!r} in org {org!r} returned no "
                f"reference item id (response: {response!r})"
            )
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
    ) -> Dict[str, Any]:
        """
        Deliver on-disk image files to the Nines product identified by ``sku``
        in the effective org (``org_slug`` when given, else the configured
        slug): upsert the reference item (once per ``(org, SKU)`` for this
        client's lifetime), then append every image non-destructively as inline
        base64. ``images`` is ``[(path, upload_filename, tags)]``; every file
        must carry a jpeg/png/webp/gif extension. Raises on any API failure -
        callers decide whether that fails their operation.
        """
        org = org_slug or self.org_slug
        cached = (org, sku) in self.item_ids
        item_id = self.item_ids.get((org, sku)) or await self.upsert_item(
            sku, product_name, org_slug=org
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
                raise NinesAPIError(
                    f"could not read {path!r} for Nines delivery: {exc}"
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
                "gone (404); re-upserting and retrying"
            )
            self.item_ids.pop((org, sku), None)
            item_id = await self.upsert_item(sku, product_name, org_slug=org)
            response = await append(item_id)

        return {
            "reference_item_id": item_id,
            "external_id": sku,
            "added_count": response.get("added_count"),
            "images_count": response.get("images_count"),
        }
