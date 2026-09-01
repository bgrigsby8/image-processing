"""
color_correction.py
--------------------
A Viam camera component that color-corrects images from a source camera using
a 3x3 Color Correction Matrix (CCM) fitted from a Calibrite / X-Rite
ColorChecker Classic.

Two ways to get corrected images out of this component:

1. Streaming path - ``get_images`` proxies the source camera, applies the CCM
   to every JPEG/PNG frame, and returns the corrected images (names preserved).
   This is what the control tab, data manager, and vision services use.

2. DoCommand path - the studio RAW workflow, for cameras (the PTP model, or
   the Canon CCAPI module) whose full-resolution stills are exposed through
   DoCommand rather than the streaming ``Images`` method:

       {"capture": {"white_balance": "camera",
                    "output_formats": ["tiff16", "jpeg"]}}
           -> trigger a still on the source camera; if it's a RAW (CR3/NEF/...)
              downloaded to disk, demosaic it to 16-bit *linear*, apply white
              balance + the CCM, and write rendered exports next to it - leaving
              the RAW untouched as the master. Returns the export paths, a JSON
              sidecar path recording the development, and a small base64 JPEG
              preview (the full image stays on disk).

   The source still arrives either inline as ``image_base64`` (small JPEGs from
   CCAPI) or as a downloaded file path in ``saved_to`` - the PTP RAW handoff.
   Wire the PTP component as this model's ``camera`` dependency and give PTP a
   ``download_dir`` so its captures land on disk where this model can read them.

       {"capture": {"defer": true}}
       {"capture_result": {"id": "<capture_id>", "wait_sec": 60}}
           -> pipelined capture for rigs that move between shots: ``capture``
              with ``defer`` returns {"capture_id", "status", "camera_path"}
              as soon as the shutter has fired (exposure done - the rig is
              free to move), while the USB download, half-size demosaic, CCM,
              and preview encode continue in the background.
              ``capture_result`` then returns {"source_path", "image_base64",
              ...} (or {"status": "pending"} if not done within ``wait_sec``).
              Deferred captures never write exports or a sidecar - run
              ``develop`` on the returned ``source_path`` when the files are
              actually needed. Requires the ptp model (its ``trigger``
              command) as the source camera.

   This is a non-destructive, Capture One-style pipeline: 16-bit linear math,
   no auto-brightness, the original RAW preserved, adjustments recorded in a
   sidecar. See image_io.py for the decode/export details and color-space
   notes, and calibration.py for the ColorChecker detection / CCM math.

   Relevant config attributes: ``output_dir`` (default: next to the source),
   ``output_formats`` (default ["tiff16", "jpeg", "png16", "png8"]),
   ``jpeg_quality`` (95), ``white_balance`` ("camera"), ``exposure_stops`` (0.0 -
   paste the value `calibrate_color` reports to render at the calibrated
   brightness), ``tone`` ("none" - delivery look: "none" is colour-accurate /
   colorimetric, "c1" matches Capture One's brightness, "medium"/"bright" are
   lighter hand-tuned lifts; all applied to luminance only so hue is preserved),
   ``sharpen`` ("none" - capture sharpening: "light"/"medium"/"strong",
   since RAW is soft before sharpening), ``demosaic`` ("DHT" - RAW demosaic
   algorithm, sharper than libraw's stock AHD), ``write_sidecar`` (true),
   and ``delete_after_upload`` (false).

   ``capture``, ``develop``, and ``preview`` also take a per-call ``ccm``
   option — a 3x3 nested list applied instead of the configured matrix for
   this call only. The config ``ccm`` stays the default when the option is
   absent.

       {"develop": {"path": "/photos/IMG_0042.CR3"}}
       {"develop": {"paths": ["/photos/a.CR3", "/photos/b.CR3"]}}
           -> develop existing RAW/image file(s) already on disk through the
              same pipeline, with no camera trigger. Takes the same
              white_balance / exposure_stops / output_formats / output_dir
              options as ``capture``. A single ``path`` returns that file's
              result; ``paths`` returns {"developed": [...], "count": N}.

       {"develop": {"path": "/photos/IMG_0042.CR3",
                    "crop": {"x": 0.1, "y": 0.0, "w": 0.6, "h": 1.0},
                    "output_stem": "IMG_0042_crop-2"}}
           -> the cropping path. ``crop`` is normalized to the decoded frame
              (0-1, top-left origin) so a rect an operator drew on the small
              preview applies unchanged to the full-res decode; it's applied
              before the colour math, so it also makes the develop cheaper. The
              RAW master is never touched. ``output_stem`` renames the exports
              and sidecar, which is what lets one RAW be developed twice - an
              uncropped master plus one or more cropped variants - without the
              second pass overwriting the first's files.

       {"preview": {"path": "/photos/IMG_0042.CR3",
                    "crop": {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
                    "max_dim": 1600}}
           -> a display-only JPEG (base64) of a file on disk, optionally cropped.
              Writes nothing, so it is safe to call repeatedly while an operator
              adjusts a crop. Cropping an already-downsized preview in the browser
              leaves only (preview size x crop fraction) pixels, which looks soft
              for a tight crop; this crops a *fresh* decode instead, so the result
              fills ``max_dim`` however small the region is. Decodes at half
              resolution when that still clears ``max_dim``, full when it doesn't.

       {"delete": {"paths": ["/photos/a.CR3", "/photos/a.jpg"]}}
           -> remove files from this machine's disk: skipped captures the
              operator discarded, or sets already safe in the cloud. Guarded -
              only files inside the configured ``output_dir`` can be deleted.
              For kept shots, prefer ``delete_after_upload`` (config attribute
              or per-``upload`` option), which removes each file only after
              its upload succeeds.

Calibration:

       {"calibrate_color": {}}
           -> grab a live-view frame, auto-detect the ColorChecker (cv2.mcc),
              fit a CCM, and return it. No white balance (no RAW from live view).
       {"calibrate_color": {"use_capture": true}}
           -> trigger a full-resolution RAW still, auto-detect the chart,
              measure white balance from the raw CFA under the neutral patches
              ([r,g,b,g2] for rawpy's user_wb), and fit the CCM under that same
              white balance. Returns both.
       {"calibrate_color": {"path": "/photos/chart.CR3"}}
           -> same, from a RAW already on disk (no camera trigger).

The chart is detected automatically anywhere in frame (cv2.mcc, from
opencv-contrib) - no clicking patch centres. Pass ``patch_centers`` (24 [x,y])
to override detection, or ``compute_wb: false`` to skip white balance.
`calibrate_color` returns the fitted 3x3 ``ccm`` and the 4-value
``white_balance``; copy them into the component's ``ccm`` / ``white_balance``
config attributes to make the calibration persist across restarts.
"""

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image
from typing_extensions import Self

from viam.app.data_client import DataClient
from viam.components.camera import Camera
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image
from viam.rpc.dial import Credentials, DialOptions, _dial_app
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.datasync import (
    DataType,
    FileData,
    FileUploadRequest,
    FileUploadResponse,
    UploadMetadata,
)
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from models.calibration import (
    CCM_FIT_WEIGHTS,
    REFERENCE_SRGB,
    ColorCorrector,
    PatchSampler,
    _fit_ccm,
    _neutral_brightness_report,
    delta_e76,
    detect_colorchecker,
)
from models.image_io import (
    DEFAULT_DEMOSAIC,
    DEMOSAIC_ALGORITHMS,
    EXPORT_FORMATS,
    SHARPEN_OPTIONS,
    TONE_OPTIONS,
    compute_raw_wb_multipliers,
    crop_linear,
    export_renditions,
    image_dimensions,
    is_raw,
    linear_to_jpeg_base64,
    linear_to_srgb,
    load_linear_rgb,
    render_raw_for_detection,
    srgb_to_linear,
)
# Default delivery set when `output_formats` isn't configured. Override in
# config to trim it (e.g. just ["tiff16", "jpeg"] for a master + proof).
DEFAULT_OUTPUT_FORMATS = ["tiff16", "jpeg", "png16", "png8"]

# app.viam.com rejects gRPC messages over 32 MiB, and the SDK's `file_upload`
# ships the whole file as a single message - too small for a CR3 (~53 MB) or a
# 16-bit TIFF (~250 MB). The FileUpload RPC is client-streaming, so we send
# the file ourselves in chunks safely under that cap.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# Image <-> base64 / PIL helpers (used on the DoCommand boundary)
# ---------------------------------------------------------------------------

def _base64_to_rgb(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded image (JPEG/PNG) into an (H, W, 3) uint8 RGB array."""
    raw = base64.b64decode(image_base64)
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(pil)


class ColorCorrection(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "image-processing"), "color-correction"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Create a new instance of this Camera component.

        ``EasyResource.new`` only constructs the instance - it does *not* call
        ``reconfigure``, and viam-server only calls ``reconfigure`` on later
        config changes, not on the initial add. So we must configure here, or
        ``self.camera``/``self.corrector`` won't exist when the first request
        arrives.
        """
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate config and declare the source camera as a required dependency."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        if not camera:
            raise ValueError("Missing required attribute `camera` in config")

        ccm = attrs.get("ccm")
        if ccm is not None and np.array(ccm, dtype=np.float32).shape != (3, 3):
            raise ValueError("`ccm` must be a 3x3 matrix")

        formats = attrs.get("output_formats")
        if formats is not None:
            unknown = [f for f in formats if f not in EXPORT_FORMATS]
            if unknown:
                raise ValueError(
                    f"unknown `output_formats` {unknown}; valid: "
                    f"{sorted(EXPORT_FORMATS)}"
                )

        output_dir = attrs.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError("`output_dir` must be a string path")

        exposure_stops = attrs.get("exposure_stops")
        if exposure_stops is not None and not isinstance(exposure_stops, (int, float)):
            raise ValueError("`exposure_stops` must be a number (stops of exposure)")

        tone = attrs.get("tone")
        if tone is not None and tone not in TONE_OPTIONS:
            raise ValueError(f"`tone` must be one of {list(TONE_OPTIONS)}")

        sharpen = attrs.get("sharpen")
        if sharpen is not None and sharpen not in SHARPEN_OPTIONS:
            raise ValueError(f"`sharpen` must be one of {list(SHARPEN_OPTIONS)}")

        demosaic = attrs.get("demosaic")
        if demosaic is not None and demosaic not in DEMOSAIC_ALGORITHMS:
            raise ValueError(f"`demosaic` must be one of {list(DEMOSAIC_ALGORITHMS)}")

        return [str(camera)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Wire up the source camera and load the CCM from the `ccm` attribute."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        source = dependencies.get(Camera.get_resource_name(str(camera)))
        if source is None:
            raise ValueError(f"Could not resolve source camera dependency `{camera}`")
        self.camera: Camera = source

        # The CCM lives inline in the `ccm` config attribute. Run the
        # `calibrate_color` DoCommand to compute one, then copy the returned
        # matrix into this attribute to make the correction persist.
        ccm = attrs.get("ccm")
        if ccm is not None:
            self.corrector = ColorCorrector(np.array(ccm, dtype=np.float32))
        else:
            self.corrector = ColorCorrector.identity()
            self.logger.info("No `ccm` configured; passing images through uncorrected")

        # Studio export settings (used by the `capture` DoCommand RAW pipeline).
        # output_dir defaults to wherever the source file was downloaded.
        self._output_dir: Optional[str] = attrs.get("output_dir") or None
        self._output_formats: List[str] = list(
            attrs.get("output_formats") or DEFAULT_OUTPUT_FORMATS
        )
        self._jpeg_quality: int = int(attrs.get("jpeg_quality", 95))
        self._white_balance = attrs.get("white_balance", "camera")
        # Default exposure compensation (stops) applied at the raw stage, the
        # digital counterpart to flash power. `calibrate_color` reports the
        # offset the chart implied vs. the reference; paste it here (alongside
        # `ccm` / `white_balance`) to render every capture at the calibrated
        # brightness when the flash can't reach the reference optically. Per-call
        # `exposure_stops` on capture/develop still overrides this.
        self._exposure_stops: float = float(attrs.get("exposure_stops", 0.0))
        # Optional delivery "look" layered on the colour-accurate render: "none"
        # (default) is pure colorimetric output; "c1" matches Capture One's
        # brightness, "medium"/"bright" are lighter hand-tuned lifts. The curve
        # is applied to luminance only, so hue/saturation are untouched - only
        # lightness/contrast changes. Applied to every export and the preview.
        self._tone: str = attrs.get("tone") or "none"
        # Capture sharpening ("none"/"light"/"medium"/"strong"): RAW is soft
        # before sharpening, so an unsharpened export looks blurry next to a
        # Capture One / Lightroom render. Default "none" (opt-in). And the
        # demosaic algorithm used to decode RAW (DHT default - sharper than
        # libraw's stock AHD). Both feed every export and the preview.
        self._sharpen: str = attrs.get("sharpen") or "none"
        self._demosaic: str = attrs.get("demosaic") or DEFAULT_DEMOSAIC
        self._write_sidecar: bool = bool(attrs.get("write_sidecar", True))

        # Local-disk hygiene, mirroring ptp's `delete_after_download`: once a
        # file is confirmed in the cloud, the local copy is redundant. Files
        # that fail to upload are kept for retry.
        self._delete_after_upload: bool = bool(attrs.get("delete_after_upload", False))

        # The `upload` DoCommand authenticates to the cloud with the API key
        # Viam injects into every module process (VIAM_API_KEY / VIAM_API_KEY_ID),
        # so no credentials are configured here. part_id falls back to the
        # machine's env var. The data client is created lazily and reused.
        self._part_id: Optional[str] = (
            attrs.get("part_id") or os.environ.get("VIAM_MACHINE_PART_ID") or None
        )
        # Dropping the old client without closing its channel leaks the gRPC
        # connection: grpclib complains "Unclosed connection" on stderr when
        # the channel is garbage-collected, which viam-server surfaces as an
        # error log.
        self._close_data_client()
        self._data_client: Optional[DataClient] = None

        # Bound the cloud round-trips so a stuck auth dial or a stalled file
        # transfer surfaces as a clear error / per-file failure (the file is
        # kept for retry) instead of hanging the submit forever. Timeouts are
        # per-file, not per-batch, so a large shoot never trips a single global
        # deadline. The dial only happens on the first upload, which is the
        # usual place a submit silently wedges when the machine can't reach the
        # cloud.
        self._upload_dial_timeout_s: float = float(
            attrs.get("upload_dial_timeout_s", 30.0)
        )
        self._upload_file_timeout_s: float = float(
            attrs.get("upload_file_timeout_s", 180.0)
        )

        # In-flight deferred captures (`capture` with `defer: true`), keyed by
        # the capture_id handed back to the caller. Preserved across
        # reconfigure so a mid-sequence config change doesn't orphan results.
        self._pending_captures: Dict[str, "asyncio.Task"] = getattr(
            self, "_pending_captures", {}
        )
        self._capture_seq: int = getattr(self, "_capture_seq", 0)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _correct_viam_image(self, image: ViamImage) -> ViamImage:
        """Apply the CCM to a single ViamImage, preserving its mime type.

        Approximation, by design: a CCM calibrated from RAW is strictly valid
        on linear sensor data, but these streamed frames are camera JPEGs that
        already carry in-camera WB and a picture-style tone curve which
        ``srgb_to_linear`` doesn't undo. Good enough for monitoring; the
        colour-accurate path is the RAW capture/develop pipeline.
        """
        pil_image = viam_to_pil_image(image).convert("RGB")
        corrected = self.corrector.apply_to_rgb(np.array(pil_image))
        mime = image.mime_type if image.mime_type == CameraMimeType.PNG else CameraMimeType.JPEG
        return pil_to_viam_image(Image.fromarray(corrected), mime)

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        images, metadata = await self.camera.get_images(
            filter_source_names=filter_source_names,
            extra=extra,
            timeout=timeout,
            **kwargs,
        )

        corrected: List[NamedImage] = []
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                viam_img = self._correct_viam_image(image)
                corrected.append(NamedImage(image.name, viam_img.data, viam_img.mime_type))
            else:
                # Pass non-image payloads (e.g. depth) through untouched.
                self.logger.debug(
                    f"Passing through image '{image.name}' with uncorrectable "
                    f"mime type {image.mime_type}"
                )
                corrected.append(image)

        return corrected, metadata

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        # Color correction doesn't apply to point clouds; proxy the source.
        return await self.camera.get_point_cloud(extra=extra, timeout=timeout, **kwargs)

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        # Report the source camera's properties; we don't change resolution,
        # mime types, or intrinsics.
        return await self.camera.get_properties(timeout=timeout, **kwargs)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        resp: Dict[str, ValueTypes] = {}

        if "calibrate_color" in command:
            resp["calibrate_color"] = await self._calibrate_color(
                command.get("calibrate_color") or {}, timeout
            )

        if "capture" in command:
            resp["capture"] = await self._capture_corrected(
                command.get("capture") or {}, timeout
            )

        if "capture_result" in command:
            resp["capture_result"] = await self._capture_result(
                command.get("capture_result") or {}
            )

        if "develop" in command:
            resp["develop"] = await self._develop(command.get("develop") or {})

        if "preview" in command:
            resp["preview"] = await self._preview(command.get("preview") or {})

        if "upload" in command:
            resp["upload"] = await self._upload(command.get("upload") or {})

        if "delete" in command:
            resp["delete"] = self._delete_local(command.get("delete") or {})

        if not resp:
            # Not one of ours: forward verbatim to the source camera so
            # clients can reach its commands (focus, status, settings)
            # through this wrapper. The source rejects unknown commands
            # itself. Commands both models define (capture, delete) never
            # get here - the branches above claim them first. Only the
            # command and timeout are forwarded: kwargs holds server-side
            # call context (grpc metadata) that a client call can't accept.
            return await self.camera.do_command(command, timeout=timeout)
        return resp

    # ------------------------------------------------------------------
    # DoCommand handlers
    # ------------------------------------------------------------------

    def _linear_from_capture_response(
        self,
        capture: Any,
        white_balance: Any,
        exposure_stops: float,
        half_size: bool = False,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Turn a source camera's ``capture`` DoCommand response into a
        **linear-light** float RGB array (sRGB primaries) plus the source path.

        Two shapes are supported, in priority order:

        * ``image_base64`` - an inline JPEG/PNG, small enough to ship over gRPC
          (the Canon CCAPI flow). Decoded as 8-bit sRGB and linearized; ``None``
          path since there's no file on disk.
        * ``saved_to`` (or ``path``) - a file the source wrote to disk. This is
          the PTP model's handoff for full-resolution stills, including RAW
          (CR3/NEF/ARW/...). It's demosaiced to 16-bit linear by
          ``image_io.load_linear_rgb`` (applying white balance / exposure at the
          raw stage), with no precision lost before color correction.
        """
        if not isinstance(capture, Mapping):
            raise ValueError("source camera `capture` returned an unexpected response")

        image_b64 = capture.get("image_base64")
        if image_b64:
            rgb8 = _base64_to_rgb(image_b64).astype(np.float32) / 255.0
            return srgb_to_linear(rgb8).astype(np.float32), None

        path = capture.get("saved_to") or capture.get("path")
        if path:
            linear = load_linear_rgb(
                str(path), white_balance=white_balance,
                exposure_stops=exposure_stops, half_size=half_size,
                demosaic=self._demosaic,
            )
            return linear, str(path)

        raise ValueError(
            "source camera `capture` returned neither an `image_base64` field "
            "nor a `saved_to` path; if the source is the PTP camera, configure "
            "its `download_dir` so captures are written to disk"
        )

    def _corrector_for(self, raw_ccm: Any) -> ColorCorrector:
        """
        Resolve a per-call ``ccm`` option into the corrector to apply: the
        configured ``self.corrector`` when the option is absent, or a one-off
        ``ColorCorrector`` built from the given 3x3 matrix that replaces the
        configured matrix for this call only (the config ``ccm`` is untouched).
        """
        if raw_ccm is None:
            return self.corrector
        try:
            return ColorCorrector(np.array(raw_ccm, dtype=np.float32))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"`ccm` must be a 3x3 nested list of numbers: {exc}"
            ) from exc

    async def _acquire_calibration_source(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Get the source to calibrate from, as ``(raw_path, rgb8)``.

        ``raw_path`` is set when a RAW file is on disk - the path that unlocks
        raw-CFA white balance. Otherwise ``rgb8`` is an 8-bit sRGB frame for
        CCM-only calibration. Exactly one is non-None.

        Resolution order: explicit ``path`` -> ``use_capture`` (trigger a still
        on the source, prefer its ``saved_to`` RAW) -> the streaming frame.
        """
        def _file_to_rgb8(p: str) -> np.ndarray:
            linear = load_linear_rgb(
                str(p), white_balance="camera", demosaic=self._demosaic
            )
            return (linear_to_srgb(linear) * 255.0).clip(0, 255).astype(np.uint8)

        path = opts.get("path")
        if path:
            if is_raw(str(path)):
                return str(path), None
            return None, _file_to_rgb8(str(path))

        if opts.get("use_capture"):
            capture_opts = opts.get("capture_options", {"af": True})
            source_resp = await self.camera.do_command(
                {"capture": capture_opts}, timeout=timeout
            )
            capture = source_resp.get("capture", source_resp)
            if isinstance(capture, Mapping):
                p = capture.get("saved_to") or capture.get("path")
                if p and is_raw(str(p)):
                    return str(p), None
                if p:
                    return None, _file_to_rgb8(str(p))
                b64 = capture.get("image_base64")
                if b64:
                    return None, _base64_to_rgb(b64)
            raise ValueError(
                "source `capture` returned nothing usable for calibration "
                "(no RAW `saved_to`, file `path`, or inline `image_base64`)"
            )

        images, _ = await self.camera.get_images(timeout=timeout)
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                return None, np.array(viam_to_pil_image(image).convert("RGB"))
        raise ValueError("source camera returned no JPEG/PNG image to use")

    async def _calibrate_color(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Auto-calibrate from a ColorChecker frame: detect the chart (cv2.mcc),
        measure white balance from the raw CFA, and fit a CCM under that same
        white balance. Both are applied to this component immediately and
        returned so they can be copied into the ``ccm`` / ``white_balance``
        config attributes to persist across restarts.

        Options (all optional):
          ``use_capture``    trigger a full-res still on the source (needed for
                             white balance - the RAW must be on disk).
          ``path``           calibrate from a RAW/image file already on disk.
          ``capture_options``forwarded to the source ``capture`` (e.g. {"af": true}).
          ``compute_wb``     derive white balance from the chart (default true).
          ``patch_centers``  24 [x, y] coords to override auto-detection.
          ``radius``         patch sampling half-width in px (default: ~15% of
                             the detected patch size, or 10 with manual centers).

        Returns ``ccm`` (pure-colour, ~unity-gain), ``white_balance``
        ([r,g,b,g2] or null), ``exposure_stops`` (the brightness offset the chart
        implied vs. the reference - pass it back as ``exposure_stops`` on
        capture/develop to render at the ColorChecker's nominal brightness),
        ``neutral_brightness`` (as-shot 0-255 sRGB picker readout per neutral
        patch with its reference target - adjust the light power until measured
        matches reference to hit nominal brightness without touching camera
        exposure), and a ``delta_e`` report in CIE delta-E*ab 1976 (mean below
        ~3 is a solid calibration; ``after`` is exposure-normalised, so it
        reflects pure colour accuracy).
        """
        raw_path, rgb8 = await self._acquire_calibration_source(opts, timeout)
        compute_wb = bool(opts.get("compute_wb", True))
        radius = int(opts["radius"]) if "radius" in opts else None
        manual_centers = opts.get("patch_centers")

        # Image used to *locate* the patches. For a RAW we render it unrotated
        # (user_flip=0) so the centres map straight onto the CFA and the linear
        # CCM render below.
        detect_img = render_raw_for_detection(raw_path) if raw_path else rgb8

        if manual_centers is not None:
            centers = np.array(
                [(int(x), int(y)) for x, y in manual_centers], dtype=np.float32
            )
            neutral_boxes = None
            radius = radius if radius is not None else 10
        else:
            detection = detect_colorchecker(detect_img)
            if detection is None:
                raise ValueError(
                    "could not auto-detect the ColorChecker; ensure the whole "
                    "chart is visible and unobstructed, or pass `patch_centers`"
                )
            if radius is None:
                radius = int(detection["suggested_radius"])
            centers = detection["centers"]
            neutral_boxes = detection["neutral_boxes_norm"]

        # White balance from the raw Bayer/CFA under the neutral patches.
        wb: Optional[List[float]] = None
        wb_note: Optional[str] = None
        if compute_wb:
            if raw_path and neutral_boxes:
                wb = compute_raw_wb_multipliers(raw_path, neutral_boxes)
            elif raw_path:
                wb_note = (
                    "skipped: white balance needs auto-detected neutral patches; "
                    "omit `patch_centers` to enable it"
                )
            else:
                wb_note = (
                    "skipped: raw-CFA white balance needs a RAW capture - set "
                    "`use_capture: true` with a RAW source, or pass a RAW `path`"
                )

        # Fit the CCM on patches developed with the SAME white balance the
        # captures will use, so the matrix and the WB stay consistent.
        reference_linear = srgb_to_linear(REFERENCE_SRGB)
        if raw_path:
            linear = load_linear_rgb(
                raw_path,
                white_balance=(wb if wb is not None else "camera"),
                user_flip=0,  # match detect_img so `centers` line up
                demosaic=self._demosaic,
            )
            measured_linear = PatchSampler.sample_linear_at_centers(linear, centers, radius)
        else:
            measured_linear = PatchSampler.sample_at_centers(
                detect_img, [(int(x), int(y)) for x, y in centers], radius
            )

        # Decouple exposure (a single scalar) from colour (the matrix). The chart
        # is typically exposed below the reference's nominal brightness; fitting
        # the CCM directly makes it absorb that gain (diagonal >> 1), which then
        # brightens *every* developed frame and clips highlights early. Instead we
        # scale the measured patches so the neutral ramp matches the reference
        # luminance, fit the CCM on that (keeping it ~unity-gain, pure colour),
        # and report the implied exposure offset for the caller to dial in via
        # `exposure_stops`. Neutral 8 / 6.5 / 5 / 3.5 - skip the clip-prone white
        # and noisy black ends of the ramp.
        neutral_fit = [19, 20, 21, 22]
        meas_neutral = measured_linear[neutral_fit].reshape(-1)
        ref_neutral = reference_linear[neutral_fit].reshape(-1)
        energy = float(np.dot(meas_neutral, meas_neutral))
        exposure_scale = float(np.dot(meas_neutral, ref_neutral) / energy) if energy > 0 else 1.0
        measured_fit = measured_linear * exposure_scale

        # Weighted fit: ~1/Y per-patch weights so shadow accuracy isn't
        # sacrificed to numerically-large highlight errors, with the
        # out-of-gamut patches (whose targets are clipped approximations)
        # down-weighted. See CCM_FIT_WEIGHTS in calibration.py.
        ccm = _fit_ccm(measured_fit, reference_linear, weights=CCM_FIT_WEIGHTS)
        corrector = ColorCorrector(ccm)

        def _delta_e_stats(values: np.ndarray) -> Dict[str, float]:
            d = delta_e76(np.clip(values, 0, 1), reference_linear)
            return {"mean": float(d.mean()), "max": float(d.max())}

        # "after" is exposure-normalised, so it reflects pure colour accuracy
        # independent of how bright you choose to render (via exposure_stops).
        report = {
            "before": _delta_e_stats(measured_linear),
            "after": _delta_e_stats(measured_fit @ ccm.T),
        }
        exposure_stops = float(np.log2(exposure_scale)) if exposure_scale > 0 else 0.0
        neutral_brightness = _neutral_brightness_report(measured_linear)

        self.corrector = corrector
        if wb is not None:
            # Subsequent capture/develop default to this WB unless overridden.
            self._white_balance = wb

        self.logger.info(
            f"Calibrated CCM (delta-E mean {report['before']['mean']:.1f} -> "
            f"{report['after']['mean']:.1f}, exposure {exposure_stops:+.2f} stops, "
            f"neutral 6.5 as-shot {neutral_brightness['neutral_6_5']['measured']:.0f}"
            f"/{neutral_brightness['neutral_6_5']['reference']:.0f})"
            + (f"; white balance [{', '.join(f'{v:.3f}' for v in wb)}]" if wb else "")
            + "; copy `ccm`"
            + (" and `white_balance`" if wb else "")
            + " into the component config to persist"
        )
        result: Dict[str, ValueTypes] = {
            "ccm": corrector.ccm.tolist(),
            "white_balance": wb,
            "exposure_stops": exposure_stops,
            "neutral_brightness": neutral_brightness,
            "delta_e": report,
        }
        if wb_note:
            result["white_balance_note"] = wb_note
        return result

    async def _capture_corrected(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Studio capture: trigger a full-resolution still on the source camera,
        develop it through the 16-bit linear pipeline (white balance + CCM),
        and write rendered exports - leaving any RAW original untouched.

        ``opts`` (all optional):
          ``capture_options``  forwarded to the source's ``capture`` (e.g. {"af": true})
          ``white_balance``    "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops``   exposure compensation applied at the raw stage
          ``ccm``              3x3 nested list applied instead of the configured
                               matrix for this call only
          ``tone``             delivery look: "none" (colour-accurate) | "c1"
                               (matches Capture One) | "medium" | "bright"
                               (lighter lifts); luminance-only, hue preserved
          ``sharpen``          capture sharpening: "none" | "light" | "medium"
                               | "strong"
          ``demosaic``         RAW demosaic algorithm (DHT/AHD/AAHD/DCB/VNG/PPG)
          ``output_formats``   subset of tiff16/tiff8/jpeg/png16/png8; pass []
                               to skip exports (preview-only capture - develop
                               the RAW later with the ``develop`` command)
          ``output_dir``       where to write exports (default: next to the source file)
          ``defer``            true -> return as soon as the shutter has fired
                               (the rig is free to move); download/decode/preview
                               continue in the background and are fetched with
                               ``capture_result``. Requires a source camera with
                               a ``trigger`` DoCommand (the ptp model). Deferred
                               captures never write exports or a sidecar - run
                               ``develop`` on the RAW for those.

        Returns the written export paths, the sidecar path, and a small base64
        JPEG preview (not the full-res image - that stays on disk). With
        ``defer`` it instead returns {"capture_id", "status": "pending",
        "camera_path"} immediately after the shutter fires.
        """
        if opts.get("defer"):
            return await self._capture_deferred(opts, timeout)

        capture_opts = opts.get("capture_options", {"af": True})
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        corrector = self._corrector_for(opts.get("ccm"))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)
        # When nothing is being exported, the decode only feeds the preview -
        # a half-resolution demosaic is ~4x faster and indistinguishable there.
        preview_only = not formats

        start = time.perf_counter()
        source_resp = await self.camera.do_command({"capture": capture_opts}, timeout=timeout)
        capture = source_resp.get("capture", source_resp)
        self.logger.debug(
            f"[timing] source camera capture (incl. download): "
            f"{time.perf_counter() - start:.2f}s"
        )

        t_decode = time.perf_counter()
        # The decode and develop/export steps are seconds of pure CPU; run them
        # in a worker thread so the event loop keeps serving other requests.
        linear, source_path = await asyncio.to_thread(
            self._linear_from_capture_response,
            capture, white_balance, exposure_stops, preview_only,
        )
        self.logger.debug(
            f"[timing] decode to linear RGB (incl. white balance): "
            f"{time.perf_counter() - t_decode:.2f}s"
        )
        result = await asyncio.to_thread(
            self._develop_one,
            linear, source_path, white_balance, exposure_stops, formats,
            out_dir_override, True, tone, sharpen,
            corrector=corrector,
        )
        self.logger.debug(
            f"[timing] capture pipeline total: {time.perf_counter() - start:.2f}s"
        )
        return result

    async def _capture_deferred(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Fire the shutter and return as soon as the exposure is done, so the
        caller can move the rig while the slow parts (USB download, demosaic,
        preview encode) run in a background task. The source's lock serializes
        camera access, so a background download naturally queues ahead of the
        next pose's trigger.
        """
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        # Resolve (and validate) the per-call `ccm` override before firing the
        # shutter, so a malformed matrix never wastes an exposure.
        corrector = self._corrector_for(opts.get("ccm"))
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)

        start = time.perf_counter()
        try:
            resp = await self.camera.do_command(
                {"trigger": opts.get("capture_options", {})}, timeout=timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"deferred capture needs a source camera with a `trigger` "
                f"DoCommand (the ptp model); triggering failed: {exc}"
            ) from exc
        trig = resp.get("trigger") or {}
        camera_path = trig.get("path")
        if not camera_path:
            raise ValueError("source camera `trigger` returned no `path`")
        self.logger.debug(
            f"[timing] deferred capture trigger (shutter + settle): "
            f"{time.perf_counter() - start:.2f}s"
        )

        self._capture_seq += 1
        stem = os.path.splitext(os.path.basename(str(camera_path)))[0]
        capture_id = f"{self._capture_seq}-{stem}"
        self._pending_captures[capture_id] = asyncio.create_task(
            self._finish_deferred_capture(
                capture_id, str(camera_path), white_balance, exposure_stops,
                tone, sharpen, corrector,
            )
        )
        # Drop completed-and-collected stragglers if a caller never fetched
        # them, so an unattended sequence can't grow the table without bound.
        if len(self._pending_captures) > 64:
            for key in [
                k for k, t in self._pending_captures.items() if t.done()
            ][:-64]:
                self._pending_captures.pop(key, None)

        return {
            "capture_id": capture_id,
            "status": "pending",
            "camera_path": str(camera_path),
        }

    async def _finish_deferred_capture(
        self,
        capture_id: str,
        camera_path: str,
        white_balance: Any,
        exposure_stops: float,
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
        corrector: Optional[ColorCorrector] = None,
    ) -> Dict[str, ValueTypes]:
        """Background half of a deferred capture: download the still from the
        camera, decode at half size, apply the CCM, and build the preview. No
        exports or sidecar - the RAW on disk is the handoff to ``develop``.
        ``corrector`` is the per-call CCM override the caller resolved before
        the shutter fired; None means the configured matrix."""
        if corrector is None:
            corrector = self.corrector
        start = time.perf_counter()
        resp = await self.camera.do_command({"download": {"path": camera_path}})
        meta = resp.get("download") or {}
        saved = meta.get("saved_to")
        if not saved:
            raise ValueError(
                f"source camera did not save {camera_path!r} to disk; configure "
                f"its `download_dir` so deferred captures can be developed later"
            )
        linear = await asyncio.to_thread(
            load_linear_rgb, str(saved),
            white_balance=white_balance, exposure_stops=exposure_stops,
            half_size=True, demosaic=self._demosaic,
        )
        corrected = await asyncio.to_thread(corrector.apply_to_linear, linear)
        preview = await asyncio.to_thread(
            linear_to_jpeg_base64, corrected, tone=tone, sharpen=sharpen
        )
        self.logger.debug(
            f"[timing] deferred capture {capture_id} background "
            f"(download + decode + preview): {time.perf_counter() - start:.2f}s"
        )
        return {
            "capture_id": capture_id,
            "status": "done",
            "source_path": str(saved),
            "image_base64": preview,
            "mime_type": CameraMimeType.JPEG.value,
            "ccm_applied": not corrector.is_identity,
            "color_space": "sRGB",
        }

    async def _capture_result(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Fetch the result of a deferred capture.

        ``opts``:
          ``id``        the capture_id returned by ``capture`` with ``defer`` (required)
          ``wait_sec``  how long to wait for the background work (default 60;
                        0 polls). Returns {"status": "pending"} on timeout -
                        call again to keep waiting.
        """
        capture_id = opts.get("id") or opts.get("capture_id")
        if not capture_id:
            raise ValueError(
                "`capture_result` needs the `id` returned by `capture` with `defer`"
            )
        capture_id = str(capture_id)
        task = self._pending_captures.get(capture_id)
        if task is None:
            raise ValueError(
                f"unknown capture id {capture_id!r}; it may have already been "
                f"collected, or the module restarted since the capture"
            )
        wait_sec = float(opts.get("wait_sec", 60.0))
        try:
            # shield() so a timeout here doesn't cancel the background work.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_sec)
        except asyncio.TimeoutError:
            return {"capture_id": capture_id, "status": "pending"}
        except Exception as exc:
            self._pending_captures.pop(capture_id, None)
            raise RuntimeError(f"deferred capture {capture_id} failed: {exc}") from exc
        self._pending_captures.pop(capture_id, None)
        return result

    async def _develop(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Develop existing image file(s) already on disk - no camera trigger.
        Point this at a RAW (CR3/NEF/ARW/...) or any JPEG/PNG/TIFF and it runs
        the same 16-bit linear pipeline (white balance + CCM) and writes the
        rendered exports + sidecar, leaving the original untouched.

        ``opts``:
          ``path``           a single file path (returns that file's result), OR
          ``paths``          a list of file paths (returns {"developed": [...]})
          ``white_balance``  "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops`` exposure compensation applied at the raw stage
          ``ccm``            3x3 nested list applied instead of the configured
                             matrix for this call only
          ``tone``           delivery look: "none" | "c1" | "medium" | "bright"
          ``sharpen``        capture sharpening: "none"|"light"|"medium"|"strong"
          ``demosaic``       RAW demosaic algorithm (DHT/AHD/AAHD/DCB/VNG/PPG)
          ``output_formats`` subset of tiff16/tiff8/jpeg/png16/png8
          ``output_dir``     where to write exports (default: next to each file)
          ``crop``           {"x","y","w","h"} normalized to the decoded frame
                             (0-1, top-left origin) - exports only this region.
                             Normalized so a rect drawn on a preview applies
                             unchanged to the full-res decode. Omit for the full
                             frame. The source file is never modified.
          ``output_stem``    override the export/sidecar filename stem (default:
                             the source file's). Lets the same source be
                             developed more than once - e.g. an uncropped master
                             plus a cropped variant - without the second pass
                             overwriting the first's exports.
        """
        raw_paths = opts.get("paths")
        single = raw_paths is None
        if single:
            one = opts.get("path")
            if not one:
                raise ValueError(
                    "`develop` needs a `path` (string) or `paths` (list of strings)"
                )
            raw_paths = [one]
        paths = [str(p) for p in raw_paths]

        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        corrector = self._corrector_for(opts.get("ccm"))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)
        crop = self._parse_crop(opts.get("crop"))
        output_stem = opts.get("output_stem")
        output_stem = str(output_stem) if output_stem else None
        if output_stem and len(paths) > 1:
            raise ValueError(
                "`output_stem` names a single set of exports, so it can't be "
                f"used with {len(paths)} `paths` (they would overwrite each "
                "other); develop one path per command instead"
            )

        results: List[Mapping[str, ValueTypes]] = []
        for path in paths:
            t_file = time.perf_counter()
            # Decode + export are seconds of pure CPU per file; keep them off
            # the event loop so other requests stay responsive mid-batch.
            linear = await asyncio.to_thread(
                load_linear_rgb,
                path, white_balance=white_balance, exposure_stops=exposure_stops,
                demosaic=self._demosaic,
            )
            results.append(
                await asyncio.to_thread(
                    self._develop_one,
                    linear, path, white_balance, exposure_stops, formats,
                    out_dir_override,
                    # Skip the per-file base64 preview in batch mode to keep the
                    # response small; a single develop still returns its preview.
                    include_preview=single,
                    tone=tone,
                    sharpen=sharpen,
                    crop=crop,
                    output_stem=output_stem,
                    corrector=corrector,
                )
            )
            self.logger.debug(
                f"[timing] develop {os.path.basename(path)} total: "
                f"{time.perf_counter() - t_file:.2f}s"
            )

        if single:
            return results[0]
        return {"developed": results, "count": len(results)}

    async def _preview(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Render a display-only JPEG of a file on disk, optionally cropped. Writes
        nothing — no exports, no sidecar — so it's safe to call repeatedly while an
        operator adjusts a crop.

        The point is resolution: a crop taken from an existing 1024px preview only
        has 1024 x (crop fraction) pixels left, which looks terrible for a tight
        crop. Here the crop comes off a fresh decode of the RAW instead, so the
        returned JPEG fills ``max_dim`` no matter how small the region is.

        ``opts``:
          ``path``           file to render (required)
          ``crop``           {"x","y","w","h"} normalized; omit for the full frame
          ``max_dim``        longest edge of the returned JPEG (default 1024)
          ``ccm``            3x3 nested list applied instead of the configured
                             matrix for this call only
          ``white_balance`` / ``exposure_stops`` / ``tone`` / ``sharpen``
                             as for ``develop``; default to the configured values
                             so the preview matches what a develop would produce
        """
        path = opts.get("path")
        if not path:
            raise ValueError("`preview` needs a `path`")
        path = str(path)

        crop = self._parse_crop(opts.get("crop"))
        max_dim = int(opts.get("max_dim", 1024))
        if max_dim <= 0:
            raise ValueError(f"`preview` needs a positive `max_dim`, got {max_dim}")
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        corrector = self._corrector_for(opts.get("ccm"))
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)

        start = time.perf_counter()
        half_size = await asyncio.to_thread(self._half_size_suffices, path, crop, max_dim)
        linear = await asyncio.to_thread(
            load_linear_rgb,
            path, white_balance=white_balance, exposure_stops=exposure_stops,
            half_size=half_size, demosaic=self._demosaic,
        )
        if crop is not None:
            linear = crop_linear(linear, *crop)
        corrected = await asyncio.to_thread(corrector.apply_to_linear, linear)
        image_base64 = await asyncio.to_thread(
            linear_to_jpeg_base64, corrected, max_dim, self._jpeg_quality,
            tone, sharpen,
        )
        self.logger.debug(
            f"[timing] preview {os.path.basename(path)} "
            f"({'half' if half_size else 'full'} decode, "
            f"{corrected.shape[1]}x{corrected.shape[0]} before encode): "
            f"{time.perf_counter() - start:.2f}s"
        )

        result: Dict[str, ValueTypes] = {
            "source_path": path,
            "image_base64": image_base64,
            "mime_type": CameraMimeType.JPEG.value,
            "half_size": half_size,
            "ccm_applied": not corrector.is_identity,
            "color_space": "sRGB",
        }
        if crop is not None:
            result["crop"] = list(crop)
        return result

    def _half_size_suffices(
        self,
        path: str,
        crop: Optional[Tuple[float, float, float, float]],
        max_dim: int,
    ) -> bool:
        """
        Whether a half-resolution demosaic still leaves ``max_dim`` pixels on the
        (cropped or full) frame's long edge. Half size is ~4x faster, so it's the
        default — but a tight crop, or a large ``max_dim`` against a full frame,
        needs the full decode or the preview is soft, which is the whole problem
        this path exists to avoid.

        Errs toward the full decode: an unreadable header, or a RAW whose reported
        orientation is ambiguous, costs time rather than quality.
        """
        try:
            width, height = image_dimensions(path)
        except Exception as exc:  # noqa: BLE001 - a probe failure must not fail the preview
            self.logger.debug(f"could not probe {path} for preview sizing: {exc}")
            return False
        if not width or not height:
            return False
        cw, ch = (crop[2], crop[3]) if crop is not None else (1.0, 1.0)
        # A 90-degree EXIF rotation would pair the crop fractions with the other
        # axis; take the smaller of the two readings so an ambiguous orientation
        # picks the higher-quality decode.
        long_edge = min(
            max(cw * width, ch * height),
            max(cw * height, ch * width),
        )
        return long_edge / 2 >= max_dim

    def _develop_one(
        self,
        linear: np.ndarray,
        source_path: Optional[str],
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        out_dir_override: Optional[str],
        include_preview: bool = True,
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
        crop: Optional[Tuple[float, float, float, float]] = None,
        output_stem: Optional[str] = None,
        corrector: Optional[ColorCorrector] = None,
    ) -> Dict[str, ValueTypes]:
        """
        Shared core for ``capture`` and ``develop``: apply the CCM in linear
        light, write the rendered exports (non-destructively) and a sidecar, and
        return the result. ``linear`` is linear-light float RGB; ``source_path``
        is the originating file (or None for an inline base64 capture).

        ``crop`` is a normalized (x, y, w, h) rect applied before the color math;
        ``output_stem`` overrides the export/sidecar filename stem so a cropped
        variant can sit next to the uncropped master without clobbering it;
        ``corrector`` is the per-call CCM override (None means the configured
        matrix).
        """
        if corrector is None:
            corrector = self.corrector
        if crop is not None:
            linear = crop_linear(linear, *crop)

        t_ccm = time.perf_counter()
        corrected = corrector.apply_to_linear(linear)
        self.logger.debug(
            f"[timing] apply color correction (CCM): {time.perf_counter() - t_ccm:.2f}s"
        )

        # Exports land alongside the source file unless an output_dir is set.
        out_dir = out_dir_override or (
            os.path.dirname(source_path) if source_path else None
        )
        stem = output_stem or (
            os.path.splitext(os.path.basename(source_path))[0]
            if source_path else "capture"
        )
        # A RAW source (.cr3/.nef/...) never collides with our .tif/.jpg/.png
        # exports, so its name is preserved. But if the source is itself a
        # JPEG/PNG/TIFF, a same-name export would overwrite the original - so
        # suffix the exports to keep the pipeline non-destructive. An explicit
        # `output_stem` is the caller's business: they already chose a distinct
        # name, so don't second-guess it.
        if source_path and not output_stem and not is_raw(source_path):
            stem = stem + "_corrected"
        exports: Dict[str, str] = {}
        if out_dir:
            t_export = time.perf_counter()
            exports = export_renditions(
                corrected, out_dir, stem, formats,
                quality=self._jpeg_quality, tone=tone, sharpen=sharpen,
            )
            self.logger.debug(
                f"[timing] export {len(exports)} format(s): "
                f"{time.perf_counter() - t_export:.2f}s"
            )
            self.logger.info(f"exported {list(exports)} for {stem} to {out_dir}")
        else:
            self.logger.warning(
                "no `output_dir` configured and source has no path; "
                "returning a preview only (nothing written to disk)"
            )

        sidecar = None
        if self._write_sidecar and source_path:
            sidecar = self._write_sidecar_file(
                source_path, white_balance, exposure_stops, formats, exports,
                tone, sharpen, crop, corrector=corrector,
                # With a stem override the sidecar has to follow the exports, or
                # a cropped variant would overwrite the master's record.
                dest=(
                    os.path.join(out_dir, stem + ".json")
                    if output_stem and out_dir else None
                ),
            )

        result: Dict[str, ValueTypes] = {
            "source_path": source_path,
            "exports": exports,
            "sidecar": sidecar,
            "ccm_applied": not corrector.is_identity,
            "color_space": "sRGB",
        }
        if crop is not None:
            result["crop"] = list(crop)
        if include_preview:
            # Report what the exports actually cover, so a caller that cropped
            # can confirm the rect landed where it drew it.
            result["width"] = int(corrected.shape[1])
            result["height"] = int(corrected.shape[0])
        if include_preview:
            result["image_base64"] = linear_to_jpeg_base64(
                corrected, tone=tone, sharpen=sharpen
            )
            result["mime_type"] = CameraMimeType.JPEG.value
        return result

    def _write_sidecar_file(
        self,
        source_path: str,
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        exports: Mapping[str, str],
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
        crop: Optional[Tuple[float, float, float, float]] = None,
        dest: Optional[str] = None,
        corrector: Optional[ColorCorrector] = None,
    ) -> str:
        """
        Write a ``<stem>.json`` sidecar next to the (untouched) source file
        recording exactly how it was developed - the non-destructive record that
        lets a capture be reproduced or re-exported later. ``dest`` overrides
        where it lands, for variants that don't share the source's stem;
        ``corrector`` is the per-call CCM override actually applied (None means
        the configured matrix), so the record always names the matrix used.
        """
        if corrector is None:
            corrector = self.corrector
        sidecar_path = dest or (os.path.splitext(source_path)[0] + ".json")
        record = {
            "source": os.path.basename(source_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "white_balance": white_balance,
            "exposure_stops": exposure_stops,
            "crop": (
                dict(zip(("x", "y", "w", "h"), crop)) if crop is not None else None
            ),
            "tone": tone or "none",
            "sharpen": sharpen or "none",
            "demosaic": self._demosaic,
            "ccm": corrector.ccm.tolist(),
            "ccm_source": "config" if corrector is self.corrector else "override",
            "ccm_applied": not corrector.is_identity,
            "color_space": "sRGB",
            "output_formats": list(formats),
            "exports": {k: os.path.basename(v) for k, v in exports.items()},
        }
        with open(sidecar_path, "w") as f:
            json.dump(record, f, indent=2)
        return sidecar_path

    @staticmethod
    def _parse_crop(
        raw: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Validate a ``crop`` option into a normalized (x, y, w, h) tuple, or
        ``None`` for "no crop". Accepts the {"x","y","w","h"} mapping the webapp
        sends or a 4-element sequence.

        A crop that covers the whole frame is normalized away to ``None`` so the
        no-op case skips the crop path entirely. Anything malformed raises rather
        than silently developing an unexpected region - a wrong crop is invisible
        in the response but wrong on disk.
        """
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            missing = [k for k in ("x", "y", "w", "h") if k not in raw]
            if missing:
                raise ValueError(
                    f"`crop` is missing {missing}; it needs all of x, y, w, h "
                    "as fractions of the frame (0-1, top-left origin)"
                )
            values = [raw["x"], raw["y"], raw["w"], raw["h"]]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 4:
                raise ValueError(
                    f"`crop` as a list needs exactly 4 values [x, y, w, h], got {len(raw)}"
                )
            values = list(raw)
        else:
            raise ValueError(
                f"`crop` must be an object with x/y/w/h or a 4-element list, "
                f"got {type(raw).__name__}"
            )

        try:
            x, y, w, h = (float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`crop` values must be numbers: {exc}") from exc

        if w <= 0 or h <= 0:
            raise ValueError(
                f"`crop` width and height must be positive, got w={w}, h={h}"
            )
        # Tolerate float dust from the browser's normalized rect, but reject a
        # rect that genuinely runs off the frame.
        eps = 1e-6
        if x < -eps or y < -eps or x + w > 1 + eps or y + h > 1 + eps:
            raise ValueError(
                f"`crop` must lie within the frame as fractions of 0-1, got "
                f"x={x}, y={y}, w={w}, h={h}"
            )
        x, y = max(0.0, x), max(0.0, y)
        w, h = min(w, 1.0 - x), min(h, 1.0 - y)
        if (x, y, w, h) == (0.0, 0.0, 1.0, 1.0):
            return None
        return (x, y, w, h)

    def _close_data_client(self) -> None:
        """Close the cached cloud channel, if one was ever dialed. Idempotent."""
        # getattr: on the first reconfigure the attribute doesn't exist yet.
        client = getattr(self, "_data_client", None)
        self._data_client = None
        if client is None:
            return
        try:
            client._channel.close()
        except Exception as exc:
            self.logger.debug(f"Closing app.viam.com channel failed: {exc}")

    async def _get_data_client(self) -> DataClient:
        """
        Lazily build (and cache) a cloud ``DataClient`` from the API key Viam
        injects into the module process (``VIAM_API_KEY`` / ``VIAM_API_KEY_ID``).

        We dial the app channel directly rather than via
        ``ViamClient.create_from_env_vars``: in viam-sdk 0.77.0 that path
        authenticates the channel inside ``_dial_app`` and then authenticates a
        *second* time, which the server rejects with "already authenticated;
        cannot re-authenticate". ``_dial_app`` alone performs the single, correct
        auth, and the resulting channel carries the bearer token we hand to the
        ``DataClient``.
        """
        if self._data_client is not None:
            return self._data_client
        api_key = os.environ.get("VIAM_API_KEY")
        api_key_id = os.environ.get("VIAM_API_KEY_ID")
        if not api_key or not api_key_id:
            raise ValueError(
                "`upload` could not authenticate: VIAM_API_KEY / VIAM_API_KEY_ID "
                "were not present in the module environment. This requires "
                "running on a cloud-connected machine."
            )
        dial_options = DialOptions(
            credentials=Credentials(type="api-key", payload=api_key),
            auth_entity=api_key_id,
        )
        self.logger.info(
            "`upload` authenticating to app.viam.com "
            f"(dial timeout {self._upload_dial_timeout_s:.0f}s)"
        )
        t_dial = time.perf_counter()
        dial_task = asyncio.ensure_future(_dial_app("app.viam.com", dial_options))
        try:
            channel = await asyncio.wait_for(
                dial_task,
                timeout=self._upload_dial_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            # wait_for cancels the dial, but the dial can complete in the race
            # window before the cancel lands; close that channel rather than
            # leak it (grpclib logs "Unclosed connection" to stderr on GC).
            if dial_task.done() and not dial_task.cancelled() and dial_task.exception() is None:
                dial_task.result().close()
            raise TimeoutError(
                f"`upload` timed out after {self._upload_dial_timeout_s:.0f}s "
                "dialing app.viam.com to authenticate — the machine may be "
                "offline or unable to reach the cloud."
            ) from exc
        self.logger.info(
            f"`upload` authenticated in {time.perf_counter() - t_dial:.2f}s"
        )
        metadata = getattr(channel, "_metadata", {})
        self._data_client = DataClient(channel, metadata)
        return self._data_client

    async def _upload(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Upload files already on disk to Viam, tagged for later retrieval.

        The full-resolution captures (CR3 + the rendered TIFF/JPEG exports + the
        JSON sidecar) live on this machine's filesystem; this ships them straight
        to the cloud so they never have to travel back through the browser. The
        webapp passes every path that shares a capture's filename stem, so a
        single selected shot uploads as a complete set under one SKU tag.

        ``opts``:
          ``paths``               list of file paths on disk to upload (required)
          ``tags``                tags to attach to every uploaded file (e.g. SKU)
          ``name``                operator-chosen file name stem (no dir, no
                                  extension) that replaces the camera capture
                                  stem on every file in this set; each file keeps
                                  its full post-stem suffix (``_16.png``,
                                  ``.cr3``, ``.json``, ...). When absent, the
                                  camera's on-disk basename is used.
          ``part_id``             override the configured / env machine part id
          ``component_name``      camera name to associate the data with (optional)
          ``delete_after_upload`` override the config attribute: remove each
                                  local file once its upload succeeds (failed
                                  uploads keep their files for retry)
        """
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`upload` needs a non-empty `paths` list")
        paths = [str(p) for p in raw_paths]
        tags = [str(t) for t in (opts.get("tags") or [])]
        delete_after = bool(opts.get("delete_after_upload", self._delete_after_upload))

        name = opts.get("name")
        name = str(name) if name else None      # falsy/empty -> keep current behavior
        # All paths in one upload call share the capture stem (e.g. "IMG_0042").
        # commonprefix over splitext-stems lands cleanly on the bare stem even
        # when _16 variants and a .json sidecar are mixed in - a basename-level
        # commonprefix would land mid-token on extension-only sets.
        capture_stem = (
            os.path.commonprefix([os.path.splitext(os.path.basename(p))[0] for p in paths])
            if name else None
        )

        part_id = opts.get("part_id") or self._part_id
        if not part_id:
            raise ValueError(
                "no part id available for upload; set `part_id` in config or pass "
                "it in the command (VIAM_MACHINE_PART_ID was not set)"
            )
        component_name = opts.get("component_name") or self.name

        data_client = await self._get_data_client()

        uploaded: List[str] = []
        failed: List[Dict[str, str]] = []
        for i, path in enumerate(paths):
            try:
                size = os.path.getsize(path)
                self.logger.info(
                    f"uploading {os.path.basename(path)} ({size / 1e6:.1f} MB) "
                    f"[{i + 1}/{len(paths)}] (timeout {self._upload_file_timeout_s:.0f}s)"
                )
                t_upload = time.perf_counter()
                file_name = (
                    self._renamed_basename(path, name, capture_stem)
                    if name else None
                )
                await asyncio.wait_for(
                    self._file_upload_chunked(
                        data_client,
                        path,
                        part_id=str(part_id),
                        component_name=str(component_name),
                        tags=tags or None,
                        file_name=file_name,
                    ),
                    timeout=self._upload_file_timeout_s,
                )
                self.logger.info(
                    f"[timing] uploaded {os.path.basename(path)} "
                    f"({size / 1e6:.1f} MB) in {time.perf_counter() - t_upload:.2f}s"
                )
                uploaded.append(path)
            except asyncio.TimeoutError:
                # File kept for retry — a per-file deadline means one stalled
                # transfer can't wedge the whole submit.
                msg = (
                    f"upload timed out after {self._upload_file_timeout_s:.0f}s"
                )
                self.logger.error(f"failed to upload {path}: {msg}")
                failed.append({"path": path, "error": msg})
            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                self.logger.error(f"failed to upload {path}: {exc}")
                failed.append({"path": path, "error": str(exc)})

        deleted: List[str] = []
        if delete_after:
            for path in uploaded:
                try:
                    os.remove(path)
                    deleted.append(path)
                except OSError as exc:
                    # The upload itself succeeded - don't let a cleanup
                    # hiccup mark the file as failed.
                    self.logger.warning(f"uploaded but could not delete {path}: {exc}")

        self.logger.info(
            f"uploaded {len(uploaded)}/{len(paths)} file(s)"
            + (f" with tags {tags}" if tags else "")
            + (f", deleted {len(deleted)} local cop(ies)" if delete_after else "")
        )
        return {
            "uploaded": uploaded,
            "count": len(uploaded),
            "failed": failed,
            "deleted": deleted,
        }

    def _delete_local(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Delete files from this machine's disk - the cleanup half of the studio
        flow. Captures the operator skipped (never developed or uploaded) have
        no other exit path and would otherwise accumulate in the download dir
        forever; the webapp sends their paths here at submit time.

        Guarded: only files inside the configured ``output_dir`` may be
        deleted, so a caller can't reach arbitrary paths on the host. Requires
        ``output_dir`` to be set (without it there is no boundary to enforce).

        ``opts``:
          ``paths``  list of file paths to delete (required)

        Already-missing files are reported in ``missing`` rather than treated
        as errors, so retried cleanups stay idempotent.
        """
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`delete` needs a non-empty `paths` list")
        if not self._output_dir:
            raise ValueError(
                "`delete` requires `output_dir` to be configured: it only "
                "removes files inside that directory"
            )
        root = os.path.realpath(self._output_dir)

        deleted: List[str] = []
        missing: List[str] = []
        failed: List[Dict[str, str]] = []
        for raw in raw_paths:
            path = str(raw)
            # realpath also resolves symlinks, so a link inside output_dir
            # pointing elsewhere can't smuggle a delete outside the boundary.
            real = os.path.realpath(path)
            if os.path.commonpath([real, root]) != root:
                self.logger.warning(f"refusing to delete outside output_dir: {path}")
                failed.append(
                    {"path": path, "error": f"outside output_dir {self._output_dir}"}
                )
                continue
            try:
                os.remove(real)
                deleted.append(path)
            except FileNotFoundError:
                missing.append(path)
            except OSError as exc:
                self.logger.error(f"failed to delete {path}: {exc}")
                failed.append({"path": path, "error": str(exc)})

        self.logger.info(
            f"deleted {len(deleted)}/{len(raw_paths)} local file(s)"
            + (f", {len(missing)} already gone" if missing else "")
        )
        return {
            "deleted": deleted,
            "count": len(deleted),
            "missing": missing,
            "failed": failed,
        }

    @staticmethod
    async def _file_upload_chunked(
        data_client: DataClient,
        path: str,
        *,
        part_id: str,
        component_name: str,
        tags: Optional[List[str]],
        file_name: Optional[str] = None,
    ) -> str:
        """
        Stream a file to Viam over the client-streaming FileUpload RPC in
        UPLOAD_CHUNK_BYTES pieces. ``DataClient.file_upload`` sends the whole
        file as one gRPC message, which app.viam.com rejects past 32 MiB -
        silently dropping every CR3 and TIFF from a submit. Chunking also
        keeps a 250 MB TIFF from being held in memory all at once.
        """
        metadata = UploadMetadata(
            part_id=part_id,
            component_type="rdk:component:camera",
            component_name=component_name,
            type=DataType.DATA_TYPE_FILE,
            file_name=file_name or os.path.basename(path),
            file_extension=os.path.splitext(path)[1],  # e.g. ".cr3", ".tif"
            tags=tags,
        )
        async with data_client._data_sync_client.FileUpload.open(  # noqa: SLF001
            metadata=data_client._metadata  # noqa: SLF001
        ) as stream:
            await stream.send_message(FileUploadRequest(metadata=metadata))
            with open(path, "rb") as f:
                chunk = f.read(UPLOAD_CHUNK_BYTES)
                while True:
                    next_chunk = f.read(UPLOAD_CHUNK_BYTES)
                    # An empty file still needs one (empty) FileData message.
                    await stream.send_message(
                        FileUploadRequest(file_contents=FileData(data=chunk)),
                        end=not next_chunk,
                    )
                    if not next_chunk:
                        break
                    chunk = next_chunk
            response: Optional[FileUploadResponse] = await stream.recv_message()
            if not response:
                # Raises the appropriate gRPC error for the failed upload.
                await stream.recv_trailing_metadata()
                raise TypeError("FileUpload response cannot be empty")
            return response.binary_data_id

    # ------------------------------------------------------------------
    # Upload naming
    # ------------------------------------------------------------------

    @staticmethod
    def _renamed_basename(
        path: str, name: Optional[str], capture_stem: Optional[str]
    ) -> str:
        """
        Apply the operator-chosen upload ``name`` to one file of a capture set:
        the shared ``capture_stem`` prefix is swapped for ``name``, preserving
        the file's post-stem suffix (``_16.tif``, ``.json``, ...). With no
        ``name`` the on-disk basename is returned unchanged, so every file in
        one upload set ends up named consistently.
        """
        base = os.path.basename(path)
        if not name or capture_stem is None:
            return base
        return name + base[len(capture_stem):]

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return await self.camera.get_geometries(extra=extra, timeout=timeout)

    async def close(self):
        self._close_data_client()
