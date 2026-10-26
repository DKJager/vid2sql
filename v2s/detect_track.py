# build 7fd77e00de7f
"""Moving-object detection and tracking.

By default detection runs **locally** with OpenCV 5 (``v2s.core_cv``):
MOG2 background subtraction, contour extraction and nearest-centroid
tracking. A remote vision service is an optional alternative, selected only
when ``V2S_VISION_ENDPOINT`` is set (``V2S_VISION_MODE=local`` forces local).
``OPENCV_DNN_MODEL_DIR`` is only forwarded to a remote service as
configuration; this module never loads, vendors or downloads model weights.

Local detector options (used by the agent loop): ``stabilize`` (bool) cancels
global camera translation before background subtraction, and
``var_threshold`` (float) sets the MOG2 variance threshold.
``measure_camera_motion`` reports the median global camera translation of a
segment (OpenCV phase correlation) so a caller can plan those options.

Audit note: ``process_video_segment`` accepts a ``detector`` argument with a
working default so callers (including ``v2s/agent.py``) match this signature
whether or not they pass one.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

#: Default detector label forwarded to the external service. The service owns
#: the actual engine choice; this repo stores no engine or provider detail.
DEFAULT_DETECTOR = "service_default"

#: Label recorded when detection runs locally with OpenCV 5.
LOCAL_DETECTOR = "opencv5_mog2_centroid"

#: Valid characters and format for model-dir path segments.
_PATH_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9._\-][a-zA-Z0-9 ._\-]*$")
_MAX_PATH_LEN = 256

#: Plausible URI schemes for video_uri validation (issue 193).
_ALLOWED_URI_SCHEMES = frozenset({"http", "https", "s3", "file", "gs", "stub"})

#: Allowed schemes for V2S_VISION_ENDPOINT (must be HTTP or HTTPS).
_VALID_ENDPOINT_SCHEMES = frozenset({"http", "https"})


class DetectConfigError(RuntimeError):
    """Raised when a required environment variable is missing at call time."""


class DetectServiceError(RuntimeError):
    """Raised when the vision service is unreachable or returns bad data."""


def _use_remote() -> bool:
    if os.environ.get("V2S_VISION_MODE", "").strip().lower() == "local":
        return False
    return bool(os.environ.get("V2S_VISION_ENDPOINT", "").strip())


def _vision_endpoint() -> str:
    value = os.environ.get("V2S_VISION_ENDPOINT", "").strip()
    if not value:
        raise DetectConfigError(
            "V2S_VISION_ENDPOINT is not set (remote vision service is optional; "
            "unset means detection runs locally with OpenCV 5)."
        )
    # Issue 202: lightweight URL scheme check so that
    # accidentally omitting a scheme (e.g. "localhost:8080") fails early
    # with a clear message rather than later inside urlopen.
    if "://" not in value:
        raise DetectConfigError(
            f"V2S_VISION_ENDPOINT must be a full URL with a scheme (http or https); "
            f"got {value!r}"
        )
    scheme = value.split("://", 1)[0].lower()
    if scheme not in _VALID_ENDPOINT_SCHEMES:
        raise DetectConfigError(
            f"V2S_VISION_ENDPOINT scheme {scheme!r} is not allowed; "
            f"expected one of {sorted(_VALID_ENDPOINT_SCHEMES)}"
        )
    return value.rstrip("/")


def _model_dir() -> str:
    """On-box model directory used by the external service side.

    Read lazily so importing this module never raises when it is unset; when
    unset, the service falls back to its own configured default.

    Returns:
        Validated directory path, or empty string when unset.

    Raises:
        DetectConfigError: if the value is set but fails format/safety checks.
    """
    raw_value = os.environ.get("OPENCV_DNN_MODEL_DIR", "").strip()
    if not raw_value:
        return ""

    if len(raw_value) > _MAX_PATH_LEN:
        raise DetectConfigError(
            f"OPENCV_DNN_MODEL_DIR must be at most {_MAX_PATH_LEN} characters long"
        )

    # Split into components, allowing leading and trailing slashes.
    # We accept absolute POSIX paths, relative paths, and simple names.
    # Empty components (e.g. double slashes) are rejected.
    components = [c for c in raw_value.split("/") if c != ""]
    if not components:
        raise DetectConfigError(
            "OPENCV_DNN_MODEL_DIR must contain at least one path component"
        )

    for comp in components:
        if comp in (".", ".."):
            raise DetectConfigError(
                f"OPENCV_DNN_MODEL_DIR contains forbidden component '{comp}'; "
                "parent traversal and self-references are not permitted"
            )
        if not _PATH_SEGMENT_RE.match(comp):
            raise DetectConfigError(
                f"OPENCV_DNN_MODEL_DIR component '{comp}' contains invalid characters; "
                "only alphanumeric, space, period, underscore, and hyphen are allowed"
            )

    return raw_value


def _vision_timeout() -> float:
    """Read the V2S_VISION_TIMEOUT env var or return 180.0 (issue 157)."""
    raw = os.environ.get("V2S_VISION_TIMEOUT", "").strip()
    if not raw:
        return 180.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise DetectConfigError(
            f"V2S_VISION_TIMEOUT must be a number; got {raw!r}"
        ) from exc
    if value <= 0:
        raise DetectConfigError(
            f"V2S_VISION_TIMEOUT must be positive; got {value}"
        )
    return value


def _validate_video_uri(video_uri: str) -> None:
    """Lightweight validation so malformed URIs fail fast (issue 193)."""
    if not video_uri or not video_uri.strip():
        raise DetectConfigError("video_uri must be a non-empty string")
    # A bare filesystem path is a local file.
    if "://" not in video_uri:
        return
    scheme = video_uri.split("://", 1)[0].lower()
    if scheme not in _ALLOWED_URI_SCHEMES:
        raise DetectConfigError(
            f"video_uri scheme {scheme!r} is not allowed; expected one of {sorted(_ALLOWED_URI_SCHEMES)}"
        )


def _default_transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{_vision_endpoint()}{path}"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("V2S_TOKEN", "").strip()
    # Issue 208: only add the Authorization header when the token is non-empty
    # after stripping, so whitespace-only tokens are never sent.
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    # Use the configurable timeout (issue 157) instead of a hard-coded value.
    timeout = payload.get("options", {}).get("timeout", _vision_timeout())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Issue 153: catch HTTPError separately to include status and body detail
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            # limit body length to avoid flooding logs with large responses
            if len(detail) > 500:
                detail = detail[:500] + "..."
        except Exception:
            detail = "(body could not be read)"
        raise DetectServiceError(
            f"vision service returned HTTP {exc.code} {exc.reason}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Issue 159: catch TimeoutError and OSError in addition to URLError.
        raise DetectServiceError(f"vision service call failed: {exc}") from exc
    try:
        decoded = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise DetectServiceError(
            f"vision service returned non-JSON body: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise DetectServiceError("vision service returned a non-object JSON body")
    return decoded


def _validate_bounding_box(box: Any) -> None:
    """Raise DetectServiceError if 'box' is not a plausible detection box."""
    if box is None:
        # A box is optional: no table stores it. Validate only when present.
        return
    if isinstance(box, list):
        if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
            raise DetectServiceError(
                "vision service returned a list bounding_box that is not a 4-number array"
            )
        return
    if isinstance(box, dict):
        # At minimum require keys x, y, w, h (or similar numeric keys).
        for key in ("x", "y", "w", "h"):
            val = box.get(key)
            if val is None:
                raise DetectServiceError(
                    f"vision service bounding_box dict is missing required key '{key}'"
                )
            if not isinstance(val, (int, float)):
                raise DetectServiceError(
                    f"vision service bounding_box dict key '{key}' is not numeric"
                )
        return
    raise DetectServiceError(
        f"vision service returned unsupported bounding_box type {type(box).__name__}"
    )


def _coerce_float(value: Any, label: str) -> float:
    """Safely convert *value* to float, raising a clear error on failure."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DetectServiceError(
            f"vision service returned non-numeric {label} value: {value!r}"
        ) from exc


def _normalise_detections(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("detections", [])
    if not isinstance(raw, list):
        raise DetectServiceError("vision service 'detections' is not a list")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise DetectServiceError("detection entry is not an object")

        # Issue 200 / Issue 217: explicitly check confidence type and clamp to [0, 1].
        raw_confidence = item.get("confidence", 0.0)
        confidence = _coerce_float(raw_confidence, "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise DetectServiceError(
                f"vision service returned confidence {confidence} outside allowed range [0.0, 1.0]"
            )
        # Clamp to [0.0, 1.0] instead of raising for sub-precision outliers.
        confidence = max(0.0, min(1.0, confidence))

        # Issue 201: reject negative timestamp_s / Issue 213: safe float conversion.
        raw_timestamp = item.get("timestamp_s", 0.0)
        timestamp_s = _coerce_float(raw_timestamp, "timestamp_s")
        if timestamp_s < 0:
            raise DetectServiceError(
                f"vision service returned negative timestamp_s: {timestamp_s}"
            )

        # Issue 206: validate bounding_box before storing the row.
        bounding_box = item.get("bounding_box")
        _validate_bounding_box(bounding_box)

        row = {
            "track_id": str(item.get("track_id", "")),
            "class_label": str(item.get("class_label", "unknown")),
            "confidence": confidence,
            "timestamp_s": timestamp_s,
            "bounding_box": bounding_box,
        }
        rows.append(row)
    return rows


_DETECT_PATH = "/detect_track"


def process_video_segment(
    video_uri: str,
    segment: dict[str, Any],
    detector: str = DEFAULT_DETECTOR,
    transport: Transport | None = None,
    **options: Any,
) -> list[dict[str, Any]]:
    """Run detection/tracking on one segment via the external vision service.

    Args:
        video_uri: resolvable URI of the source video.
        segment: descriptor with ``start_s`` and ``end_s`` (from sampling).
        detector: detector selector forwarded to the service. Has a working
            default (``DEFAULT_DETECTOR``) so callers may omit it.
        transport: optional stub callable ``(path, payload) -> dict`` for
            tests; it receives the service-relative path ``"/detect_track"``.
            When ``None``, detection runs locally with OpenCV 5, or via the
            optional remote service when ``V2S_VISION_ENDPOINT`` is set.
        **options: pass-through service hints.  Recognised option keys:
            ``timeout`` (float, seconds) overrides the default/V2S_VISION_TIMEOUT.
            On the local OpenCV 5 path, ``stabilize`` (bool) and
            ``var_threshold`` (float) are forwarded to the detector; other
            keys are ignored there.

    Returns:
        A list of normalised detection rows with ``track_id``, ``class_label``,
        ``confidence``, ``timestamp_s`` and ``bounding_box``.
    """
    # Fail on malformed video_uri (issue 193).
    _validate_video_uri(video_uri)

    # Issue 158: require start_s and end_s; missing bounds mean the caller's
    # segment descriptor is incomplete.
    if "start_s" not in segment:
        raise DetectConfigError("segment descriptor is missing required key 'start_s'")
    if "end_s" not in segment:
        raise DetectConfigError("segment descriptor is missing required key 'end_s'")

    # Issue 213: safe float conversion for start_s / end_s.
    start_s = _coerce_float(segment["start_s"], "start_s")
    end_s = _coerce_float(segment["end_s"], "end_s")

    # Issue 202: validate segment bounds.
    if start_s < 0:
        raise DetectConfigError(
            f"segment start_s must be >= 0; got {start_s}"
        )
    if end_s <= start_s:
        raise DetectConfigError(
            f"segment end_s ({end_s}) must be greater than start_s ({start_s})"
        )

    segment_id = segment.get("segment_id")
    payload: dict[str, Any] = {
        "video_uri": video_uri,
        "segment_start_s": start_s,
        "segment_end_s": end_s,
        "detector": detector,
    }
    if segment_id:
        payload["segment_id"] = str(segment_id)

    if transport is None and not _use_remote():
        return _local_detect(video_uri, start_s, end_s, **options)

    model_dir = _model_dir()
    if model_dir:
        payload["model_dir"] = model_dir
    if options:
        payload["options"] = dict(options)

    send = transport if transport is not None else _default_transport
    response = send(_DETECT_PATH, payload)
    if not isinstance(response, dict):
        raise DetectServiceError("detection response is not an object")
    return _normalise_detections(response)


def _resolve_local(video_uri: str) -> tuple[Any, str]:
    """Import the OpenCV 5 core and resolve *video_uri* to a local path."""
    from . import sample as _sample

    try:
        core = _sample._import_core()
    except _sample.SampleConfigError as exc:
        raise DetectConfigError(str(exc)) from exc
    try:
        path = _sample.local_path(video_uri)
    except _sample.SampleServiceError as exc:
        raise DetectConfigError(str(exc)) from exc
    return core, path


def _local_detector_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """Pick the recognised local detector options; ignore everything else."""
    kwargs: dict[str, Any] = {}
    if "stabilize" in options:
        kwargs["stabilize"] = bool(options["stabilize"])
    if "var_threshold" in options:
        try:
            kwargs["var_threshold"] = float(options["var_threshold"])
        except (TypeError, ValueError) as exc:
            raise DetectConfigError(
                f"option var_threshold must be a number; got {options['var_threshold']!r}"
            ) from exc
    return kwargs


def _local_detect(
    video_uri: str, start_s: float, end_s: float, **options: Any
) -> list[dict[str, Any]]:
    """Run OpenCV 5 background-subtraction tracking on one segment window.

    Recognised options ``stabilize`` (bool) and ``var_threshold`` (float) are
    forwarded to ``core.track_detections``; other options are ignored.
    """
    core, path = _resolve_local(video_uri)
    kwargs = _local_detector_kwargs(options)
    try:
        rows = core.track_detections(path, start_s=start_s, end_s=end_s, **kwargs)
    except FileNotFoundError as exc:
        raise DetectServiceError(str(exc)) from exc
    return _normalise_detections({"detections": rows})


def measure_camera_motion(video_uri: str, start_s: float, end_s: float) -> float:
    """Median global camera translation (pixels at analysis width) of a segment.

    Returns 0.0 when the remote vision service is in use (no local frames).
    Otherwise resolves the local path exactly as ``_local_detect`` does and
    measures motion with OpenCV 5 phase correlation via
    ``core.estimate_camera_motion``.
    """
    if _use_remote():
        return 0.0
    core, path = _resolve_local(video_uri)
    try:
        return float(core.estimate_camera_motion(path, start_s=start_s, end_s=end_s))
    except FileNotFoundError as exc:
        raise DetectServiceError(str(exc)) from exc
