# build 7fd77e00de7f
"""Scene-change and motion segment sampling.

By default sampling runs **locally** with OpenCV 5 (``v2s.core_cv``): frame
histogram comparison for hard scene cuts and Farneback dense optical flow for
motion. A remote vision service is an optional alternative, selected only when
``V2S_VISION_ENDPOINT`` is set.

Configuration (all read at call time, never at import):
    V2S_VISION_ENDPOINT -- optional base URL of a remote vision service. When
                           set, sampling is delegated to it.
    V2S_VISION_MODE     -- set to ``local`` to force local OpenCV even when an
                           endpoint is configured.
    V2S_TOKEN           -- optional bearer token for the remote service.
    V2S_SAMPLE_PATH     -- path appended to the endpoint (default "/sample").
    V2S_TIMEOUT         -- remote request timeout in seconds (default 120.0).

The ``transport`` parameter lets tests inject a stub callable
``(path, payload) -> dict``; no environment variable is needed in that case.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import unquote, urlparse

#: A transport receives the service-relative path (e.g. ``"/sample"``) and the
#: JSON payload, and returns the decoded JSON object.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

_DEFAULT_SAMPLE_PATH = "/sample"
_DEFAULT_TIMEOUT = 120.0


class SampleConfigError(RuntimeError):
    """Raised when configuration or input is invalid at call time."""


class SampleServiceError(RuntimeError):
    """Raised when sampling fails or returns bad data."""


@dataclass
class Segment:
    """A sampled time window worth analysing, in seconds from video start."""

    start_s: float
    end_s: float
    reason: str = "scene_or_motion"
    motion_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "reason": self.reason,
            "motion_score": self.motion_score,
        }


def _remote_endpoint() -> str:
    """Return the configured remote endpoint, or ``""`` when unset."""
    value = os.environ.get("V2S_VISION_ENDPOINT", "").strip()
    if not value:
        return ""
    if value.split("://", 1)[0].lower() not in ("http", "https") or "://" not in value:
        raise SampleConfigError(
            "V2S_VISION_ENDPOINT must be a full http(s) URL; got "
            f"{value!r}"
        )
    return value.rstrip("/")


def _use_remote() -> bool:
    if os.environ.get("V2S_VISION_MODE", "").strip().lower() == "local":
        return False
    return bool(_remote_endpoint())


def _sample_path() -> str:
    return os.environ.get("V2S_SAMPLE_PATH", _DEFAULT_SAMPLE_PATH).strip() or _DEFAULT_SAMPLE_PATH


def _timeout() -> float:
    raw = os.environ.get("V2S_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        raise SampleConfigError(f"V2S_TIMEOUT must be a valid number, got: {raw!r}")
    if parsed <= 0:
        raise SampleConfigError(f"V2S_TIMEOUT must be positive, got: {parsed!r}")
    return parsed


def _http_transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = _remote_endpoint()
    if not endpoint:
        raise SampleConfigError("V2S_VISION_ENDPOINT is not set")
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("V2S_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{endpoint}{path}", data=body, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SampleServiceError(
            f"vision service returned HTTP {exc.code} {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SampleServiceError(f"vision service call failed: {exc}") from exc
    try:
        decoded = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise SampleServiceError(f"vision service returned non-JSON body: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SampleServiceError("vision service returned a non-object JSON body")
    return decoded


def local_path(video_uri: str) -> str:
    """Map a plain path or ``file://`` URI to a filesystem path.

    Any other scheme (``s3://``, ``stub://`` ...) cannot be opened locally.
    """
    uri = video_uri.strip()
    if "://" not in uri:
        return uri
    parsed = urlparse(uri)
    if parsed.scheme.lower() == "file":
        return unquote(parsed.path)
    raise SampleServiceError(
        f"local sampling cannot open {parsed.scheme!r} URIs; download the file "
        "first, or configure a remote vision service"
    )


def _import_core():
    try:
        from . import core_cv
    except ImportError as exc:  # opencv / numpy not installed
        raise SampleConfigError(
            "local sampling needs opencv-python-headless (see requirements.txt)"
        ) from exc
    return core_cv


def probe_duration_s(video_uri: str) -> float:
    """Best-effort video duration in seconds; 0.0 when unknown."""
    try:
        return float(_import_core().probe_duration(local_path(video_uri)))
    except Exception:
        return 0.0


def _local_sample(video_uri: str, options: dict[str, Any]) -> dict[str, Any]:
    core = _import_core()
    path = local_path(video_uri)
    min_len_s = float(options.get("min_len_s", 0.5))
    try:
        raw = core.sample_segments(path, min_len_s=min_len_s)
    except FileNotFoundError as exc:
        raise SampleServiceError(str(exc)) from exc
    return {
        "segments": [
            {
                "start_s": r["t_start"],
                "end_s": r["t_end"],
                "reason": "scene_or_motion",
                "motion_score": r["score"],
            }
            for r in raw
        ]
    }


def _parse_segments(response: dict[str, Any]) -> list[Segment]:
    if not isinstance(response, dict):
        raise SampleServiceError("sampling response is not an object")
    raw = response.get("segments")
    if not isinstance(raw, list):
        raise SampleServiceError("sampling response missing 'segments' list")
    segments: list[Segment] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SampleServiceError("segment entry is not an object")
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
            motion_score = float(item.get("motion_score", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise SampleServiceError(
                f"segment entry missing or non-numeric start_s/end_s/motion_score: {item!r}"
            ) from exc
        if not (math.isfinite(start_s) and math.isfinite(end_s) and math.isfinite(motion_score)):
            raise SampleServiceError(f"segment values must be finite numbers: {item!r}")
        if end_s < start_s:
            raise SampleServiceError(f"segment end before start: {item!r}")
        segments.append(
            Segment(
                start_s=start_s,
                end_s=end_s,
                reason=str(item.get("reason", "scene_or_motion")),
                motion_score=motion_score,
            )
        )
    return segments


def sample_segments(
    video_uri: str,
    transport: Transport | None = None,
    **options: Any,
) -> list[Segment]:
    """Return sampled segments for a video.

    Args:
        video_uri: local path or ``file://`` URI (local mode), or any URI the
            remote service understands (remote mode / injected transport).
        transport: optional callable ``(path, payload) -> dict``. When given it
            is used instead of both local OpenCV and the remote service; tests
            inject a deterministic stub here.
        **options: sampling hints (local mode honours ``min_len_s``).
    """
    if not video_uri or not video_uri.strip():
        raise SampleConfigError("video_uri must be a non-empty string")
    uri = video_uri.strip()
    if transport is None and not _use_remote():
        return _parse_segments(_local_sample(uri, options))

    payload: dict[str, Any] = {
        "video_uri": uri,
        "video_duration_s": probe_duration_s(uri),
    }
    if options:
        payload["options"] = dict(options)
    send = transport if transport is not None else _http_transport
    path = _sample_path()
    try:
        response = send(path, payload)
    except (SampleServiceError, SampleConfigError):
        raise
    except OSError as exc:
        raise SampleServiceError(f"sampling transport failed: {exc}") from exc
    return _parse_segments(response)


def sample_video(
    video_uri: str,
    transport: Transport | None = None,
    **options: Any,
) -> list[dict[str, Any]]:
    """Dict-returning convenience wrapper used by the agent loop."""
    return [s.to_dict() for s in sample_segments(video_uri, transport, **options)]
