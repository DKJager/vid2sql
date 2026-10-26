# build 7fd77e00de7f
"""Lazy HTTP client for the private reasoning service.

This module is a thin wrapper only. The private 3Dogs service (crowd read,
reconciliation support, RSI) runs entirely outside this repository and is
reached through a configurable endpoint. No model names, provider names,
prompts, scoring methods, or weights exist here.

Configuration (read lazily, only when a call is actually made):
    V2S_ENDPOINT -- base URL of the private reasoning service.
    V2S_TOKEN    -- bearer token for that service.
    V2S_TIMEOUT  -- optional request timeout in seconds (default 60).

Neither variable is read at import time, so tests can import this module with
stubs and no environment set.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any
from urllib.parse import urljoin

# Safety cap on response size to prevent OOM on a misbehaving endpoint.
# 16 MiB is generous for JSON rows and still bounds memory usage.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MiB

# Allowed schemes for V2S_ENDPOINT.  We refuse file://, ftp://, etc. so the
# service boundary stays tight.
_ALLOWED_SCHEMES = frozenset(("http", "https"))

# Validation bounds for segment timestamps (seconds).  Reasonable clips are
# shorter than 24 hours; negative or inverted values are always wrong.
_MAX_SEGMENT_DURATION_S = 86400  # 24 hours


class ClientConfigError(RuntimeError):
    """Raised when a required environment variable is missing at call time."""


class ClientServiceError(RuntimeError):
    """Raised when the private service returns an error or unreachable."""


def _endpoint() -> str:
    value = os.environ.get("V2S_ENDPOINT", "").strip()
    if not value:
        raise ClientConfigError(
            "V2S_ENDPOINT is not set (empty after trimming). It is read lazily "
            "at call time; set it to the base URL of the private reasoning "
            "service before calling."
        )
    # Tighten scheme validation to prevent accidental use of non-HTTP(S) URLs.
    # urlopen supports file:// and ftp://, which would silently bypass the
    # intended service boundary.
    if "://" in value:
        scheme = value.split("://", 1)[0].lower()
    else:
        # Without a scheme, urlopen will likely fail, but we still reject
        # because we require an explicit http/https prefix.
        scheme = ""
    if scheme not in _ALLOWED_SCHEMES:
        raise ClientConfigError(
            f"V2S_ENDPOINT must start with http:// or https:// "
            f"(got {value!r})."
        )
    return value.rstrip("/")


def _token() -> str:
    value = os.environ.get("V2S_TOKEN", "").strip()
    if not value:
        raise ClientConfigError(
            "V2S_TOKEN is not set (empty after trimming). Provide the service "
            "bearer token via the environment; tokens are never stored in files."
        )
    return value


def _timeout() -> float:
    """Read V2S_TIMEOUT lazily; provide a default of 60 seconds."""
    raw = os.environ.get("V2S_TIMEOUT", "60").strip()
    try:
        return float(raw)
    except ValueError:
        raise ClientConfigError(
            f"V2S_TIMEOUT must be a float (got {raw!r})."
        )


def _normalise_path(path: str) -> str:
    """Drop exactly one leading slash.  Paths must be single-slash-prefixed."""
    return path[1:] if path.startswith("/") else path


def _validate_video_uri(video_uri: Any) -> str:
    """Return *video_uri* as a non-empty, stripped string or raise ValueError."""
    if not isinstance(video_uri, str):
        raise TypeError(f"video_uri must be str, got {type(video_uri).__name__}")
    value = video_uri.strip()
    if not value:
        raise ValueError("video_uri must be a non-empty string after stripping")
    return value


def _validate_segments(segments: list[dict[str, Any]]) -> None:
    """Client-side validation for segment descriptors.

    Raises ValueError with a deterministic message when any segment fails
    basic shape and range checks, so callers never waste a round-trip on
    trivially invalid data (issue #188).
    """
    if not isinstance(segments, list):
        raise TypeError(f"segments must be a list, got {type(segments).__name__}")
    if not segments:
        raise ValueError("segments must be non-empty")
    required = {"start_s", "end_s"}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise TypeError(f"segment {i} must be a dict, got {type(seg).__name__}")
        missing = required - seg.keys()
        if missing:
            raise ValueError(
                f"segment {i} is missing required keys: {sorted(missing)}"
            )
        start = seg["start_s"]
        end = seg["end_s"]
        for key in ("start_s", "end_s"):
            val = seg[key]
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"segment {i} {key!r} must be a number, "
                    f"got {type(val).__name__} ({val!r})"
                )
            if val < 0:
                raise ValueError(
                    f"segment {i} {key!r} must be >= 0, got {val!r}"
                )
            if val > _MAX_SEGMENT_DURATION_S:
                raise ValueError(
                    f"segment {i} {key!r} exceeds maximum of "
                    f"{_MAX_SEGMENT_DURATION_S} s, got {val!r}"
                )
        if start >= end:
            raise ValueError(
                f"segment {i} start_s ({start!r}) must be less than "
                f"end_s ({end!r})"
            )


def _post(path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """POST JSON to the private service and return the decoded JSON body.

    This is the single internal transport function through which every public
    API call flows.  It resolves the full URL from the endpoint read via
    ``_endpoint()`` and the relative *path*, serialises *payload* as UTF-8
    JSON, attaches a Bearer token from ``_token()``, applies a configurable
    *timeout*, and decodes the JSON response.

    Args:
        path: service path, optionally preceded by a single ``/``.
        payload: JSON-serialisable dict to POST as the request body.
        timeout: optional override in seconds; falls back to ``V2S_TIMEOUT``
            then 60 s.

    Returns:
        Decoded JSON response body as a dict.

    Raises:
        ClientConfigError: when V2S_ENDPOINT or V2S_TOKEN is missing at call
            time.
        ClientServiceError: for HTTP errors, network failures, or non-JSON
            responses.
    """
    base = _endpoint()
    # Ensure exactly one slash between the base and the normalised path.
    normalised = _normalise_path(path)
    url = urljoin(base + "/" if not base.endswith("/") else base, normalised)
    body = json.dumps(payload).encode("utf-8")
    if timeout is None:
        timeout = _timeout()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_token()}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES).decode("utf-8")
            # If the stream is still readable, the body exceeded our cap.
            if response.read(1):
                raise ClientServiceError(
                    f"private service response exceeded the {_MAX_RESPONSE_BYTES} byte limit "
                    f"and was truncated to protect memory."
                )
    except urllib.error.HTTPError as exc:
        # Preserve the status code and a truncated body for diagnostic clues
        # (issue #191).  Callers can now distinguish 401, 429, 503, etc.
        body_preview = ""
        try:
            body_bytes = exc.read(_MAX_RESPONSE_BYTES)
            body_text = body_bytes.decode("utf-8", errors="replace")
            body_preview = body_text[:500]
        except Exception:
            body_preview = "<unreadable>"
        raise ClientServiceError(
            f"private service returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise ClientServiceError(f"private service call failed: {exc}") from exc
    try:
        decoded = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ClientServiceError(
            f"private service returned non-JSON body: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ClientServiceError("private service returned a non-object JSON body")
    return decoded


def crowd_read(video_uri: str,
               segments: list[dict[str, Any]],
               timeout: float | None = None) -> dict[str, Any]:
    """Ask the private service for a crowd read of sampled segments.

    Args:
        video_uri: resolvable URI of the source video.
        segments: non-empty list of segment descriptors, each carrying at
            least ``start_s`` and ``end_s``.
        timeout: optional per-call override for the request timeout
            (issue #192).

    Returns:
        Service response object. The exact contract is owned by the private
        service; this wrapper passes it through unchanged.

    Raises:
        TypeError, ValueError: if arguments fail client-side validation
            (issues #181, #188).
    """
    video_uri = _validate_video_uri(video_uri)
    _validate_segments(segments)
    return _post("/crowd_read", {"video_uri": video_uri, "segments": segments},
                 timeout=timeout)


def reconcile(local_rows: list[dict[str, Any]],
              crowd_rows: list[dict[str, Any]],
              timeout: float | None = None) -> dict[str, Any]:
    """Ask the private service to reconcile local detections with crowd rows.

    Args:
        local_rows: list of row dicts from local detection.
        crowd_rows: list of row dicts from the crowd read.
        timeout: optional per-call override for the request timeout
            (issue #192).

    Returns:
        Reconciled rows, each carrying a calibrated ``confidence``.

    Raises:
        TypeError: if *local_rows* or *crowd_rows* is not a list of dicts
            (issue #181).
    """
    if not isinstance(local_rows, list):
        raise TypeError(f"local_rows must be a list, got {type(local_rows).__name__}")
    if not isinstance(crowd_rows, list):
        raise TypeError(f"crowd_rows must be a list, got {type(crowd_rows).__name__}")
    for label, rows in [("local_rows", local_rows), ("crowd_rows", crowd_rows)]:
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(
                    f"{label}[{i}] must be a dict, got {type(row).__name__}"
                )
    return _post(
        "/reconcile", {"local_rows": local_rows, "crowd_rows": crowd_rows},
        timeout=timeout,
    )


def submit_feedback(video_uri: str,
                    flagged_rows: list[dict[str, Any]],
                    timeout: float | None = None) -> dict[str, Any]:
    """Send flagged-wrong rows to the private service for RSI improvement.

    Args:
        video_uri: resolvable URI of the source video.
        flagged_rows: non-empty list of row dicts flagged as incorrect.
        timeout: optional per-call override for the request timeout
            (issue #192).

    This is an explicit, caller-initiated action only; nothing is reported
    automatically.

    Raises:
        TypeError, ValueError: if arguments fail client-side validation
            (issue #181).
    """
    video_uri = _validate_video_uri(video_uri)
    if not isinstance(flagged_rows, list):
        raise TypeError(f"flagged_rows must be a list, got {type(flagged_rows).__name__}")
    if not flagged_rows:
        raise ValueError("flagged_rows must be non-empty")
    for i, row in enumerate(flagged_rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"flagged_rows[{i}] must be a dict, got {type(row).__name__}"
            )
    return _post(
        "/feedback", {"video_uri": video_uri, "flagged_rows": flagged_rows},
        timeout=timeout,
    )


# Stable public API for downstream tests and tooling.
__all__ = [
    "ClientConfigError",
    "ClientServiceError",
    "crowd_read",
    "reconcile",
    "submit_feedback",
]


def is_configured() -> bool:
    """True when the optional private service endpoint is set (read at call time)."""
    return bool(os.environ.get("V2S_ENDPOINT", "").strip())


def __getattr__(name: str) -> Any:
    # Module-level ``V2S_ENDPOINT`` is resolved on access, never at import.
    if name == "V2S_ENDPOINT":
        return os.environ.get("V2S_ENDPOINT", "").strip()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
