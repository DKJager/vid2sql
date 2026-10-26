# build 7fd77e00de7f
"""Agent loop: sample -> detect/track -> crowd read -> reconcile -> write.

Thin-wrapper orchestration only. OpenCV 5 sampling and detection/tracking run
in the external vision service (via ``v2s.sample`` and ``v2s.detect_track``);
the crowd read, reconciliation, and RSI run in the private 3Dogs service (via
``v2s.client``). No engine, model, provider detail, prompt, or scoring method
exists in this repository.

Agent layer (per segment, local detector only, on by default):
  observe camera motion (OpenCV phase correlation) -> plan detector settings
  (``v2s.policy.plan_detection``) -> detect with those settings -> observe the
  detections (track statistics) -> decide: accept, retry once with
  stabilisation, or flag for human review.  Every step is written to the
  ``agent_trace`` table; flagged segments go to ``review_queue`` with status
  ``pending``.  Disable with ``process_clip(..., agent=False)`` or the
  environment variable ``V2S_AGENT=off`` (read at call time).

Timestamp semantics (read before querying ``events``):
  - ``videos.processed_at_utc`` is always the wall-clock moment the clip was
    processed by this loop.
  - ``videos.recording_start_utc`` and ``events.start_timestamp_utc`` /
    ``events.end_timestamp_utc`` are recording wall-clock values ONLY when the
    clip object supplies a ``recording_start_utc`` attribute (ISO 8601).
  - When the clip does NOT supply ``recording_start_utc``, the loop has no
    way to know when the footage was recorded.  It then anchors the clip at
    ``processed_at_utc``, so ``recording_start_utc`` equals
    ``processed_at_utc`` and every event timestamp is **processing-relative**
    (``processed_at_utc`` + offset in seconds into the clip), NOT the original
    recording time.  Such rows are identifiable by
    ``videos.recording_start_utc = videos.processed_at_utc``; the in-video
    offsets (``segments.start_s`` / ``end_s``, ``objects.first_seen_s`` /
    ``last_seen_s``) remain exact in both cases.  A warning is logged for every
    clip processed this way, and callers that need true recording time can pass
    ``require_recording_start=True`` to ``process_clip`` to fail instead of
    falling back.

Audit notes:
  - No environment variable is read at import time; all configuration is
    resolved lazily at call time inside the collaborating modules.
  - ``process_video_segment`` is always invoked with its ``detector``
    argument satisfied (Audit Fix #4); it also has a working default.
  - Tests inject a stub crowd-read/reconcile client and stub transports so no
    real network calls occur.
"""

from __future__ import annotations

import json
import locale
import logging
import os
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from . import client as private_client
from . import detect_track, policy, sample

logger = logging.getLogger(__name__)


class CrowdClient(Protocol):
    """Contract for the private-service client used by the loop."""

    def crowd_read(
        self, video_uri: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    def reconcile(
        self,
        local_rows: list[dict[str, Any]],
        crowd_rows: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class TranscriptClient(Protocol):
    """Contract for an optional external transcript service.

    The agent does not implement speech-to-text locally; demos that require
    transcript output MUST inject a client conforming to this protocol.
    When no client is provided, a built-in no-op stub is used and zero
    transcript rows are written.
    """

    def transcribe(
        self, video_uri: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return transcript rows for the given segments.

        Each row must include at least ``t`` (seconds) and ``text`` keys.
        """
        ...


class LocalCrowdClient:
    """Offline fallback used when no private service is configured.

    ``crowd_read`` contributes no rows and ``reconcile`` returns the local
    detections unchanged, so the loop stays fully functional with local
    OpenCV 5 results alone.
    """

    def crowd_read(
        self, video_uri: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {"rows": []}

    def reconcile(
        self,
        local_rows: list[dict[str, Any]],
        crowd_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {"rows": [dict(r) for r in local_rows] + [dict(r) for r in crowd_rows]}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float safely, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _agent_enabled(agent: bool) -> bool:
    """Agent layer is OFF when *agent* is False or V2S_AGENT=off (call time)."""
    if not agent:
        return False
    return os.environ.get("V2S_AGENT", "").strip().lower() != "off"


def _trace_row(
    video_id: str,
    segment_id: str,
    step: str,
    observation: dict[str, Any],
    decision: str,
    reason: str,
) -> dict[str, Any]:
    """Build one ``agent_trace`` row."""
    return {
        "trace_id": _new_id("trc"),
        "video_id": video_id,
        "segment_id": segment_id,
        "step": step,
        "observation": json.dumps(observation, sort_keys=True),
        "decision": decision,
        "reason": reason,
        "created_at_utc": _utc_now_iso(),
    }


def _as_detection_list(raw: Any, segment_id: str) -> list[dict[str, Any]]:
    """Guard against non-list return from process_video_segment (issue 172)."""
    if isinstance(raw, list):
        return raw
    logger.debug(
        "segment_id=%s :: process_video_segment returned %s (expected list); "
        "falling back to empty list",
        segment_id,
        type(raw).__name__,
    )
    return []


def _local_rows_from_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shape service detection rows for the reconciliation request."""
    return [
        {
            "track_id": str(det.get("track_id", "")),
            "class_label": str(det.get("class_label", "unknown")),
            "confidence": _clamp_confidence(det.get("confidence", 0.0)),
            "timestamp_s": _safe_float(det.get("timestamp_s", 0.0)),
        }
        for det in detections
    ]


def _build_object_rows(
    video_id: str, reconciled: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate reconciled detections into per-track object rows.

    ``max_count`` reflects the peak number of detections that share the same
    second-of-video timestamp for a (track_id, class_label) key — a simple
    approximation of instantaneous co-occurrence density.
    """
    tracks: dict[tuple[str, str], dict[str, Any]] = {}
    per_second_counts: dict[tuple[str, str, int], set[str]] = {}

    for row in reconciled:
        # Audit Fix #182: guard against non-dict entries in the reconciled list
        if not isinstance(row, dict):
            logger.debug("_build_object_rows: skipping non-dict row %r", row)
            continue

        track_id = str(row.get("track_id", "")) or "untracked"
        class_label = str(row.get("class_label", "unknown"))
        timestamp_s = _safe_float(row.get("timestamp_s", 0.0))
        second_bucket = int(timestamp_s)
        confidence = _clamp_confidence(row.get("confidence", 0.0))

        # Per-track summary accumulation
        key = (track_id, class_label)
        entry = tracks.get(key)
        if entry is None:
            tracks[key] = {
                "track_id": track_id,
                "class_label": class_label,
                "first_seen_s": timestamp_s,
                "last_seen_s": timestamp_s,
                "max_count": 1,
                "confidence": confidence,
            }
        else:
            entry["first_seen_s"] = min(entry["first_seen_s"], timestamp_s)
            entry["last_seen_s"] = max(entry["last_seen_s"], timestamp_s)
            entry["confidence"] = max(entry["confidence"], confidence)

        # Track per-second distinct detections for max_count
        count_key = (track_id, class_label, second_bucket)
        per_second_counts.setdefault(count_key, set()).add(
            f"{timestamp_s:.3f}"  # use full timestamp to count detection instances
        )

    # Back-fill max_count from per-second peaks
    for (track_id, class_label, second_bucket), stamps in per_second_counts.items():
        key = (track_id, class_label)
        if key in tracks:
            tracks[key]["max_count"] = max(tracks[key]["max_count"], len(stamps))

    return [
        {
            "object_id": _new_id("obj"),
            "video_id": video_id,
            "track_id": entry["track_id"],
            "class_label": entry["class_label"],
            "first_seen_s": entry["first_seen_s"],
            "last_seen_s": entry["last_seen_s"],
            "max_count": entry["max_count"],
            "confidence": entry["confidence"],
        }
        for entry in sorted(tracks.values(), key=lambda e: (e["track_id"], e["class_label"]))
    ]


def _build_event_rows(
    video_id: str,
    segment_id: str,
    segment: dict[str, Any],
    video_start_utc: str,
    reconciled: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive segment-level presence events from reconciled rows.

    Events are rule-based presence aggregations only; no predictive
    behavioral analytics is claimed or implemented here.

    Event timestamps are ``video_start_utc`` plus the segment boundaries
    (``start_s`` / ``end_s``), so downstream consumers see the span of time
    covered by the segment relative to that anchor.

    .. important::
       ``video_start_utc`` is expected to represent the **recording start** of
       the source video in ISO 8601.  When the caller could not determine the
       recording start and passes the processing timestamp instead
       (``process_clip`` does this only when the clip has no
       ``recording_start_utc`` and ``require_recording_start`` is False), the
       resulting ``start_timestamp_utc`` / ``end_timestamp_utc`` values are
       **processing-relative**, not recording wall-clock.  Such rows can be
       recognised through ``videos.recording_start_utc = videos.processed_at_utc``
       for the same ``video_id``.  The in-video offsets in ``segments`` and
       ``objects`` are unaffected.  This limitation must be disclosed wherever
       event timestamps are presented as recording time.
    """
    start_s = _safe_float(segment.get("start_s", 0.0))
    end_s = _safe_float(segment.get("end_s", 0.0))

    # video_start_utc is ISO 8601; convert to datetime for safe arithmetic
    try:
        anchor_dt = datetime.fromisoformat(
            video_start_utc.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        anchor_dt = datetime.now(timezone.utc)
        logger.warning(
            "video_id=%s segment_id=%s :: video_start_utc %r is not ISO 8601; "
            "event timestamps for this segment are anchored at the current "
            "processing time, not the recording time",
            video_id,
            segment_id,
            video_start_utc,
        )

    start_dt = anchor_dt + timedelta(seconds=start_s)
    end_dt = anchor_dt + timedelta(seconds=end_s)

    segment_start_utc = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    segment_end_utc = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    by_class: dict[str, list[float]] = {}
    for row in reconciled:
        class_label = str(row.get("class_label", "unknown"))
        by_class.setdefault(class_label, []).append(
            _clamp_confidence(row.get("confidence", 0.0))
        )
    events: list[dict[str, Any]] = []
    for class_label, confidences in sorted(by_class.items()):
        aggregate = sum(confidences) / len(confidences) if confidences else 0.0
        events.append(
            {
                "event_id": _new_id("evt"),
                "video_id": video_id,
                "segment_id": segment_id,
                "event_type": f"presence:{class_label}",
                "start_timestamp_utc": segment_start_utc,
                "end_timestamp_utc": segment_end_utc,
                "confidence": round(aggregate, 6),
            }
        )
    return events


def _build_count_rows(
    video_id: str, reconciled: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-minute per-class *distinct track* counts from reconciled detections.

    Counts distinct ``track_id`` values that appear in each
    (minute, class_label) bucket.  Detections with an empty or missing
    ``track_id`` are treated as an "untracked" singleton so they still
    contribute one unit count rather than being silently dropped.

    The ``confidence`` field stores the **mean** of the individual detection
    confidences that fell into each bucket (not peak, not mode).  Downstream
    consumers who need a different aggregation should compute it from the
    raw detection stream.
    """
    buckets: dict[tuple[int, str], set[str]] = {}
    confidences_per_bucket: dict[tuple[int, str], list[float]] = {}

    for row in reconciled:
        # Audit Fix #183: use safe coercion for timestamp_s to avoid crashes
        # on non-numeric values that survive earlier sanitization.
        timestamp_s = _safe_float(row.get("timestamp_s", 0.0))
        minute_index = int(timestamp_s // 60)
        class_label = str(row.get("class_label", "unknown"))
        track_id = str(row.get("track_id", "")) or "untracked"
        confidence = _clamp_confidence(row.get("confidence", 0.0))

        key = (minute_index, class_label)
        buckets.setdefault(key, set()).add(track_id)
        confidences_per_bucket.setdefault(key, []).append(confidence)

    rows: list[dict[str, Any]] = []
    for (minute_index, class_label), track_set in sorted(buckets.items()):
        confidences = confidences_per_bucket.get((minute_index, class_label), [])
        # Mean confidence – see docstring for semantics
        aggregate = sum(confidences) / len(confidences) if confidences else 0.0
        rows.append(
            {
                "count_id": _new_id("cnt"),
                "video_id": video_id,
                "minute_index": minute_index,
                "class_label": class_label,
                "count": len(track_set),
                "confidence": round(aggregate, 6),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Issue 139: built-in no-op transcript stub so the agent can still emit zero
# rows without hard-coding an empty list deep inside the loop body.
# ---------------------------------------------------------------------------
class _NoOpTranscriptClient:
    """Transcript stub used when no external transcript service is injected."""

    def transcribe(
        self, video_uri: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Issue 175: validate clip shape early and produce a clear error message.
# ---------------------------------------------------------------------------
_REQUIRED_CLIP_ATTRS = ("uri", "filename", "source_url", "license")


def _validate_clip_shape(clip: Any) -> None:
    """Raise a descriptive :exc:`TypeError` if *clip* lacks a required attr."""
    missing = [attr for attr in _REQUIRED_CLIP_ATTRS if not hasattr(clip, attr)]
    if missing:
        msg = (
            f"clip must expose {_REQUIRED_CLIP_ATTRS!r} attributes. "
            f"Missing: {missing!r}. "
            f"Received type {type(clip).__name__}."
        )
        raise TypeError(msg)


def process_clip(
    clip: Any,
    writer: Any,
    detector: str = detect_track.DEFAULT_DETECTOR,
    crowd_client: CrowdClient | None = None,
    transcript_client: TranscriptClient | None = None,
    sample_transport: sample.Transport | None = None,
    detect_transport: detect_track.Transport | None = None,
    require_recording_start: bool = False,
    agent: bool = True,
) -> dict[str, Any]:
    """Run the full agent loop for one clip and write all table rows.

    Args:
        clip: an :class:`v2s.ingest.ClipWorkItem` (or any object with ``uri``,
            ``filename``, ``source_url``, ``license`` and optionally
            ``recording_start_utc`` attributes).  ``recording_start_utc`` must
            be the ISO 8601 UTC wall-clock time at which the footage was
            recorded.  If it is absent or ``None``, the recording time is
            unknown: the clip is anchored at ``processed_at_utc`` instead, so
            ``videos.recording_start_utc`` equals ``videos.processed_at_utc``
            and all ``events.*_timestamp_utc`` values are
            **processing-relative** (processing time + seconds into the clip),
            not recording wall-clock.  A warning is logged whenever this
            fallback is taken (see :issue:`170`).  In-video offsets in
            ``segments`` and ``objects`` are exact regardless.
        writer: object with ``write_rows(table, rows) -> int`` (e.g.
            :class:`v2s.sqlwriter.SqliteWriter`).
        detector: detector selector forwarded to the external vision service.
            Passed explicitly so the call always satisfies the
            ``process_video_segment`` signature (Audit Fix #4).
        crowd_client: optional stub implementing :class:`CrowdClient`; when
            ``None``, the real private-service client module is used and its
            configuration is resolved lazily at call time.
        transcript_client: optional stub implementing
            :class:`TranscriptClient`; when ``None``, a no-op stub is used
            and zero transcript rows are written.
        sample_transport / detect_transport: optional stub transports for
            the external vision service, used by tests.
        require_recording_start: when ``True``, refuse to process a clip that
            does not supply ``recording_start_utc`` (raises :exc:`ValueError`
            before any row is written) instead of falling back to
            processing-relative event timestamps.  Defaults to ``False`` to
            keep existing callers working.
        agent: when ``True`` (default) and the environment variable
            ``V2S_AGENT`` is not ``off`` (case-insensitive, read at call
            time), the agent layer runs for every segment processed by the
            local detector (``detect_transport is None``): measure camera
            motion, plan detector settings, detect, review track statistics,
            optionally retry once with stabilisation, and flag segments for
            human review.  Steps go to ``agent_trace``; flags to
            ``review_queue``.  When off, behaviour is unchanged.

    Returns:
        A dict with keys:

        * ``video_id`` – the identifier assigned to this clip (Issue 174).
        * ``counts`` – mapping of table name to rows written (always includes
          ``agent_trace`` and ``review_queue``).

    Raises:
        TypeError: if *clip* lacks a required attribute (Issue 175).
        ValueError: if ``require_recording_start`` is ``True`` and the clip has
            no ``recording_start_utc``.

    Note:
        If the sample service returns an empty segment list, a video row is
        still written with ``processed_at_utc`` but no segment, event, object
        or count rows are produced.  A warning is emitted to make this visible
        in logs.  (Audit Fix #180)
    """
    # -- Issue 175: validate clip shape before any side-effects --
    _validate_clip_shape(clip)

    agent_on = _agent_enabled(agent)

    if crowd_client is not None:
        service = crowd_client
    elif private_client.is_configured():
        service = private_client
    else:
        # No private service configured: use the local single-reader fallback.
        service = LocalCrowdClient()
    txn = transcript_client if transcript_client is not None else _NoOpTranscriptClient()

    # -- Issue 178: capture a single processing timestamp for the entire clip --
    processing_ts = _utc_now_iso()

    video_id = _new_id("vid")

    # -- Resolve recording-start timestamp (issue 170) --
    # Prefer an explicit `recording_start_utc`.  If the clip does not supply
    # one, the recording time is unknown.  Either fail (strict mode) or fall
    # back to `processing_ts` and make the limitation visible: the resulting
    # event timestamps are processing-relative, and the video row records the
    # same value in both `recording_start_utc` and `processed_at_utc` so
    # queries can tell these rows apart from clips with a known recording time.
    recording_start_utc = getattr(clip, "recording_start_utc", None)
    if recording_start_utc is None:
        if require_recording_start:
            raise ValueError(
                f"clip {clip.filename!r} ({clip.uri!r}) has no recording_start_utc "
                "and require_recording_start=True; refusing to write "
                "processing-relative event timestamps"
            )
        recording_start_utc = processing_ts
        logger.warning(
            "video_id=%s :: clip %r has no recording_start_utc; anchoring at "
            "processed_at_utc=%s. events.start_timestamp_utc/end_timestamp_utc "
            "for this video are processing-relative, NOT recording wall-clock "
            "(videos.recording_start_utc = videos.processed_at_utc marks this)",
            video_id,
            clip.filename,
            recording_start_utc,
        )

    video_row = {
        "video_id": video_id,
        "uri": clip.uri,
        "filename": clip.filename,
        "source_url": clip.source_url,
        "license": clip.license,
        "processed_at_utc": processing_ts,
        "recording_start_utc": recording_start_utc,
    }

    # Always write the video row first so partial progress is preserved (Issue 143).
    writer.write_rows("videos", [video_row])

    segments = sample.sample_video(clip.uri, transport=sample_transport)

    # Audit Fix #168: handle empty segment list gracefully
    if not segments:
        warnings.warn(
            f"sample_video returned no segments for URI {clip.uri!r} "
            f"(video_id={video_id}); only the video row will be written.",
            stacklevel=2,
        )
        return {
            "video_id": video_id,
            "counts": {
                "videos": 1,
                "segments": 0,
                "events": 0,
                "objects": 0,
                "counts": 0,
                "transcript": 0,
                "agent_trace": 0,
                "review_queue": 0,
            },
        }

    segment_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    all_reconciled: list[dict[str, Any]] = []
    transcript_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for segment in segments:
        segment_id = _new_id("seg")
        segment = {**segment, "segment_id": segment_id}
        # Audit Fix #166: sanitize segment timestamps with safe coercion
        segment_start = _safe_float(segment.get("start_s", 0.0))
        segment_end = _safe_float(segment.get("end_s", 0.0))
        reason = str(segment.get("reason", "scene_or_motion"))
        # motion_score validation (issue 171): safe-coerce with warning on non-numeric
        try:
            motion_score = _safe_float(segment.get("motion_score", 0.0))
        except Exception:
            motion_score = 0.0
            logger.debug(
                "segment_id=%s :: motion_score coercion failed; defaulting to 0.0",
                segment_id,
            )

        # Issue 166: populate transcript data when a transcriber is available
        seg_transcript = txn.transcribe(clip.uri, [segment])
        if isinstance(seg_transcript, list):
            transcript_rows.extend(seg_transcript)

        # The agent layer runs only when the detector runs locally.
        segment_agent = agent_on and detect_transport is None

        try:
            plan = None
            if segment_agent:
                # (1) OBSERVE camera motion with OpenCV phase correlation.
                try:
                    motion = _safe_float(
                        detect_track.measure_camera_motion(
                            clip.uri, segment_start, segment_end
                        )
                    )
                except Exception as motion_exc:
                    logger.warning(
                        "segment_id=%s :: camera motion measurement failed "
                        "(%s: %s); using 0.0",
                        segment_id, type(motion_exc).__name__, motion_exc,
                    )
                    motion = 0.0
                # (2) PLAN detector settings from the observation.
                plan = policy.plan_detection(motion)
                trace_rows.append(
                    _trace_row(
                        video_id,
                        segment_id,
                        "plan",
                        {"camera_motion_px": round(motion, 4)},
                        "stabilize" if plan.stabilize else "default",
                        plan.reason,
                    )
                )

            # (3) DETECT with the planned settings (none when the agent is off).
            detect_options = dict(plan.options) if plan is not None else {}
            raw_detections = detect_track.process_video_segment(
                clip.uri,
                segment,
                detector=detector,
                transport=detect_transport,
                **detect_options,
            )
            # Guard against non-list return (issue 172)
            detections = _as_detection_list(raw_detections, segment_id)

            if plan is not None:
                # (4) OBSERVE the detections, (5) DECIDE.
                obs = policy.observe(detections, segment_start, segment_end)
                decision = policy.review_decision(obs, plan.stabilize)
                trace_rows.append(
                    _trace_row(
                        video_id,
                        segment_id,
                        "review",
                        obs.as_dict(),
                        decision.action,
                        decision.reason,
                    )
                )
                if decision.action == "retry_stabilized":
                    raw_retry = detect_track.process_video_segment(
                        clip.uri,
                        segment,
                        detector=detector,
                        transport=detect_transport,
                        stabilize=True,
                        var_threshold=policy.VAR_THRESHOLD_STABILIZED,
                    )
                    retry_detections = _as_detection_list(raw_retry, segment_id)
                    obs2 = policy.observe(
                        retry_detections, segment_start, segment_end
                    )
                    if policy.prefer_retry(obs, obs2):
                        detections = retry_detections
                        obs_used = obs2
                        retry_decision = "accept_retry"
                    else:
                        obs_used = obs
                        retry_decision = "keep_first"
                    trace_rows.append(
                        _trace_row(
                            video_id,
                            segment_id,
                            "retry",
                            {"first": obs.as_dict(), "retry": obs2.as_dict()},
                            retry_decision,
                            f"First pass produced {obs.n_tracks} tracks; "
                            f"stabilised retry produced {obs2.n_tracks} tracks.",
                        )
                    )
                    decision = policy.review_decision(obs_used, True)
                    trace_rows.append(
                        _trace_row(
                            video_id,
                            segment_id,
                            "review",
                            obs_used.as_dict(),
                            decision.action,
                            decision.reason,
                        )
                    )
                if decision.action == "flag_review":
                    review_rows.append(
                        {
                            "review_id": _new_id("rev"),
                            "video_id": video_id,
                            "segment_id": segment_id,
                            "reason": decision.reason,
                            "status": "pending",
                            "created_at_utc": _utc_now_iso(),
                        }
                    )

            local_rows = _local_rows_from_detections(detections)

            crowd_response = service.crowd_read(clip.uri, [segment])
            crowd_rows = crowd_response.get("rows") if isinstance(crowd_response, dict) else None
            if not isinstance(crowd_rows, list):
                # Audit Fix #167: emit warning on type mismatch instead of silent fallback
                logger.debug(
                    "segment_id=%s :: crowd_read response missing 'rows' key; "
                    "payload type=%s; discarding crowd data for this segment",
                    segment_id,
                    type(crowd_rows).__name__ if crowd_rows is not None else "None",
                )
                warnings.warn(
                    f"crowd_read for segment {segment_id}: response missing 'rows' list; "
                    f"received type {type(crowd_rows).__name__ if crowd_rows is not None else 'None'}, "
                    f"discarding crowd data for this segment",
                    stacklevel=2,
                )
                crowd_rows = []

            reconciled_response = service.reconcile(local_rows, crowd_rows)
            # Audit Fix #184: ensure reconcile() returned a dict before calling .get()
            if not isinstance(reconciled_response, dict):
                logger.warning(
                    "segment_id=%s :: reconcile returned non-dict type %s; falling back to local_rows",
                    segment_id,
                    type(reconciled_response).__name__,
                )
                reconciled_response = {}

            # Fix #164: explicit guard instead of silent dict.get fallback
            reconciled = reconciled_response.get("rows") if isinstance(reconciled_response, dict) else None
            if not isinstance(reconciled, list):
                # Issue 173: debug log when reconcile rows key is missing or malformed
                logger.debug(
                    "segment_id=%s :: reconcile response missing 'rows' key; "
                    "payload type=%s; falling back to local_rows (%d detections)",
                    segment_id,
                    type(reconciled).__name__ if reconciled is not None else "None",
                )
                warnings.warn(
                    f"reconcile for segment {segment_id}: response missing 'rows' list; "
                    f"received type {type(reconciled).__name__ if reconciled is not None else 'None'}, "
                    f"falling back to local_rows ({len(local_rows)} detections)",
                    stacklevel=2,
                )
                reconciled = local_rows

            # Audit Fix #170: validate reconciled row structure before accessing keys
            valid_reconciled = [r for r in reconciled if isinstance(r, dict)]
            for row in valid_reconciled:
                all_reconciled.append(row)

            event_rows.extend(
                _build_event_rows(
                    video_id,
                    segment_id,
                    segment,
                    # Pass the resolved anchor: the clip's recording start when
                    # known, otherwise processed_at_utc (processing-relative
                    # event timestamps; warned about above).
                    recording_start_utc,
                    valid_reconciled,
                )
            )

        except Exception as exc:
            logger.warning(
                "segment_id=%s :: processing failed (%s: %s)",
                segment_id, type(exc).__name__, exc,
            )
            reason = f"error:{reason}"
            # Still record the segment so it is visible in the output tables
            # (Issue 143).

        segment_rows.append(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "start_s": segment_start,
                "end_s": segment_end,
                "reason": reason,
                "motion_score": motion_score,
            }
        )

    object_rows = _build_object_rows(video_id, all_reconciled)
    count_rows = _build_count_rows(video_id, all_reconciled)

    batch: dict[str, list[dict[str, Any]]] = {
        "segments": segment_rows,
        "events": event_rows,
        "objects": object_rows,
        "counts": count_rows,
        "transcript": transcript_rows,
        "agent_trace": trace_rows,
        "review_queue": review_rows,
    }
    write_counts: dict[str, int] = {"videos": 1}
    for table, rows in batch.items():
        if rows:
            write_counts[table] = writer.write_rows(table, rows)
        else:
            write_counts[table] = 0

    # -- Issue 174: return video_id alongside per-table write counts --
    return {"video_id": video_id, "counts": write_counts}


def process_clips(
    clips: list[Any],
    writer: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the agent loop over a work list; returns per-clip write counts.

    Keys are derived from each clip's ``video_id`` (Issue 140) to avoid
    silent overwrites when duplicate filenames appear.  If duplicate
    ``video_id`` values are encountered across clips, a suffix is appended
    so every entry is preserved.
    """
    result: dict[str, Any] = {}
    seen_ids: set[str] = set()
    for idx, clip in enumerate(clips):
        per_clip = process_clip(clip, writer, **kwargs)
        video_id = per_clip["video_id"]
        key = str(video_id)
        if key in seen_ids:
            suffix = 2
            while f"{key}__{suffix}" in seen_ids:
                suffix += 1
            key = f"{key}__{suffix}"
        seen_ids.add(key)
        # Issue 176: store the full return dict (video_id + counts)
        result[key] = per_clip
    return result
