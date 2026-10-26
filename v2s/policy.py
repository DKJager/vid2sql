# build 7fd77e00de7f
"""Agent decision policy: turn OpenCV observations into plans and decisions.

Standard library only (no OpenCV import). This module holds the pure logic of
the agent loop run per segment:

  (1) OBSERVE camera motion with OpenCV (phase correlation, done elsewhere)
  (2) PLAN the detector settings from it          -> ``plan_detection``
  (3) DETECT with those settings (done elsewhere)
  (4) OBSERVE the detections (track statistics)   -> ``observe``
  (5) DECIDE: accept, retry once with stabilisation, or flag for a human
                                                  -> ``review_decision`` /
                                                     ``prefer_retry``

Every function here is deterministic and side-effect free so it can be unit
tested offline with hand-made detection rows.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# Camera motion (pixels at the analysis width) at or above which the detector
# runs with stabilisation.
CAMERA_MOTION_STABILIZE_PX = 6.0
# MOG2 varThreshold used together with stabilisation.
VAR_THRESHOLD_STABILIZED = 40.0
# Peak number of simultaneous tracks at or above which counts are unreliable.
PEAK_REVIEW_TRACKS = 40
# Tracks shorter than this (seconds) count as "short".
SHORT_TRACK_S = 0.5
# Share of short tracks at or above which the segment looks fragmented.
SHORT_FRACTION_REVIEW = 0.6
# Minimum number of tracks before the fragmentation rule is applied.
MIN_TRACKS_FOR_JUDGEMENT = 10


@dataclass(frozen=True)
class Observation:
    """Summary statistics of the detections of one segment."""

    n_tracks: int
    peak_simultaneous: int
    median_track_s: float
    short_track_fraction: float
    duration_s: float

    def as_dict(self) -> dict:
        return {
            "n_tracks": int(self.n_tracks),
            "peak_simultaneous": int(self.peak_simultaneous),
            "median_track_s": round(float(self.median_track_s), 4),
            "short_track_fraction": round(float(self.short_track_fraction), 4),
            "duration_s": round(float(self.duration_s), 4),
        }


def observe(detections: list[dict], start_s: float, end_s: float) -> Observation:
    """Compute track statistics from detection rows.

    Each row carries ``track_id`` and ``timestamp_s``. A track's length is its
    last timestamp minus its first timestamp. ``peak_simultaneous`` is the
    largest number of distinct track ids sharing one ``timestamp_s`` value.
    """
    duration_s = max(float(end_s) - float(start_s), 0.0)

    first_last: dict[object, list[float]] = {}
    per_timestamp: dict[float, set] = {}
    for row in detections or []:
        track_id = row["track_id"]
        ts = float(row["timestamp_s"])
        span = first_last.get(track_id)
        if span is None:
            first_last[track_id] = [ts, ts]
        else:
            if ts < span[0]:
                span[0] = ts
            if ts > span[1]:
                span[1] = ts
        per_timestamp.setdefault(ts, set()).add(track_id)

    if not first_last:
        return Observation(
            n_tracks=0,
            peak_simultaneous=0,
            median_track_s=0.0,
            short_track_fraction=0.0,
            duration_s=duration_s,
        )

    lengths = [last - first for first, last in first_last.values()]
    n_tracks = len(lengths)
    n_short = sum(1 for length in lengths if length < SHORT_TRACK_S)
    peak = max(len(ids) for ids in per_timestamp.values())

    return Observation(
        n_tracks=n_tracks,
        peak_simultaneous=peak,
        median_track_s=float(statistics.median(lengths)),
        short_track_fraction=n_short / n_tracks,
        duration_s=duration_s,
    )


@dataclass(frozen=True)
class Plan:
    """Detector settings chosen from the measured camera motion."""

    stabilize: bool
    options: dict
    reason: str


def plan_detection(camera_motion_px: float) -> Plan:
    """Choose detector options from the measured camera motion."""
    motion = float(camera_motion_px)
    if motion >= CAMERA_MOTION_STABILIZE_PX:
        return Plan(
            True,
            {"stabilize": True, "var_threshold": VAR_THRESHOLD_STABILIZED},
            f"Camera motion {motion:.2f} px is at or above "
            f"{CAMERA_MOTION_STABILIZE_PX} px, so the detector runs with "
            f"stabilisation and varThreshold {VAR_THRESHOLD_STABILIZED}.",
        )
    return Plan(
        False,
        {},
        f"Camera motion {motion:.2f} px is below "
        f"{CAMERA_MOTION_STABILIZE_PX} px, so the detector runs with default settings.",
    )


@dataclass(frozen=True)
class Decision:
    """Outcome of reviewing one segment: accept, retry_stabilized or flag_review."""

    action: str
    reason: str


def review_decision(obs: Observation, stabilized: bool) -> Decision:
    """Decide what to do with a segment given its observation."""
    if obs.peak_simultaneous >= PEAK_REVIEW_TRACKS:
        return Decision(
            "flag_review",
            f"Peak of {obs.peak_simultaneous} simultaneous tracks reaches "
            f"{PEAK_REVIEW_TRACKS}; counts are unreliable in dense scenes.",
        )
    if (obs.n_tracks >= MIN_TRACKS_FOR_JUDGEMENT
            and obs.short_track_fraction >= SHORT_FRACTION_REVIEW):
        if stabilized:
            return Decision(
                "flag_review",
                f"{obs.short_track_fraction:.2f} of {obs.n_tracks} tracks are "
                f"shorter than {SHORT_TRACK_S} s even after stabilisation; "
                f"tracks are still fragmenting.",
            )
        return Decision(
            "retry_stabilized",
            f"{obs.short_track_fraction:.2f} of {obs.n_tracks} tracks are "
            f"shorter than {SHORT_TRACK_S} s; retrying once with stabilisation.",
        )
    return Decision(
        "accept",
        f"{obs.n_tracks} tracks, peak {obs.peak_simultaneous} simultaneous, "
        f"short-track fraction {obs.short_track_fraction:.2f}; accepted.",
    )


def prefer_retry(first: Observation, second: Observation) -> bool:
    """True when the retry has strictly fewer tracks and a strictly lower short fraction."""
    return (second.n_tracks < first.n_tracks
            and second.short_track_fraction < first.short_track_fraction)
