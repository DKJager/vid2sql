# vid2sql: rule-based natural-language question -> read-only SQL (local mode).
"""Translate a plain question into a SELECT over the eight tables.

The eight tables are the six video tables plus the two agent tables,
``agent_trace`` (what the agent decided at each step) and ``review_queue``
(what is waiting for human review).

This is deliberate pattern matching, not a language model: every supported
question maps to one fixed, reviewed query. Anything else raises
``UnsupportedQuestion`` rather than guessing. Used by the web endpoint and by
``scripts/ask.py``; it never runs a query itself.
"""

from __future__ import annotations

import re

#: Largest second considered by the per-second occupancy query.
_MAX_SECONDS = 3600


class UnsupportedQuestion(ValueError):
    """The question matches none of the supported patterns."""


_OCCUPANCY = (
    "WITH RECURSIVE s(t) AS (SELECT 0 UNION ALL SELECT t + 1 FROM s"
    f" WHERE t < {_MAX_SECONDS}),"
    " occ AS (SELECT o.video_id AS video_id, s.t AS second, COUNT(*) AS active_tracks"
    " FROM s JOIN objects o ON o.first_seen_s <= s.t AND o.last_seen_s >= s.t"
    " GROUP BY o.video_id, s.t)"
)

_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "needs_review",
        re.compile(r"\breview\b", re.I),
        "SELECT r.review_id AS review_id, v.filename AS filename, r.video_id AS video_id,"
        " r.segment_id AS segment_id, r.reason AS reason, r.status AS status"
        " FROM review_queue r JOIN videos v ON v.video_id = r.video_id"
        " ORDER BY r.created_at_utc, r.review_id",
    ),
    (
        "agent_decisions",
        re.compile(r"\bagent\b.*\b(decid|decision)|\b(decid|decision).*\bagent\b", re.I),
        "SELECT step, decision, COUNT(*) AS n FROM agent_trace"
        " GROUP BY step, decision ORDER BY step, decision",
    ),
    (
        "busiest_second",
        re.compile(r"\b(busiest|most (crowded|active|busy)|peak|rush)\b.*\b(second|moment|time|when)\b"
                   r"|\bwhen\b.*\b(busiest|most (crowded|active|busy)|peak)\b", re.I),
        _OCCUPANCY + " SELECT v.filename AS filename, occ.second AS second,"
        " occ.active_tracks AS active_tracks FROM occ JOIN videos v ON v.video_id = occ.video_id"
        " WHERE occ.active_tracks = (SELECT MAX(o2.active_tracks) FROM occ o2"
        " WHERE o2.video_id = occ.video_id) GROUP BY occ.video_id ORDER BY v.filename",
    ),
    (
        "longest_tracks",
        re.compile(r"\b(longest|longest-lived|stays?|stayed|lingers?)\b.*\b(track|object|visible|seen|screen|frame)", re.I),
        "SELECT v.filename AS filename, o.track_id AS track_id,"
        " ROUND(o.last_seen_s - o.first_seen_s, 2) AS seconds_visible,"
        " o.first_seen_s AS first_seen_s, o.last_seen_s AS last_seen_s"
        " FROM objects o JOIN videos v ON v.video_id = o.video_id"
        " ORDER BY seconds_visible DESC LIMIT 5",
    ),
    (
        "most_activity",
        re.compile(r"\b(which|what)\b.*\b(video|clip)\b.*\b(most|least|more|fewest|busiest|quietest)\b"
                   r"|\b(most|least|busiest|quietest)\b.*\b(video|clip)\b", re.I),
        "SELECT v.filename AS filename, COUNT(o.object_id) AS moving_objects"
        " FROM videos v LEFT JOIN objects o ON o.video_id = v.video_id"
        " GROUP BY v.video_id ORDER BY moving_objects DESC",
    ),
    (
        "objects_per_video",
        re.compile(r"\bhow many\b.*\b(object|objects|track|tracks|people|pedestrians?|vehicles?|cars?|things)\b", re.I),
        "SELECT v.filename AS filename, COUNT(o.object_id) AS moving_objects"
        " FROM videos v LEFT JOIN objects o ON o.video_id = v.video_id"
        " GROUP BY v.video_id ORDER BY v.filename",
    ),
    (
        "confidence",
        re.compile(r"\b(confiden(ce|t)|how (sure|certain|reliable)|trust)\b", re.I),
        "SELECT v.filename AS filename, ROUND(AVG(o.confidence), 3) AS mean_confidence,"
        " ROUND(MIN(o.confidence), 3) AS min_confidence, ROUND(MAX(o.confidence), 3) AS max_confidence"
        " FROM objects o JOIN videos v ON v.video_id = o.video_id GROUP BY o.video_id ORDER BY v.filename",
    ),
    (
        "scenes",
        re.compile(r"\b(scenes?|segments?|cuts?)\b", re.I),
        "SELECT v.filename AS filename, s.start_s AS start_s, s.end_s AS end_s,"
        " s.motion_score AS motion_score FROM segments s JOIN videos v ON v.video_id = s.video_id"
        " ORDER BY v.filename, s.start_s",
    ),
    (
        "provenance",
        re.compile(r"\b(licen[cs]e|source|attribution|where .* from|who (made|shot|filmed))\b", re.I),
        "SELECT filename, license, source_url FROM videos ORDER BY filename",
    ),
)


def translate(question: str) -> tuple[str, str]:
    """Return ``(sql, pattern_name)`` for a supported question."""
    text = (question or "").strip()
    if not text:
        raise UnsupportedQuestion("empty question")
    for name, pattern, sql in _PATTERNS:
        if pattern.search(text):
            return sql, name
    raise UnsupportedQuestion(
        "unsupported question; supported topics: busiest moment, longest-visible"
        " objects, which video has the most activity, how many moving objects,"
        " confidence, scenes, licence and source, what needs human review,"
        " what the agent decided"
    )
