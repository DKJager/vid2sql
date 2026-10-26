"""Unit tests for v2s/policy.py (agent layer, section C of the contract).

Offline, standard library + pytest only. Detection rows are hand-made dicts
with "track_id" and "timestamp_s". Every policy constant is exercised at its
boundary value (exactly on the threshold and just below it).
"""

from __future__ import annotations

import dataclasses

import pytest

from v2s import policy
from v2s.policy import Decision, Observation, Plan


# ---------------------------------------------------------------- helpers


def _rows(pairs):
    """Build detection rows from (track_id, timestamp_s) pairs."""
    return [{"track_id": tid, "timestamp_s": ts} for tid, ts in pairs]


def _obs(
    n_tracks=0,
    peak_simultaneous=0,
    median_track_s=0.0,
    short_track_fraction=0.0,
    duration_s=10.0,
):
    return Observation(
        n_tracks=n_tracks,
        peak_simultaneous=peak_simultaneous,
        median_track_s=median_track_s,
        short_track_fraction=short_track_fraction,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------- constants


def test_constants_have_contract_values():
    assert policy.CAMERA_MOTION_STABILIZE_PX == 6.0
    assert policy.VAR_THRESHOLD_STABILIZED == 40.0
    assert policy.PEAK_REVIEW_TRACKS == 40
    assert policy.SHORT_TRACK_S == 0.5
    assert policy.SHORT_FRACTION_REVIEW == 0.6
    assert policy.MIN_TRACKS_FOR_JUDGEMENT == 10


def test_policy_does_not_import_opencv():
    assert not hasattr(policy, "cv2")


# ---------------------------------------------------------------- observe


def test_observe_empty_input_is_all_zeros():
    obs = policy.observe([], 2.0, 2.0)
    assert obs.n_tracks == 0
    assert obs.peak_simultaneous == 0
    assert obs.median_track_s == 0.0
    assert obs.short_track_fraction == 0.0
    assert obs.duration_s == 0.0


def test_observe_counts_tracks_and_lengths():
    # track 1: 0.0 -> 2.0 (2.0 s), track 2: 1.0 -> 2.0 (1.0 s), track 3: single row (0.0 s)
    rows = _rows([(1, 0.0), (1, 1.0), (1, 2.0), (2, 1.0), (2, 2.0), (3, 3.0)])
    obs = policy.observe(rows, 0.0, 4.0)
    assert obs.n_tracks == 3
    assert obs.median_track_s == pytest.approx(1.0)
    # only track 3 (0.0 s) is shorter than SHORT_TRACK_S
    assert obs.short_track_fraction == pytest.approx(1 / 3)
    assert obs.duration_s == pytest.approx(4.0)


def test_observe_track_length_uses_first_and_last_timestamp_regardless_of_order():
    rows = _rows([(7, 3.0), (7, 1.0), (7, 2.0)])
    obs = policy.observe(rows, 0.0, 5.0)
    assert obs.n_tracks == 1
    assert obs.median_track_s == pytest.approx(2.0)
    assert obs.short_track_fraction == 0.0


def test_observe_short_track_boundary_exactly_at_threshold_is_not_short():
    # length exactly SHORT_TRACK_S (0.5) -> not short; length 0.49 -> short
    rows = _rows([(1, 1.0), (1, 1.5), (2, 1.0), (2, 1.49)])
    obs = policy.observe(rows, 0.0, 2.0)
    assert obs.n_tracks == 2
    assert obs.short_track_fraction == pytest.approx(0.5)


def test_observe_all_tracks_at_threshold_gives_zero_short_fraction():
    rows = _rows([(1, 0.0), (1, 0.5), (2, 2.0), (2, 2.5)])
    obs = policy.observe(rows, 0.0, 3.0)
    assert obs.short_track_fraction == 0.0


def test_observe_peak_simultaneous_counts_distinct_tracks_per_timestamp():
    rows = _rows(
        [
            (1, 0.0),
            (2, 0.0),
            (1, 0.1),
            (2, 0.1),
            (3, 0.1),
            (3, 0.1),  # duplicate row of the same track at the same timestamp
            (1, 0.2),
        ]
    )
    obs = policy.observe(rows, 0.0, 1.0)
    assert obs.peak_simultaneous == 3
    assert obs.n_tracks == 3


def test_observe_duplicate_rows_of_one_track_do_not_inflate_peak():
    rows = _rows([(5, 1.0), (5, 1.0), (5, 1.0)])
    obs = policy.observe(rows, 0.0, 2.0)
    assert obs.peak_simultaneous == 1
    assert obs.n_tracks == 1


def test_observe_duration_is_clamped_at_zero():
    rows = _rows([(1, 0.0), (1, 1.0)])
    assert policy.observe(rows, 5.0, 3.0).duration_s == 0.0
    assert policy.observe(rows, 1.5, 4.0).duration_s == pytest.approx(2.5)


def test_observation_as_dict_has_five_keys_and_rounds_floats():
    obs = _obs(
        n_tracks=3,
        peak_simultaneous=2,
        median_track_s=1.234567,
        short_track_fraction=1 / 3,
        duration_s=4.000049,
    )
    d = obs.as_dict()
    assert set(d) == {
        "n_tracks",
        "peak_simultaneous",
        "median_track_s",
        "short_track_fraction",
        "duration_s",
    }
    assert d["n_tracks"] == 3
    assert d["peak_simultaneous"] == 2
    assert d["median_track_s"] == 1.2346
    assert d["short_track_fraction"] == 0.3333
    assert d["duration_s"] == 4.0


def test_observation_is_frozen():
    obs = _obs()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.n_tracks = 5  # type: ignore[misc]


# ---------------------------------------------------------------- plan_detection


def test_plan_stabilizes_exactly_at_threshold():
    plan = policy.plan_detection(policy.CAMERA_MOTION_STABILIZE_PX)
    assert isinstance(plan, Plan)
    assert plan.stabilize is True
    assert plan.options == {
        "stabilize": True,
        "var_threshold": policy.VAR_THRESHOLD_STABILIZED,
    }
    assert "6.00" in plan.reason


def test_plan_default_just_below_threshold():
    plan = policy.plan_detection(5.99)
    assert plan.stabilize is False
    assert plan.options == {}
    assert isinstance(plan.reason, str) and plan.reason


def test_plan_default_for_steady_camera():
    plan = policy.plan_detection(0.0)
    assert plan.stabilize is False
    assert plan.options == {}
    assert plan.reason


def test_plan_reason_quotes_measured_value_with_two_decimals():
    plan = policy.plan_detection(7.14159)
    assert plan.stabilize is True
    assert "7.14" in plan.reason
    assert plan.options["var_threshold"] == 40.0


def test_plan_is_frozen():
    plan = policy.plan_detection(2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.stabilize = False  # type: ignore[misc]


# ---------------------------------------------------------------- review_decision


@pytest.mark.parametrize("stabilized", [False, True])
def test_review_flags_dense_scene_exactly_at_peak_threshold(stabilized):
    obs = _obs(n_tracks=50, peak_simultaneous=policy.PEAK_REVIEW_TRACKS)
    d = policy.review_decision(obs, stabilized)
    assert isinstance(d, Decision)
    assert d.action == "flag_review"
    assert d.reason


def test_review_accepts_just_below_peak_threshold():
    obs = _obs(n_tracks=45, peak_simultaneous=policy.PEAK_REVIEW_TRACKS - 1)
    d = policy.review_decision(obs, False)
    assert d.action == "accept"
    assert d.reason


def test_review_dense_scene_takes_precedence_over_fragmentation():
    obs = _obs(n_tracks=60, peak_simultaneous=40, short_track_fraction=0.9)
    assert policy.review_decision(obs, False).action == "flag_review"


def test_review_retries_fragmented_unstabilized_at_both_thresholds():
    obs = _obs(
        n_tracks=policy.MIN_TRACKS_FOR_JUDGEMENT,
        peak_simultaneous=3,
        short_track_fraction=policy.SHORT_FRACTION_REVIEW,
    )
    d = policy.review_decision(obs, False)
    assert d.action == "retry_stabilized"
    assert d.reason


def test_review_flags_fragmented_after_stabilization():
    obs = _obs(
        n_tracks=policy.MIN_TRACKS_FOR_JUDGEMENT,
        peak_simultaneous=3,
        short_track_fraction=policy.SHORT_FRACTION_REVIEW,
    )
    d = policy.review_decision(obs, True)
    assert d.action == "flag_review"
    assert d.reason


@pytest.mark.parametrize("stabilized", [False, True])
def test_review_accepts_just_below_short_fraction_threshold(stabilized):
    obs = _obs(n_tracks=20, peak_simultaneous=3, short_track_fraction=0.59)
    assert policy.review_decision(obs, stabilized).action == "accept"


@pytest.mark.parametrize("stabilized", [False, True])
def test_review_accepts_below_min_tracks_even_if_all_short(stabilized):
    obs = _obs(
        n_tracks=policy.MIN_TRACKS_FOR_JUDGEMENT - 1,
        peak_simultaneous=3,
        short_track_fraction=1.0,
    )
    assert policy.review_decision(obs, stabilized).action == "accept"


def test_review_accepts_empty_observation():
    d = policy.review_decision(policy.observe([], 0.0, 4.0), False)
    assert d.action == "accept"
    assert d.reason


def test_review_reason_quotes_measured_numbers():
    obs = _obs(n_tracks=55, peak_simultaneous=42)
    d = policy.review_decision(obs, False)
    assert "42" in d.reason


def test_review_from_observed_rows_end_to_end():
    # 10 tracks, 6 of them shorter than 0.5 s -> fraction exactly 0.6
    pairs = []
    for tid in range(6):
        pairs += [(tid, float(tid)), (tid, float(tid) + 0.2)]
    for tid in range(6, 10):
        pairs += [(tid, float(tid)), (tid, float(tid) + 1.0)]
    obs = policy.observe(_rows(pairs), 0.0, 12.0)
    assert obs.n_tracks == 10
    assert obs.short_track_fraction == pytest.approx(0.6)
    assert policy.review_decision(obs, False).action == "retry_stabilized"
    assert policy.review_decision(obs, True).action == "flag_review"


def test_decision_is_frozen():
    d = policy.review_decision(_obs(), False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = "flag_review"  # type: ignore[misc]


# ---------------------------------------------------------------- prefer_retry


def test_prefer_retry_when_fewer_tracks_and_lower_fraction():
    first = _obs(n_tracks=20, short_track_fraction=0.7)
    second = _obs(n_tracks=12, short_track_fraction=0.3)
    assert policy.prefer_retry(first, second) is True


def test_prefer_retry_false_when_track_count_equal():
    first = _obs(n_tracks=20, short_track_fraction=0.7)
    second = _obs(n_tracks=20, short_track_fraction=0.3)
    assert policy.prefer_retry(first, second) is False


def test_prefer_retry_false_when_fraction_equal():
    first = _obs(n_tracks=20, short_track_fraction=0.7)
    second = _obs(n_tracks=12, short_track_fraction=0.7)
    assert policy.prefer_retry(first, second) is False


def test_prefer_retry_false_when_retry_is_worse():
    first = _obs(n_tracks=12, short_track_fraction=0.3)
    second = _obs(n_tracks=20, short_track_fraction=0.7)
    assert policy.prefer_retry(first, second) is False


def test_prefer_retry_false_when_only_one_metric_improves():
    first = _obs(n_tracks=20, short_track_fraction=0.3)
    second = _obs(n_tracks=12, short_track_fraction=0.7)
    assert policy.prefer_retry(first, second) is False
    assert policy.prefer_retry(second, first) is False


def test_prefer_retry_false_for_identical_observations():
    obs = _obs(n_tracks=15, short_track_fraction=0.65)
    assert policy.prefer_retry(obs, obs) is False
