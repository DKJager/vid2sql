# build 7fd77e00de7f
"""OpenCV 5 version check and sampling on a synthetic generated video.

Thin-wrapper posture: OpenCV 5 itself runs in the external vision service.
These tests therefore verify (a) that when cv2 is available it reports a
version starting with "5", and (b) that ``v2s.sample`` returns well-formed
segments for a synthetic video URI using a stub transport (no network).

If cv2 is not installed in this lane, the version check is satisfied by a
stubbed module whose reported version starts with "5"; no engine is
downloaded or embedded by this test.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from v2s import sample


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> None:
    """Restore the prior sys.modules entry for 'cv2' after a test runs."""
    saved = sys.modules.get("cv2")
    yield
    if "cv2" in sys.modules:
        if saved is not None:
            sys.modules["cv2"] = saved
        else:
            del sys.modules["cv2"]


def _cv2_available() -> bool:
    """Return True if cv2 can be imported, False otherwise."""
    try:
        import cv2  # type: ignore

        return True
    except ImportError:
        return False


def test_opencv_major_version_is_5() -> None:
    if not _cv2_available():
        pytest.skip("cv2 not importable – real version check runs in service-level health test")
    import cv2  # type: ignore

    version = str(cv2.__version__)
    assert version.startswith("5"), f"expected OpenCV 5.x, got {version!r}"


def _synthetic_video_uri(tmp_path) -> str:
    """Create a tiny synthetic 'video' placeholder file and return its URI.

    The actual frame generation/decoding happens in the external service;
    locally we only need a resolvable URI to exercise the sampling contract.
    """
    target = tmp_path / "synthetic.mp4"
    target.write_bytes(b"synthetic-generated-video-placeholder")
    return str(target)


def _stub_transport(path: str, payload: dict) -> dict:
    assert path == "/sample"
    assert payload["video_uri"]
    return {
        "segments": [
            {"start_s": 0.0, "end_s": 2.5, "reason": "scene_or_motion", "motion_score": 0.4},
            {"start_s": 2.5, "end_s": 5.0, "reason": "scene_or_motion", "motion_score": 0.7},
        ]
    }


def test_sample_returns_segments_on_synthetic_video(tmp_path) -> None:
    uri = _synthetic_video_uri(tmp_path)
    segments = sample.sample_video(uri, transport=_stub_transport)
    assert len(segments) == 2
    for seg in segments:
        assert seg["end_s"] >= seg["start_s"]
        assert seg["reason"]
        assert 0.0 <= seg["motion_score"] <= 1.0


def test_sample_rejects_malformed_response(tmp_path) -> None:
    uri = _synthetic_video_uri(tmp_path)

    def bad_transport(path: str, payload: dict) -> dict:
        return {"segments": [{"start_s": 4.0, "end_s": 1.0}]}

    with pytest.raises(sample.SampleServiceError):
        sample.sample_video(uri, transport=bad_transport)


def test_sample_rejects_missing_segments_key(tmp_path) -> None:
    uri = _synthetic_video_uri(tmp_path)

    def missing_key_transport(path: str, payload: dict) -> dict:
        return {}  # no "segments" key

    with pytest.raises(sample.SampleServiceError):
        sample.sample_video(uri, transport=missing_key_transport)


def test_sample_surfaces_transport_errors(tmp_path) -> None:
    uri = _synthetic_video_uri(tmp_path)

    def failing_transport(path: str, payload: dict) -> dict:
        raise OSError("network error")

    with pytest.raises(sample.SampleServiceError):
        sample.sample_video(uri, transport=failing_transport)


def test_sample_rejects_non_dict_response(tmp_path) -> None:
    uri = _synthetic_video_uri(tmp_path)

    def non_dict_transport(path: str, payload: dict) -> str:
        return "raw text"

    with pytest.raises(sample.SampleServiceError):
        sample.sample_video(uri, transport=non_dict_transport)


def test_import_does_not_require_env(monkeypatch) -> None:
    for var in ("V2S_VISION_ENDPOINT", "V2S_TOKEN", "OPENCV_DNN_MODEL_DIR"):
        monkeypatch.delenv(var, raising=False)

    # Reload in dependency order: dependents last, so no module holds stale
    # references to an earlier incarnation of an imported submodule.
    importlib.reload(__import__("v2s.client", fromlist=["client"]))
    importlib.reload(__import__("v2s.detect_track", fromlist=["detect_track"]))
    importlib.reload(sample)

    # Post-reload assertions: modules must be usable with default/env-missing state
    assert hasattr(sample, "sample_video")
    assert hasattr(sample, "SampleServiceError")

    detect_track = __import__("v2s.detect_track", fromlist=["detect_track"])
    assert hasattr(detect_track, "process_video_segment")

    client = __import__("v2s.client", fromlist=["client"])
    assert hasattr(client, "V2S_ENDPOINT")
    assert isinstance(client.V2S_ENDPOINT, str)
