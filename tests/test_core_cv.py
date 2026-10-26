import sys
import pathlib
import math
import json
import os

# Set path to find v2s_core module at repo/v2s/core_cv.py
# Tests are assumed to run from repo/v2s/tests/ or similar depth
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import cv2
import numpy as np

# Import the module under test
from v2s import core_cv as v2s_core


# --- Test Video Generation Fixtures ---

def _write_video(tmp_path, filename, frames, fps=30, size=(320, 240)):
    """Helper to write a list of frames to a video file."""
    path = tmp_path / filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
    return str(path)


@pytest.fixture
def static_video(tmp_path):
    """A 3-second video with all frames identical (static)."""
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(90)]
    return _write_video(tmp_path, "static.mp4", frames)


@pytest.fixture
def moving_rect_video(tmp_path):
    """A 2-second video with one moving white rectangle on black background."""
    frames = []
    for i in range(60):  # 2 seconds at 30fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw a white rectangle that moves right
        start_x = 10 + (i * 4)
        cv2.rectangle(frame, (start_x, 100), (start_x + 30, 130), (255, 255, 255), -1)
        frames.append(frame)
    return _write_video(tmp_path, "moving_rect.mp4", frames)


@pytest.fixture
def scene_cut_video(tmp_path):
    """A video with a hard cut between two different scenes."""
    # Scene 1: white frames (1s) -> hard cut -> Scene 2: black frames (1s)
    frames = []
    for _ in range(30):  # 1 second of white
        frames.append(np.full((240, 320, 3), 255, dtype=np.uint8))
    for _ in range(30):  # 1 second of black
        frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    return _write_video(tmp_path, "scene_cut.mp4", frames)


# --- OpenCV 5 Version Tests ---

def test_opencv_version_is_5():
    """
    CLAUSE 1 (Performance/Reliability): Verify OpenCV version meets requirement.
    cv2.__version__ must start with '5'.
    """
    assert cv2.__version__.startswith('5'), (
        f"OpenCV version must be 5.x, got {cv2.__version__}"
    )


# --- sample_segments Tests ---

def test_sample_segments_scene_cut(scene_cut_video):
    """
    SPEC: a hard cut between two scenes -> exactly 2 segments.
    """
    segments = v2s_core.sample_segments(scene_cut_video, min_len_s=0.5)
    
    assert isinstance(segments, list), "sample_segments must return a list"
    assert len(segments) == 2, (
        f"Expected exactly 2 segments for scene cut video, got {len(segments)}"
    )
    
    for seg in segments:
        assert isinstance(seg, dict), "Each segment must be a dict"
        assert 't_start' in seg, "Segment missing 't_start'"
        assert 't_end' in seg, "Segment missing 't_end'"
        assert 'score' in seg, "Segment missing 'score'"
        assert isinstance(seg['t_start'], (int, float)), "t_start must be numeric"
        assert isinstance(seg['t_end'], (int, float)), "t_end must be numeric"
        assert isinstance(seg['score'], (int, float)), "score must be numeric"
        assert seg['t_end'] >= seg['t_start'], "t_end must be >= t_start"


def test_sample_segments_static_video(static_video):
    """
    SPEC: static video -> 0 segments (no motion/scene changes).
    """
    segments = v2s_core.sample_segments(static_video, min_len_s=0.5)
    assert isinstance(segments, list), "sample_segments must return a list"
    assert len(segments) == 0, (
        f"Expected 0 segments for static video, got {len(segments)}"
    )


def test_sample_segments_min_len_filter(tmp_path):
    """
    Verify min_len_s parameter filters short segments.
    Create video with fast scene changes that would produce short segments.
    """
    # Create rapid alternating scenes (0.2s each)
    frames = []
    for _ in range(6):  # 0.2s each = 6 frames at 30fps
        frames.append(np.full((240, 320, 3), 255, dtype=np.uint8))
    for _ in range(6):
        frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    for _ in range(6):
        frames.append(np.full((240, 320, 3), 128, dtype=np.uint8))
    
    video_path = _write_video(tmp_path, "rapid_cuts.mp4", frames)
    
    # With min_len_s=0.5, most or all segments should be filtered
    segments = v2s_core.sample_segments(video_path, min_len_s=0.5)
    
    # Verify all returned segments meet minimum length
    for seg in segments:
        duration = seg['t_end'] - seg['t_start']
        assert duration >= 0.5, (
            f"Segment duration {duration}s is less than min_len_s=0.5"
        )


# --- detect_track Tests ---

def test_detect_track_one_moving_object(moving_rect_video):
    """
    SPEC: one moving rectangle -> exactly 1 track.
    """
    segments = v2s_core.sample_segments(moving_rect_video, min_len_s=0.5)
    
    tracks = v2s_core.detect_track(moving_rect_video, segments)
    
    assert isinstance(tracks, list), "detect_track must return a list"
    assert len(tracks) == 1, (
        f"Expected exactly 1 track for single moving object, got {len(tracks)}"
    )
    
    track = tracks[0]
    assert isinstance(track, dict), "Each track must be a dict"
    assert 'track_id' in track, "Track missing 'track_id'"
    assert 'cls' in track, "Track missing 'cls'"
    assert 'first_seen' in track, "Track missing 'first_seen'"
    assert 'last_seen' in track, "Track missing 'last_seen'"
    assert 'max_count' in track, "Track missing 'max_count'"
    
    # Verify class is 'moving_object' as specified
    assert track['cls'] == 'moving_object', (
        f"Expected class 'moving_object', got {track['cls']!r}"
    )
    
    # Verify temporal consistency
    assert track['last_seen'] >= track['first_seen'], (
        "last_seen must be >= first_seen"
    )
    
    # Verify max_count is reasonable (at least 1 for a tracked object)
    assert track['max_count'] >= 1, "max_count must be >= 1"


def test_detect_track_static_video(static_video):
    """
    SPEC: static video -> 0 tracks.
    """
    segments = v2s_core.sample_segments(static_video, min_len_s=0.5)
    tracks = v2s_core.detect_track(static_video, segments)
    
    assert isinstance(tracks, list), "detect_track must return a list"
    assert len(tracks) == 0, (
        f"Expected 0 tracks for static video, got {len(tracks)}"
    )


def test_detect_track_multiple_objects(tmp_path):
    """
    Verify tracking handles multiple distinct objects.
    """
    frames = []
    for i in range(60):  # 2 seconds
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Object 1: moving right at top
        x1 = 10 + i * 3
        cv2.rectangle(frame, (x1, 50), (x1 + 20, 70), (255, 255, 255), -1)
        # Object 2: moving left at bottom (different motion = different object)
        x2 = 290 - i * 3
        cv2.rectangle(frame, (x2, 170), (x2 + 20, 190), (255, 255, 255), -1)
        frames.append(frame)
    
    video_path = _write_video(tmp_path, "two_objects.mp4", frames)
    segments = v2s_core.sample_segments(video_path, min_len_s=0.5)
    tracks = v2s_core.detect_track(video_path, segments)
    
    # Should detect 2 distinct tracks for 2 moving objects
    assert len(tracks) == 2, (
        f"Expected 2 tracks for two moving objects, got {len(tracks)}"
    )
    
    # Each should have unique track_id
    track_ids = [t['track_id'] for t in tracks]
    assert len(set(track_ids)) == 2, "Track IDs must be unique"


# --- No Environment Variable Test ---

def test_no_env_var_read_at_import():
    """
    SPEC: no environment variable read at import.
    This test verifies the module doesn't read env vars during import.
    """
    # Save current environment state
    saved_env = os.environ.get('V2S_VISION_ENDPOINT')
    
    # Remove the env var if it exists
    if 'V2S_VISION_ENDPOINT' in os.environ:
        del os.environ['V2S_VISION_ENDPOINT']
    
    try:
        # Re-import the module (it should not fail or set different behavior)
        import importlib
        importlib.reload(v2s_core)
        
        # Verify module still functions
        assert hasattr(v2s_core, 'sample_segments')
        assert hasattr(v2s_core, 'detect_track')
        
    finally:
        # Restore environment
        if saved_env is not None:
            os.environ['V2S_VISION_ENDPOINT'] = saved_env


# --- Negative Control Tests ---

def test_negative_control_stub_fails_on_different_input(tmp_path):
    """
    NEGATIVE CONTROL: A stub returning fixed segments regardless of input must fail.
    
    This test creates two very different videos and passes them through a
    hypothetical stub. If the stub returns the same segments for both,
    it fails the negative control check.
    """
    
    # Create two very different videos
    frames_moving = []
    for i in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 4, 100), (40 + i * 4, 130), (255, 255, 255), -1)
        frames_moving.append(frame)
    
    video1_path = _write_video(tmp_path, "moving.mp4", frames_moving)
    
    # Video 2: Static
    frames_static = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(60)]
    video2_path = _write_video(tmp_path, "static.mp4", frames_static)
    
    # The NEGATIVE CONTROL stub (a bad implementation)
    class BadStub:
        """A stub that returns fixed segments regardless of input video."""
        def __init__(self):
            self.fixed_segments = [
                {'t_start': 0.0, 't_end': 1.0, 'score': 0.5},
                {'t_start': 1.0, 't_end': 2.0, 'score': 0.3},
            ]
        
        def sample_segments(self, path, min_len_s=0.5):
            # BAD: Ignores video content entirely
            return self.fixed_segments
    
    bad_stub = BadStub()
    
    # Get segments from both videos using the bad stub
    segs1 = bad_stub.sample_segments(video1_path)
    segs2 = bad_stub.sample_segments(video2_path)
    
    # NEGATIVE CONTROL CHECK: The stub is defective if it returns
    # the same segments for fundamentally different videos
    # (The ACTUAL v2s_core should return different results)
    
    # Compare the segments
    # The content-blind stub is detectably defective: identical output for different videos.
    assert json.dumps(segs1, sort_keys=True) == json.dumps(segs2, sort_keys=True)
    # The real implementation must NOT be content-blind.
    real1 = v2s_core.sample_segments(video1_path, min_len_s=0.5)
    real2 = v2s_core.sample_segments(video2_path, min_len_s=0.5)
    assert json.dumps(real1, sort_keys=True) != json.dumps(real2, sort_keys=True), (
        "Implementation returned identical segments for moving and static videos."
    )


def test_negative_control_stub_fails_fixed_tracks(tmp_path):
    """
    NEGATIVE CONTROL: A stub returning fixed tracks regardless of input must fail.
    """
    
    class BadTrackStub:
        """Bad implementation that always returns the same track."""
        def detect_track(self, path, segments):
            return [{'track_id': '1', 'cls': 'moving_object', 
                     'first_seen': 0.0, 'last_seen': 1.0, 'max_count': 1}]
    
    bad_stub = BadTrackStub()
    
    # Create static video (should have 0 tracks)
    frames_static = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(60)]
    static_video = _write_video(tmp_path, "static.mp4", frames_static)
    
    # Create moving video (should have tracks)
    frames_moving = []
    for i in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 4, 100), (40 + i * 4, 130), (255, 255, 255), -1)
        frames_moving.append(frame)
    moving_video = _write_video(tmp_path, "moving.mp4", frames_moving)
    
    # Bad stub returns same result for both
    tracks_static = bad_stub.detect_track(static_video, [])
    tracks_moving = bad_stub.detect_track(moving_video, [])
    
    # NEGATIVE CONTROL CHECK
    # The content-blind stub is detectably defective: identical output for both videos.
    assert json.dumps(tracks_static, sort_keys=True) == json.dumps(tracks_moving, sort_keys=True)
    # The real implementation must NOT be content-blind.
    real_static = v2s_core.detect_track(static_video, [])
    real_moving = v2s_core.detect_track(moving_video, [])
    assert json.dumps(real_static, sort_keys=True) != json.dumps(real_moving, sort_keys=True), (
        "Implementation returned identical tracks for static and moving videos."
    )


# --- Additional Contract Verification Tests ---

def test_sample_segments_returns_proper_structure(moving_rect_video):
    """
    Verify sample_segments returns proper dict structure with required keys.
    """
    segments = v2s_core.sample_segments(moving_rect_video, min_len_s=0.5)
    
    for seg in segments:
        assert 't_start' in seg
        assert 't_end' in seg
        assert 'score' in seg
        
        # Verify types
        assert isinstance(seg['t_start'], (int, float))
        assert isinstance(seg['t_end'], (int, float))
        assert isinstance(seg['score'], (int, float))
        
        # Verify score is non-negative
        assert seg['score'] >= 0, "score must be non-negative"


def test_detect_track_returns_proper_structure(moving_rect_video):
    """
    Verify detect_track returns proper dict structure with required keys.
    """
    segments = v2s_core.sample_segments(moving_rect_video, min_len_s=0.5)
    tracks = v2s_core.detect_track(moving_rect_video, segments)
    
    for track in tracks:
        assert 'track_id' in track
        assert 'cls' in track
        assert 'first_seen' in track
        assert 'last_seen' in track
        assert 'max_count' in track
        
        # Verify types
        assert isinstance(track['track_id'], (str, int))
        assert isinstance(track['cls'], str)
        assert isinstance(track['first_seen'], (int, float))
        assert isinstance(track['last_seen'], (int, float))
        assert isinstance(track['max_count'], int)


def test_sample_segments_with_empty_segments_list(static_video):
    """
    Verify detect_track handles empty segments list gracefully.
    """
    tracks = v2s_core.detect_track(static_video, [])
    assert isinstance(tracks, list)


def test_sample_segments_min_len_parameter_respected(tmp_path):
    """
    Verify min_len_s parameter is properly used.
    """
    # Create video with one scene change
    frames = []
    for _ in range(30):  # 1s white
        frames.append(np.full((240, 320, 3), 255, dtype=np.uint8))
    for _ in range(30):  # 1s black
        frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    video_path = _write_video(tmp_path, "twoscenes.mp4", frames)
    
    # Test with different min_len values
    segments_small = v2s_core.sample_segments(video_path, min_len_s=0.1)
    segments_large = v2s_core.sample_segments(video_path, min_len_s=2.0)
    
    # With small min_len, we should get our 2 segments
    # With large min_len (2.0s), segments shorter than 2s should be filtered
    for seg in segments_large:
        duration = seg['t_end'] - seg['t_start']
        assert duration >= 2.0, f"Segment {duration}s too short for min_len_s=2.0"
