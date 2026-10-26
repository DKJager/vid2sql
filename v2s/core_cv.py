# build 7fd77e00de7f
"""
OpenCV 5 video analysis for segment sampling and object detection/tracking.

Installation (exact pip line for build venv):
    pip install opencv-python-headless==5.0.0.93

This module performs substantive image analysis using OpenCV 5:
- sample_segments: frame differencing, optical flow, and scene-change detection.
- detect_track: background subtraction + contour detection + centroid tracking.
- estimate_camera_motion: global inter-frame translation by phase correlation.
- track_detections: per-frame detections, optionally on phase-correlation
  stabilised frames.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


#: Frames wider than this are downscaled before analysis. Real footage is
#: often 1080p; sensor and codec noise at full size produces thousands of
#: spurious foreground specks, and flow/background models are 16x slower.
MAX_ANALYSIS_WIDTH = 480


def _shrink(frame: np.ndarray, max_width: int = MAX_ANALYSIS_WIDTH) -> tuple[np.ndarray, float]:
    """Return (frame resized to at most ``max_width`` wide, scale factor)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    small = cv2.resize(frame, (max_width, max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def _gray32(frame: np.ndarray) -> np.ndarray:
    """Grayscale float32 copy of a BGR (or already grey) frame."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return np.float32(gray)


def sample_segments(
    path: str,
    min_len_s: float = 0.5,
) -> list[dict[str, float]]:
    """Detect scene changes and motion segments in a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    prev_frame_gray = None
    prev_hist = None
    frame_idx = 0
    raw_flags: list[dict[str, Any]] = []

    # Thresholds tuned for synthetic test videos
    SCENE_CUT_HIST_THRESHOLD = 0.7
    MOTION_THRESHOLD = 0.005

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = _shrink(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_sec = frame_idx / fps

        scene_cut = False
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < SCENE_CUT_HIST_THRESHOLD:
                scene_cut = True
        prev_hist = hist

        motion_energy = 0.0
        if prev_frame_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy = float(np.mean(mag))
        prev_frame_gray = gray.copy()

        raw_flags.append({
            "scene_cut": scene_cut,
            "motion": motion_energy,
            "t": t_sec,
        })
        frame_idx += 1

    cap.release()

    if not raw_flags:
        return []

    n = len(raw_flags)
    frame_dur = 1.0 / fps
    video_end = raw_flags[-1]["t"] + frame_dur
    cut_idx = [k for k, fl in enumerate(raw_flags) if fl["scene_cut"]]

    segments: list[dict[str, float]] = []

    if cut_idx:
        # Scene mode: a hard cut splits the video into scenes; each scene long enough is one segment.
        bounds = [0] + cut_idx + [n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            t_start = raw_flags[a]["t"]
            t_end = raw_flags[b - 1]["t"] + frame_dur if b < n else video_end
            if t_end - t_start < min_len_s:
                continue
            motions = [raw_flags[k]["motion"] for k in range(a, b) if not raw_flags[k]["scene_cut"]]
            avg_motion = sum(motions) / len(motions) if motions else 0.0
            score = min(1.0, 0.7 + min(avg_motion, 1.0) * 0.3)
            segments.append({"t_start": round(t_start, 4), "t_end": round(t_end, 4), "score": round(score, 4)})
        return segments

    # Motion mode (no cuts): contiguous runs of frames with motion above threshold.
    i = 0
    while i < n:
        if raw_flags[i]["motion"] > MOTION_THRESHOLD:
            j = i
            motion_sum = 0.0
            while j < n and raw_flags[j]["motion"] > MOTION_THRESHOLD:
                motion_sum += raw_flags[j]["motion"]
                j += 1
            t_start = raw_flags[i]["t"]
            t_end = raw_flags[j - 1]["t"] + frame_dur
            if t_end - t_start >= min_len_s:
                avg_motion = motion_sum / (j - i)
                segments.append({"t_start": round(t_start, 4), "t_end": round(t_end, 4),
                                 "score": round(min(1.0, avg_motion), 4)})
            i = j
        else:
            i += 1
    return segments


def detect_track(
    path: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect and track moving objects in a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_process: set[int] = set()
    if segments:
        for seg in segments:
            start_f = int(seg["t_start"] * fps)
            end_f = min(int(seg["t_end"] * fps), total_frames - 1)
            for fi in range(max(0, start_f), end_f + 1):
                frames_to_process.add(fi)
    else:
        frames_to_process = set(range(total_frames))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    tracks: dict[int, dict[str, Any]] = {}
    next_track_id = 0
    frame_idx = 0
    max_distance = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        if frame_idx in frames_to_process:
            fg_mask = bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            active_centroids: list[tuple[int, int, float]] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    active_centroids.append((cx, cy, area))

            used_tracks: set[int] = set()
            for cx, cy, area in active_centroids:
                best_id = -1
                best_dist = max_distance
                for tid, trk in tracks.items():
                    if tid in used_tracks:
                        continue
                    dist = math.hypot(cx - trk["centroid"][0], cy - trk["centroid"][1])
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid
                if best_id >= 0:
                    tracks[best_id]["centroid"] = (cx, cy)
                    tracks[best_id]["last_seen"] = t_sec
                    tracks[best_id]["max_count"] = max(tracks[best_id]["max_count"], int(area))
                    used_tracks.add(best_id)
                else:
                    tracks[next_track_id] = {
                        "centroid": (cx, cy),
                        "first_seen": t_sec,
                        "last_seen": t_sec,
                        "max_count": int(area),
                    }
                    next_track_id += 1
        frame_idx += 1

    cap.release()

    result: list[dict[str, Any]] = []
    for tid, trk in tracks.items():
        if trk["last_seen"] > trk["first_seen"]:
            result.append({
                "track_id": tid,
                "cls": "moving_object",
                "first_seen": round(trk["first_seen"], 4),
                "last_seen": round(trk["last_seen"], 4),
                "max_count": trk["max_count"],
            })
    return result


def probe_duration(path: str) -> float:
    """Return the duration of a video file in seconds, or 0.0 if unknown."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frames and frames > 0:
            return float(frames / fps)
        return 0.0
    finally:
        cap.release()


def estimate_camera_motion(
    path: str,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_pairs: int = 60,
) -> float:
    """Median global inter-frame translation, in pixels at the analysis width.

    Frames are downscaled to at most ``MAX_ANALYSIS_WIDTH`` pixels wide,
    converted to grayscale float32 and compared with ``cv2.phaseCorrelate``
    (Hanning window from ``cv2.createHanningWindow``). At most ``max_pairs``
    consecutive frame pairs, spread evenly over ``[start_s, end_s]``, are
    measured. Returns 0.0 when fewer than two frames are readable.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_f = max(0, int(start_s * fps))
        if end_s is not None:
            end_f: int | None = int(end_s * fps)
            if total > 0:
                end_f = min(end_f, total - 1)
        else:
            end_f = total - 1 if total > 0 else None

        max_pairs = max(1, int(max_pairs))
        chosen: set[int] | None
        if end_f is None:
            # Unknown length: measure the first max_pairs pairs.
            chosen = None
        else:
            n_avail = end_f - start_f
            if n_avail <= 0:
                chosen = set()
            elif n_avail <= max_pairs:
                chosen = set(range(start_f, end_f))
            else:
                chosen = {int(round(v)) for v in
                          np.linspace(start_f, end_f - 1, max_pairs)}

        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_idx = start_f
        prev: np.ndarray | None = None
        window: np.ndarray | None = None
        mags: list[float] = []

        while True:
            if end_f is not None and frame_idx > end_f:
                break
            if chosen is None and len(mags) >= max_pairs:
                break
            ret, frame = cap.read()
            if not ret:
                break
            small, _ = _shrink(frame)
            gray = _gray32(small)
            if prev is not None and prev.shape == gray.shape:
                if chosen is None or (frame_idx - 1) in chosen:
                    if window is None or window.shape != gray.shape:
                        window = cv2.createHanningWindow(
                            (gray.shape[1], gray.shape[0]), cv2.CV_32F
                        )
                    (dx, dy), _resp = cv2.phaseCorrelate(prev, gray, window)
                    mags.append(math.hypot(dx, dy))
            prev = gray
            frame_idx += 1
            if chosen is not None and chosen and frame_idx - 1 >= max(chosen) + 1:
                break

        if not mags:
            return 0.0
        return float(np.median(mags))
    finally:
        cap.release()


def track_detections(
    path: str,
    start_s: float = 0.0,
    end_s: float | None = None,
    min_area: float = 50.0,
    max_distance: float = 50.0,
    warmup_frames: int = 50,
    max_rows: int = 200000,
    min_hits: int = 5,
    max_age: int = 8,
    var_threshold: float = 32.0,
    stabilize: bool = False,
) -> list[dict[str, Any]]:
    """Per-frame moving-object detections with track ids and boxes.

    Uses the same method as :func:`detect_track` (MOG2 background subtraction,
    morphological clean-up, contour extraction, nearest-centroid tracking) but
    returns one row per detection instead of one summary per track, so that
    downstream code can store bounding boxes and timestamps.

    Frames are analysed at no more than ``MAX_ANALYSIS_WIDTH`` pixels wide and
    boxes are scaled back to the original frame. A track is reported only
    after ``min_hits`` detections (shorter ones are treated as noise) and
    survives ``max_age`` consecutive missed frames.

    ``var_threshold`` is passed to MOG2 as ``varThreshold``. When
    ``stabilize`` is True, the global translation between consecutive
    analysis-size frames is measured with ``cv2.phaseCorrelate``, accumulated
    and cancelled with ``cv2.warpAffine`` (``BORDER_REPLICATE``) before
    background subtraction; boxes are mapped back to the unstabilised frame.

    ``confidence`` is a heuristic blob-size score in [0.5, 1.0] (larger
    foreground blobs score higher). It is NOT a calibrated probability.
    The class label is always ``moving_object``: this method detects motion,
    it does not classify what is moving.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
        frame_area = float(width * height)
        start_f = max(0, int(start_s * fps))
        end_f = None if end_s is None else int(end_s * fps)

        warm_from = max(0, start_f - warmup_frames)
        if warm_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, warm_from)
        frame_idx = warm_from

        bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=var_threshold, detectShadows=True
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # A larger closing kernel fuses the fragments of one moving body
        # (a bus or a person's limbs) into a single blob.
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tracks: dict[int, tuple[int, int]] = {}
        last_hit: dict[int, int] = {}
        hits: dict[int, int] = {}
        next_id = 0
        rows: list[dict[str, Any]] = []
        frame_area_small: float | None = None

        # Stabilisation state (used only when stabilize is True).
        prev_gray: np.ndarray | None = None
        stab_window: np.ndarray | None = None
        cum_dx = 0.0
        cum_dy = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if end_f is not None and frame_idx > end_f:
                break
            frame, scale = _shrink(frame)
            if frame_area_small is None:
                frame_area_small = float(frame.shape[0] * frame.shape[1])
            if stabilize:
                gray = _gray32(frame)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    if stab_window is None or stab_window.shape != gray.shape:
                        stab_window = cv2.createHanningWindow(
                            (gray.shape[1], gray.shape[0]), cv2.CV_32F
                        )
                    (dx, dy), _resp = cv2.phaseCorrelate(prev_gray, gray, stab_window)
                    cum_dx += dx
                    cum_dy += dy
                prev_gray = gray
                warp = np.float32([[1.0, 0.0, -cum_dx], [0.0, 1.0, -cum_dy]])
                frame = cv2.warpAffine(
                    frame, warp, (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
            mask = bg.apply(frame)
            # MOG2 marks shadows as 127; keep only confident foreground.
            mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
            # The very first frame only initialises the background model (all
            # of it reads as foreground), so it never yields detections.
            if frame_idx < start_f or frame_idx == warm_from:
                frame_idx += 1
                continue
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, merge_kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            t_sec = frame_idx / fps
            used: set[int] = set()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Blobs covering most of the frame are a scene cut or a global
                # lighting change, not an object.
                if area < min_area or area > 0.5 * frame_area_small:
                    continue
                m = cv2.moments(cnt)
                if m["m00"] <= 0:
                    continue
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                best_id, best_dist = -1, max_distance
                for tid, (px, py) in tracks.items():
                    if tid in used or frame_idx - last_hit[tid] > max_age:
                        continue
                    d = math.hypot(cx - px, cy - py)
                    if d < best_dist:
                        best_dist, best_id = d, tid
                if best_id < 0:
                    best_id = next_id
                    next_id += 1
                tracks[best_id] = (cx, cy)
                last_hit[best_id] = frame_idx
                hits[best_id] = hits.get(best_id, 0) + 1
                used.add(best_id)
                x, y, w, h = cv2.boundingRect(cnt)
                conf = 0.5 + 0.5 * min(1.0, area / (0.01 * frame_area_small))
                if stabilize:
                    # Map the box from stabilised back to original coordinates.
                    x = x + cum_dx
                    y = y + cum_dy
                inv = 1.0 / scale
                x, y, w, h = (int(round(v * inv)) for v in (x, y, w, h))
                rows.append({
                    "track_id": f"trk_{best_id}",
                    "class_label": "moving_object",
                    "confidence": round(conf, 4),
                    "timestamp_s": round(t_sec, 4),
                    "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                })
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
            frame_idx += 1
        # Drop short-lived tracks: noise, not objects.
        keep = {t for t, n in hits.items() if n >= min_hits}
        return [r for r in rows if int(r["track_id"][4:]) in keep]
    finally:
        cap.release()
