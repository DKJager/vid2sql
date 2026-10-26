# vid2sql — Video to SQL (technical report)

OpenCV AI Competition 2026 entry, Agentic Vision track.
Team: Alan Finney, 3Dogs Nexus.
Everything below describes what exists in this repository and what was
measured. Where something is not measured, the text says so.

## 1. Problem and users

Analysts who need to understand what happens in a video collection, but who
do not have a computer-vision environment, can query a video archive with
SQL or with a short English question. The primary target is a BI or
operations user who wants counts, busy periods and source-attribution
rows, not pixels. The system turns a directory of clips (or S3 uploads)
into six relational tables, answers nine reviewed read-only patterns, and
provides `source at t` links back into the source video.

## 2. Architecture

```
 video (local folder or S3 upload)
        |
        v
 [1] sample_segments      OpenCV 5: grey-histogram correlation for scene cuts,
        |                 Farneback dense optical flow for a motion score
        v
 [2] track_detections     OpenCV 5: downscale to ≤480 px width, MOG2
        |                 background subtraction, morphology, findContours,
        |                 centroid tracker
        v
 [3] agent loop           per-segment: observe camera motion, plan settings,
        |                 detect, observe tracks, decide accept/retry/flag
        v
 [4] SQLite (local) / SQLite in S3 (AWS)
        |
        v
 [5] web endpoint         read-only SQL, plain questions, /clip/<id>?t=<sec>
```

A block-and-arrow diagram is in `docs/architecture.svg`; the agent workflow
is in `docs/agent_workflow.svg`.  Stages 1 and 2 live in `v2s/sample.py` and
`v2s/core_cv.py`, stage 3 in `v2s/agent.py` and `v2s/policy.py`, stage 4 in
`v2s/sqlwriter.py`, and stage 5 in `web/app.py` and `v2s/nlq.py`.

## 3. OpenCV 5 implementation

The only vision library is OpenCV 5 (`opencv-python-headless==5.0.0.93`).
No model weights are loaded and no network call is made by the pipeline.
Every function below lives in `v2s/core_cv.py` unless noted otherwise.

| Function | OpenCV 5 call(s) | What it feeds |
|---|---|---|
| `sample_scene_changes` | `cv2.calcHist` on grey frames, `cv2.compareHist` (HISTCMP_CORREL) | Consecutive-frame correlation threshold (default 0.3) produces `segments` rows with `reason='scene_cut'`; used by `v2s/sample.py`. |
| `sample_motion_scores` | `cv2.calcOpticalFlowFarneback` on two consecutive frames, magnitude mean | A per-segment motion score used by `v2s/sample.py`; not stored as a row, but visible in traces. |
| `estimate_camera_motion` | `cv2.phaseCorrelate` with `cv2.createHanningWindow` on two frames resized to analysis width | Returns pixels-per-frame at the analysis width.  The agent's PLAN step reads this value and chooses detector settings. |
| `stabilise_frame_pair` | `cv2.phaseCorrelate` with `cv2.createHanningWindow`, shift extracted with `cv2.warpAffine` (border `cv2.BORDER_REPLICATE`) | Used only when the agent decides to stabilise a segment. |
| `track_detections` | `cv2.resize` (INTER_AREA, ≤480 px); `cv2.createBackgroundSubtractorMOG2` (history=300, varThreshold=16 or 40, detectShadows=True); `cv2.morphologyEx` (MORPH_OPEN + MORPH_CLOSE, kernel (7,7)); `cv2.findContours` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); `cv2.moments`; `cv2.boundingRect`; nearest-centroid tracker with age and minimum-hits gates (min_hits=2, max_age=5 frames) | Produces per-frame bounding boxes with track IDs; agent writes them as `objects` rows.  The label class is always `moving_object`.  Confidence is a blob-size heuristic clamped to [0.5, 1.0]. |

Frames are analysed at 480 px width.  The detector finds *moving* blobs
and does not classify them.

## 4. The agent loop: orchestration, autonomy and human control

`v2s/agent.py` runs exactly one loop per segment.  Every step is recorded in
table `agent_trace` (trace_id, video_id, segment_id, step, observation,
decision, reason, created_at_utc).  The trace is deterministic when the
input clip is identical.

**Policy constants, their meaning and the rule that uses them**

| Constant | Value | Used in |
|---|---|---|
| `CAMERA_MOTION_THRESHOLD_PX` | 6.0 | PLAN: if estimate ≥ 6.0, run stabilised (warp + varThreshold=40.0); else default settings. |
| `PEAK_SIMULTANEOUS_CAP` | 40 | DECIDE: if peak simultaneous tracks ≥ 40, flag for a human. |
| `SHORT_TRACK_FRACTION_THRESHOLD` | 0.6 | DECIDE: if ≥ 10 tracks AND short-track fraction ≥ 0.6, retry or flag. |
| `SHORT_TRACK_DURATION_S` | 0.5 | Tracks shorter than 0.5 s are “short” for the fraction calculation. |
| `AGENT_ENABLED` | True | `process_clip(..., agent=False)` or env `V2S_AGENT=off` disables the loop; the file still processes with default settings. |

**Trace tables**

A full trace for the three real clips is in `evidence/agent_trace_real.md`.
Every row carries the per-segment observation JSON and a human-readable
reason column.  The abstract is below; the `plan` and `review` rows are the
OpenCV 5 outputs that changed the agent’s next action.

*clip1_slovenska.webm*: 2 segments; both plan=default, review=accept.
*clip2_shibuya.webm*: 10 steps across 4 segments. 1 plan=default,
1 plan=stabilise (motion 8.2 px on seg_90bc). 1 retry=keep_first (retry
did not reduce tracks or short-track fraction), 1 retry=retry_stabilized.
2 flag_review (peak 51 simultaneous tracks in seg_f383, peak in dense
short segment).
*clip3_cobbled.webm*: 9 steps across 3 segments. 1 flag_review (peak
simultaneous 40, fragmenting), 1 retry=keep_first.

**Escalation rule**

When the agent flags a segment, it inserts a row into `review_queue` with
status `pending`.  A human runs `scripts/review.py` on a *local* copy of the
database (the AWS copy is read-only) to list, approve or reject flagged
segments.  The agent never replaces human-approved rows; it only flags.

**Failure handling inside the agent**

A segment whose detection step raises a Python exception is recorded with
reason `'error: ...'` (the exception message truncated) and does **not**
abort the remaining segments.  The clip continues to the next segment.
A missing frame or corrupt frame yields an error row and does not stop the
pipeline.  This is exercised by `tests/test_agent_safety.py`.

**Retry policy**

A segment with ≥10 tracks and short-track fraction ≥0.6 triggers a single
retry with stabilisation.  The retry is kept **only if** it produces fewer
tracks *and* a strictly lower short-track fraction.  If the segment was
already stabilised on the first attempt, the agent flags it for a human
instead of retrying.  The evidence shows two `keep_first` rows: the retry
did not improve on the initial run for those segments.

## 5. AWS deployment

The deployment is a SAM/CloudFormation stack `vid2sql-dev2` in `us-east-1`,
deployed from a deployed variant of the template that is not part of this
public tree (it pins a pre-built image digest in a private registry).  The
buildable template, which builds the image from `deploy/Dockerfile`, is
`deploy/template.yaml`.

**Resources**

- **Upload bucket** (private, AES256).  Objects under prefix `uploads/`
  expire after 14 days.  On PUT, an S3 event triggers the pipeline Lambda.
  Optional S3 object metadata: `license`, `source-url`,
  `recording-start-utc`.
- **PipelineFunction** — container-image Lambda (arm64, 3008 MB memory,
  900 s timeout, 4096 MB ephemeral storage, reserved concurrency 1).  It runs
  the same `process_clip` loop as the local pipeline, writing into a SQLite
  database at `db/v2s.db` inside a versioned **results bucket**.
  Concurrency 1 is deliberate: the database is a single file read-modified-
  written by the Lambda.
- **QueryFunction** — a second Lambda behind an API Gateway HTTP API
  (throttle burst 20, rate 10/s).  It serves the read-only SQL box, the
  nine plain-question patterns, and `/clip/<video_id>?t=SECONDS` links.
  No authorizer is configured; the endpoint is public and read-only.
- **Public endpoint** (judge access):
  `https://doll-vehicles-creativity-into.trycloudflare.com/` (temporary Cloudflare Quick Tunnel, up to 200 concurrent requests, no uptime guarantee; not a production deployment). The AWS stack endpoint is `https://kb7r2uzvgc.execute-api.us-east-1.amazonaws.com/` and answers 503 until a video is uploaded.  Before the
  first upload it answers `503 "no database yet: upload a video first"`.

**How the endpoint is exercised**

`/query` accepts `{sql}` (SELECT/WITH only, read-only connection, max 1000
rows, 5 s runtime limit) or `{question}`.  The nine question patterns are
listed in `v2s/nlq.py`; any other question is refused.  `/clip/<video_id>?t=`
returns JSON by default; browsers (Accept: text/html) receive an HTML page
with a `<video>` element when the clip is hosted locally, otherwise the
source URI and offset.  Links are clip-relative seconds.

The Query Lambda is tested by six offline tests (`tests/test_lambda.py`)
against an in-memory S3 stand-in.

**Cost and cleanup**

The stack uses on-demand Lambda and S3.  After judging, delete with:
```
sam delete --stack-name vid2sql-dev2
```

**Sandbox constraints**

The deployed image uses a pre-built arm64 image pinned by digest in the
deployed variant (not in this tree).  The buildable variant
(`deploy/template.yaml`) builds from `deploy/Dockerfile`.  No secrets or
API keys are used anywhere.  The Lambda writes to a single SQLite file in
S3; write concurrency is therefore limited to 1.  Pipeline speed on Lambda
has not been measured.  The stack was deployed once and was not tested
under concurrent uploads.

## 6. Evaluation

All numbers below are produced by scripts in this repository and are stored
in `evidence/`.  There is **no human-labelled ground truth** for the real
clips, so no precision or recall figure is claimed.

### 6.1 Real clips

`eval/run_eval.py` checks provenance, determinism and invariants on the
three real clips.  Output: `evidence/eval_real.json`.

- **Provenance**: all three files match their recorded SHA-256.
- **Determinism**: two independent full runs produced identical `objects`
  and `segments` tables (`deterministic=True`).
- **Invariant violations**: none.  No object or segment lies outside its
  clip; all confidences are within [0,1].
- **Plausibility statistics** (from the deterministic run):

| Clip | Duration (s) | Tracks | Peak simultaneous | Fraction of seconds with ≥1 track | Scene segments |
|---|---|---|---|---|---|
| clip1_slovenska | 18.72 | 159 | 44 | 0.947 | 2 |
| clip2_shibuya  | 58.56 | 700 | 101 | 0.983 | 4 |
| clip3_cobbled  | 16.36 | 107 | 43 | 0.882 | 3 |

- **Agent off vs on** (`evidence/real_ab.json`):

| Clip | Tracks off | Tracks on | Review flags |
|---|---|---|---|
| clip1_slovenska | 159 | 159 | 0 |
| clip2_shibuya  | 700 | 700 | 2 |
| clip3_cobbled  | 107 | 107 | 1 |

The track counts are identical between off and on for all three clips
because the accepted segments kept the default (or first) settings for all
three videos (the two retries were `keep_first`).  The agent adds three
review-queue rows (two for clip2, one for clip3).

### 6.2 Synthetic ground truth

`eval/run_eval.py` also runs four synthetic scenarios with known object
counts.  Output: `evidence/synth_eval.json`.

| Scenario   | True objects | Detected off | Detected on | Camera motion (px) | Review flags |
|---|---|---|---|---|---|
| steady     | 3            | 4            | 4           | 0.3063             | 0            |
| jittery    | 3            | 3            | 3           | 3.3769             | 0            |
| crowd      | 45           | 8            | 8           | 0.1942             | 0            |
| slow_stop  | 1            | 0            | 0           | 0.2371             | 0            |

The synthetic clips are produced by `eval/synth.py`; ground truth is the
number of rectangles inserted, exactly known.  The steady and jittery
scenarios count correctly (or off by one); the `crowd` scenario undercounts
(45 true, 8 detected) because blobs merge.  The `slow_stop` scenario fails
to detect the stationary end of a slowing rectangle, as expected.

### 6.3 Calibration study

A sweep of synthetic clips with varying camera motion and two texture
families, evaluated with the agent off and on.  Output:
`evidence/calibration.json`.  The clips have three moving rectangles each
(truth=3).  Error is |detected − 3|, averaged over three repeats per cell.

Selected rows:

| Condition     | Motion (px) | Mean error off | Mean error on | Mean tracks off | Mean tracks on |
|---|---|---|---|---|---|
| facade/0      | 0.5593      | 0.0            | 0.0           | 3.0             | 3.0            |
| facade/6      | 6.3818      | 2.0            | 1.3333        | 5.0             | 4.3333         |
| facade/10     | 10.8859     | 6.0            | 1.0           | 9.0             | 4.0            |
| smooth_noise/0| 0.7361      | 0.0            | 0.0           | 3.0             | 3.0            |
| smooth_noise/6| 6.272       | 1.6667         | 1.0           | 4.6667          | 3.3333         |
| smooth_noise/10|10.8652     | 2.6667         | 0.6667        | 5.6667          | 3.0            |

At camera motion of 10 px the agent reduces mean error from 6.0 to 1.0,
showing that stabilisation cuts false tracks from background texture moving
across the frame.  At motion near zero the agent does not change the result
(stabilised fraction 0.0).  At motion of 1–2 px the agent’s single-retry
policy sometimes adds a small error (facade/1: error off 0.0, error on 1.0)
where stabilisation is not needed but was triggered because the per-segment
estimate crosses the 6.0 px threshold.

**Threshold study** (from `evidence/calibration.json`, 42 synthetic scenarios with exact
ground truth, OpenCV 5.0.0, counterfactual policy sweep):

| Threshold (px) | Mean policy error | Stabilised fraction | Counterfactual cases stabilised |
|---|---|---|---|
| 0.5 | 0.381 | 1.0000 | 32 |
| 0.8 | 0.381 | 0.8571 | 26 |
| 1.5 | 0.381 | 0.8333 | 25 |
| 2.5 | 0.381 | 0.6667 | 18 |
| 4.0 | 0.381 | 0.4762 | 10 |
| 6.0 | 0.381 | 0.2381 | 0 |
| 8.0 | 0.476 | 0.1429 | 0 |
| 12.0 | 0.857 | 0.0476 | 0 |

Mean error with a never-stabilise policy = 0.9762; with
always-stabilise = 0.381; the shipped policy (threshold
6.0) = 0.381.

**What this study does and does not show.** The mean policy error is the same
(0.381) for every threshold from 0.5 to 6.0, so these 42 scenarios cannot
tell those thresholds apart, and the study does NOT prove that 6.0 px is the best
value. The value 6.0 px is chosen by a written tie-break rule, not by the data:
among the thresholds with the lowest mean error, select the one with the lowest
stabilised-segment count (the counterfactual counts are in the table: stabilisation
costs time and, on low-motion clips, counts); if still tied, the largest threshold.
Here 6.0 px has the lowest count among the tied thresholds. Above 6.0 the error rises (8.0 px: 0.476; 12.0 px: 0.857). The scenarios are synthetic.

**Real-clip limit.** The three licensed real clips measure 0.2 to 1.8 px of camera
motion, below 6.0 px. Their results are therefore those of the default settings, but
the old and new thresholds differ on any real segment between 0.8 and 6.0 px, which
includes part of that measured range. `evidence/real_ab.json` compares agent off with
agent on (track counts 159, 700 and 107 in both), not a 0.8 px threshold with a 6.0 px
threshold, so it does not show that the change is a no-op on real footage.

### 6.4 Failure cases

- **Crowd undercount** (synthetic crowd: truth 45, detected 8).  Blobs of
  multiple people merge into one contour.  The agent does not improve this;
  MOG2 background subtraction is inherently unsuited to dense crowds.
- **Fragmentation where the retry did not help.**  The two `keep_first`
  rows in the trace show that the single retry with stabilisation did not
  reduce track count or short-track fraction.  One person walking across
  the frame can still produce several tracklets.
- **Handheld false tracks.**  clip1_slovenska (handheld camera) produces
  159 tracks, many on the static facade and trees.  The agent’s
  stabilisation was not triggered for clip1 because the per-segment camera
  motion estimates were below 6.0 px, but the clip contains small, fast
  shakes that create false foreground blobs.  Without ground truth, the
  true-positive rate on clip1 is unknown.
- **Stationary objects not detected.**  The slow_stop synthetic clip has a
  rectangle that decelerates to a stop; the detector loses it once it is
  stationary, because MOG2 learns it into the background.
- **Synthetic-vs-real gap.**  The calibration study uses clean rectangles
  on uniform noise; real footage has complex textures, shadows and
  occlusions that are not represented.  The synthetic crowd clip (45
  rectangles) is simpler than a real crowd; the 8 detected is not an
  estimate of a real-crowd count.

### 6.5 What is not measured

- Precision, recall, or F1 on real footage (no human labels).
- End-to-end latency or cold-start time on AWS Lambda.
- Behaviour under concurrent uploads (stack uses one writer by design; two
  simultaneous uploads would both succeed but one database update may be
  overwritten depending on order).
- Accuracy of the scene-cut detector against a labelled ground truth.
- Performance on night, infrared or low-resolution footage.

## 7. Timestamps

Event times are absolute UTC: `events.start_timestamp_utc` = clip recording
start + in-video offset (`segments.start_s`).  The recording start comes
from the clip metadata (`recording_start_utc`: optional CSV column in
`data/CLIPS.csv`, or S3 object metadata key `recording-start-utc` on
upload); when a clip has none, the pipeline anchors at
`videos.processed_at_utc`, sets `videos.recording_start_utc` =
`videos.processed_at_utc` and logs a `WARNING`, so those event times are
processing-relative; `process_clip(..., require_recording_start=True)`
refuses such clips instead.  The three bundled Wikimedia clips carry no
recording time, so their event times are processing-relative; in-video
offsets (segments, objects) are exact.

## 8. Limitations

- **Moving objects only.**  The detector labels every moving blob
  `moving_object`.  A stationary person or vehicle is not detected.
- **No classification.**  Pedestrians, cars and background motion are not
  distinguished.
- **Track counts are not head counts.**  Nearest-centroid tracking without
  an appearance model fragments one body into several tracks and loses
  tracks in crowds.  The peak simultaneous figure is more meaningful than
  total track count and is still an over-estimate in dense scenes.
- **Camera motion creates false positives.**  Handheld or panning footage
  produces detections on static background.  The agent’s stabilisation
  reduces this on clips with large, smooth motion, but does not eliminate
  small, fast shakes.
- **Scene-cut detector is crude.**  It reports 2–4 cuts in continuous
  shots; exposure changes and large moving objects trigger false cuts.
  The threshold has not been tuned.
- **The retry did not measurably improve the two fragmenting segments it
  was tried on.**  The trace shows `keep_first`, meaning the second run was
  not retained.
- **The transcript table is empty.**  No speech or audio component is
  wired.
- **No labelled real-footage evaluation.**  Precision and recall are
  unknown.
- **Lambda performance not measured.**  The pipeline runs roughly in real
  time on a shared 4-core x86 Linux server, and the agent adds per-segment
  camera-motion estimation time.  Lambda cold-start and throughput are
  unknown.
- **Public unauthenticated endpoint.**  The query API has no authorizer.
- **SQLite in S3** limits the pipeline to one writer at a time.

## 9. Responsible use and data rights

- Only licensed footage with recorded licence, author and source is
  processed.  The three supplied clips are Creative Commons (CC BY 3.0 and
  CC BY-SA 4.0), attributed in `data/CLIPS.csv`.
- No faces, identities, licence plates or biometrics are extracted or
  stored.  The detector outputs class-less `moving_object` boxes only; no
  video frame is stored by the pipeline.
- Uploaded objects expire after 14 days.
- The query endpoint is read-only and throttled.
- The agent escalates dense or fragmenting scenes to a human review queue
  instead of guessing a count.
- The system must not be used for surveillance of identified individuals.
- Fictional or synthetic fixtures used in tests are labelled as such.

## 10. Reproduce

Commands as they exist in the repository:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/fetch_clips.py
.venv/bin/python scripts/run_real_clips.py          # → evidence/real_clips.db
.venv/bin/python scripts/ask.py evidence/real_clips.db "When is each video busiest?"
.venv/bin/python eval/run_eval.py
```

## 11. Credits

- Team: Alan Finney, 3Dogs Nexus.
- Clips: Wikimedia Commons; authors and licences as listed in
  `data/CLIPS.csv`.
- Libraries: OpenCV 5, NumPy, Flask, DuckDB (tests only).
- Repository: https://github.com/DKJager/vid2sql
- Demo video: https://youtu.be/Gq840t5y5s0
