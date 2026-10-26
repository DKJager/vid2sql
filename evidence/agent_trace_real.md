# Agent trace report

plan = OpenCV 5 camera-motion estimate chose the detector settings; review = OpenCV 5 detections were judged;
retry = second detector call; flag_review = human approval request

## Video vid_391dfcadf61d40db: clip1_slovenska.webm

### Segment seg_e91b49ffbf574998 (0.00 s - 3.04 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 3.04 | plan | default | Camera motion 1.83 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=1.8291 |
| 0.00 | 3.04 | review | accept | 55 tracks, peak 28 simultaneous, short-track fraction 0.35; accepted. | duration_s=3.04, median_track_s=0.76, n_tracks=55, peak_simultaneous=28, short_track_fraction=0.3455 |

### Segment seg_bd2be058825c41a7 (3.04 s - 18.64 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 3.04 | 18.64 | plan | default | Camera motion 0.21 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.2055 |
| 3.04 | 18.64 | review | accept | 140 tracks, peak 22 simultaneous, short-track fraction 0.29; accepted. | duration_s=15.6, median_track_s=0.88, n_tracks=140, peak_simultaneous=22, short_track_fraction=0.2857 |

## Video vid_3b1bbe835ebb471a: clip2_shibuya.webm

### Segment seg_f38332ddacaf4947 (0.08 s - 45.76 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 0.08 | 45.76 | plan | default | Camera motion 0.56 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.5627 |
| 0.08 | 45.76 | review | flag_review | Peak of 51 simultaneous tracks reaches 40; counts are unreliable in dense scenes. | duration_s=45.68, median_track_s=1.36, n_tracks=671, peak_simultaneous=51, short_track_fraction=0.1908 |

### Segment seg_c16a7a1a96cf49bd (45.76 s - 52.00 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 45.76 | 52.00 | plan | default | Camera motion 0.05 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.0518 |
| 45.76 | 52.00 | review | accept | 73 tracks, peak 21 simultaneous, short-track fraction 0.37; accepted. | duration_s=6.24, median_track_s=0.8, n_tracks=73, peak_simultaneous=21, short_track_fraction=0.3699 |

### Segment seg_90bc25466db24701 (52.00 s - 54.36 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 52.00 | 54.36 | plan | default | Camera motion 0.53 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.5342 |
| 52.00 | 54.36 | review | accept | 25 tracks, peak 14 simultaneous, short-track fraction 0.40; accepted. | duration_s=2.36, median_track_s=0.72, n_tracks=25, peak_simultaneous=14, short_track_fraction=0.4 |

### Segment seg_4dfa0d7071084478 (54.36 s - 58.56 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 54.36 | 58.56 | plan | default | Camera motion 0.04 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.0394 |
| 54.36 | 58.56 | review | retry_stabilized | 0.71 of 17 tracks are shorter than 0.5 s; retrying once with stabilisation. | duration_s=4.2, median_track_s=0.36, n_tracks=17, peak_simultaneous=7, short_track_fraction=0.7059 |
| 54.36 | 58.56 | retry | keep_first | First pass produced 17 tracks; stabilised retry produced 32 tracks. | first: duration_s=4.2, median_track_s=0.36, n_tracks=17, peak_simultaneous=7, short_track_fraction=0.7059 ; retry: duration_s=4.2, median_track_s=0.28, n_tracks=32, peak_simultaneous=12, short_track_fraction=0.8438 |
| 54.36 | 58.56 | review | flag_review | 0.71 of 17 tracks are shorter than 0.5 s even after stabilisation; tracks are still fragmenting. | duration_s=4.2, median_track_s=0.36, n_tracks=17, peak_simultaneous=7, short_track_fraction=0.7059 |

## Video vid_e6d450ed50d64e0f: clip3_cobbled.webm

### Segment seg_d143e51b51e0432c (0.00 s - 2.40 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 2.40 | plan | default | Camera motion 0.43 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.4346 |
| 0.00 | 2.40 | review | accept | 39 tracks, peak 24 simultaneous, short-track fraction 0.33; accepted. | duration_s=2.4, median_track_s=0.56, n_tracks=39, peak_simultaneous=24, short_track_fraction=0.3333 |

### Segment seg_b0748d311e214860 (3.04 s - 13.36 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 3.04 | 13.36 | plan | default | Camera motion 0.23 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.2337 |
| 3.04 | 13.36 | review | accept | 92 tracks, peak 29 simultaneous, short-track fraction 0.25; accepted. | duration_s=10.32, median_track_s=0.9, n_tracks=92, peak_simultaneous=29, short_track_fraction=0.25 |

### Segment seg_24e7532b75ab4ddd (13.64 s - 16.32 s)

| start_s | end_s | step | decision | reason | observation |
| --- | --- | --- | --- | --- | --- |
| 13.64 | 16.32 | plan | default | Camera motion 0.09 px is below 6.0 px, so the detector runs with default settings. | camera_motion_px=0.088 |
| 13.64 | 16.32 | review | retry_stabilized | 0.71 of 24 tracks are shorter than 0.5 s; retrying once with stabilisation. | duration_s=2.68, median_track_s=0.38, n_tracks=24, peak_simultaneous=16, short_track_fraction=0.7083 |
| 13.64 | 16.32 | review | flag_review | 0.71 of 24 tracks are shorter than 0.5 s even after stabilisation; tracks are still fragmenting. | duration_s=2.68, median_track_s=0.38, n_tracks=24, peak_simultaneous=16, short_track_fraction=0.7083 |
| 13.64 | 16.32 | retry | keep_first | First pass produced 24 tracks; stabilised retry produced 24 tracks. | first: duration_s=2.68, median_track_s=0.38, n_tracks=24, peak_simultaneous=16, short_track_fraction=0.7083 ; retry: duration_s=2.68, median_track_s=0.3, n_tracks=24, peak_simultaneous=15, short_track_fraction=0.6667 |

## Review queue (human approval requests)

| review_id | video | segment | start_s | end_s | reason | status | created_at_utc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rev_26fd85c5736b4232 | vid_3b1bbe835ebb471a | seg_f38332ddacaf4947 | 0.08 | 45.76 | Peak of 51 simultaneous tracks reaches 40; counts are unreliable in dense scenes. | pending | 2026-10-20T16:25:32Z |
| rev_f84530171e2e423e | vid_3b1bbe835ebb471a | seg_4dfa0d7071084478 | 54.36 | 58.56 | 0.71 of 17 tracks are shorter than 0.5 s even after stabilisation; tracks are still fragmenting. | pending | 2026-10-20T16:26:10Z |
| rev_5670942ccefd44ce | vid_e6d450ed50d64e0f | seg_24e7532b75ab4ddd | 13.64 | 16.32 | 0.71 of 24 tracks are shorter than 0.5 s even after stabilisation; tracks are still fragmenting. | pending | 2026-10-20T16:27:47Z |
