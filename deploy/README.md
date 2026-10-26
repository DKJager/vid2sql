# vid2sql on AWS (deployed stack vid2sql-dev2, us-east-1)

The stack was deployed with SAM from a deployed variant of the template
(not included in this public tree: it pins a pre-built image digest in a private
registry). The buildable template is `deploy/template.yaml`; it produces the same
resources from `deploy/Dockerfile` and the Lambda entry points.

Public endpoint (judge access, read-only, throttled):
https://kb7r2uzvgc.execute-api.us-east-1.amazonaws.com/ (AWS stack endpoint; answers 503 'no database yet' until a video is uploaded). A temporary demo with data runs at https://doll-vehicles-creativity-into.trycloudflare.com/ (Cloudflare Quick Tunnel, no uptime guarantee).

Before the first upload it answers `503` with the message
`no database yet: upload a video first`.

## What is deployed

| Resource | Purpose |
|---|---|
| Upload bucket (S3, private, AES256) | videos go under `uploads/`; objects expire after 14 days |
| PipelineFunction (container-image Lambda, arm64, 3008 MB memory, 900 s timeout, 4096 MB ephemeral storage, reserved concurrency 1) | on each S3 upload event runs the OpenCV 5 agent loop and writes rows into `db/v2s.db` in the results bucket |
| Results bucket (S3, private, versioned) | holds the single SQLite database `db/v2s.db` |
| QueryFunction + HTTP API (throttle burst 20, rate 10 per second) | read-only endpoint: plain questions, SQL box, `/clip/<video_id>?t=SECONDS` links |
| Optional S3 object metadata keys on upload | `license`, `source-url`, `recording-start-utc`; see timestamp wording below |

The image is built for arm64 from `deploy/Dockerfile` (dependencies pinned in
`deploy/requirements-lambda.txt`). No secrets or API keys are used. The query
endpoint has no authorizer; it is a public read-only demo.

Event times are absolute UTC: `events.start_timestamp_utc = clip recording start + in-video offset (segments.start_s)`. The recording start comes from the clip metadata (`recording_start_utc`: optional CSV column in `data/CLIPS.csv`, or S3 object metadata key `recording-start-utc` on upload); when a clip has none, the pipeline anchors at `videos.processed_at_utc`, sets `videos.recording_start_utc = videos.processed_at_utc` and logs a WARNING, so those event times are processing-relative; `process_clip(..., require_recording_start=True)` refuses such clips instead. The three bundled Wikimedia clips carry no recording time, so their event times are processing-relative; in-video offsets (segments, objects) are exact.

## Files in this folder

`template.yaml` — buildable SAM template; build with `sam build` using
`deploy/Dockerfile`.

`Dockerfile` — the container recipe (arm64, multi-stage, no Docker-in-Docker).

`requirements-lambda.txt` — pinned Lambda dependencies.

`lambda_query.py`, `lambda_pipeline.py` — Lambda entry points, covered by
`tests/test_lambda_handlers.py`.

## Reproduce in your own account

1. Install the AWS SAM CLI and Docker.  On an x86 build host you must
   enable arm64 emulation **before** `sam build`:
   `docker run --privileged --rm tonistiigi/binfmt --install arm64`

2. From the repository root, validate:
   `sam validate --template deploy/template.yaml --lint`

3. Build:
   `sam build --template deploy/template.yaml`

4. Deploy:
   `sam deploy --guided --stack-name vid2sql`
   Use a lowercase stack name (it is used to derive bucket names).
   Accept the other defaults; allow SAM to create the image repository
   when asked.
   If the deploy is rejected over `ReservedConcurrentExecutions`, delete
   that line from `deploy/template.yaml` and run `sam build` / `sam deploy`
   again.  The consequence is that concurrent uploads can race on the
   single-writer `db/v2s.db`.

5. Note the stack outputs: `WebEndpoint`, `UploadBucketName`.

6. Upload a clip with the three optional metadata keys:
   ```
   aws s3 cp data/clips/clip3_cobbled.webm \
     s3://<UploadBucketName>/uploads/ \
     --metadata '{"license":"CC BY 3.0","source-url":"https://commons.wikimedia.org/wiki/File:Traffic_on_a_cobbled_street_-_soundscape.webm","recording-start-utc":"2026-01-01T00:00:00Z"}'
   ```

7. Smoke-test the endpoint:
   - `GET /` — returns a short OK message.
   - `POST /query` with body `{"sql": "SELECT COUNT(*) FROM videos"}` —
     returns rows after the first upload finishes.
   - `POST /query` with body `{"question": "What needs review?"}` —
     returns `503` before the first upload, rows from `review_queue`
     afterwards.

## Operating notes

- **Cold start:** the QueryFunction downloads the database from S3 on first
  request; it is cached in the module for the lifetime of the warm container.
- **Concurrency:** the pipeline function has reserved concurrency 1 because
  it read-modify-writes a single SQLite file in S3.
- **Throttle:** the HTTP API allows a burst of 20 requests, sustained 10/s.
- **Public endpoint:** read-only, no authorizer; delete the stack after
  judging with `sam delete --stack-name vid2sql-dev2`.
- **Upload expiry:** objects under `uploads/` are deleted after 14 days.

## Known limits

- Moving objects only; a stationary person is not detected.  No
  classification — all detections are labelled `moving_object`.
- Track counts are inflated by fragmentation and are not head counts.
- Handheld or shaky footage creates false tracks.  Camera motion on clip1
  stayed below 6 px, so stabilisation was not triggered; tracks were
  unchanged (159/159).
- The scene-cut detector is crude and reports false cuts on continuous
  clips.
- The retry step (stabilise again) did not improve the two fragmenting
  segments it was tried on; the trace shows `keep_first`.
- Crowd undercount: synthetic crowd truth 45, agent reported fewer tracks.
- The `transcript` table is always empty (no speech component).
- No evaluation on labelled real footage.
- Lambda performance is untested.
- The query endpoint is unauthenticated and public.
- SQLite in S3 means a single writer at a time; concurrent uploads can
  corrupt the database if reserved concurrency is removed.

## Responsible use

Only licensed footage with recorded licence and source is processed.  No
faces, identities, licence plates or biometrics are extracted or stored
(class-less `moving_object` boxes only; no frames are saved).  Uploads
expire after 14 days.  The endpoint is read-only and throttled.  The agent
flags dense or fragmenting scenes for human review instead of guessing.
Do not use for surveillance of individuals.  Fictional and synthetic
fixtures are labelled as such.
