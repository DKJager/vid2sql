# vid2sql: offline tests of the Lambda entry points with an in-memory S3 fake.
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cv2 = pytest.importorskip("cv2")

from deploy import lambda_pipeline, lambda_query  # noqa: E402


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def download_file(self, bucket, key, dest):
        if (bucket, key) not in self.objects:
            err = type("E", (Exception,), {})()
            err.response = {"Error": {"Code": "404"}}
            raise err
        Path(dest).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, src, bucket, key):
        self.objects[(bucket, key)] = Path(src).read_bytes()

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            err = type("E", (Exception,), {})()
            err.response = {"Error": {"Code": "404"}}
            raise err
        data = self.objects[(Bucket, Key)]
        return {"ETag": str(hash(data)), "Metadata": {"license": "CC0"}}


@pytest.fixture
def video_bytes(tmp_path) -> bytes:
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(90):
        f = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(f, (20 + 4 * i, 100), (50 + 4 * i, 130), (255, 255, 255), -1)
        w.write(f)
    w.release()
    return path.read_bytes()


@pytest.fixture
def s3(monkeypatch, tmp_path, video_bytes):
    fake = FakeS3()
    fake.objects[("up", "uploads/car.mp4")] = video_bytes
    monkeypatch.setenv("RESULTS_BUCKET", "res")
    monkeypatch.setattr(lambda_pipeline, "_s3", lambda: fake)
    monkeypatch.setattr(lambda_query, "_s3", lambda: fake)
    monkeypatch.setattr(lambda_query, "LOCAL_DB", str(tmp_path / "q.db"))
    monkeypatch.setattr(lambda_query, "_state", {"etag": None, "client": None})
    monkeypatch.setenv("V2S_DB_PATH", str(tmp_path / "q.db"))
    return fake


def _event(path, method="GET", body=None):
    ev = {"rawPath": path, "rawQueryString": "", "requestContext": {"http": {"method": method}},
          "headers": {"content-type": "application/json"}}
    if body is not None:
        ev["body"] = json.dumps(body)
    return ev


def test_upload_event_runs_the_pipeline_and_publishes_the_database(s3):
    ev = {"Records": [{"s3": {"bucket": {"name": "up"}, "object": {"key": "uploads/car.mp4"}}}]}
    out = lambda_pipeline.handler(ev)
    assert out["processed"][0]["counts"]["objects"] >= 1
    assert ("res", "db/v2s.db") in s3.objects


def test_second_upload_appends_to_the_same_database(s3):
    s3.objects[("up", "uploads/car2.mp4")] = s3.objects[("up", "uploads/car.mp4")]
    for k in ("uploads/car.mp4", "uploads/car2.mp4"):
        lambda_pipeline.process_object("up", k, s3)
    resp = lambda_query.handler(_event("/query", "POST", {"question": "How many moving objects are in each video?"}))
    rows = json.loads(resp["body"])["rows"]
    assert {r["filename"] for r in rows} == {"car.mp4", "car2.mp4"}


def test_query_endpoint_answers_with_the_stored_uri(s3):
    lambda_pipeline.process_object("up", "uploads/car.mp4", s3)
    resp = lambda_query.handler(_event("/query", "POST", {"sql": "SELECT uri, license FROM videos"}))
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["rows"] == [{"uri": "s3://up/uploads/car.mp4", "license": "CC0"}]


def test_query_endpoint_stays_read_only(s3):
    lambda_pipeline.process_object("up", "uploads/car.mp4", s3)
    resp = lambda_query.handler(_event("/query", "POST", {"sql": "DELETE FROM videos"}))
    assert resp["statusCode"] == 400


def test_query_before_any_upload_is_a_clear_503(s3):
    resp = lambda_query.handler(_event("/query", "POST", {"sql": "SELECT 1"}))
    assert resp["statusCode"] == 503


def test_non_video_objects_are_skipped(s3):
    assert "skipped" in lambda_pipeline.process_object("up", "uploads/notes.txt", s3)
    assert ("res", "db/v2s.db") not in s3.objects
