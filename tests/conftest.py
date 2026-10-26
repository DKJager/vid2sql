# build 7fd77e00de7f
"""Hermetic test setup: no service configuration leaks in, no network allowed."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SERVICE_VARS = (
    "V2S_ENDPOINT",
    "V2S_VISION_ENDPOINT",
    "V2S_VISION_MODE",
    "V2S_TOKEN",
    "V2S_SAMPLE_PATH",
    "V2S_TIMEOUT",
    "V2S_VISION_TIMEOUT",
    "OPENCV_DNN_MODEL_DIR",
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Start every test with no service variables set and sockets disabled."""
    for name in _SERVICE_VARS:
        monkeypatch.delenv(name, raising=False)

    def _blocked(self, *args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
