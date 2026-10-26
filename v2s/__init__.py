# build 7fd77e00de7f
"""vid2sql turns video into SQL tables.

Vision (scene cuts, motion, camera-motion estimate, background-subtraction
detection, tracking) runs locally on OpenCV 5 in ``v2s.core_cv``.
``v2s.agent`` runs the agent loop (observe, plan, detect, decide, human
review, ``v2s.policy``). Optional remote vision/reasoning services are used
only when ``V2S_VISION_ENDPOINT`` / ``V2S_ENDPOINT`` are set.

Import safety: importing this package (or any submodule) must never raise
because an environment variable is missing. Secrets and endpoint settings are
read lazily at call time only.
"""

__all__ = [
    "agent",
    "client",
    "detect_track",
    "ingest",
    "policy",
    "sample",
    "sqlwriter",
]

__version__ = "0.1.0"
