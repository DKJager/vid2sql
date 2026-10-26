# build 7fd77e00de7f
"""Resolve a local folder or S3 prefix of videos into a work list.

Thin wrapper: this module only enumerates candidate video objects and joins
them against the clip registry (``data/CLIPS.csv``) so every processed clip
carries a recorded source URL and licence. No video decoding happens here;
perception is delegated to the external vision service.

The clip registry path is read lazily via ``V2S_CLIPS_CSV`` (default
``data/CLIPS.csv``) so importing this module never touches the filesystem.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

_DEFAULT_CLIPS_CSV = Path("data") / "CLIPS.csv"

_log = logging.getLogger(__name__)
# Module logger used by normalise_recording_start (same logger object as _log).
logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"filename", "source_url", "license"}


@dataclass
class ClipWorkItem:
    """One video to process, with registry metadata when available."""

    uri: str
    filename: str
    source_url: str = "NOT FOUND"
    license: str = "NOT FOUND"
    extra: dict[str, str] = field(default_factory=dict)
    recording_start_utc: str | None = None


def normalise_recording_start(value: str | None) -> str | None:
    """Normalise an ISO 8601 UTC recording start to ``YYYY-MM-DDTHH:MM:SSZ``.

    ``None`` or an empty/blank value returns ``None``. A non-empty value must
    end in ``Z`` or ``+00:00`` and be accepted by ``datetime.fromisoformat``
    after replacing a trailing ``Z`` with ``+00:00``. Invalid values are
    ignored (``None``) and a warning is logged.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning("Ignoring invalid recording_start_utc %r: not a string.", value)
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        candidate = text[:-1] + "+00:00"
    elif text.endswith("+00:00"):
        candidate = text
    else:
        logger.warning(
            "Ignoring invalid recording_start_utc %r: must be an ISO 8601 UTC "
            "timestamp ending in 'Z' or '+00:00'.",
            value,
        )
        return None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        logger.warning(
            "Ignoring invalid recording_start_utc %r: not a valid ISO 8601 timestamp.",
            value,
        )
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        logger.warning(
            "Ignoring invalid recording_start_utc %r: timestamp is not UTC.",
            value,
        )
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _clips_csv_path() -> Path:
    return Path(os.environ.get("V2S_CLIPS_CSV", str(_DEFAULT_CLIPS_CSV)))


def load_registry(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load the clip registry keyed by filename.

    Rows whose licence is missing are kept but marked ``NOT FOUND`` per the
    claims policy; such clips are excluded from processing by callers.

    .. warning::

       *Issue 161* — This function keys on the basename column only. When two
       clips share the same ``filename`` value (e.g. video from different
       folders or buckets) only the **last** row in the CSV survives. Callers
       routing multi-folder ingests *must* ensure unique filenames or use a
       registry with fully-qualified paths in the ``filename`` column.
    """
    registry_path = path if path is not None else _clips_csv_path()
    registry: dict[str, dict[str, str]] = {}

    # Issue 211: warn when the registry file is missing
    if not registry_path.is_file():
        msg = (
            f"Clip registry not found at {registry_path}. "
            "No metadata will be loaded, so all clips will have licence "
            "\"NOT FOUND\" and will be excluded during ingestion. "
        )
        if path is not None or os.environ.get("V2S_CLIPS_CSV"):
            msg += (
                "A custom path was configured via V2S_CLIPS_CSV environment "
                "variable or function argument. Verify the file exists."
            )
        else:
            msg += "Create data/CLIPS.csv with columns: filename,source_url,license."
        _log.warning(msg)
        return registry

    seen: dict[str, int] = {}
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        # Issue 212: validate that required columns exist in the CSV header
        header_columns = set(reader.fieldnames or [])
        missing_required = _REQUIRED_COLUMNS - header_columns
        if missing_required:
            _log.warning(
                "Clip registry %s is missing required columns: %s. "
                "These values will default to 'NOT FOUND' for all rows. "
                "Expected columns: filename, source_url, license.",
                registry_path,
                ", ".join(sorted(missing_required)),
            )

        # Capture all column names present in the header so we can forward
        # unknown columns into ClipWorkItem.extra.  This is a best-effort
        # strategy; it intentionally avoids coupling this module to a fixed
        # schema beyond the three mandated columns.
        known_columns = {"filename", "source_url", "license"}
        extra_columns = [c for c in (reader.fieldnames or []) if c not in known_columns]
        line_no = 1  # header line
        for row in reader:
            line_no += 1
            filename = (row.get("filename") or "").strip()
            if not filename:
                _log.debug("Skipping row %d in %s: empty filename.", line_no, registry_path)
                continue
            if filename in seen:
                _log.warning(
                    "Duplicate filename %r at line %d (first seen at line %d) "
                    "— overwriting metadata with latest row.",
                    filename,
                    line_no,
                    seen[filename],
                )
            seen[filename] = line_no
            meta: dict[str, str] = {
                "source_url": (row.get("source_url") or "").strip() or "NOT FOUND",
                "license": (row.get("license") or "").strip() or "NOT FOUND",
            }
            # Committee #1430: optional recording start (absent/empty -> None).
            recording_start = normalise_recording_start(row.get("recording_start_utc"))
            if recording_start is not None:
                meta["recording_start_utc"] = recording_start
            # Issue 166: populate `extra` from any additional CSV columns.
            if extra_columns:
                extra = {}
                for col in extra_columns:
                    val = (row.get(col) or "").strip()
                    if val:
                        extra[col] = val
                if extra:
                    meta["extra"] = extra  # stored as key in the meta dict, packed later
            registry[filename] = meta
    return registry


def _is_video_name(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def _with_registry(uri: str, filename: str,
                   registry: dict[str, dict[str, str]]) -> ClipWorkItem:
    meta = registry.get(filename, {})
    raw_extra = meta.get("extra", {})
    # `extra` was stored as a plain dict in the meta dict; unpack it.
    extra = raw_extra if isinstance(raw_extra, dict) else {}
    return ClipWorkItem(
        uri=uri,
        filename=filename,
        source_url=meta.get("source_url", "NOT FOUND"),
        license=meta.get("license", "NOT FOUND"),
        extra=extra,
        recording_start_utc=meta.get("recording_start_utc"),
    )


def _build_items(source_objects: Iterable[tuple[str, str]],
                 registry: dict[str, dict[str, str]]) -> list[ClipWorkItem]:
    """Build work items and filter out clips whose registry licence is NOT FOUND.

    Issue 167: the licence gate is now applied here so that *every* code path
    that produces work items (local, S3, or future plug-in sources) honours the
    data rule automatically.  Callers that need the unfiltered list can use
    ``load_registry`` and construct ``ClipWorkItem`` themselves.
    """
    items: list[ClipWorkItem] = []
    for uri, key in source_objects:
        filename = Path(key).name
        meta = registry.get(filename, {})
        licence = meta.get("license", "NOT FOUND")
        if licence == "NOT FOUND":
            _log.info(
                "Skipping %s: licence metadata is NOT FOUND. "
                "Add a row to data/CLIPS.csv with a valid licence to process it.",
                filename,
            )
            continue
        items.append(_with_registry(uri, filename, registry))
    return items


def ingest_local(folder: str | os.PathLike[str], recursive: bool = False) -> list[ClipWorkItem]:
    """Enumerate video files under a local folder.

    Only clips with a recorded licence in ``data/CLIPS.csv`` are returned.

    Parameters
    ----------
    folder : str or Path
        Root directory.
    recursive : bool, default ``False``
        When ``False``, only immediate children are scanned (the historical
        semantics). When ``True``, the entire directory tree is walked,
        matching the recursive behaviour of ``ingest_s3``.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"local video folder not found: {root}")
    registry = load_registry()
    source_objects: list[tuple[str, str]] = []
    paths = root.rglob("*") if recursive else root.iterdir()
    for path in sorted(paths):
        if path.is_file() and _is_video_name(path.name):
            source_objects.append((str(path), path.name))
    return _build_items(source_objects, registry)


def _list_s3_uris(bucket: str, prefix: str) -> Iterable[tuple[str, str]]:
    """Yield (uri, key) for video objects under an S3 prefix.

    boto3 is imported lazily so that importing this module and running tests
    never requires AWS credentials or the boto3 package to be active.

    Issue 207: a missing boto3 package is surfaced with a clear, actionable
    message rather than a bare ``ModuleNotFoundError``.
    """
    try:
        import boto3  # lazy: only when S3 ingestion is explicitly requested
    except ImportError as exc:
        raise RuntimeError(
            "S3 ingestion requires the boto3 package. "
            "Install it with: pip install boto3"
        ) from exc

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if _is_video_name(key):
                yield f"s3://{bucket}/{key}", key


def _normalise_s3_prefix(prefix: str) -> str:
    """Issue 209: enforce a trailing slash on S3 prefixes unless the prefix is
    empty, so ``list_objects_v2`` with ``Prefix="folder"`` does not
    accidentally match ``folder2/`` objects.  An empty string (whole-bucket
    listing) is intentionally left untouched."""
    if prefix and not prefix.endswith("/"):
        _log.debug(
            "Appending trailing '/' to S3 prefix %r for correct object-key "
            "matching. Use the exact prefix (with trailing slash) to suppress "
            "this behaviour.",
            prefix,
        )
        return prefix + "/"
    return prefix


def ingest_s3(s3_uri: str) -> list[ClipWorkItem]:
    """Enumerate video objects under an ``s3://bucket/prefix`` URI.

    The prefix *must* be non-empty (i.e. ``s3://bucket/path/to/``). Passing a
    bare ``s3://bucket`` (which would implicitly list every object in the
    bucket) is rejected; use ``s3://bucket/`` explicitly if you intend to scan
    the whole bucket.  (Issue 164)

    By default a trailing ``/`` is appended to a non-empty prefix that does not
    already end with one, which avoids the S3 ``list_objects_v2`` behaviour of
    returning objects whose key starts with the prefix string (e.g.
    ``Prefix="folder"`` would also return ``folder2/file.mp4``).  (Issue 209)
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got: {s3_uri!r}")
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"s3 URI has no bucket: {s3_uri!r}")
    # Issue 164: guard against accidental whole-bucket listing.
    if prefix == "" and not s3_uri.endswith("/"):
        raise ValueError(
            f"Empty prefix rejected for safety (would list entire bucket): "
            f"{s3_uri!r}. Append '/' (s3://bucket/) if you intend to scan the "
            f"whole bucket."
        )
    prefix = _normalise_s3_prefix(prefix)
    registry = load_registry()
    return _build_items(_list_s3_uris(bucket, prefix), registry)


def ingest(source: str, recursive: bool = False) -> list[ClipWorkItem]:
    """Ingest from a local folder path or an ``s3://`` prefix.

    Clips whose registry licence is ``NOT FOUND`` are excluded, per the data
    rule that only footage with an explicit licence permitting analysis and
    redistribution of derived tables may be processed.

    The ``recursive`` flag controls local behaviour only (S3 prefix listing is
    always recursive over the given prefix — :issue:`163`).
    """
    if source.startswith("s3://"):
        return ingest_s3(source)
    return ingest_local(source, recursive=recursive)
