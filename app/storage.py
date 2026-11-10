"""Disk usage reporting and cleanup for the three folders that grow.

ClipForge accumulates data in three places, and only the first is obvious:

* ``clips/`` — rendered shorts. Small each, but every generation adds several.
* ``downloads/`` — the SOURCE videos (uploaded or fetched). By far the biggest
  items; a single long video can be over a gigabyte.
* ``transcripts/`` — cached Whisper output. Tiny by comparison, but it is what
  makes re-running a video instant, so it is worth keeping even when the clips
  are gone.

Everything here is destructive, so every path is rebuilt from a validated id and
checked to still live under the directory it belongs to. Ids never come from the
caller as paths, only as hex strings.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import history
from .paths import CLIPS_DIR, DEFAULT_CLIPS_DIR, DOWNLOADS_DIR, TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

# Source ids and clip ids are both 32-char hex. Anything else is rejected before
# it can be turned into a path.
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class StorageError(ValueError):
    """A cleanup request named something that does not exist or is not valid."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _valid(item_id: str) -> str:
    item_id = (item_id or "").strip().lower()
    if not _ID_RE.match(item_id):
        raise StorageError("That is not a valid id.")
    return item_id


def _dir_size(path: Path) -> int:
    """Total bytes of every file under ``path`` (0 if it isn't there)."""
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _source_files(source_id: str) -> List[Path]:
    """The video file(s) for a source id — ``downloads/<id>.*``."""
    return [p for p in DOWNLOADS_DIR.glob(f"{source_id}.*") if p.is_file()]


def _transcript_files(source_id: str) -> List[Path]:
    """Cached transcripts for a source — ``transcripts/<id>__<lang>.json``."""
    return [p for p in TRANSCRIPTS_DIR.glob(f"{source_id}__*.json") if p.is_file()]


def _under(path: Path, parent: Path) -> bool:
    """True if ``path`` really resolves to somewhere inside ``parent``."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return parent.resolve() == resolved or parent.resolve() in resolved.parents


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def scan() -> dict:
    """Report what is on disk, grouped so the UI can offer sensible deletes.

    Generations are linked back to their source video when the history entry
    records one (entries written before that was tracked simply won't link).
    """
    entries = history.list_entries()

    generations: List[dict] = []
    by_source: Dict[str, List[str]] = {}
    for entry in entries:
        clip_id = entry.get("clip_id") or ""
        base = history.entry_dir(entry)
        folder = base / clip_id
        source_id = entry.get("source_id") or ""
        generations.append(
            {
                "clip_id": clip_id,
                "created": entry.get("created"),
                "source": entry.get("source"),
                "source_id": source_id or None,
                "clip_count": len(entry.get("clips") or []),
                "bytes": _dir_size(folder),
                "folder": str(folder),
                "settings": entry.get("settings") or {},
            }
        )
        if source_id:
            by_source.setdefault(source_id, []).append(clip_id)

    # Every source video on disk, whether uploaded or fetched from a URL.
    sources: List[dict] = []
    seen_ids = set()
    for video in sorted(DOWNLOADS_DIR.glob("*")):
        if not video.is_file() or video.suffix == ".part":
            continue
        source_id = video.stem
        if not _ID_RE.match(source_id) or source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        transcripts = _transcript_files(source_id)
        sources.append(
            {
                "source_id": source_id,
                "ext": video.suffix,
                "bytes": video.stat().st_size,
                "modified": video.stat().st_mtime,
                "transcript_bytes": sum(t.stat().st_size for t in transcripts),
                "transcript_count": len(transcripts),
                "generations": by_source.get(source_id, []),
            }
        )
    sources.sort(key=lambda s: s["bytes"], reverse=True)

    # Transcripts whose video is gone. Harmless and small, but they are dead
    # weight once the footage they describe cannot be cut any more.
    orphans: List[dict] = []
    for tpath in sorted(TRANSCRIPTS_DIR.glob("*__*.json")):
        source_id = tpath.name.split("__", 1)[0]
        if source_id in seen_ids:
            continue
        orphans.append({
            "source_id": source_id,
            "name": tpath.name,
            "bytes": tpath.stat().st_size,
        })

    clip_roots = {CLIPS_DIR.resolve(), DEFAULT_CLIPS_DIR.resolve()}
    for entry in entries:
        try:
            clip_roots.add(history.entry_dir(entry).resolve())
        except OSError:
            pass

    return {
        "totals": {
            "clips_bytes": sum(_dir_size(root) for root in clip_roots),
            "downloads_bytes": _dir_size(DOWNLOADS_DIR),
            "transcripts_bytes": _dir_size(TRANSCRIPTS_DIR),
        },
        "generations": generations,
        "sources": sources,
        "orphan_transcripts": orphans,
    }


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
def delete_generation(clip_id: str) -> dict:
    """Delete a whole generation: every rendered clip plus its history entry."""
    clip_id = _valid(clip_id)
    entry = next(
        (e for e in history.list_entries() if e.get("clip_id") == clip_id), None
    )
    base = history.entry_dir(entry) if entry else CLIPS_DIR
    folder = base / clip_id
    freed = _dir_size(folder)

    if folder.is_dir() and _under(folder, base):
        shutil.rmtree(folder, ignore_errors=True)

    # Drop the entry. remove_clip() prunes an entry once its last clip goes, so
    # walking the recorded indices leaves history consistent either way.
    for clip in list(entry.get("clips") or []) if entry else []:
        history.remove_clip(clip_id, clip.get("index", -1))

    logger.info("Deleted generation %s (%d bytes)", clip_id, freed)
    return {"deleted": "generation", "clip_id": clip_id, "freed_bytes": freed}


def delete_source(source_id: str, keep_transcript: bool = False) -> dict:
    """Delete a source video, and by default its cached transcript too.

    Keeping the transcript is offered because it is tiny and still readable — but
    without the video nothing can be re-rendered from it, so removing it is the
    normal choice.
    """
    source_id = _valid(source_id)
    freed = 0

    videos = _source_files(source_id)
    if not videos and not _transcript_files(source_id):
        raise StorageError("That video is not on disk any more.")

    for video in videos:
        if not _under(video, DOWNLOADS_DIR):
            continue
        freed += video.stat().st_size
        try:
            video.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", video, exc)

    removed_transcripts = 0
    if not keep_transcript:
        for tpath in _transcript_files(source_id):
            if not _under(tpath, TRANSCRIPTS_DIR):
                continue
            freed += tpath.stat().st_size
            try:
                tpath.unlink()
                removed_transcripts += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", tpath, exc)

    logger.info("Deleted source %s (%d bytes)", source_id, freed)
    return {
        "deleted": "source",
        "source_id": source_id,
        "freed_bytes": freed,
        "transcripts_removed": removed_transcripts,
    }


def delete_transcripts(source_id: str) -> dict:
    """Delete only the cached transcript(s) for a source, leaving the video."""
    source_id = _valid(source_id)
    freed = 0
    count = 0
    for tpath in _transcript_files(source_id):
        if not _under(tpath, TRANSCRIPTS_DIR):
            continue
        freed += tpath.stat().st_size
        try:
            tpath.unlink()
            count += 1
        except OSError as exc:
            logger.warning("Could not delete %s: %s", tpath, exc)

    # Drop it from the in-memory cache too, or the next run reuses a transcript
    # whose file the user just deleted.
    try:
        from . import pretranscribe

        pretranscribe.forget(source_id)
    except Exception:  # noqa: BLE001 - cache eviction must never block a delete
        logger.debug("Could not evict cached transcript for %s", source_id, exc_info=True)

    return {
        "deleted": "transcripts",
        "source_id": source_id,
        "freed_bytes": freed,
        "transcripts_removed": count,
    }


def purge(target: str) -> dict:
    """Bulk cleanup. ``target`` is 'clips', 'sources', or 'orphan_transcripts'.

    Deliberately narrow: there is no "delete everything" here, so a mis-click can
    only ever take out one category.
    """
    report = scan()
    freed = 0
    removed = 0

    if target == "clips":
        for gen in report["generations"]:
            freed += delete_generation(gen["clip_id"])["freed_bytes"]
            removed += 1

    elif target == "sources":
        for source in report["sources"]:
            freed += delete_source(source["source_id"])["freed_bytes"]
            removed += 1

    elif target == "orphan_transcripts":
        for orphan in report["orphan_transcripts"]:
            path = TRANSCRIPTS_DIR / orphan["name"]
            if not _under(path, TRANSCRIPTS_DIR) or not path.is_file():
                continue
            freed += path.stat().st_size
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)

    else:
        raise StorageError(f"Unknown cleanup target {target!r}.")

    logger.info("Purged %s: %d item(s), %d bytes", target, removed, freed)
    return {"deleted": target, "removed": removed, "freed_bytes": freed}
