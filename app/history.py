"""Persistent history of generated clips.

Jobs live only in memory (cleared on restart), but the rendered clips survive on
disk. This module records a small JSON manifest next to them so the UI can show a
"Video clips" history that persists across restarts, and can remove clips or
reveal them in the OS file manager.

Every generation appends one entry (a job and its clips). The store is a single
JSON file guarded by a lock; this is a local, single-user tool so that is plenty.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .paths import CLIPS_DIR, DATA_DIR, DEFAULT_CLIPS_DIR

logger = logging.getLogger(__name__)

# The history lives OUTSIDE the clips folder, because that folder is something
# the user can move (e.g. onto a synced drive). Keeping the index here means past
# generations stay listed and deletable across such a move.
HISTORY_FILE = DATA_DIR / "history.json"
_LEGACY_HISTORY_FILE = DEFAULT_CLIPS_DIR / "history.json"
_LOCK = threading.Lock()

# clip_id is a 32-char hex uuid; index is a small non-negative int. Validating
# both means a request can never reach a path outside a clips directory.
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _migrate_legacy_history() -> None:
    """Move a pre-existing clips/history.json to its new home, once."""
    if HISTORY_FILE.exists() or not _LEGACY_HISTORY_FILE.exists():
        return
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_LEGACY_HISTORY_FILE, HISTORY_FILE)
        logger.info("Moved clip history to %s", HISTORY_FILE)
    except OSError:
        logger.warning("Could not migrate the old clip history.", exc_info=True)


_migrate_legacy_history()


def entry_dir(entry: dict) -> Path:
    """The clips directory an entry's files actually live in.

    Entries record the directory they were rendered into, so moving the clips
    folder doesn't strand everything made before the move. Entries written before
    the setting existed have no record, and were necessarily in the default spot.
    """
    raw = (entry or {}).get("clips_dir")
    return Path(raw) if raw else DEFAULT_CLIPS_DIR


def clip_path(clip_id: str, index: int, base: Optional[Path] = None) -> Optional[Path]:
    """Resolve ``<clips dir>/<clip_id>/<index>.mp4``, or None if invalid.

    ``base`` names the directory to look in; it defaults to the current one. Pass
    an entry's own directory (see :func:`entry_dir`) when working with history.
    """
    if not _ID_RE.match(clip_id or "") or not isinstance(index, int) or index < 0:
        return None
    root = (base or CLIPS_DIR).resolve()
    path = (root / clip_id / f"{index}.mp4").resolve()
    # Defence in depth: the resolved path must stay under that directory.
    if root not in path.parents:
        return None
    return path


def find_entry_dir(clip_id: str) -> Path:
    """Look up which directory a generation was rendered into."""
    for entry in _load():
        if entry.get("clip_id") == clip_id:
            return entry_dir(entry)
    return CLIPS_DIR


def _load() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        logger.warning("history.json was unreadable; starting a fresh history.")
        return []


def _save(entries: list) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not write history.json: %s", exc)


def add_entry(
    clip_id: str,
    source: str,
    settings: dict,
    clips: List[dict],
    source_type: str = "url",
    source_id: str = "",
) -> None:
    """Record one finished generation (a job and the clips it produced).

    ``source`` is a human label (the URL, or an uploaded file's name) and
    ``source_type`` is "url" or "upload" so the UI can show the right icon.
    ``source_id`` is the id of the video file in downloads/, so the storage
    screen can tell which generations came from which source (entries written
    before this was recorded simply have no link).
    """
    entry = {
        "clip_id": clip_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "source_type": source_type,
        "source_id": source_id,
        # Where these files were written. Recorded so a later change to the
        # clips folder doesn't strand them.
        "clips_dir": str(CLIPS_DIR),
        "settings": settings,
        "clips": clips,
    }
    with _LOCK:
        entries = _load()
        entries.append(entry)
        _save(entries)


def list_entries() -> list:
    """All entries, newest first, each clip annotated with whether it still exists."""
    with _LOCK:
        entries = _load()
    for e in entries:
        base = entry_dir(e)
        for c in e.get("clips", []):
            p = clip_path(e.get("clip_id", ""), c.get("index", -1), base)
            c["exists"] = bool(p and p.exists())
    entries.sort(key=lambda e: e.get("created", ""), reverse=True)
    return entries


def get_clip(clip_id: str, index: int):
    """``(entry, clip)`` for one clip in history, or ``(None, None)``."""
    with _LOCK:
        for entry in _load():
            if entry.get("clip_id") != clip_id:
                continue
            for clip in entry.get("clips", []):
                if clip.get("index") == index:
                    return entry, clip
    return None, None


def update_clip(clip_id: str, index: int, fields: dict) -> Optional[dict]:
    """Merge ``fields`` into one clip's history record; return the new record."""
    with _LOCK:
        entries = _load()
        for entry in entries:
            if entry.get("clip_id") != clip_id:
                continue
            for clip in entry.get("clips", []):
                if clip.get("index") == index:
                    clip.update(fields)
                    _save(entries)
                    return dict(clip)
    return None


def remove_clip(clip_id: str, index: int) -> bool:
    """Delete one clip file and drop it from history. Returns True if a file went."""
    path = clip_path(clip_id, index, find_entry_dir(clip_id))
    deleted = False
    if path and path.exists():
        try:
            path.unlink()
            deleted = True
        except OSError as exc:
            logger.warning("Could not delete clip %s/%s: %s", clip_id, index, exc)

    with _LOCK:
        entries = _load()
        for e in entries:
            if e.get("clip_id") == clip_id:
                e["clips"] = [c for c in e.get("clips", []) if c.get("index") != index]
        # Drop entries that have no clips left, and clean their (now empty) folder.
        kept = []
        for e in entries:
            if e.get("clips"):
                kept.append(e)
            else:
                folder = entry_dir(e) / e.get("clip_id", "")
                if _ID_RE.match(e.get("clip_id", "")) and folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
        _save(kept)
    return deleted
