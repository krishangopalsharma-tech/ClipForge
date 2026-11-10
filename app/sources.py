"""Has this video been cut before?

Cutting a long video twice by accident is the most expensive mistake the UI
allows: a transcription plus a render per clip, repeated for nothing. The
history already records every finished generation and the source it came from,
so this module asks it, instead of keeping a second index that could drift.

Four independent signals, and the answer says which fired, so the user can judge:

* ``same_file``      the very same file (uploads are named by a hash of their
                     bytes, so an equal source id is an equal file).
* ``youtube``        the same YouTube video id, however the link was written.
* ``title``          the same file name once case, punctuation and spacing are
                     ignored. Deliberately NOT stripping "(HD)", "1080p" or
                     "full episode": those are often the only difference between
                     two real uploads, and a false "you already did this" teaches
                     people to click through the warning.
* ``size_duration``  the same byte size and a duration within a second: the same
                     video re-saved under another name.

It is advisory. Nothing here refuses a job, and any failure reads as "no
duplicates", so this can never be the reason a submit fails.
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .paths import DOWNLOADS_DIR

logger = logging.getLogger(__name__)

DURATION_TOLERANCE_SECONDS = 1.0
MIN_TITLE_CHARS = 4          # "a", "v1" and the like would match everything
MAX_RESULTS = 3

_YOUTUBE_ID = re.compile(
    r"(?:youtu\.be/|(?:www\.|m\.|music\.)?youtube(?:-nocookie)?\.com/"
    r"(?:watch\?(?:[^#\s]*&)?v=|shorts/|embed/|live/|v/))([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
)


def youtube_id(url: Optional[str]) -> Optional[str]:
    """The 11-character video id in a YouTube link, or None."""
    match = _YOUTUBE_ID.search(url or "")
    return match.group(1) if match else None


def normalize_title(name: Optional[str]) -> str:
    """A file name as a recognisable title: no extension, case or punctuation."""
    stem = Path(str(name or "")).stem if re.search(r"\.[A-Za-z0-9]{2,5}$", str(name or "")) else str(name or "")
    text = unicodedata.normalize("NFKC", stem).lower()
    # Keep letters, digits and combining marks. A regex \W would split Hindi
    # words at every vowel sign (matra), which are marks, not letters.
    text = "".join(ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Facts about a source file on disk
# --------------------------------------------------------------------------- #
def _file_for(source_id: str) -> Optional[Path]:
    if not re.fullmatch(r"[0-9a-f]{32}", source_id or ""):
        return None
    matches = sorted(DOWNLOADS_DIR.glob(f"{source_id}.*"))
    return matches[0] if matches else None


@functools.lru_cache(maxsize=256)
def _probe_duration(path: str, size: int, mtime: float) -> Optional[float]:
    """Duration via ffprobe. ``size`` and ``mtime`` only key the cache."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001 - a probe problem is "unknown", not an error
        return None


def file_info(source_id: str) -> Tuple[Optional[int], Optional[float]]:
    """(size in bytes, duration in seconds) of a source on disk; None where unknown."""
    path = _file_for(source_id)
    if path is None:
        return None, None
    try:
        stat = path.stat()
    except OSError:
        return None, None
    return stat.st_size, _probe_duration(str(path), stat.st_size, stat.st_mtime)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def find_duplicates(
    entries: List[dict],
    *,
    source_id: Optional[str] = None,
    video_url: Optional[str] = None,
    filename: Optional[str] = None,
    size: Optional[int] = None,
    duration: Optional[float] = None,
    info: Callable[[str], Tuple[Optional[int], Optional[float]]] = file_info,
) -> List[dict]:
    """History entries that look like the same source, newest first.

    Each result names the signals that fired. ``size``/``duration`` default to
    the facts of ``source_id``'s file when that is on disk.
    """
    wanted_yt = youtube_id(video_url)
    wanted_title = normalize_title(filename)
    if len(wanted_title) < MIN_TITLE_CHARS:
        wanted_title = ""
    if source_id and (size is None or duration is None):
        disk_size, disk_duration = info(source_id)
        size = disk_size if size is None else size
        duration = disk_duration if duration is None else duration

    found: List[dict] = []
    for entry in sorted(entries, key=lambda e: e.get("created", ""), reverse=True):
        signals: List[str] = []
        entry_source = entry.get("source_id") or ""
        if source_id and entry_source == source_id:
            signals.append("same_file")

        label = entry.get("source") or ""
        if entry.get("source_type", "url") == "url":
            if wanted_yt and youtube_id(label) == wanted_yt:
                signals.append("youtube")
        elif wanted_title and normalize_title(label) == wanted_title:
            signals.append("title")

        if size is not None and entry_source and "same_file" not in signals:
            other_size, other_duration = info(entry_source)
            if (
                other_size is not None and other_size == size
                and duration is not None and other_duration is not None
                and abs(other_duration - duration) <= DURATION_TOLERANCE_SECONDS
            ):
                signals.append("size_duration")

        if signals:
            found.append({
                "signals": signals,
                "clip_id": entry.get("clip_id"),
                "created": entry.get("created"),
                "source": label,
                "source_type": entry.get("source_type"),
                "clip_count": len(entry.get("clips") or []),
            })
        if len(found) >= MAX_RESULTS:
            break
    return found


def check(**kwargs) -> List[dict]:
    """:func:`find_duplicates` over the real history. Never raises."""
    try:
        from . import history

        return find_duplicates(history.list_entries(), **kwargs)
    except Exception:  # noqa: BLE001 - advisory only
        logger.warning("Duplicate-source check failed; reporting none.", exc_info=True)
        return []
