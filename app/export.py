"""Download a whole generation at once: every clip in a ZIP, plus a CSV.

The CSV is a tracking sheet. Each row is one clip with the copy the chat wrote
for it (title, hook, description, tags, score), how it was cut, and where it came
from, followed by blank columns (``platform``, ``posted_at``, ``post_url``,
``status``) to fill in as clips are uploaded to YouTube, Instagram or anywhere
else, so "which of these went up, and where?" has an answer.

The ``filename`` column is the same name the clip has inside the ZIP, so a row
and its file can always be matched.

Written with a UTF-8 byte-order mark so Excel opens Hindi and other non-Latin
text correctly instead of as mojibake. Cells that begin with ``= + - @`` are
prefixed with an apostrophe: titles and descriptions come from an AI chat, and a
spreadsheet would otherwise run one as a formula.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from . import history

CSV_COLUMNS = [
    "filename", "title", "hook", "description", "tags", "score",
    "start_s", "end_s", "duration_s", "language", "aspect_ratio", "framing",
    "caption_style", "picked_by", "source", "source_id", "generated_at",
    "clip_id", "index",
    # To fill in as clips go out.
    "platform", "posted_at", "post_url", "status",
]

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def _cell(value) -> str:
    """A CSV cell as text, defused against spreadsheet formulas."""
    if value is None:
        return ""
    text = str(value) if not isinstance(value, float) else f"{value:g}"
    return "'" + text if text.startswith(_FORMULA_START) else text


def clip_file(entry: dict, clip: dict) -> Optional[Path]:
    """The clip's mp4 on disk, or None if it is gone."""
    path = history.clip_path(entry.get("clip_id", ""), clip.get("index", -1), history.entry_dir(entry))
    return path if path and path.is_file() else None


def _zip_name(clip: dict, used: set) -> str:
    """A safe, unique file name for the clip inside the ZIP."""
    name = _UNSAFE_NAME.sub(" ", str(clip.get("filename") or "")).strip()
    if not name.lower().endswith(".mp4"):
        name = f"{name or 'clip ' + str(int(clip.get('index', 0)) + 1)}.mp4"
    stem, dot, ext = name.rpartition(".")
    candidate, n = name, 2
    while candidate.lower() in used:
        candidate = f"{stem} ({n}).{ext}"
        n += 1
    used.add(candidate.lower())
    return candidate


def _hashtags(description) -> str | None:
    """``#a #b`` in a description as ``a, b`` — Movie Shorts writes its tags there."""
    tags = re.findall(r"#(\w+)", str(description or ""))
    seen: list = []
    for tag in tags:
        if tag.lower() not in (t.lower() for t in seen):
            seen.append(tag)
    return ", ".join(seen) or None


def _row(entry: dict, clip: dict, name: str) -> dict:
    recipe = clip.get("recipe") or {}
    settings = entry.get("settings") or {}
    start = recipe.get("start", clip.get("start"))
    end = recipe.get("end", clip.get("end"))
    try:
        duration = round(float(end) - float(start), 2)
    except (TypeError, ValueError):
        duration = None
    reframe = clip.get("reframe") or {}
    return {
        "filename": name,
        "title": clip.get("title"),
        "hook": clip.get("hook"),
        "description": clip.get("description"),
        "tags": clip.get("tags") or _hashtags(clip.get("description")),
        "score": clip.get("score"),
        "start_s": start,
        "end_s": end,
        "duration_s": duration,
        "language": clip.get("language") or settings.get("language"),
        "aspect_ratio": recipe.get("aspect_ratio") or settings.get("aspect_ratio"),
        "framing": reframe.get("mode") or recipe.get("fit_mode") or settings.get("fit_mode"),
        "caption_style": recipe.get("caption_style") or settings.get("caption_style"),
        "picked_by": clip.get("picked_by"),
        "source": entry.get("source"),
        "source_id": entry.get("source_id"),
        "generated_at": entry.get("created"),
        "clip_id": entry.get("clip_id"),
        "index": clip.get("index"),
        "platform": "",
        "posted_at": "",
        "post_url": "",
        "status": "not_posted",
    }


def collect(entries: Iterable[dict]) -> List[Tuple[dict, dict, Path, str]]:
    """``(entry, clip, file, zip_name)`` for every clip whose file still exists."""
    out: List[Tuple[dict, dict, Path, str]] = []
    used: set = set()
    for entry in entries:
        for clip in sorted(entry.get("clips") or [], key=lambda c: c.get("index", 0)):
            path = clip_file(entry, clip)
            if path is not None:
                out.append((entry, clip, path, _zip_name(clip, used)))
    return out


def build_csv(items: List[Tuple[dict, dict, Path, str]]) -> bytes:
    """The tracking sheet as UTF-8 (with BOM) bytes."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for entry, clip, _path, name in items:
        row = _row(entry, clip, name)
        writer.writerow({k: _cell(row.get(k)) for k in CSV_COLUMNS})
    return buf.getvalue().encode("utf-8-sig")


def build_zip(items: List[Tuple[dict, dict, Path, str]], dest: Path) -> int:
    """Write the clips and ``clips.csv`` to ``dest``. Returns how many clips went in.

    Stored, not deflated: an mp4 is already compressed, so deflating costs
    seconds and saves nothing.
    """
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for _entry, _clip, path, name in items:
            zf.write(path, arcname=name)
        zf.writestr("clips.csv", build_csv(items))
    return len(items)
