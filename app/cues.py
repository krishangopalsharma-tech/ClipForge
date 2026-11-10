"""Subtitle and timestamped-text files, as caption cues.

``uploads.parse_srt`` reads plain SubRip. This module reads the other two files an
editor actually hands over:

* **WebVTT** (``.vtt``).
* **Timestamped text** (``.txt``): a time at the start of each line, then what
  was said. That includes the timecode Premiere Pro, Resolve and Avid export,
  ``00:00:20:03`` (hours:minutes:seconds:FRAMES), alone or as a range,
  ``00:00:20:03 - 00:00:21:12``. A parser that only knew ``H:MM:SS`` stopped at
  ``00:00:20`` and made ``03 - 00:00:21:12`` the first words of every cue.

Timecode frames are converted with the video's real frame rate when the caller
knows it (``fps``); without it the rate is inferred from the highest frame number
in the file, which can only err by a few hundredths of a second.

The result is a whisper-shaped transcript with the words of each cue spread
across it, which is all the picker and the captions need until ``app.verify``
replaces those spread times with measured ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MIN_CUE_SECONDS = 0.2
# A last cue with nothing after it to bound it. Caption lines run 2-6 s.
DEFAULT_TAIL_SECONDS = 4.0


class CueError(ValueError):
    """Unreadable input; the message is shown to the user as it is."""


@dataclass
class Cue:
    start: float
    end: float
    text: str


# ``0:14``, ``[00:14]``, ``1:02:33``, ``00:00:14.500``, ``00:00:20:03`` (frames),
# alone or as a range. Anchored at the start of a line: a time in the middle of a
# sentence ("we did it at 3:30") is speech, not a cue boundary.
_STAMP = (
    r"(?:(?P<{p}h>\d{{1,2}}):)?(?P<{p}m>\d{{1,2}}):(?P<{p}s>\d{{2}})"
    r"(?:[.,](?P<{p}frac>\d{{1,3}})|:(?P<{p}ff>\d{{2}}))?"
)
_TS = re.compile(
    r"^[\s\[\(\-•]*"
    + _STAMP.format(p="a")
    + r"(?:\s*(?:-->|[-–—]|to)\s*"
    + _STAMP.format(p="b")
    + r")?[\s\]\)\-:]*"
)


def frame_rate_guess(max_frame: int) -> float:
    """A frame rate from the highest frame number seen (timecode never says it)."""
    if max_frame >= 50:
        return 60.0
    if max_frame >= 30:
        return 50.0
    if max_frame >= 25:
        return 30.0
    return 25.0


def _seconds(match: re.Match, prefix: str, fps: float) -> float:
    h, m, s = (match.group(prefix + k) for k in ("h", "m", "s"))
    frac, ff = match.group(prefix + "frac"), match.group(prefix + "ff")
    total = int(m) * 60 + int(s) + (int(h) * 3600 if h else 0)
    if frac:
        total += float(f"0.{frac}")
    if ff:
        total += int(ff) / fps
    return float(total)


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #
_SRT_TIMING = re.compile(r"\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}")


def looks_like_vtt(text: str) -> bool:
    return (text or "").lstrip("﻿ \t\r\n").startswith("WEBVTT")


def looks_like_srt(text: str) -> bool:
    """SubRip: timing lines with a comma-or-dot millisecond field and no WEBVTT header."""
    return not looks_like_vtt(text) and bool(_SRT_TIMING.search(text or ""))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
_VTT_TIMING = re.compile(
    r"((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)


def _clock(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_vtt(text: str) -> List[Cue]:
    """WebVTT cues. Header, NOTE/STYLE blocks, cue ids and settings are ignored."""
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", (text or "").replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            m = _VTT_TIMING.search(line)
            if not m:
                continue
            start, end = _clock(m.group(1)), _clock(m.group(2))
            body = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:]))
            body = re.sub(r"\s+", " ", body).strip()
            if body:
                if end <= start:
                    end = start + MIN_CUE_SECONDS
                cues.append(Cue(start, end, body))
            break
    return cues


def parse_timestamped_text(text: str, fps: Optional[float] = None) -> List[Cue]:
    """A time, then the line, repeated; the timestamp may be on its own line.

    A cue runs until the next timestamp, except when the line gives a range, whose
    end is used as it is (the silence between two ranged cues is real).
    """
    blocks = []  # [match, [text lines]]
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        match = _TS.match(line)
        if match:
            blocks.append([match, [line[match.end():]]])
        elif blocks:
            blocks[-1][1].append(line)
        # A line before the first timestamp is a header ("Transcript"); dropped.

    if fps is None:
        max_frame = max(
            (int(m.group(p + "ff")) for m, _ in blocks for p in ("a", "b") if m.group(p + "ff")),
            default=0,
        )
        fps = frame_rate_guess(max_frame)

    cues: List[Cue] = []
    for i, (match, lines) in enumerate(blocks):
        body = re.sub(r"\s+", " ", " ".join(p.strip() for p in lines if p.strip())).strip()
        if not body:
            continue
        start = _seconds(match, "a", fps)
        if match.group("bm") is not None:
            end = _seconds(match, "b", fps)
        elif i + 1 < len(blocks):
            end = _seconds(blocks[i + 1][0], "a", fps)
        else:
            end = start + DEFAULT_TAIL_SECONDS
        if end <= start:
            end = start + MIN_CUE_SECONDS
        cues.append(Cue(start, end, body))
    return cues


def parse_cues(text: str, filename: str = "", fps: Optional[float] = None) -> List[Cue]:
    """VTT or timestamped text, decided by content (the name only breaks ties).

    Raises:
        CueError: if nothing usable was found.
    """
    if not (text or "").strip():
        raise CueError("That file is empty.")
    if looks_like_vtt(text) or (filename or "").lower().endswith(".vtt"):
        cues = parse_vtt(text)
        if not cues:
            raise CueError("No subtitle cues found in that WebVTT file.")
        return cues
    cues = parse_timestamped_text(text, fps)
    if not cues:
        raise CueError(
            "No timestamps found. Each line should start with a time, like "
            "'00:00:20:03 what was said' or '0:14 what was said'."
        )
    return cues


# --------------------------------------------------------------------------- #
# Cues -> transcript
# --------------------------------------------------------------------------- #
def cues_to_transcript(cues: List[Cue], language: str = "en") -> dict:
    """A whisper-shaped transcript: one segment per cue, its words spread across it.

    Words share the cue's span in proportion to their length, which reads more
    naturally than equal slices (a long word is said for longer than "a").
    """
    segments: List[dict] = []
    words: List[dict] = []
    for seg_id, cue in enumerate(cues):
        tokens = cue.text.split()
        segments.append({
            "id": seg_id, "start": round(cue.start, 3), "end": round(cue.end, 3),
            "text": cue.text, "speaker": None,
        })
        if not tokens:
            continue
        span = max(MIN_CUE_SECONDS, cue.end - cue.start)
        weights = [max(1, len(t)) for t in tokens]
        total = float(sum(weights))
        cursor = cue.start
        for token, weight in zip(tokens, weights):
            width = span * weight / total
            words.append({
                "word": token, "start": round(cursor, 3), "end": round(cursor + width, 3),
                "speaker": None,
            })
            cursor += width
    return {
        "language": language,
        "duration": float(max((c.end for c in cues), default=0.0)),
        "text": " ".join(c.text for c in cues),
        "words": words,
        "segments": segments,
        "source": "imported_text",
    }
