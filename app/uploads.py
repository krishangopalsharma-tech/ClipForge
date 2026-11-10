"""Local file uploads as an alternative video source to a URL.

A user can upload a video straight from their machine instead of providing a URL.
We stream the file to ``downloads/<id><ext>`` (the same place yt-dlp writes to)
and hand the pipeline a reference (``upload_id``) it can resolve back to a path.

The id is the first 32 hex chars of the file's SHA-256, so the SAME video always
gets the SAME id. That matters because transcripts are cached under the source id
(``transcripts/<id>__<lang>.json``): with a random id, re-uploading a video you
already transcribed would silently re-run Whisper on it. Content addressing makes
the cache self-healing — lose the id, upload again, and the existing transcript is
found instantly. It also de-duplicates the downloads folder for free.

Resolution only ever globs ``downloads/<id>.*`` — so a caller can never use the
id to reach a file outside the downloads directory (no path traversal).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import BinaryIO, Optional, Union

from . import cues
from .models import InvalidVideoURLError
from .paths import DOWNLOADS_DIR, TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

# Container formats ffmpeg/whisper can handle. Extension is a cheap first gate;
# ffmpeg is the real validator when it tries to read the file.
ALLOWED_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".flv", ".wmv", ".mpeg", ".mpg", ".ts", ".m2ts",
}

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def save_upload(filename: str, fileobj: BinaryIO) -> dict:
    """Stream an uploaded file to downloads/ and return its reference.

    Args:
        filename: the client-supplied name (used only for its extension/label).
        fileobj: a binary file-like object to copy from.

    Returns:
        ``{"upload_id": <hex>, "filename": <name>, "ext": <ext>, "reused": <bool>}``.
        ``reused`` is True when this exact file was already on disk, which means
        any transcript cached for it is still valid.

    Raises:
        InvalidVideoURLError: unsupported extension or empty file.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise InvalidVideoURLError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a video "
            f"file ({', '.join(sorted(ALLOWED_EXTS))})."
        )

    # Stream to a scratch name while hashing, since the final name IS the hash and
    # therefore isn't known until the last byte has been read. One pass over the
    # data, so a large upload is not read twice.
    temp = DOWNLOADS_DIR / f"upload-{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with temp.open("wb") as out:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        logger.exception("Failed to save upload %s", filename)
        raise InvalidVideoURLError(f"Could not save the uploaded file: {exc}") from exc

    if size == 0:
        temp.unlink(missing_ok=True)
        raise InvalidVideoURLError("The uploaded file was empty.")

    # 32 hex chars keeps the id the same shape the rest of the app validates.
    upload_id = digest.hexdigest()[:32]
    dest = DOWNLOADS_DIR / f"{upload_id}{ext}"

    reused = dest.exists() and dest.stat().st_size == size
    if reused:
        # Identical file already here — keep it (its transcript is still cached)
        # and drop the duplicate we just wrote.
        temp.unlink(missing_ok=True)
        logger.info(
            "Upload %s matches existing source %s — reusing it (no re-transcribe).",
            filename, dest.name,
        )
    else:
        try:
            temp.replace(dest)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            logger.exception("Failed to finalise upload %s", filename)
            raise InvalidVideoURLError(
                f"Could not save the uploaded file: {exc}"
            ) from exc
        logger.info("Saved upload %s -> %s (%d bytes)", filename, dest.name, size)

    return {
        "upload_id": upload_id,
        "filename": Path(filename or "video").name,
        "ext": ext,
        "reused": reused,
    }


def resolve_upload(upload_id: str) -> Path:
    """Map an ``upload_id`` back to the saved file path.

    Raises:
        InvalidVideoURLError: malformed id or file no longer present.
    """
    if not upload_id or not _ID_RE.match(upload_id):
        raise InvalidVideoURLError("Invalid upload reference.")
    matches = sorted(DOWNLOADS_DIR.glob(f"{upload_id}.*"))
    if not matches:
        raise InvalidVideoURLError(
            "The uploaded file could not be found. Please upload it again."
        )
    return matches[0]


def _parse_srt_timestamp(ts: str) -> float:
    """Convert SRT timestamp ('00:01:23,456' or '01:23.456') to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    return float(ts)


def parse_srt(srt_text: str) -> dict:
    """Parse standard SRT subtitle text into ClipForge's transcript format."""
    srt_text = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", srt_text.strip())

    segments: list[dict] = []
    words: list[dict] = []
    text_parts: list[str] = []
    seg_id = 0

    ts_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        ts_match = None
        ts_line_idx = -1
        for idx, line in enumerate(lines):
            m = ts_pattern.search(line)
            if m:
                ts_match = m
                ts_line_idx = idx
                break

        if not ts_match or ts_line_idx == -1:
            continue

        start_sec = _parse_srt_timestamp(ts_match.group(1))
        end_sec = _parse_srt_timestamp(ts_match.group(2))
        if end_sec <= start_sec:
            end_sec = start_sec + 1.0

        dialogue_lines = lines[ts_line_idx + 1 :]
        raw_text = " ".join(dialogue_lines).strip()
        clean_text = re.sub(r"<[^>]+>", "", raw_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if not clean_text:
            continue

        speaker = None
        speaker_match = re.match(
            r"^(?:\[([A-Za-z0-9 _\-]+)\]|\(([^)]+)\)|([A-Za-z0-9 _\-]+):)\s*(.*)$",
            clean_text,
        )
        if speaker_match:
            speaker = (
                speaker_match.group(1)
                or speaker_match.group(2)
                or speaker_match.group(3)
                or ""
            ).strip()
            body = (speaker_match.group(4) or "").strip()
            if body:
                clean_text = body

        segments.append(
            {
                "id": seg_id,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": clean_text,
                "speaker": speaker,
            }
        )
        text_parts.append(clean_text)

        # Generate synthetic word timestamps so word reveal & animations work
        raw_words = [w for w in clean_text.split() if w.strip()]
        num_words = len(raw_words)
        dur = max(0.2, end_sec - start_sec)
        if num_words > 0:
            word_dur = dur / num_words
            for wi, w in enumerate(raw_words):
                w_start = start_sec + wi * word_dur
                w_end = start_sec + (wi + 1) * word_dur
                words.append(
                    {
                        "word": w,
                        "start": round(w_start, 3),
                        "end": round(w_end, 3),
                        "speaker": speaker,
                    }
                )
        seg_id += 1

    duration = max((s["end"] for s in segments), default=0.0)
    return {
        "language": "en",
        "duration": float(duration),
        "text": " ".join(text_parts).strip(),
        "words": words,
        "segments": segments,
    }


def probe_fps(video: Path) -> Optional[float]:
    """The video's frame rate, or None. Editor timecode frames are counted in it."""
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30,
        )
        num, _, den = out.stdout.strip().partition("/")
        fps = float(num) / float(den or 1)
        return fps if 1.0 <= fps <= 240.0 else None
    except Exception:  # noqa: BLE001 - a guess is made from the file instead
        return None


def save_srt_for_source(
    source_id: str,
    srt_content: Union[str, bytes],
    filename: str = "",
    fps: Optional[float] = None,
) -> dict:
    """Parse a subtitle or timestamped-text file and save it as the transcript for source_id.

    Reads SubRip, WebVTT and plain text with timestamps, including an editor's
    ``HH:MM:SS:FF`` timecode (``fps`` is the video's frame rate, for the frames).
    The file as imported is also kept as ``<id>__srt.json``, so a later
    verification (``app.verify``) can compare against it or put it back."""
    if isinstance(srt_content, bytes):
        try:
            srt_text = srt_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            srt_text = srt_content.decode("latin-1", errors="replace")
    else:
        srt_text = srt_content

    kind = "srt"
    if cues.looks_like_srt(srt_text):
        transcript = parse_srt(srt_text)
    else:
        try:
            parsed = cues.parse_cues(srt_text, filename, fps)
        except cues.CueError as exc:
            raise InvalidVideoURLError(str(exc)) from exc
        transcript = cues.cues_to_transcript(parsed)
        kind = "vtt" if cues.looks_like_vtt(srt_text) else "text"
    if not transcript["segments"]:
        raise InvalidVideoURLError(
            "No valid subtitle segments could be parsed from that file."
        )

    out_file = TRANSCRIPTS_DIR / f"{source_id}__auto.json"
    out_file.write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (TRANSCRIPTS_DIR / f"{source_id}__srt.json").write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )

    from . import pretranscribe

    with pretranscribe._CACHE_LOCK:
        pretranscribe._CACHE[f"{source_id}__auto"] = transcript
        pretranscribe._CACHE[f"{source_id}__en"] = transcript

    return {
        "source_id": source_id,
        "segments": len(transcript["segments"]),
        "duration": transcript["duration"],
        "words": len(transcript["words"]),
        "format": kind,
    }

