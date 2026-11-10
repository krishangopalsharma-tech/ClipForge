"""Voiceover from a local Kokoro-FastAPI server, cached on disk.

Kokoro runs as its OWN process (``start-gpu.ps1`` in the Kokoro-FastAPI folder,
listening on port 8880), not as a library in this app. That is deliberate: the
``kokoro`` package needs PyTorch, and ClipForge ships an embedded Python that
has neither the room nor the CUDA build for it. Talking to it over HTTP keeps
this module to the standard library, and keeps the GPU free for Whisper between
renders.

``speak()`` asks for one line of narration and returns the WAV plus REAL
per-word timings (Kokoro measures them; we never guess by dividing the duration
by the word count), so captions land on the syllable. Every line is cached by
content, so re-rendering a short costs no synthesis at all.

Ported from the same client used by the quote-reels project, including its two
hard-won Windows fixes — see ``_request`` and ``fix_wav``.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from .paths import DATA_DIR

logger = logging.getLogger(__name__)

# Where Kokoro listens. Override with CLIPFORGE_KOKORO_URL to point at another host.
KOKORO_URL = os.environ.get("CLIPFORGE_KOKORO_URL") or "http://127.0.0.1:8880"

TTS_CACHE_DIR = Path(os.environ.get("CLIPFORGE_TTS_CACHE_DIR") or (DATA_DIR / "tts"))

DEFAULT_VOICE = "am_michael"

# One line can take a while: the FIRST request after Kokoro starts also loads the model.
TIMEOUT = 120

# Bumped when the way a line is voiced changes, so stale audio is never reused.
CACHE_VERSION = 1

START_HINT = (
    "Start Kokoro first: run start-gpu.ps1 in the Kokoro-FastAPI folder "
    "(it listens on port 8880)."
)


class KokoroError(RuntimeError):
    """Kokoro is unreachable, refused the request, or sent something unusable."""


def _request(
    path: str,
    payload: Optional[dict] = None,
    url: Optional[str] = None,
    timeout: int = TIMEOUT,
    tries: int = 2,
) -> Any:
    """JSON from Kokoro: GET without a payload, POST with one. One retry on a reset.

    Plain ``http.client`` rather than ``urllib``: urllib always sends
    ``Connection: close``, and Kokoro on Windows then stalls for ~19 s and
    resets about one request in ten.
    """
    base = (url or KOKORO_URL).rstrip("/")
    parts = urlsplit(base)
    data = json.dumps(payload) if payload is not None else None
    for attempt in range(tries):
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
        try:
            conn.request(
                "POST" if data is not None else "GET",
                parts.path + path,
                data,
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            body = resp.read()
        except OSError as exc:
            if attempt + 1 < tries and isinstance(exc, ConnectionResetError):
                time.sleep(1)
                continue
            raise KokoroError(f"Can't reach Kokoro at {base}. {START_HINT} ({exc})") from None
        finally:
            conn.close()
        if resp.status >= 400:
            detail = body.decode("utf-8", "replace")[:300]
            raise KokoroError(f"Kokoro refused the request ({resp.status}): {detail}")
        try:
            return json.loads(body)
        except ValueError:
            raise KokoroError("Kokoro's reply wasn't JSON.") from None
    raise KokoroError(f"Can't reach Kokoro at {base}. {START_HINT}")


def list_voices(url: Optional[str] = None) -> List[str]:
    """Voice ids the server offers (``am_michael``, ``af_heart``, ...)."""
    voices = _request("/v1/audio/voices", url=url, timeout=10).get("voices", [])
    return [v["id"] if isinstance(v, dict) else v for v in voices]


def status(url: Optional[str] = None) -> dict:
    """``{"ok": True, "voices": [...]}`` when Kokoro answers, else ``{"ok": False, ...}``."""
    try:
        return {"ok": True, "voices": list_voices(url)}
    except KokoroError as exc:
        return {"ok": False, "error": str(exc)}


def fix_wav(data: bytes) -> tuple[bytes, float]:
    """A WAV with correct chunk sizes, and its length in seconds.

    Kokoro writes ``0xFFFFFFFF`` as the RIFF and data sizes (it streams), which
    makes Python's ``wave`` module report hours of audio and makes ffmpeg read
    far past the real samples. The data chunk is taken to run to end of file.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise KokoroError("Kokoro didn't return a WAV file.")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = data[pos + 8:pos + 8 + size]
        elif cid == b"data":
            if fmt is None:
                break
            streaming = size == 0xFFFFFFFF or pos + 8 + size > len(data)
            pcm = data[pos + 8:] if streaming else data[pos + 8:pos + 8 + size]
            channels, rate = struct.unpack("<HI", fmt[2:8])
            bits = struct.unpack("<H", fmt[14:16])[0]
            frame = channels * bits // 8
            pcm = pcm[:len(pcm) - len(pcm) % frame]
            header = (
                b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE"
                + b"fmt " + struct.pack("<I", len(fmt)) + fmt
                + b"data" + struct.pack("<I", len(pcm))
            )
            return header + pcm, len(pcm) / (rate * frame)
        pos += 8 + size + (size & 1)
    raise KokoroError("Kokoro's WAV has no audio data.")


def cache_key(text: str, voice: str, speed: float, normalize: bool) -> str:
    raw = json.dumps(
        [CACHE_VERSION, text, voice, round(float(speed), 3), bool(normalize)],
        ensure_ascii=False,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def speak(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    normalize: bool = True,
    url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Voice one line of narration.

    Returns ``{"path": <wav>, "duration": <seconds>, "words": [...], "cached": bool}``
    where each word is ``{"word", "start", "end"}`` in seconds from the start of
    the WAV. Punctuation comes back as its own "word".
    """
    text = " ".join((text or "").split())
    if not text:
        raise KokoroError("Nothing to say: the narration text is empty.")

    folder = Path(cache_dir or TTS_CACHE_DIR)
    key = cache_key(text, voice, speed, normalize)
    wav_path = folder / f"{key}.wav"
    meta_path = folder / f"{key}.json"

    if wav_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {
                "path": wav_path,
                "duration": meta["duration"],
                "words": meta["words"],
                "cached": True,
            }
        except (OSError, ValueError, KeyError):
            pass  # a damaged entry is simply voiced again below

    resp = _request(
        "/dev/captioned_speech",
        {
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav",
            "stream": False,
            "return_timestamps": True,
            "normalization_options": {"normalize": bool(normalize)},
        },
        url=url,
    )
    try:
        audio, duration = fix_wav(base64.b64decode(resp["audio"]))
    except (KeyError, ValueError, TypeError):
        raise KokoroError("Kokoro's reply had no audio.") from None

    # A closing mark's time can run a little past the audio; keep every time inside it.
    words = [
        {
            "word": t["word"],
            "start": round(min(t["start_time"], duration), 3),
            "end": round(min(t["end_time"], duration), 3),
        }
        for t in resp.get("timestamps") or []
    ]
    meta = {
        "text": text,
        "voice": voice,
        "speed": speed,
        "normalize": bool(normalize),
        "duration": round(duration, 3),
        "words": words,
    }

    # Written under temporary names, then moved into place, so parallel renders
    # never read half a file.
    folder.mkdir(parents=True, exist_ok=True)
    payloads = (
        (wav_path, audio),
        (meta_path, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")),
    )
    for path, blob in payloads:
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, path)

    logger.info("kokoro voiced %.1fs for %d chars (%s)", duration, len(text), voice)
    return {"path": wav_path, "duration": meta["duration"], "words": words, "cached": False}
