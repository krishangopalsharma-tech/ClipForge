"""Pure helpers around speech recognition: which Whisper model fits a language,
and whether a transcript holds enough speech to be worth picking clips from.

Kept free of heavy imports (no faster-whisper, no CUDA) so they are cheap to
import from the API layer and trivial to unit-test.
"""

from __future__ import annotations

import os
from typing import Optional

# The model for everything that is not Hindustani. 'medium' is what ClipForge
# has always shipped; override with CLIPFORGE_WHISPER_MODEL (e.g. "large-v3").
DEFAULT_MODEL = os.environ.get("CLIPFORGE_WHISPER_MODEL", "medium")

# The model for Hindi / Urdu / Hinglish. Whisper's smaller builds are markedly
# weaker on these low-resource languages and on Hindi-English code-switching
# (wrong names, merged and invented words), and Hinglish is exactly that. Cost:
# a one-time ~3 GB download and ~3 GB of VRAM instead of ~1.5 GB, and slower
# decoding. Override with CLIPFORGE_WHISPER_MODEL_HINDUSTANI; set it to the same
# value as the default to turn the swap off.
HINDUSTANI_MODEL = os.environ.get("CLIPFORGE_WHISPER_MODEL_HINDUSTANI", "large-v3")

# Language values (as the UI sends them) that need the Hindustani model.
HINDUSTANI_LANGUAGES = {"hi", "ur", "hinglish"}

# Below this much speech there is nothing for a clip picker to read. Music-only
# footage and a recording with the mic muted transcribe to a handful of stray
# words, which is worse than silence: the selector scores those words and cuts
# around them.
MIN_SPEECH_WORDS = 8
MIN_SPEECH_WORDS_PER_MIN = 5.0


def _fit_for_hindustani(model_name: str) -> bool:
    """True for a model already as capable as large-v3 on Hindustani.

    A plain ``large-v2``/``large-v3`` is. The turbo and distilled builds are not
    (their decoder is shrunk, and that is what loses the low-resource
    languages), nor are medium and below. A path or a custom name is left alone:
    whoever set it knows what it is.
    """
    name = (model_name or "").strip().lower()
    if os.sep in name or "/" in name or name.startswith("."):
        return True
    if "turbo" in name or "distil" in name:
        return False
    return name.startswith("large")


def model_for(language: Optional[str]) -> str:
    """The Whisper model name to use for this UI language value."""
    lang = (language or "").strip().lower()
    if lang in HINDUSTANI_LANGUAGES and not _fit_for_hindustani(DEFAULT_MODEL):
        return HINDUSTANI_MODEL
    return DEFAULT_MODEL


def is_hindustani(language: Optional[str]) -> bool:
    """True for a language code that wants the Hindustani model."""
    return (language or "").strip().lower() in HINDUSTANI_LANGUAGES


def speech_report(transcript: Optional[dict]) -> dict:
    """How much speech a transcript holds: words, words per minute, sparse or not.

    Works on any transcript shape ClipForge produces (Whisper, or an imported
    .srt): the flat ``words`` list when there is one, else the segment text.
    """
    transcript = transcript or {}
    words = transcript.get("words")
    if isinstance(words, list) and words:
        count = len(words)
    else:
        count = sum(
            len(str(seg.get("text") or "").split())
            for seg in (transcript.get("segments") or [])
        )

    try:
        duration = float(transcript.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        ends = [float(s.get("end") or 0.0) for s in (transcript.get("segments") or [])]
        duration = max(ends, default=0.0)

    per_minute = (count / (duration / 60.0)) if duration > 0 else 0.0
    sparse = count < MIN_SPEECH_WORDS or (duration > 0 and per_minute < MIN_SPEECH_WORDS_PER_MIN)
    return {
        "words": count,
        "per_minute": round(per_minute, 1),
        "minutes": round(duration / 60.0, 1),
        "sparse": bool(sparse),
    }


def speech_is_sparse(transcript: Optional[dict]) -> bool:
    """True when the transcript has too little speech to pick clips from."""
    return speech_report(transcript)["sparse"]


def model_hint(transcript: Optional[dict]) -> Optional[str]:
    """A tip when a Hindustani source was decoded by a model that is not fit for it.

    The usual cause is leaving the language on auto-detect: the model has to be
    chosen before the language is known, so detection runs on the default one.
    """
    transcript = transcript or {}
    if not is_hindustani(transcript.get("language")):
        return None
    used = str(transcript.get("whisper_model") or "")
    if used and _fit_for_hindustani(used):
        return None
    if not _fit_for_hindustani(HINDUSTANI_MODEL):
        return None  # the swap is switched off; advising it would be wrong
    label = used or "the default model"
    return (
        f"This sounds like Hindi/Urdu but was transcribed with {label}. "
        f"Choose Hindi or Hinglish as the language and transcribe again to use "
        f"{HINDUSTANI_MODEL}, which is more accurate on these languages."
    )
