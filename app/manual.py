"""Manual clip selection — hand the transcript to any LLM, get windows back.

ClipForge's built-in selector is a local heuristic (see :mod:`app.selector`).
This module supports a different workflow: export the transcript, let a strong
external model (Claude, ChatGPT, Gemini, whatever the user prefers) read it in
their own chat, and feed the chosen moments back in as JSON.

The key design choice is that the model NEVER supplies a timestamp. It picks
SEGMENT IDS from the exported transcript, and we resolve those ids back to the
exact times Whisper already recorded. Language models drift when asked to emit
precise numbers, and a clip that starts three seconds late loses the hook — so
the model decides *which* moment and this module decides *when* it starts.

The resolved edges are then snapped onto real word boundaries plus a little of
the surrounding silence, so a cut lands in a pause instead of clipping a word.

Nothing here talks to a network. It formats text and validates JSON.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from typing import Any, List, Optional, Tuple

from . import profiles

# A resolved clip must be long enough to be worth rendering and short enough to
# stay a short. These are deliberately wider than the selector's own bounds —
# the user picked these moments on purpose, so we only reject the absurd.
MIN_MANUAL_LEN = 3.0
MAX_MANUAL_LEN = 900.0

# Fields the chat reply may carry beyond the times. Kept short so one odd reply
# cannot bloat the history file or the UI.
MAX_TITLE_CHARS = 100
MAX_HOOK_CHARS = 120
MAX_DESCRIPTION_CHARS = 500
MAX_TAGS = 15

# Names the UI may send for "which chat answered". Anything else is "other".
KNOWN_PICKERS = {"claude", "chatgpt", "gemini", "other"}


class ManualWindowError(ValueError):
    """The supplied clip JSON could not be used (bad shape, ids, or lengths)."""


# --------------------------------------------------------------------------- #
# Export: transcript -> numbered text for an LLM
# --------------------------------------------------------------------------- #
def _mmss(seconds: float) -> str:
    """Format seconds as M:SS (or H:MM:SS past an hour) for human reading."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_transcript_text(transcript: dict, source_id: str = "") -> str:
    """Render a transcript as numbered, timestamped lines for an LLM to read.

    Word-level timings are deliberately dropped. They are essential for burning
    captions but useless for judging a moment, and they are the bulk of the
    stored JSON — a 21-minute video is ~0.3 MB on disk but ~17 KB here, which is
    the difference between "fits comfortably in a chat" and "does not".
    """
    segments = transcript.get("segments") or []
    duration = float(transcript.get("duration") or 0.0)
    language = transcript.get("language") or "unknown"

    lines = [
        "# ClipForge transcript",
        f"# source_id: {source_id}",
        f"# duration: {duration:.1f}s ({_mmss(duration)})",
        f"# language: {language}",
        f"# segments: {len(segments)}",
        "#",
        "# Each line is: [id] start-end  text",
        "# Refer to moments by their [id]. Do NOT invent timestamps.",
        "",
    ]
    for i, seg in enumerate(segments):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        text = re.sub(r"\s+", " ", (seg.get("text") or "")).strip()
        lines.append(f"[{i}] {_mmss(start)}-{_mmss(end)}  {text}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
# Built here (not in the page) so the contract and the validator can never
# drift apart. These are plain strings, not str.format templates, because the
# JSON examples are full of braces.
COPY_FIELDS = """COPY FIELDS:
- title: max 70 characters.
- hook: on-screen text for the first seconds, max 8 words.
- description: 1 sentence teasing the payoff, then the hashtags.
- tags: 8-15 comma-separated tags, no "#".
- score: 0-100, how likely a cold viewer is to stop scrolling and watch to the end.
- reason: one line on why this moment works (for me, not for viewers)."""


def _count_sentence(num_clips: Optional[int], count_band: Optional[Tuple[int, int]]) -> Tuple[str, str]:
    """(how-many sentence, closing rule) for the chosen count or band."""
    if num_clips:
        n = int(num_clips)
        return (
            f"Pick the {n} moments with the strongest chance of performing well as a standalone short.",
            f"- Return exactly {n} clips unless the transcript genuinely cannot support that many.",
        )
    lo, hi = count_band or (3, 8)
    return (
        f"Pick {lo} to {hi} moments with the strongest chance of performing well as a standalone short.",
        f"- Return between {lo} and {hi} clips. Quality over quantity: never pad with a "
        "clip you would not post yourself.",
    )


def _general_prompt(count_line: str, count_rule: str, length_line: str, audience_block: str) -> str:
    parts = [
        "You are a senior short-form video editor selecting clips from a long video "
        "for YouTube Shorts, TikTok and Instagram Reels.",
    ]
    if audience_block:
        parts.append(audience_block)
    parts.append(
        """I am giving you a transcript. Every line looks like:

    [id] start-end  spoken text"""
    )
    parts.append(
        f"""{count_line}

What makes a moment work:
- It hooks in the first 2 seconds — a question, a claim, a number, a contradiction, tension.
- It is self-contained. A viewer with zero context understands it.
- It pays off. A setup without its punchline is a dead clip; include both.
- It starts on a clean sentence start and ends after the payoff lands, not mid-thought.
- No two clips make the same point or tell the same joke.

Avoid: intros, outros, sponsor reads, housekeeping, rambling, anything that needs earlier context to make sense.

{length_line}"""
    )
    parts.append(COPY_FIELDS)
    parts.append(
        f"""CRITICAL — how to answer:
Do NOT write timestamps. Refer to lines by their [id] only. Reply with ONLY this JSON, no prose, no markdown fence:

{{
  "clips": [
    {{
      "start_seg": 142,
      "end_seg": 158,
      "title": "...",
      "hook": "...",
      "description": "...",
      "tags": "...",
      "score": 85,
      "reason": "..."
    }}
  ]
}}

Rules:
- start_seg and end_seg are ids that appear in the transcript.
- start_seg must be less than or equal to end_seg.
- Order the clips best-first.
{count_rule}"""
    )
    return "\n\n".join(parts) + "\n"


def _series_prompt(count_line: str, count_rule: str, length_line: str, audience_block: str) -> str:
    parts = [
        "You are an expert viral editor selecting high-retention clips from a TV series, "
        "drama, or movie for YouTube Shorts, TikTok, and Instagram Reels "
        "(similar to viral edits on Unit7 EditZ).",
    ]
    if audience_block:
        parts.append(audience_block)
    parts.append(
        """I am giving you a dialogue transcript. Every line looks like:

    [id] start-end  spoken text"""
    )
    count_line = count_line.replace(
        "strongest chance of performing well as a standalone short",
        "HIGHEST potential to go viral",
    )
    parts.append(
        f"""{count_line}

What makes a viral TV series short:
1. THE HOOK: Immediate tension, witty banter, a power dynamic, an accusation, or a provocative question in the first 2 seconds.
2. THE DIALOGUE BUILDUP: Escalating conflict, humorous exchange, or emotional stake between the characters.
3. THE PAYOFF / MIC-DROP CLIMAX: A punchline, cold flex, mic-drop statement, or sudden twist that hits hard before the edit cuts.
4. SPEAKER IDENTIFICATION: Identify the main characters speaking so the video can use character-specific neon glowing captions (e.g. Cyan for Character A, Gold for Character B).
5. CLIMAX SEGMENT: Pinpoint the EXACT line where the punchline lands so a beat-drop character flash montage can be triggered.

{length_line}"""
    )
    parts.append(COPY_FIELDS)
    parts.append(
        f"""CRITICAL — Reply with ONLY this JSON, no conversational prose, no markdown fences outside the JSON:

{{
  "clips": [
    {{
      "start_seg": 142,
      "end_seg": 165,
      "climax_seg": 164,
      "title": "...",
      "hook": "...",
      "description": "...",
      "tags": "...",
      "score": 90,
      "characters": ["Character A", "Character B"],
      "speaker_segments": {{
        "Character A": [142, 143, 146, 150, 163],
        "Character B": [144, 145, 148, 164]
      }},
      "punchline_words": ["superpower", "rich", "arrested"],
      "reason": "Intense power dynamic ending with a comedic mic-drop flex."
    }}
  ]
}}

Rules:
- start_seg, end_seg, and climax_seg must be valid [id] numbers from the transcript.
- start_seg <= climax_seg <= end_seg.
- Assign dialogue line ids in the clip to their respective character in speaker_segments.
- punchline_words should be 1-4 high-impact words spoken at the climax that deserve intense red glow emphasis.
- Order the clips best-first.
{count_rule}"""
    )
    return "\n\n".join(parts) + "\n"


def build_prompt(
    num_clips: Optional[int] = 5,
    clip_length: Optional[float] = 45.0,
    mode: str = "general",
    audience: Optional[str] = None,
    count_band: Optional[Tuple[int, int]] = None,
) -> str:
    """Return the paste-ready instruction block.

    ``num_clips`` fixes the count; leave it ``None`` and pass ``count_band``
    (from :func:`profiles.clip_count_band`) to let the video's length decide.
    ``audience`` is an already-resolved profile key (``"in"`` / ``"us"``) or
    ``None`` for the generic prompt. ``clip_length`` of ``None`` uses the
    audience's own length band.
    """
    lo, hi = profiles.length_band(audience, clip_length)
    if clip_length:
        length_line = (
            f"Target length: about {int(round(float(clip_length)))} seconds each "
            f"(between {lo} and {hi}). Going somewhat over is fine if the payoff needs it."
        )
    else:
        length_line = (
            f"Target length: {lo} to {hi} seconds each. "
            "Going somewhat over is fine if the payoff needs it."
        )
    count_line, count_rule = _count_sentence(num_clips, count_band)
    audience_block = profiles.context(audience)
    builder = _series_prompt if mode in ("series", "drama", "edits") else _general_prompt
    return builder(count_line, count_rule, length_line, audience_block)


# --------------------------------------------------------------------------- #
# Import: LLM JSON -> validated, resolved windows
# --------------------------------------------------------------------------- #
def parse_windows_json(raw: str) -> List[dict]:
    """Pull a clip list out of whatever the user pasted.

    Models wrap JSON in prose or a markdown fence more often than not, so we
    recover the first balanced object/array rather than demanding clean input.
    """
    if not raw or not raw.strip():
        raise ManualWindowError("No clip JSON was provided.")

    text = raw.strip()
    # Strip a markdown fence if present.
    fence = re.match(r"^`{3}(?:json)?\s*(.*?)\s*`{3}$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        # Fall back to the first {...} or [...] block embedded in prose.
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise ManualWindowError(
                "That doesn't look like JSON. Paste the whole JSON block the "
                "model replied with."
            )
        try:
            data = json.loads(match.group(1))
        except ValueError as exc:
            raise ManualWindowError(f"The JSON could not be parsed: {exc}") from exc

    if isinstance(data, dict):
        for key in ("clips", "shorts", "windows", "moments", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # A single clip object on its own is a reasonable thing to paste.
            data = [data]

    if not isinstance(data, list):
        raise ManualWindowError(
            'Expected a list of clips, e.g. {"clips": [ ... ]}.'
        )
    if not data:
        raise ManualWindowError("The clip list is empty.")
    return data


def _coerce_index(value: Any) -> Optional[int]:
    """Read a segment id that may arrive as 12, '12', or '[12]'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return None


def _to_seconds(value: Any) -> float:
    """Accept 83.5, "83.5", "1:23.5" or "0:01:23.5" as seconds."""
    if isinstance(value, bool):
        raise ValueError("not a time")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        total = 0.0
        for part in text.split(":"):
            total = total * 60 + float(part)
        return total
    return float(text)


def _derive_title(text: str, max_words: int = 8) -> str:
    """Build a readable title from a clip's opening words (fallback only)."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return "Clip"
    title = " ".join(cleaned.split(" ")[:max_words]).strip(" ,.;:-")
    if not title:
        return "Clip"
    if title.islower() or title.isupper():
        title = title.capitalize()
    return title


# --------------------------------------------------------------------------- #
# Snapping: put the cut on real word edges
# --------------------------------------------------------------------------- #
def _word_edges(words: List[dict]) -> Tuple[List[float], List[float]]:
    """Sorted (starts, ends) of every word, tolerant of either key spelling."""
    starts: List[float] = []
    ends: List[float] = []
    for w in words or []:
        try:
            s = float(w.get("start", w.get("s")))
            e = float(w.get("end", w.get("e")))
        except (TypeError, ValueError):
            continue
        starts.append(s)
        ends.append(e)
    starts.sort()
    ends.sort()
    return starts, ends


def snap_to_words(
    start: float,
    end: float,
    words: List[dict],
    duration: float = 0.0,
    search_window: float = 1.5,
    max_lead: float = 0.35,
    max_tail: float = 0.45,
) -> Tuple[float, float]:
    """Move a clip's edges onto word boundaries plus a bit of the silence around.

    The start goes to the nearest word start (within ``search_window`` s) and
    leads into the pause before it by at most ``max_lead``; the end goes to the
    nearest word end and trails into the pause after it by at most
    ``max_tail``. Half the gap at most, so the cut never reaches the
    neighbouring word. With no word near an edge, that edge is left alone.

    Snapping its own output returns the same pair, so a window that is resolved
    once for the preview and again at render time does not creep.
    """
    start, end = float(start), float(end)
    starts, ends = _word_edges(words)
    if not starts:
        return round(start, 3), round(end, 3)

    new_start = start
    lo = bisect_left(starts, start - search_window)
    hi = bisect_right(starts, start + search_window)
    if lo < hi:
        word_start = min(starts[lo:hi], key=lambda s: abs(s - start))
        before = bisect_right(ends, word_start) - 1
        if before >= 0:
            gap = max(0.0, word_start - ends[before])
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    new_end = end
    lo = bisect_left(ends, end - search_window)
    hi = bisect_right(ends, end + search_window)
    if lo < hi:
        word_end = min(ends[lo:hi], key=lambda e: abs(e - end))
        after = bisect_left(starts, word_end)
        if after < len(starts):
            gap = max(0.0, starts[after] - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = word_end + tail
        if duration and duration > 0:
            new_end = min(new_end, float(duration))

    if new_end <= new_start:
        return round(start, 3), round(end, 3)
    return round(new_start, 3), round(new_end, 3)


# --------------------------------------------------------------------------- #
# Copy fields
# --------------------------------------------------------------------------- #
def _clean_text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit] if text else None


def _clean_tags(value: Any) -> Optional[str]:
    """Tags as one 'a, b, c' string, no '#', no repeats, at most MAX_TAGS."""
    if not value:
        return None
    if isinstance(value, str):
        items = re.split(r"[,\n]+", value)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        return None
    seen: List[str] = []
    for item in items:
        tag = item.strip().lstrip("#").strip()
        if tag and tag.lower() not in (t.lower() for t in seen):
            seen.append(tag)
    return ", ".join(seen[:MAX_TAGS]) or None


def _clean_score(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return int(max(0, min(100, round(score))))


def normalise_picker(value: Any) -> str:
    """Which chat answered, as one of KNOWN_PICKERS, or 'chat' when unstated."""
    text = str(value or "").strip().lower()
    if not text:
        return "chat"
    return text if text in KNOWN_PICKERS else "other"


# --------------------------------------------------------------------------- #
# Resolve
# --------------------------------------------------------------------------- #
def resolve_with_report(
    transcript: dict,
    entries: List[dict],
    length_band: Optional[Tuple[int, int]] = None,
    picked_by: Optional[str] = None,
) -> Tuple[List[dict], List[str]]:
    """Turn model-supplied clip entries into (windows, skipped-clip problems).

    A reply that silently loses 2 of 5 clips looks like the model gave 3, so
    the problems of the clips that were dropped are returned, not discarded.

    Segment ids are resolved against the transcript's own timings, so the times
    are exactly what Whisper heard. Entries that give raw ``start``/``end``
    seconds are accepted too (the model may answer in seconds, or the JSON may
    be hand-written), but ids are the supported path. Both are then snapped onto
    word edges unless the entry says it already was (``snapped``).

    ``length_band`` is only used to *warn* about a clip well outside the length
    the audience wants; the user picked these moments on purpose, so nothing is
    rejected for it. ``picked_by`` names the chat that answered.

    Raises:
        ManualWindowError: if nothing usable survives validation. The message
            names the offending clip so the user can fix it in their chat.
    """
    segments = transcript.get("segments") or []
    if not segments:
        raise ManualWindowError(
            "This video has no transcript segments to resolve clip ids against."
        )

    n = len(segments)
    duration = float(transcript.get("duration") or segments[-1].get("end") or 0.0)
    words = transcript.get("words") or []
    default_picker = normalise_picker(picked_by)

    windows: List[dict] = []
    problems: List[str] = []

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            problems.append(
                f"Clip {position}: expected an object, got {type(entry).__name__}."
            )
            continue

        start_seg = _coerce_index(
            entry.get("start_seg", entry.get("start_segment", entry.get("from_seg")))
        )
        end_seg = _coerce_index(
            entry.get("end_seg", entry.get("end_segment", entry.get("to_seg")))
        )

        if start_seg is not None or end_seg is not None:
            # Segment-id path (the supported one).
            if start_seg is None or end_seg is None:
                problems.append(f"Clip {position}: needs both start_seg and end_seg.")
                continue
            if not (0 <= start_seg < n) or not (0 <= end_seg < n):
                problems.append(
                    f"Clip {position}: segment id out of range "
                    f"({start_seg}-{end_seg}; transcript has 0-{n - 1})."
                )
                continue
            if start_seg > end_seg:
                problems.append(
                    f"Clip {position}: start_seg {start_seg} is after end_seg {end_seg}."
                )
                continue
            start = float(segments[start_seg].get("start") or 0.0)
            end = float(segments[end_seg].get("end") or 0.0)
            body = " ".join(
                (segments[i].get("text") or "").strip()
                for i in range(start_seg, end_seg + 1)
            ).strip()
        else:
            # Raw-seconds fallback.
            try:
                start = _to_seconds(entry["start"])
                end = _to_seconds(entry["end"])
            except (KeyError, TypeError, ValueError):
                problems.append(
                    f"Clip {position}: no start_seg/end_seg, and no usable "
                    f"start/end seconds."
                )
                continue
            start = max(0.0, start)
            body = ""

        if duration > 0:
            end = min(end, duration)

        if words and not entry.get("snapped"):
            start, end = snap_to_words(start, end, words, duration)
        length = end - start

        if length < MIN_MANUAL_LEN:
            problems.append(
                f"Clip {position}: only {length:.1f}s long "
                f"(minimum {MIN_MANUAL_LEN:.0f}s)."
            )
            continue
        if length > MAX_MANUAL_LEN:
            problems.append(
                f"Clip {position}: {length:.0f}s long "
                f"(maximum {MAX_MANUAL_LEN:.0f}s)."
            )
            continue

        warnings: List[str] = []
        if length_band:
            lo, hi = length_band
            if length < lo * 0.85:
                warnings.append(f"{length:.0f}s is shorter than the {lo}-{hi}s band.")
            elif length > hi * 1.15:
                warnings.append(f"{length:.0f}s is longer than the {lo}-{hi}s band.")

        title = _clean_text(entry.get("title"), MAX_TITLE_CHARS) or _derive_title(body)

        climax_seg = _coerce_index(
            entry.get("climax_seg", entry.get("climax_segment", entry.get("punchline_seg")))
        )
        climax_time = None
        if climax_seg is not None and 0 <= climax_seg < n:
            climax_time = float(segments[climax_seg].get("end") or 0.0)
        elif entry.get("climax_time") is not None:
            try:
                climax_time = float(entry["climax_time"])
            except (TypeError, ValueError):
                climax_time = None

        raw_chars = entry.get("characters") or []
        if isinstance(raw_chars, str):
            raw_chars = [c.strip() for c in raw_chars.split(",") if c.strip()]
        characters = [str(c).strip() for c in raw_chars if str(c).strip()]

        raw_speaker_segs = (
            entry.get("speaker_segments") or entry.get("speaker_map") or {}
        )
        speaker_segments = {}
        if isinstance(raw_speaker_segs, dict):
            for spk, seg_ids in raw_speaker_segs.items():
                if isinstance(seg_ids, list):
                    clean_ids = []
                    for sid in seg_ids:
                        cid = _coerce_index(sid)
                        if cid is not None:
                            clean_ids.append(cid)
                    speaker_segments[str(spk)] = clean_ids

        raw_punch = entry.get("punchline_words") or []
        if isinstance(raw_punch, str):
            raw_punch = [w.strip() for w in raw_punch.split(",") if w.strip()]
        punchline_words = [str(w).lower().strip() for w in raw_punch if str(w).strip()]

        windows.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "snapped": True,
                "title": title,
                "hook": _clean_text(entry.get("hook") or entry.get("viral_hook_text"), MAX_HOOK_CHARS),
                "description": _clean_text(entry.get("description"), MAX_DESCRIPTION_CHARS),
                "tags": _clean_tags(entry.get("tags") or entry.get("video_tags")),
                "score": _clean_score(entry.get("score", entry.get("predicted_score"))),
                "reason": (entry.get("reason") or "").strip() or None,
                "picked_by": (
                    normalise_picker(entry["picked_by"])
                    if entry.get("picked_by") else default_picker
                ),
                "warnings": warnings,
                "text": body,
                "climax_seg": climax_seg,
                "climax_time": round(climax_time, 2) if climax_time is not None else None,
                "characters": characters,
                "speaker_segments": speaker_segments,
                "punchline_words": punchline_words,
            }
        )

    if not windows:
        detail = " ".join(problems[:5]) or "No usable clips were found."
        raise ManualWindowError(detail)

    return windows, problems


def resolve_windows(
    transcript: dict,
    entries: List[dict],
    length_band: Optional[Tuple[int, int]] = None,
    picked_by: Optional[str] = None,
) -> List[dict]:
    """Same as :func:`resolve_with_report`, for callers that only need the windows."""
    return resolve_with_report(transcript, entries, length_band, picked_by)[0]


def summarise(windows: List[dict], skipped: Optional[List[str]] = None) -> dict:
    """Small report for the UI: count, total runtime, overlaps and warnings.

    Overlapping clips are allowed — the user chose these moments deliberately and
    may well want two takes on the same passage — but they are worth flagging.
    """
    ordered = sorted(windows, key=lambda w: w["start"])
    overlaps = [
        f"{a['title']!r} and {b['title']!r}"
        for a, b in zip(ordered, ordered[1:])
        if a["end"] > b["start"]
    ]
    return {
        "count": len(windows),
        "total_seconds": round(sum(w["end"] - w["start"] for w in windows), 1),
        "overlaps": overlaps,
        "skipped": list(skipped or []),
        "warning_count": sum(len(w.get("warnings") or []) for w in windows),
    }
