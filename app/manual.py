"""Manual clip selection — hand the transcript to any LLM, get windows back.

ClipForge's built-in selector is a local heuristic (see :mod:`app.selector`).
This module supports a different workflow: export the transcript, let a strong
external model (Claude, Gemini, whatever the user prefers) read it in their own
chat, and feed the chosen moments back in as JSON.

The key design choice is that the model NEVER supplies a timestamp. It picks
SEGMENT IDS from the exported transcript, and we resolve those ids back to the
exact times Whisper already recorded. Language models drift when asked to emit
precise numbers, and a clip that starts three seconds late loses the hook — so
the model decides *which* moment and this module decides *when* it starts.

Nothing here talks to a network. It formats text and validates JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

# A resolved clip must be long enough to be worth rendering and short enough to
# stay a short. These are deliberately wider than the selector's own bounds —
# the user picked these moments on purpose, so we only reject the absurd.
MIN_MANUAL_LEN = 3.0
MAX_MANUAL_LEN = 900.0


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


# The prompt the user pastes alongside the transcript. Kept here (not in the
# page) so the contract and the validator can never drift apart.
PROMPT_TEMPLATE = """You are selecting short-form clips from a long video for YouTube Shorts and Instagram Reels.

I am giving you a transcript. Every line looks like:

    [id] start-end  spoken text

Pick the {num_clips} moments with the strongest chance of performing well as a standalone short.

What makes a moment work:
- It hooks in the first 2 seconds — a question, a claim, a number, a contradiction, tension.
- It is self-contained. A viewer with zero context understands it.
- It pays off. A setup without its punchline is a dead clip; include both.
- It starts on a clean sentence start and ends after the payoff lands, not mid-thought.

Avoid: intros, outros, sponsor reads, housekeeping, rambling, anything that needs earlier context to make sense.

Target length: {clip_length} seconds each. Going somewhat over is fine if the payoff needs it.

CRITICAL — how to answer:
Do NOT write timestamps. Refer to lines by their [id] only. Reply with ONLY this JSON, no prose, no markdown fence:

{{
  "clips": [
    {{
      "start_seg": 142,
      "end_seg": 158,
      "title": "Short punchy title, max 8 words",
      "reason": "One sentence on why this could travel."
    }}
  ]
}}

Rules:
- start_seg and end_seg are ids that appear in the transcript.
- start_seg must be less than or equal to end_seg.
- Order the clips best-first.
- Return exactly {num_clips} clips unless the transcript genuinely cannot support that many.
"""

PROMPT_TEMPLATE_SERIES = """You are an expert viral editor selecting high-retention clips from a TV series, drama, or movie for YouTube Shorts, TikTok, and Instagram Reels (similar to viral edits on Unit7 EditZ).

I am giving you a dialogue transcript. Every line looks like:

    [id] start-end  spoken text

Pick the {num_clips} moments with the HIGHEST potential to go viral.

What makes a viral TV series short:
1. THE HOOK: Immediate tension, witty banter, a power dynamic, an accusation, or a provocative question in the first 2 seconds.
2. THE DIALOGUE BUILDUP: Escalating conflict, humorous exchange, or emotional stake between the characters.
3. THE PAYOFF / MIC-DROP CLIMAX: A punchline, cold flex, mic-drop statement, or sudden twist that hits hard before the edit cuts.
4. SPEAKER IDENTIFICATION: Identify the main characters speaking so the video can use character-specific neon glowing captions (e.g. Cyan for Character A, Gold for Character B).
5. CLIMAX SEGMENT: Pinpoint the EXACT line where the punchline lands so a beat-drop character flash montage can be triggered.

Target length: {clip_length} seconds each.

CRITICAL — Reply with ONLY this JSON, no conversational prose, no markdown fences outside the JSON:

{{
  "clips": [
    {{
      "start_seg": 142,
      "end_seg": 165,
      "climax_seg": 164,
      "title": "Punchy Title, Max 8 Words",
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
- Return exactly {num_clips} clips ordered best-first.
"""


def build_prompt(
    num_clips: int = 5, clip_length: float = 45.0, mode: str = "general"
) -> str:
    """Return the paste-ready instruction block for the chosen clip count/length."""
    template = (
        PROMPT_TEMPLATE_SERIES
        if mode in ("series", "drama", "edits")
        else PROMPT_TEMPLATE
    )
    return template.format(
        num_clips=int(num_clips), clip_length=int(clip_length)
    )



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
        for key in ("clips", "windows", "moments", "results"):
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


def resolve_windows(transcript: dict, entries: List[dict]) -> List[dict]:
    """Turn model-supplied clip entries into concrete {start, end, title} windows.

    Segment ids are resolved against the transcript's own timings, so the times
    are exactly what Whisper heard. Entries that give raw ``start``/``end``
    seconds are accepted too (useful for hand-written JSON), but ids are the
    supported path.

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
                start = float(entry["start"])
                end = float(entry["end"])
            except (KeyError, TypeError, ValueError):
                problems.append(
                    f"Clip {position}: no start_seg/end_seg, and no usable "
                    f"start/end seconds."
                )
                continue
            body = ""

        if duration > 0:
            end = min(end, duration)
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

        raw_title = (entry.get("title") or "").strip()
        title = raw_title or _derive_title(body)

        climax_seg = _coerce_index(
            entry.get("climax_seg", entry.get("climax_segment", entry.get("punchline_seg")))
        )
        climax_time = None
        if climax_seg is not None and 0 <= climax_seg < n:
            climax_time = float(segments[climax_seg].get("end") or 0.0)

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
                "title": title,
                "reason": (entry.get("reason") or "").strip() or None,
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

    return windows


def summarise(windows: List[dict]) -> dict:
    """Small report for the UI: count, total runtime, and any overlaps.

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
    }
