"""Movie Shorts — the prompt an LLM reads, and the JSON it sends back.

Same contract as :mod:`app.manual`: the model picks SEGMENT IDS out of the
exported transcript and never writes a timestamp, because models drift when
asked to emit precise numbers. What is different here is the shape of a clip. A
manual clip is one window; a movie short is a TIMELINE of blocks, either

* ``dialogue``  — a slice of the film, its own audio at full volume, or
* ``voiceover`` — narration we synthesise, laid over B-roll cut from elsewhere
  in the scene.

Nothing in this module runs ffmpeg or Kokoro; it formats text and validates
JSON. :mod:`app.movie_renderer` turns a resolved clip into video.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .manual import ManualWindowError, snap_to_words

# Modes the tab offers.
SCENE_EDIT = "scene_edit"
VOICEOVER_HYBRID = "voiceover_hybrid"
MODES = (SCENE_EDIT, VOICEOVER_HYBRID)

# Grades the prompt may choose from. These are keys of ``effects.COLOR_GRADES``;
# keeping the list here (rather than offering every grade) stops the model
# picking "bw" for a movie recap.
MOVIE_GRADES = ("teal_orange", "cold_gritty", "vibrant_cinema", "none")

# A short that is shorter than this is not worth rendering; longer than this is
# past what the platforms accept as a short.
MIN_SHORT_LEN = 15.0
MAX_SHORT_LEN = 180.0

# Narration pace used to ESTIMATE a voiceover block's length before Kokoro has
# spoken it. The renderer replaces this with the measured duration.
WORDS_PER_MINUTE = 165.0

# A dialogue block is a punch-in, not a scene. Longer than this and the short
# stops being a recap.
MAX_DIALOGUE_LEN = 12.0

MAX_TITLE_CHARS = 100
MAX_BANNER_CHARS = 120
MAX_DESCRIPTION_CHARS = 500


class MovieJsonError(ManualWindowError):
    """The pasted JSON is not a usable movie-shorts plan."""


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
_SHARED_RULES = """RULES THAT APPLY TO EVERY SHORT:
- Refer to moments ONLY by their [id] from the transcript. Never write a timestamp.
- Pick moments that stand on their own: a viewer who has not seen the film must
  follow it from the first second.
- Open on the most gripping beat you can find, not on setup.
- hook_banner is the line of text held on screen at the start: max 10 words.
- grade_preset is one of: {grades}.
- Output ONE json code block and nothing else."""


def _scene_edit_prompt(count_line: str, length_line: str) -> str:
    return f"""You are editing viral movie shorts from a film's subtitle transcript.
Each transcript line is: [id] start-end  dialogue text

{count_line}
{length_line}

Build each short from the film's OWN dialogue and audio only — there is no
narrator. Cut the dead air: split a scene into several consecutive or
near-consecutive blocks so slow walks and long pauses are dropped and the
tension stays tight.

{_SHARED_RULES.format(grades=", ".join(MOVIE_GRADES))}

JSON SHAPE:
{{
  "mode": "scene_edit",
  "clips": [
    {{
      "clip_id": 1,
      "title": "short viral title",
      "hook_banner": "HE CAN SEE TWO MINUTES INTO THE FUTURE",
      "grade_preset": "teal_orange",
      "sharpen": true,
      "description": "one teasing sentence #movie #shorts",
      "timeline": [
        {{"type": "dialogue", "start_seg": 140, "end_seg": 148, "note": "the hook"}},
        {{"type": "dialogue", "start_seg": 151, "end_seg": 165, "note": "skips the slow walk"}}
      ]
    }}
  ]
}}

TRANSCRIPT:
"""


def _voiceover_hybrid_prompt(count_line: str, length_line: str) -> str:
    return f"""You are writing viral English movie-recap shorts from a film's subtitle transcript.
Each transcript line is: [id] start-end  dialogue text

{count_line}
{length_line}

Each short interleaves TWO kinds of block:

1. "voiceover" — your own narration, which we synthesise. It carries the story:
   setup, what happens next, why it matters. Write narration_text as plain
   spoken English. NEVER use brackets, stage directions, symbols or
   abbreviations — write "fifty million dollars", not "$50M". It is read aloud
   exactly as written, so anything you put in brackets gets spoken.
   Give each voiceover block a "visual_segs" list of 2 to 4 segment ids from
   that part of the film; their footage plays under the narration.

2. "dialogue" — a short slice of the film with the actors' real voices at full
   volume. Use one whenever a character says something iconic, funny or
   devastating. Keep each to a few seconds.

Alternate them. Narration should carry most of the runtime, with dialogue
blocks landing on the moments that hit hardest.

{_SHARED_RULES.format(grades=", ".join(MOVIE_GRADES))}

JSON SHAPE:
{{
  "mode": "voiceover_hybrid",
  "voice": "am_michael",
  "speed": 1.15,
  "clips": [
    {{
      "clip_id": 1,
      "title": "He Built A Fake Village To Teach His Son A Lesson",
      "hook_banner": "HIS DAD SPENT A FORTUNE TO HUMBLE HIM",
      "grade_preset": "vibrant_cinema",
      "sharpen": true,
      "description": "one teasing sentence #movie #recap #shorts",
      "timeline": [
        {{"type": "voiceover",
          "narration_text": "To cure his son's arrogance, a dying billionaire built an entire medieval village and hired a hundred actors to live in it.",
          "visual_segs": [12, 15, 19]}},
        {{"type": "dialogue", "start_seg": 24, "end_seg": 25,
          "note": "the son screaming that his father is rich"}},
        {{"type": "voiceover",
          "narration_text": "Nobody broke character. Not even when the rope went around his neck.",
          "visual_segs": [27, 30, 33]}}
      ]
    }}
  ]
}}

TRANSCRIPT:
"""


def build_movie_prompt(
    mode: str = VOICEOVER_HYBRID,
    num_clips: int = 4,
    clip_length: float = 90.0,
) -> str:
    """The paste-ready instruction block for one of the two movie modes."""
    if mode not in MODES:
        raise MovieJsonError(f"Unknown movie mode {mode!r}.")
    count_line = f"Find {int(num_clips)} shorts in this film."
    target = int(round(float(clip_length)))
    length_line = (
        f"Target length: about {target} seconds each "
        f"(never under {int(MIN_SHORT_LEN)}, never over {int(MAX_SHORT_LEN)})."
    )
    builder = _scene_edit_prompt if mode == SCENE_EDIT else _voiceover_hybrid_prompt
    return builder(count_line, length_line)


# --------------------------------------------------------------------------- #
# Import: pasted JSON -> resolved, snapped timeline
# --------------------------------------------------------------------------- #
def parse_movie_json(raw: str) -> dict:
    """Pull the payload object out of whatever was pasted.

    Models wrap JSON in prose or a markdown fence more often than not, so the
    first balanced object is recovered rather than demanding clean input.
    """
    if not raw or not raw.strip():
        raise MovieJsonError("No JSON was provided.")

    text = raw.strip()
    fence = re.match(r"^`{3}(?:json)?\s*(.*?)\s*`{3}$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except ValueError:
        data = _first_balanced_object(text)

    if not isinstance(data, dict):
        raise MovieJsonError("The JSON should be an object with a \"clips\" list.")
    if not isinstance(data.get("clips"), list) or not data["clips"]:
        raise MovieJsonError("The JSON has no \"clips\" list.")
    mode = str(data.get("mode") or VOICEOVER_HYBRID)
    if mode not in MODES:
        raise MovieJsonError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}.")
    data["mode"] = mode
    return data


def _first_balanced_object(text: str) -> Any:
    """The first {...} that parses, for replies with prose around the JSON."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise MovieJsonError("That does not contain valid JSON.")


def _clean(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit] or None


def _seg_time(segments: List[dict], index: int) -> Tuple[float, float]:
    seg = segments[index]
    return float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)


def estimate_narration_seconds(text: str, speed: float = 1.0) -> float:
    """How long Kokoro will take over ``text`` — an ESTIMATE, for the preview.

    The renderer uses the real measured duration; this only has to be close
    enough for the timeline preview to be honest about a clip's length.
    """
    words = len([w for w in re.split(r"\s+", text or "") if w])
    if not words:
        return 0.0
    return round(words / WORDS_PER_MINUTE * 60.0 / max(0.5, float(speed)), 2)


def resolve_clip(
    clip: dict,
    transcript: dict,
    mode: str,
    speed: float = 1.0,
) -> dict:
    """Validate one clip's timeline and resolve every segment id to real times.

    Returns the clip with each block carrying ``start``/``end`` (snapped onto
    word boundaries) and an ``est_duration``, plus a ``warnings`` list. Raises
    :class:`MovieJsonError` only when the clip cannot be rendered at all.
    """
    segments: List[dict] = transcript.get("segments") or []
    words: List[dict] = transcript.get("words") or []
    duration = float(transcript.get("duration") or 0.0)
    if not segments:
        raise MovieJsonError("The transcript has no segments.")
    last = len(segments) - 1

    timeline = clip.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise MovieJsonError(f"Clip {clip.get('clip_id', '?')} has an empty timeline.")

    warnings: List[str] = []
    blocks: List[dict] = []
    total = 0.0

    for position, raw_block in enumerate(timeline):
        if not isinstance(raw_block, dict):
            warnings.append(f"Block {position + 1} was not an object; skipped.")
            continue
        btype = str(raw_block.get("type") or "").strip().lower()

        if btype == "dialogue":
            try:
                first = int(raw_block["start_seg"])
                final = int(raw_block.get("end_seg", raw_block["start_seg"]))
            except (KeyError, TypeError, ValueError):
                warnings.append(f"Block {position + 1}: no usable start_seg; skipped.")
                continue
            if first > final:
                first, final = final, first
            if first < 0 or final > last:
                warnings.append(
                    f"Block {position + 1}: segments {first}-{final} are outside "
                    f"this transcript (0-{last}); skipped."
                )
                continue
            start = _seg_time(segments, first)[0]
            end = _seg_time(segments, final)[1]
            # The cut lands in a pause instead of clipping a word.
            start, end = snap_to_words(start, end, words, duration)
            length = round(end - start, 3)
            if length <= 0:
                warnings.append(f"Block {position + 1}: empty after snapping; skipped.")
                continue
            if length > MAX_DIALOGUE_LEN:
                warnings.append(
                    f"Block {position + 1}: {length:.1f}s of dialogue is long for a "
                    f"punch-in (over {MAX_DIALOGUE_LEN:.0f}s)."
                )
            blocks.append({
                "type": "dialogue",
                "start_seg": first,
                "end_seg": final,
                "start": start,
                "end": end,
                "est_duration": length,
                "note": _clean(raw_block.get("note"), 200),
            })
            total += length

        elif btype == "voiceover":
            narration = _clean(raw_block.get("narration_text"), 2000)
            if not narration:
                warnings.append(f"Block {position + 1}: voiceover with no text; skipped.")
                continue
            if re.search(r"[\[\](){}]", narration):
                # Kokoro reads brackets aloud, so strip them rather than ship it.
                narration = re.sub(r"\s*[\[\({].*?[\]\)}]\s*", " ", narration)
                narration = re.sub(r"\s+", " ", narration).strip()
                warnings.append(
                    f"Block {position + 1}: removed a bracketed aside from the "
                    "narration (Kokoro would have read it out)."
                )
            visual = raw_block.get("visual_segs")
            ids: List[int] = []
            if isinstance(visual, list):
                for value in visual:
                    try:
                        index = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index <= last:
                        ids.append(index)
            if not ids:
                warnings.append(
                    f"Block {position + 1}: no usable visual_segs, so its B-roll "
                    "will come from around the neighbouring blocks."
                )
            est = estimate_narration_seconds(narration, speed)
            blocks.append({
                "type": "voiceover",
                "narration_text": narration,
                "visual_segs": ids,
                "est_duration": est,
            })
            total += est

        else:
            warnings.append(f"Block {position + 1}: unknown type {btype!r}; skipped.")

    if not blocks:
        raise MovieJsonError(
            f"Clip {clip.get('clip_id', '?')} has no usable blocks left."
        )

    if mode == VOICEOVER_HYBRID:
        if not any(b["type"] == "voiceover" for b in blocks):
            warnings.append("No narration in this clip — it is really a scene edit.")
        if not any(b["type"] == "dialogue" for b in blocks):
            warnings.append(
                "No original dialogue in this clip — the actors are never heard."
            )

    total = round(total, 2)
    if total < MIN_SHORT_LEN:
        warnings.append(f"Only about {total:.0f}s long.")
    if total > MAX_SHORT_LEN:
        warnings.append(f"About {total:.0f}s long, past the {int(MAX_SHORT_LEN)}s limit.")

    grade = str(clip.get("grade_preset") or "teal_orange")
    if grade not in MOVIE_GRADES:
        warnings.append(f"Unknown grade {grade!r}; using teal_orange.")
        grade = "teal_orange"

    return {
        "clip_id": clip.get("clip_id"),
        "title": _clean(clip.get("title"), MAX_TITLE_CHARS),
        "hook_banner": _clean(clip.get("hook_banner"), MAX_BANNER_CHARS),
        "description": _clean(clip.get("description"), MAX_DESCRIPTION_CHARS),
        "grade_preset": grade,
        "sharpen": bool(clip.get("sharpen", True)),
        "timeline": blocks,
        "est_duration": total,
        "warnings": warnings,
    }


def resolve_payload(payload: dict, transcript: dict) -> dict:
    """Resolve every clip in a pasted payload, keeping per-clip problems local.

    A clip that cannot be rendered is reported in ``skipped`` rather than
    failing the whole paste — one bad clip out of five should not cost the
    other four.
    """
    mode = payload.get("mode") or VOICEOVER_HYBRID
    voice = str(payload.get("voice") or payload.get("kokoro_voice") or "am_michael")
    try:
        speed = float(payload.get("speed") or payload.get("kokoro_speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    speed = min(2.0, max(0.5, speed))

    clips: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for index, raw_clip in enumerate(payload["clips"]):
        if not isinstance(raw_clip, dict):
            skipped.append(f"Clip {index + 1}: not an object.")
            continue
        try:
            clips.append(resolve_clip(raw_clip, transcript, mode, speed))
        except MovieJsonError as exc:
            skipped.append(str(exc))

    if not clips:
        raise MovieJsonError(
            "None of the clips could be used. " + (" ".join(skipped) if skipped else "")
        )

    return {
        "mode": mode,
        "voice": voice,
        "speed": speed,
        "clips": clips,
        "skipped": skipped,
        "total_duration": round(sum(c["est_duration"] for c in clips), 2),
    }
