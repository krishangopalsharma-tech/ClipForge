"""Audience profiles for the chat-picked clip route.

A profile is the part of the prompt that depends on WHO is watching: the
language of the copy, the clip length that suits the platform, and what a clip
has to do to stand alone. It deliberately says nothing about the channel's
niche, because an invented line about the viewer steers every title the model
writes. Add a "what wins" section from your own numbers, not from taste.

Which profile applies (first match wins):
  1. an explicit choice from the UI / API (``in``, ``us``, ``none``)
  2. the transcript's language (hi/ur/hinglish/... -> ``in``, en -> ``us``)
  3. nothing: the generic prompt, no audience block.

Nothing here talks to a network.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

PROFILES = {
    "in": {
        "label": "India - Hindi / Hinglish",
        "languages": {
            "hinglish", "hi", "ur", "pa", "mr", "gu", "bn", "ne", "ta", "te",
            "kn", "ml", "as", "sa",
        },
        "clip_seconds": (20, 40),
        "context": """AUDIENCE: Hindi / Hinglish speakers watching Shorts on a phone.

WHAT THE CLIP MUST DO:
- Stand alone. Someone who has seen nothing else must understand it.
- Contain its own payoff. On comedy that means the full setup AND the
  punchline, ending 1-2 s after the laugh; a clip that cuts before the payoff
  is worthless however good the setup was.
- Open on the line that carries the moment. Never on a greeting, applause, a
  host introduction, a sponsor read or filler.

COPY STYLE (overrides any other language rule):
- Title and hook in Hinglish written in ROMAN script, the way people type it
  in a chat. Not pure English, not Devanagari.
- The hook names the situation or tension of THIS clip, not the show or the
  topic in general.
- Max 1 emoji. No "POV:" and no translated English templates.
- Hashtags: 3-5, drawn from what the clip is actually about.
- Never put words in a real person's mouth that they did not say.""",
    },
    "us": {
        "label": "USA - English",
        "languages": {"en"},
        "clip_seconds": (25, 50),
        "context": """AUDIENCE: English speakers watching Shorts on a phone.

WHAT THE CLIP MUST DO:
- One idea per clip, and it must be clear without the rest of the video.
- Open on the idea itself, not on background or context-setting.
- Land a concrete payoff: a number, a reveal, a name, or something the viewer
  could repeat to someone else.
- Cut rambling setups.

COPY STYLE (overrides any other language rule):
- Plain English, sentence case. Curiosity-driven but truthful: no fake claims,
  no ALL CAPS, 0-1 emoji.
- Hook: 3-7 words, a question or a statement about THIS moment specifically.
- Hashtags: 3-5, drawn from what the clip is actually about. No #fyp or
  #viral spam.
- Money, insurance, health or legal topics: no promises, no guarantees, no
  "you should buy/cancel X" framing.
- Never put words in a real person's mouth that they did not say.""",
    },
}

# Used when no profile matches: the old behaviour.
GENERIC_CLIP_SECONDS = (20, 45)

# Values the UI / API may send to mean "do not apply any profile".
_NO_PROFILE = {"none", "generic", "off"}


def pick(choice: Optional[str] = None, language: Optional[str] = None) -> Optional[str]:
    """Profile key (``"in"`` / ``"us"``) or ``None`` for the generic prompt.

    ``choice`` is the user's explicit pick and wins, including ``"none"``.
    ``"auto"`` or empty falls through to the transcript language.
    """
    wanted = (choice or "").strip().lower()
    if wanted in _NO_PROFILE:
        return None
    if wanted in PROFILES:
        return wanted
    lang = (language or "").strip().lower()
    for key, profile in PROFILES.items():
        if lang in profile["languages"]:
            return key
    return None


def label(key: Optional[str]) -> str:
    """Human name of a profile for the UI."""
    if key in PROFILES:
        return PROFILES[key]["label"]
    return "Generic (no audience block)"


def context(key: Optional[str]) -> str:
    """The audience block to put at the top of the prompt ('' when none)."""
    return PROFILES[key]["context"] if key in PROFILES else ""


def length_band(
    key: Optional[str], clip_length: Optional[float] = None
) -> Tuple[int, int]:
    """(min, max) seconds a clip should run.

    An explicit ``clip_length`` from the user means "around this long", so the
    band is 70%-130% of it. Without one, the profile's own band applies.
    """
    if clip_length:
        try:
            target = float(clip_length)
        except (TypeError, ValueError):
            target = 0.0
        if target > 0:
            return max(5, int(round(target * 0.7))), max(10, int(round(target * 1.3)))
    if key in PROFILES:
        return PROFILES[key]["clip_seconds"]
    return GENERIC_CLIP_SECONDS


def clip_count_band(duration: float, max_clip_seconds: float = 45.0) -> Tuple[int, int]:
    """How many clips to ask for, given how long the source is.

    A fixed "5" asked of a 4-minute clip is padding and asked of a 2-hour
    episode is starvation. The shortlist grows with the material (one slot per
    ~90 s plus two, capped at 10), limited by how many scoring windows of the
    source actually exist; the band is then a floor that grows with the
    material and a ceiling that lets a rich stretch yield more than one clip
    without inviting padding.
    """
    try:
        duration = max(0.0, float(duration))
    except (TypeError, ValueError):
        duration = 0.0
    window = max(90.0, float(max_clip_seconds) * 1.5)
    step = max(30.0, window - 30.0)  # windows overlap by 30 s
    windows = max(1, int(math.ceil(duration / step))) if duration else 1
    target = max(3, min(10, int(duration // 90) + 2))
    n = max(1, min(target, windows))
    low = max(2, min(6, n // 2 + 2))
    high = min(12, max(4, n * 2))
    return min(low, high), high
