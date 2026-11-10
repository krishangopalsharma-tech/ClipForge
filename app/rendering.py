"""Render one clip from a *recipe*, and re-render it later with changes.

Every clip ClipForge makes carries its recipe: the source it was cut from, the
times, the format, the caption style, music, look, signature and overlays. The
recipe is everything the render needs besides the source video and its cached
transcript, so changing the captions or the framing of a finished clip is "edit
the recipe, render again" — no re-download, no re-transcribe, no AI call.

The first render (jobs.py) and every later edit (``restyle_clip``) go through
the same ``render_clip``, so a clip looks the same whichever way it was made.

Each edit renders the whole clip afresh from the source rather than stacking
files on the previous render. That is slower than patching the last file, but a
re-grade never compounds on an earlier one and nothing can drift.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from . import (
    captions, flashes, history, manual, music, overlays, pretranscribe, reframe, uploads,
)
from .clipper import ClipOptions, generate_clip, target_size
from .models import (
    AspectRatio, ClipChanges, ClipGenerationError, FitMode, GenerateRequest,
    InvalidVideoURLError,
)
from .paths import WORK_DIR

logger = logging.getLogger(__name__)

RECIPE_VERSION = 1
MIN_CLIP_SECONDS = manual.MIN_MANUAL_LEN
MAX_CLIP_SECONDS = manual.MAX_MANUAL_LEN


class RestyleError(Exception):
    """An edit that could not be applied. ``status`` is the HTTP status to use."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- #
# Recipe
# --------------------------------------------------------------------------- #
def make_recipe(req: GenerateRequest, win: dict, source_id: str, start: float, end: float) -> dict:
    """Everything needed to render this clip again, as plain JSON-able data."""
    return {
        "version": RECIPE_VERSION,
        "source_id": source_id,
        # The language the transcript is cached under (the cache key), which is
        # what the user chose, not what Whisper detected.
        "transcript_language": req.language,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "aspect_ratio": req.aspect_ratio.value,
        "fit_mode": req.fit_mode.value,
        "caption_style": req.caption_style,
        "caption_overrides": req.caption_overrides.model_dump() if req.caption_overrides else None,
        "cinematic": req.cinematic.model_dump() if req.cinematic else None,
        "music_track": req.music_track,
        "music_volume": req.music_volume if req.music_volume is not None else 35.0,
        "music_duck": req.music_duck if req.music_duck is not None else 70.0,
        "music_start": req.music_start if req.music_start is not None else 0.0,
        "signature": req.signature.model_dump() if req.signature else None,
        "bar_text": req.bar_text,
        "bar_text_color": req.bar_text_color or "#FFFFFF",
        "bar_text_anim": req.bar_text_anim or "none",
        "overlays": [],
        "flash_outro": req.flash_outro,
        "flash_music_track": req.flash_music_track,
        "window": {
            "characters": list(win.get("characters") or []),
            "punchline_words": list(win.get("punchline_words") or []),
            "speaker_segments": dict(win.get("speaker_segments") or {}),
            "climax_time": win.get("climax_time"),
        },
    }


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _tag_speakers(clip_words: list, transcript: dict, speaker_segments: dict) -> None:
    """Put the speaker on each word, from the segment ids the chat assigned."""
    if not speaker_segments:
        return
    seg_to_spk = {}
    for spk, sids in speaker_segments.items():
        for sid in sids:
            seg_to_spk[sid] = spk
    all_segs = transcript.get("segments") or []
    for w in clip_words:
        if w.get("speaker"):
            continue
        for s in all_segs:
            if s["start"] <= w["start"] <= s["end"] + 0.05:
                if s["id"] in seg_to_spk:
                    w["speaker"] = seg_to_spk[s["id"]]
                break


def render_clip(
    recipe: dict,
    source_mp4: Path,
    transcript: dict,
    clip_id: str,
    index: int,
    clip_dir: Path,
    work_dir: Path,
    out_name: Optional[str] = None,
    status: Optional[Callable[[str], None]] = None,
) -> Optional[reframe.ReframePlan]:
    """Render ``recipe`` to ``clip_dir/<out_name or index.mp4>``; return the reframe plan.

    Also leaves the caption file beside the clip (``<index>.ass``) so it can be
    downloaded. Raises ClipGenerationError if ffmpeg fails.
    """
    aspect = AspectRatio(recipe["aspect_ratio"])
    fit = FitMode(recipe["fit_mode"])
    start, end = float(recipe["start"]), float(recipe["end"])
    window = recipe.get("window") or {}
    width, height = target_size(aspect, fit)

    clip_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = (clip_dir / (out_name or f"{index}.mp4")).resolve()

    # Words in this clip, copied so tagging speakers never edits the cached transcript.
    clip_words = [dict(w) for w in (transcript.get("words") or []) if w["end"] > start and w["start"] < end]
    _tag_speakers(clip_words, transcript, window.get("speaker_segments") or {})

    plan = None
    if fit in (FitMode.AUTO, FitMode.TRACK, FitMode.BLUR):
        if status:
            status("Looking at the footage to frame it...")
        plan = reframe.plan_reframe(
            source_mp4, start, end, width, height, fit.value, work_dir, f"{index}",
        )
        logger.info("clip %s/%d reframe: %s (%s)", clip_id, index, plan.mode, plan.reason)

    ass_path = work_dir / f"{index}.ass"
    captions.build_ass(
        words=clip_words,
        style_preset=recipe["caption_style"],
        video_w=width,
        video_h=height,
        out_path=ass_path,
        clip_start=start,
        overrides=recipe.get("caption_overrides"),
        fit_mode=fit.value,
        punchline_words=window.get("punchline_words") or [],
        characters=window.get("characters") or [],
    )

    music_path = None
    if recipe.get("music_track"):
        try:
            music_path = music.resolve_track(recipe["music_track"])
        except InvalidVideoURLError:
            logger.warning("music track %r not found; rendering without it.", recipe["music_track"])

    opts = ClipOptions(
        aspect_ratio=aspect,
        fit_mode=fit,
        ass_path=ass_path,
        clip_id=clip_id,
        index=index,
        bar_text=recipe.get("bar_text"),
        bar_text_color=recipe.get("bar_text_color") or "#FFFFFF",
        bar_text_anim=recipe.get("bar_text_anim") or "none",
        cinematic=recipe.get("cinematic"),
        music_path=music_path,
        music_volume=recipe.get("music_volume", 35.0),
        music_duck=recipe.get("music_duck", 70.0),
        music_start=recipe.get("music_start", 0.0),
        signature=recipe.get("signature"),
        reframe=plan,
        overlays=list(recipe.get("overlays") or []),
    )

    # Flash outro: a beat-synced montage of character stills at the climax.
    flash = recipe.get("flash_outro")
    do_flash = bool(flash or (window.get("climax_time") and flash is not False))
    rendered = False
    if do_flash:
        try:
            characters = window.get("characters") or []
            images = flashes.find_character_images(characters)
            if len(images) < 4:
                images.extend(flashes.extract_fallback_keyframes(
                    source_path=source_mp4, start=start, end=end, count=6,
                    work_dir=work_dir / f"flashes_{index}", width=width, height=height,
                ))
            if images:
                climax = window.get("climax_time")
                climax_end = float(climax) + 0.3 if climax else end
                climax_end = max(start + 2.0, min(end, climax_end))
                dialogue = work_dir / f"dialogue_{index}.mp4"
                outro = work_dir / f"outro_{index}.mp4"
                generate_clip(source_mp4, start, climax_end, opts, out_path=dialogue)
                flashes.render_flash_outro(
                    images=images, output_path=outro, width=width, height=height,
                    beat_duration=0.35,
                    music_track=flashes.resolve_edit_music(recipe.get("flash_music_track")),
                )
                flashes.concat_main_and_outro(dialogue, outro, out_path)
                rendered = True
        except Exception as exc:  # noqa: BLE001 - the plain clip is the fallback
            logger.warning("flash outro failed (%s); rendering the plain clip", exc)
    if not rendered:
        generate_clip(source_mp4, start, end, opts, out_path=out_path)

    try:
        shutil.copy2(ass_path, clip_dir / f"{index}.ass")
    except OSError:
        pass
    return plan


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #
def apply_changes(recipe: dict, changes: ClipChanges, transcript: dict) -> dict:
    """A new recipe with ``changes`` applied, validated. The input is untouched."""
    new = copy.deepcopy(recipe)
    data = changes.model_dump(mode="json", exclude_unset=True)

    for key, value in data.items():
        if key in ("start", "end"):
            continue
        if key == "overlays":
            try:
                new["overlays"] = overlays.normalize(value)
            except overlays.OverlayError as exc:
                raise RestyleError(400, str(exc)) from exc
        elif key == "music_track":
            track = (value or "").strip()
            if track:
                try:
                    music.resolve_track(track)
                except InvalidVideoURLError as exc:
                    raise RestyleError(400, str(exc)) from exc
            new["music_track"] = track or None
        else:
            new[key] = value

    if "start" in data or "end" in data:
        start = float(data.get("start", new["start"]) if data.get("start") is not None else new["start"])
        end = float(data.get("end", new["end"]) if data.get("end") is not None else new["end"])
        duration = float(transcript.get("duration") or 0.0)
        if duration > 0:
            end = min(end, duration)
        words = transcript.get("words") or []
        if words:
            start, end = manual.snap_to_words(start, end, words, duration)
        length = end - start
        if length < MIN_CLIP_SECONDS:
            raise RestyleError(400, f"A clip must be at least {MIN_CLIP_SECONDS:.0f} seconds (this one is {length:.1f}s).")
        if length > MAX_CLIP_SECONDS:
            raise RestyleError(400, f"A clip can be at most {MAX_CLIP_SECONDS:.0f} seconds.")
        new["start"], new["end"] = round(start, 3), round(end, 3)
    return new


_BUSY: set = set()
_BUSY_LOCK = threading.Lock()


def restyle_clip(clip_id: str, index: int, changes: ClipChanges) -> dict:
    """Apply ``changes`` to a finished clip and render it again in place.

    The new render replaces the old file only once it has finished, so a failed
    edit leaves the clip as it was. Returns the updated history clip.
    """
    entry, clip = history.get_clip(clip_id, index)
    if clip is None:
        raise RestyleError(404, "That clip is no longer in your history.")
    recipe = clip.get("recipe")
    if not recipe:
        raise RestyleError(
            409, "This clip was made before editing existed. Generate it again to edit it."
        )

    try:
        source = uploads.resolve_upload(recipe["source_id"])
    except InvalidVideoURLError as exc:
        raise RestyleError(409, "The source video was deleted (Storage page), so this clip cannot be re-rendered.") from exc
    transcript = pretranscribe.load(recipe["source_id"], recipe.get("transcript_language"))
    if transcript is None:
        raise RestyleError(409, "The transcript for this clip was deleted, so it cannot be re-rendered.")

    new_recipe = apply_changes(recipe, changes, transcript)

    key = (clip_id, index)
    with _BUSY_LOCK:
        if key in _BUSY:
            raise RestyleError(409, "This clip is already being re-rendered.")
        _BUSY.add(key)

    clip_dir = history.entry_dir(entry) / clip_id
    final = clip_dir / f"{index}.mp4"
    tmp_name = f"{index}.edit-{uuid.uuid4().hex[:6]}.mp4"
    work_dir = WORK_DIR / f"{clip_id}-edit-{index}-{uuid.uuid4().hex[:6]}"
    try:
        plan = render_clip(new_recipe, source, transcript, clip_id, index, clip_dir, work_dir, out_name=tmp_name)
        os.replace(clip_dir / tmp_name, final)
    except ClipGenerationError as exc:
        raise RestyleError(500, str(exc)) from exc
    finally:
        (clip_dir / tmp_name).unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        with _BUSY_LOCK:
            _BUSY.discard(key)

    version = int(clip.get("version") or 0) + 1
    fields = {
        "recipe": new_recipe,
        "start": round(new_recipe["start"], 2),
        "end": round(new_recipe["end"], 2),
        "version": version,
        "url": f"/clips/{clip_id}/{index}.mp4?v={version}",
    }
    if plan is not None:
        fields["reframe"] = plan.summary()
    updated = history.update_clip(clip_id, index, fields)
    return updated or {**clip, **fields}
