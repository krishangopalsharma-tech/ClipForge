"""Turn a resolved Movie Shorts timeline into one rendered vertical clip.

A short is assembled in a SINGLE ffmpeg command. Every slice of the film and
every narration WAV goes in as its own input, the blocks are built with the
``concat`` filter and the captions are burned onto the result. Doing it in one
pass rather than rendering blocks to disk and stitching them afterwards matters
for quality: each extra encode of already-compressed film footage costs detail
that sharpening then exaggerates.

Block by block:

* ``dialogue``  — one slice of the film, its own audio at full volume, so the
  actor's real voice and lip sync play untouched.
* ``voiceover`` — Kokoro speaks the narration, and its MEASURED length decides
  how much footage plays under it: the block's ``visual_segs`` are cut into
  equal micro-clips that add up to exactly that long. The film's own audio sits
  underneath, ducked, so ambience survives without fighting the narrator.

Captions are merged from both kinds of block onto one timeline — the real word
timings Kokoro reports for narration, the transcript's own words for dialogue.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import captions, effects, kokoro_tts, reframe
from .clipper import (
    _SQUARE_INNER, _SQUARE_RADIUS, _rel_for_filter, ensure_rounded_mask,
    target_size,
)
from .models import AspectRatio, ClipGenerationError, FitMode
from .paths import FONTS_DIR

logger = logging.getLogger(__name__)

CANVAS_W, CANVAS_H = 1080, 1920
FPS = 30

# How far the film's own audio drops under narration.
DUCK_DB = -22.0

# How hard the narration is lifted toward a steady speaking level. Kokoro's
# output is quiet next to a mastered film mix; without this the narrator is the
# quietest thing in his own recap.
NARRATION_EXPANSION = 12.5

# A micro-clip shorter than this reads as a glitch rather than a cut.
MIN_MICRO = 1.2
MAX_MICRO = 4.5

_ENCODE = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
]


def _fit_chain(
    fit: FitMode, w: int, h: int, u: str = "", plan=None,
    work_dir: Optional[Path] = None,
) -> str:
    """How one slice of film is fitted into a ``w x h`` frame, as a filter chain.

    Mirrors what ``clipper`` does for a whole clip, slice by slice, so Movie
    Shorts offers the same framing choices as the rest of the app rather than
    one hard-coded layout.

    ``square`` is not here: it needs the rounded mask as a second input, so it
    is built in :func:`_square_stage`.
    """
    if fit == FitMode.LETTERBOX:
        # Fit the whole picture on black. Crops nothing — which also means any
        # damaged edge in the source survives into the short.
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease"
            ":force_divisible_by=2,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
        )
    if fit == FitMode.BLUR or (plan is not None and plan.mode == "blur"):
        small_w, small_h = max(2, w // 8), max(2, h // 8)
        return (
            f"split[bgs{u}][fgs{u}];"
            f"[bgs{u}]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},scale={small_w}:{small_h},boxblur=4:2,"
            f"scale={w}:{h},eq=brightness=-0.06[bg{u}];"
            f"[fgs{u}]scale={w}:-2[fg{u}];"
            f"[bg{u}][fg{u}]overlay=(W-w)/2:(H-h)/2"
        )
    if plan is not None and plan.mode == "track" and plan.commands_path is not None:
        script = _rel_for_filter(plan.commands_path, work_dir or Path("."))
        return (
            f"scale={plan.scaled_w}:{h},sendcmd=f={script},"
            f"crop={w}:{h}:{plan.x0}:0"
        )
    # crop / centre: scale to cover, then take the middle.
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    )


def _look_chain(grade: str, sharpen: bool) -> str:
    """Grade + sharpening, as a comma chain, or "" when neither is on."""
    parts: List[str] = []
    grade_filter = effects.COLOR_GRADES.get(grade or "none")
    if grade_filter:
        parts.append(grade_filter)
    if sharpen:
        parts.append(
            "cas=strength=0.55,"
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.95:"
            "chroma_msize_x=3:chroma_msize_y=3:chroma_amount=0.25"
        )
    return ",".join(parts)


def _video_stage(
    in_index: int,
    out_label: str,
    grade: str,
    sharpen: bool,
    fit: FitMode = FitMode.LETTERBOX,
    w: int = CANVAS_W,
    h: int = CANVAS_H,
    plan=None,
    work_dir: Optional[Path] = None,
    mask_label: Optional[str] = None,
) -> str:
    """One input's video -> a frame-sized, framed, graded, constant-fps stream.

    ``out_label`` doubles as the unique suffix for any internal labels, so two
    slices in the same graph never collide.
    """
    look = _look_chain(grade, sharpen)
    tail = [f"fps={FPS}", "setsar=1", "format=yuv420p"]

    if fit == FitMode.SQUARE and mask_label:
        return _square_stage(in_index, out_label, look, w, h, mask_label, tail)

    fit_chain = _fit_chain(fit, w, h, out_label, plan, work_dir)
    suffix = "," + ",".join(([look] if look else []) + tail)

    # A fit chain is usually one linear chain, but blur is several ';'-joined
    # stages. Either way the FIRST stage reads the input and the LAST one gets
    # the look, the tail filters and the output label.
    stages = fit_chain.split(";")
    stages[0] = f"[{in_index}:v]{stages[0]}"
    stages[-1] = f"{stages[-1]}{suffix}[{out_label}]"
    return ";".join(stages)


def _square_stage(
    in_index: int, out_label: str, look: str, w: int, h: int,
    mask_label: str, tail: List[str],
) -> str:
    """The rounded 1:1 square centred on a black canvas, for one slice.

    Same composition as ``clipper._build_square_filter_complex`` — the grade
    goes on the SQUARE, not the canvas around it — but labelled per slice and
    taking its mask from a split of the shared mask input.
    """
    s = _SQUARE_INNER
    u = out_label
    mx, my = (w - s) // 2, (h - s) // 2
    square_chain = ",".join([f"scale={s}:{s}:force_original_aspect_ratio=increase",
                             f"crop={s}:{s}"] + ([look] if look else []))
    return ";".join([
        f"[{in_index}:v]split[base{u}][fgi{u}]",
        f"[base{u}]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},drawbox=0:0:iw:ih:black:t=fill[bg{u}]",
        f"[fgi{u}]{square_chain},format=yuva420p[sq{u}]",
        f"[{mask_label}]format=gray,scale={s}:{s}[mk{u}]",
        f"[sq{u}][mk{u}]alphamerge[r{u}]",
        f"[bg{u}][r{u}]overlay={mx}:{my}," + ",".join(tail) + f"[{out_label}]",
    ])


def _audio_stage(in_index: int, out_label: str, gain_db: float = 0.0) -> str:
    """One input's audio -> 48 kHz stereo, optionally ducked.

    ``aresample=async=1`` keeps a slice whose audio starts slightly before or
    after its video from drifting once several slices are concatenated.
    """
    chain = ["aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
             "aresample=async=1"]
    if gain_db:
        chain.append(f"volume={gain_db}dB")
    return f"[{in_index}:a]" + ",".join(chain) + f"[{out_label}]"


def _silent_audio_stage(out_label: str, seconds: float) -> str:
    """A silent stereo bed, for a slice whose source has no audio track."""
    return (
        f"anullsrc=channel_layout=stereo:sample_rate=48000,"
        f"atrim=0:{seconds:.3f},asetpts=PTS-STARTPTS[{out_label}]"
    )


def _has_audio(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True  # assume it has audio; ffmpeg will say otherwise


def _micro_windows(
    segments: List[dict],
    visual_segs: List[int],
    total: float,
    source_duration: float,
) -> List[Tuple[float, float]]:
    """Cut ``total`` seconds of B-roll across the block's visual segments.

    Each window is centred on its segment so the shot is the one the model
    meant, and they are sized to add up to exactly ``total`` — the narration's
    measured length — so the block's picture and voice end together.
    """
    if not visual_segs:
        return []
    count = max(1, min(len(visual_segs), int(total // MIN_MICRO) or 1))
    chosen = visual_segs[:count]
    each = total / len(chosen)
    # Very long narration over few segments would hold one shot too long; very
    # short narration over many would strobe. Clamp, then re-spread the rest.
    if each > MAX_MICRO and len(chosen) < len(visual_segs):
        chosen = visual_segs[:min(len(visual_segs), max(1, int(total // MAX_MICRO) + 1))]
        each = total / len(chosen)

    windows: List[Tuple[float, float]] = []
    remaining = total
    for position, seg_index in enumerate(chosen):
        seg = segments[seg_index]
        mid = (float(seg.get("start") or 0.0) + float(seg.get("end") or 0.0)) / 2.0
        length = each if position < len(chosen) - 1 else remaining
        length = max(0.2, round(length, 3))
        start = max(0.0, mid - length / 2.0)
        if source_duration:
            start = min(start, max(0.0, source_duration - length))
        windows.append((round(start, 3), length))
        remaining = round(remaining - length, 3)
    return windows


def _words_in(words: List[dict], start: float, end: float, offset: float) -> List[dict]:
    """Transcript words inside a window, moved onto the output timeline."""
    out: List[dict] = []
    for word in words:
        ws = float(word.get("start") or 0.0)
        we = float(word.get("end") or 0.0)
        if we <= start or ws >= end:
            continue
        out.append({
            "word": word.get("word", ""),
            "start": round(max(0.0, ws - start) + offset, 3),
            "end": round(max(0.0, min(we, end) - start) + offset, 3),
        })
    return out


def build_short(
    source: Path,
    clip: dict,
    transcript: dict,
    out_path: Path,
    work_dir: Path,
    voice: str = "am_michael",
    speed: float = 1.15,
    # A centre-anchored preset by default: with the picture in the middle of
    # the frame a bottom-anchored caption lands on the black bar under it.
    caption_style: str = "alex_caps",
    caption_overrides: Optional[dict] = None,
    fit_mode: FitMode = FitMode.LETTERBOX,
    aspect_ratio: AspectRatio = AspectRatio.NINE_16,
    status: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Render one resolved movie short. Returns a small report about it.

    ``clip`` is a clip as :func:`app.movie.resolve_clip` returns it: every block
    already carries real times and the dialogue cuts are already snapped onto
    word boundaries.
    """
    source = Path(source)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    segments: List[dict] = transcript.get("segments") or []
    all_words: List[dict] = transcript.get("words") or []
    source_duration = float(transcript.get("duration") or 0.0)
    source_has_audio = _has_audio(source)

    grade = clip.get("grade_preset") or "teal_orange"
    sharpen = bool(clip.get("sharpen", True))

    fit = FitMode(fit_mode)
    width, height = target_size(AspectRatio(aspect_ratio), fit)
    # Square composites the footage through a rounded mask, added as one extra
    # looped input per slice (see ``frame_stage``).
    mask_labels: List[str] = []
    mask_path = ensure_rounded_mask(_SQUARE_INNER, _SQUARE_RADIUS) if fit == FitMode.SQUARE else None

    inputs: List[str] = []          # ffmpeg -i arguments, in order
    stages: List[str] = []          # filter_complex stages
    block_labels: List[Tuple[str, str]] = []   # (video label, audio label) per block
    caption_words: List[dict] = []
    offset = 0.0                    # where we are on the output timeline
    narration_used = 0
    next_input = 0

    def add_slice(start: float, length: float) -> int:
        """Add the film, trimmed to one window, as an input. Returns its index."""
        nonlocal next_input
        inputs.extend(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source)])
        index = next_input
        next_input += 1
        return index

    def frame_stage(index: int, label: str, start: float, length: float) -> str:
        """The video stage for one slice, framed the way the user chose."""
        plan = None
        if fit in (FitMode.AUTO, FitMode.TRACK):
            # Decided per slice, because a recap cuts between shots that want
            # different framing — a wide establishing shot and a close-up.
            plan = reframe.plan_reframe(
                source, start, start + length, width, height,
                fit.value, work_dir, f"{out_path.stem}_{label}",
            )
        mask_label = None
        if fit == FitMode.SQUARE:
            # Each slice gets its own mask input, bounded to that slice's
            # length. Both halves matter: `concat` reads slices one after
            # another, so any branch it has not reached yet must END on its
            # own — an endless looped still (or a split of one) just buffers
            # until the render stalls.
            nonlocal next_input
            inputs.extend([
                "-loop", "1", "-t", f"{length:.3f}", "-i", str(mask_path.resolve()),
            ])
            mask_label = f"{next_input}:v"
            next_input += 1
            mask_labels.append(mask_label)
        return _video_stage(
            index, label, grade, sharpen, fit, width, height,
            plan=plan, work_dir=work_dir, mask_label=mask_label,
        )

    for position, block in enumerate(clip.get("timeline") or []):
        kind = block.get("type")

        if kind == "dialogue":
            start = float(block["start"])
            end = float(block["end"])
            length = max(0.05, end - start)
            index = add_slice(start, length)
            vlabel, alabel = f"bv{position}", f"ba{position}"
            stages.append(frame_stage(index, vlabel, start, length))
            if source_has_audio:
                stages.append(_audio_stage(index, alabel))
            else:
                stages.append(_silent_audio_stage(alabel, length))
            block_labels.append((vlabel, alabel))
            caption_words.extend(_words_in(all_words, start, end, offset))
            offset = round(offset + length, 3)

        elif kind == "voiceover":
            if status:
                status(f"Voicing narration {narration_used + 1}...")
            spoken = kokoro_tts.speak(
                block["narration_text"], voice=voice, speed=speed,
                cache_dir=work_dir.parent / "tts",
            )
            narration_used += 1
            duration = float(spoken["duration"])

            windows = _micro_windows(
                segments, block.get("visual_segs") or [], duration, source_duration
            )
            if not windows:
                # No usable B-roll: hold the film from where the last block left
                # off rather than dropping the narration.
                windows = [(max(0.0, offset), duration)]

            micro_v: List[str] = []
            micro_a: List[str] = []
            for micro_index, (start, length) in enumerate(windows):
                index = add_slice(start, length)
                vlabel = f"mv{position}_{micro_index}"
                alabel = f"ma{position}_{micro_index}"
                stages.append(frame_stage(index, vlabel, start, length))
                if source_has_audio:
                    stages.append(_audio_stage(index, alabel))
                else:
                    stages.append(_silent_audio_stage(alabel, length))
                micro_v.append(f"[{vlabel}]")
                micro_a.append(f"[{alabel}]")

            bedv, beda = f"bedv{position}", f"beda{position}"
            if len(windows) == 1:
                stages.append(f"{micro_v[0]}null[{bedv}]")
                stages.append(f"{micro_a[0]}anull[{beda}]")
            else:
                stages.append(
                    "".join(micro_v) + f"concat=n={len(windows)}:v=1:a=0[{bedv}]"
                )
                stages.append(
                    "".join(micro_a) + f"concat=n={len(windows)}:v=0:a=1[{beda}]"
                )

            # The narration WAV is its own input, mixed over the ducked bed.
            inputs.extend(["-i", str(spoken["path"])])
            narr_index = next_input
            next_input += 1
            nlabel = f"narr{position}"
            # Kokoro returns speech well below the level of a mastered film
            # soundtrack, so without levelling the narrator ends up quieter
            # than the footage he is talking over. ``speechnorm`` lifts him to
            # a consistent level without the pumping a compressor would add.
            stages.append(
                f"[{narr_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                f"channel_layouts=stereo,"
                f"speechnorm=e={NARRATION_EXPANSION}:r=0.0001:l=1[{nlabel}]"
            )

            vlabel, alabel = f"bv{position}", f"ba{position}"
            stages.append(f"[{bedv}]null[{vlabel}]")
            stages.append(
                f"[{beda}]volume={DUCK_DB}dB[duck{position}];"
                f"[duck{position}][{nlabel}]amix=inputs=2:duration=first:"
                f"normalize=0[{alabel}]"
            )
            block_labels.append((vlabel, alabel))

            for word in spoken["words"]:
                caption_words.append({
                    "word": word["word"],
                    "start": round(word["start"] + offset, 3),
                    "end": round(word["end"] + offset, 3),
                })
            offset = round(offset + duration, 3)

    if not block_labels:
        raise ClipGenerationError("This short has no renderable blocks.")

    # Join the blocks end to end.
    if len(block_labels) == 1:
        vout, aout = block_labels[0]
    else:
        pairs = "".join(f"[{v}][{a}]" for v, a in block_labels)
        stages.append(f"{pairs}concat=n={len(block_labels)}:v=1:a=1[cv][ca]")
        vout, aout = "cv", "ca"

    # Captions over the finished picture.
    final_v = vout
    if caption_words:
        ass_path = work_dir / f"{out_path.stem}.ass"
        captions.build_ass(
            words=caption_words,
            style_preset=caption_style,
            video_w=width,
            video_h=height,
            out_path=ass_path,
            clip_start=0.0,
            overrides=caption_overrides,
            # Square pulls the default caption margin up so captions sit under
            # the square rather than off the bottom of the canvas.
            fit_mode=fit.value,
        )
        stages.append(
            f"[{vout}]ass={_rel_for_filter(ass_path, work_dir)}:"
            f"fontsdir={_rel_for_filter(FONTS_DIR, work_dir)}[vcap]"
        )
        final_v = "vcap"

    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    cmd += inputs
    cmd += ["-filter_complex", ";".join(stages)]
    cmd += ["-map", f"[{final_v}]", "-map", f"[{aout}]"]
    cmd += ["-t", f"{offset:.3f}"]
    cmd += _ENCODE
    cmd += [str(out_path)]

    if status:
        status("Rendering the short...")
    logger.info(
        "movie short %s: %d blocks, %d inputs, %.1fs",
        out_path.name, len(block_labels), next_input, offset,
    )
    (work_dir / f"{out_path.stem}.cmd.txt").write_text(
        "\n".join(cmd), encoding="utf-8"
    )

    try:
        done = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
    except OSError as exc:
        raise ClipGenerationError(f"ffmpeg could not be started: {exc}") from exc
    if done.returncode != 0 or not out_path.exists():
        tail = "\n".join((done.stderr or "").strip().splitlines()[-12:])
        raise ClipGenerationError(f"ffmpeg failed while building the short:\n{tail}")

    return {
        "path": out_path,
        "duration": offset,
        "blocks": len(block_labels),
        "narration_blocks": narration_used,
        "words": len(caption_words),
    }


