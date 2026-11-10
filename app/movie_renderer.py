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

from . import captions, effects, kokoro_tts
from .clipper import _rel_for_filter
from .models import ClipGenerationError
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


def _letterbox_chain() -> str:
    """Fit the picture inside the frame on black — the movie-recap layout.

    Mirrors ``clipper._reframe_stage``'s letterbox branch, but as a plain filter
    chain (no labels) so it can sit inside a per-input stage here.
    """
    return (
        f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=decrease"
        ":force_divisible_by=2,"
        f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black"
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


def _video_stage(in_index: int, out_label: str, grade: str, sharpen: bool) -> str:
    """One input's video -> a canvas-sized, graded, constant-fps stream."""
    chain = [_letterbox_chain()]
    look = _look_chain(grade, sharpen)
    if look:
        chain.append(look)
    chain.append(f"fps={FPS}")
    chain.append("setsar=1")
    chain.append("format=yuv420p")
    return f"[{in_index}:v]" + ",".join(chain) + f"[{out_label}]"


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
    # A centre-anchored preset by default: in letterbox the picture sits in the
    # middle of the frame, so a bottom-anchored caption would land on the black
    # bar under it rather than on the footage.
    caption_style: str = "alex_caps",
    caption_overrides: Optional[dict] = None,
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

    for position, block in enumerate(clip.get("timeline") or []):
        kind = block.get("type")

        if kind == "dialogue":
            start = float(block["start"])
            end = float(block["end"])
            length = max(0.05, end - start)
            index = add_slice(start, length)
            vlabel, alabel = f"bv{position}", f"ba{position}"
            stages.append(_video_stage(index, vlabel, grade, sharpen))
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
                stages.append(_video_stage(index, vlabel, grade, sharpen))
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
            video_w=CANVAS_W,
            video_h=CANVAS_H,
            out_path=ass_path,
            clip_start=0.0,
            overrides=caption_overrides,
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


