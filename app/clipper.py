"""Cut, reframe, and burn captions with ffmpeg.

Each clip is re-encoded (libx264 + aac) so cuts are frame-accurate and the ASS
captions are burned into the pixels.

``crop`` scales to *cover* the chosen aspect ratio (9:16 / 16:9) and center-crops
to fill it. ``square`` renders a 9:16 canvas with the source cropped to a 1:1
square, given **soft rounded corners**, and centered on black — with an optional
title drawn above it (the "rounded square reel" look). The aspect ratio is
ignored in square mode (the canvas is always 9:16).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import effects, overlays
from .models import AspectRatio, ClipGenerationError, FitMode
from .reframe import ReframePlan
from .paths import CLIPS_DIR, FONTS_DIR, MASKS_DIR

logger = logging.getLogger(__name__)

# Target frame sizes per aspect ratio.
_TARGETS = {
    AspectRatio.NINE_16: (1080, 1920),
    AspectRatio.SIXTEEN_9: (1920, 1080),
}

# Square ("rounded reel") mode: a 9:16 canvas with a centered, rounded 1:1 square.
_SQUARE_CANVAS = (1080, 1920)   # output frame (aspect ratio is ignored)
_SQUARE_INNER = 1020            # side of the centered square
_SQUARE_RADIUS = 60             # corner radius of that square (soft, anti-aliased)


def target_size(aspect_ratio: AspectRatio, fit_mode: FitMode) -> tuple[int, int]:
    """Output (width, height): 9:16 canvas for square mode, else the aspect size."""
    if fit_mode == FitMode.SQUARE:
        return _SQUARE_CANVAS
    return _TARGETS[aspect_ratio]


def ensure_rounded_mask(size: int = _SQUARE_INNER, radius: int = _SQUARE_RADIUS) -> Path:
    """Create (once) and cache a grayscale rounded-rectangle mask via ffmpeg.

    White inside the rounded square, black outside; used by ``alphamerge`` to
    give the centered square its soft corners. Generated with ffmpeg's ``geq``
    so we need no extra image library.
    """
    path = MASKS_DIR / f"rounded_{size}_{radius}.png"
    if path.exists():
        return path
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    edge = size - 1 - radius
    # Distance from each pixel to the inner rectangle's edge (the rounded-rect SDF);
    # a ~1.5px soft ramp around `radius` anti-aliases the corners (smooth, not jaggy).
    # alpha = 255 inside, 0 outside. Commas are escaped for the filtergraph.
    expr = (
        f"255*clip(0.5+({radius}-hypot("
        f"max(0\\,{radius}-X)+max(0\\,X-{edge})\\,"
        f"max(0\\,{radius}-Y)+max(0\\,Y-{edge})))/1.5\\,0\\,1)"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={size}x{size}:d=0.1",
        "-vf", f"geq=lum='{expr}':cb=128:cr=128",
        "-frames:v", "1", str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise ClipGenerationError(
            "ffmpeg was not found on PATH (needed to build the rounded mask)."
        ) from exc
    if proc.returncode != 0 or not path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise ClipGenerationError(
            "Could not generate the rounded-corner mask:\n" + "\n".join(tail)
        )
    return path

_BAR_FONT = FONTS_DIR / "Roboto-Bold.ttf"


@dataclass
class ClipOptions:
    """Everything generate_clip needs beyond the time range."""

    aspect_ratio: AspectRatio
    fit_mode: FitMode
    ass_path: Path
    clip_id: str
    index: int
    bar_text: Optional[str] = None
    bar_text_color: str = "#FFFFFF"     # square title colour
    bar_text_anim: str = "none"         # square title entrance: none | fade | slide
    cinematic: Optional[dict] = None  # cinematic effects config (see app.effects)
    music_path: Optional[Path] = None  # background-music track to mix under the audio
    music_volume: float = 35.0         # 0-100, reels-style (ducked under speech)
    music_duck: float = 70.0           # 0-100, how hard music dips under the voice
    music_start: float = 0.0           # seconds into the track to start from (beat-aligned)
    signature: Optional[dict] = None   # burned-in watermark (see app.models.Signature)
    reframe: Optional[ReframePlan] = None  # face-follow / blur layout (see app.reframe)
    overlays: List[dict] = field(default_factory=list)  # logos / text (see app.overlays)
    logo_base: int = 1                 # ffmpeg input index of the first logo (set by generate_clip)


def _rel_for_filter(target: Path, start_dir: Path) -> str:
    """Return `target` relative to `start_dir` with forward slashes.

    ffmpeg's filtergraph parser treats ':' as an option separator and is fussy
    about Windows drive letters and spaces inside filter VALUES (the `ass`,
    `fontsdir`, and drawtext `fontfile` options). By running ffmpeg with its cwd
    set to the clip folder and passing these as *relative* paths (e.g. `0.ass`,
    `../../assets/fonts`) we avoid the drive colon and spaces entirely, so no
    fragile two-level escaping is needed. All paths live under the project root
    on the same drive, so relpath is always valid.
    """
    return os.path.relpath(str(target), str(start_dir)).replace("\\", "/")


def _escape_drawtext(text: str) -> str:
    """Escape user text for ffmpeg drawtext (the value is wrapped in quotes)."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "’")  # swap apostrophe for a typographic one to avoid quoting hell
        .replace("%", "\\%")
    )


def _ass_filter(opts: ClipOptions, work_dir: Path) -> str:
    """The caption-burn filter. Relative paths keep the drive colon/spaces out."""
    return (
        f"ass={_rel_for_filter(opts.ass_path, work_dir)}:"
        f"fontsdir={_rel_for_filter(FONTS_DIR, work_dir)}"
    )


def _caption_stage(in_label: str, opts: ClipOptions, work_dir: Path) -> str:
    """The final stage that burns the captions onto ``in_label`` -> ``[outv]``."""
    return f"[{in_label}]{_ass_filter(opts, work_dir)}[outv]"


def _signature_stages(in_label: str, sig: Optional[dict], w: int, h: int, work_dir: Path) -> tuple[list[str], str]:
    """Burn the signature/watermark text. Returns (stages, out_label).

    pos_x/pos_y are the text centre as a % of the frame; size is at 1080-wide and
    scales to the real width; opacity 0-100. No-op when disabled/empty.
    """
    if not sig or not sig.get("enabled") or not (sig.get("text") or "").strip():
        return [], in_label
    scale = w / 1080.0
    size = max(10, int(round(float(sig.get("size") or 34) * scale)))
    alpha = max(0.0, min(1.0, float(sig.get("opacity") if sig.get("opacity") is not None else 75) / 100.0))
    col = (sig.get("color") or "#FFFFFF").replace("#", "0x")
    px = max(0.0, min(1.0, float(sig.get("pos_x") if sig.get("pos_x") is not None else 50) / 100.0))
    py = max(0.0, min(1.0, float(sig.get("pos_y") if sig.get("pos_y") is not None else 92) / 100.0))
    txt = _escape_drawtext(sig["text"].strip())
    stage = (
        f"[{in_label}]drawtext=fontfile={_rel_for_filter(_BAR_FONT, work_dir)}:"
        f"text='{txt}':fontcolor={col}@{alpha:.3f}:fontsize={size}:"
        f"x=(w-text_w)*{px:.4f}:y=(h-text_h)*{py:.4f}:"
        f"borderw=2:bordercolor=black@{min(0.55, alpha):.3f}:shadowcolor=black@0.4:shadowx=1:shadowy=1[sig]"
    )
    return [stage], "sig"


def _overlay_stages(in_label: str, opts: ClipOptions, w: int, h: int, work_dir: Path) -> tuple[list[str], str]:
    """Logos (``overlay``) and text blocks (``drawtext``), in list order.

    Positions are fractions of the frame, the centre of the item, so the same
    list works on every output format. A logo is the next ffmpeg input after
    the source/mask/music (``opts.logo_base`` + its order among the logos); the
    caller added them with ``-loop 1`` so a still image lasts the whole clip.
    """
    stages: list[str] = []
    last = in_label
    logo_n = 0
    for k, item in enumerate(opts.overlays or []):
        enable = ""
        start, end = item.get("start"), item.get("end")
        if end is not None:
            enable = f":enable='between(t\\,{float(start or 0):.3f}\\,{float(end):.3f})'"
        elif start is not None:
            enable = f":enable='gte(t\\,{float(start):.3f})'"
        opacity = float(item.get("opacity", 1.0))
        label = f"ovl{k}"
        if item["type"] == "logo":
            idx = opts.logo_base + logo_n
            logo_n += 1
            width = max(8, int(round(float(item["w"]) * w)))
            stages.append(
                f"[{idx}:v]scale={width}:-2,format=rgba,"
                f"colorchannelmixer=aa={opacity:.3f}[lg{k}]"
            )
            stages.append(
                f"[{last}][lg{k}]overlay=x=W*{float(item['x']):.4f}-w/2:"
                f"y=H*{float(item['y']):.4f}-h/2{enable}[{label}]"
            )
        else:
            size = max(10, int(round(float(item["size"]) * h)))
            col = item["color"].replace("#", "0x")
            stages.append(
                f"[{last}]drawtext=fontfile={_rel_for_filter(_BAR_FONT, work_dir)}:"
                f"text='{_escape_drawtext(item['text'])}':fontcolor={col}@{opacity:.3f}:"
                f"fontsize={size}:x=w*{float(item['x']):.4f}-text_w/2:"
                f"y=h*{float(item['y']):.4f}-text_h/2:borderw=3:bordercolor=black@0.7"
                f"{enable}[{label}]"
            )
        last = label
    return stages, last


def _finish_stages(in_label: str, vw: int, vh: int, opts: ClipOptions, work_dir: Path) -> list[str]:
    """Append the cinematic stages (under the captions) then the caption burn.

    Returns the stages that take ``in_label`` -> cinematic effects -> ``[outv]``.
    With no effects this is just the single caption-burn stage, so the default
    render is unchanged. Used by crop mode, where the footage fills the whole
    frame so the effects land on the video. (Square mode applies the effects to
    the square itself — see ``_build_square_filter_complex``.)
    """
    cine_stages, cap_in = effects.cinematic_stages(opts.cinematic, in_label, vw, vh)
    sig_stages, sig_in = _signature_stages(cap_in, opts.signature, vw, vh, work_dir)
    ov_stages, ov_in = _overlay_stages(sig_in, opts, vw, vh, work_dir)
    return cine_stages + sig_stages + ov_stages + [_caption_stage(ov_in, opts, work_dir)]


def _reframe_stage(width: int, height: int, plan: Optional[ReframePlan], work_dir: Path) -> str:
    """The first stage: source -> ``[v0]`` at the output size.

    * track  scale to the output height, then a crop whose ``x`` is moved along
             the face by a ``sendcmd`` script (ffmpeg's crop accepts ``x`` as a
             runtime command).
    * blur   the whole picture at full width centred over a blurred, darkened
             copy that covers the frame. The copy is blurred at 1/8 size and
             scaled back up, which is far cheaper than blurring 1080x1920.
    * else   the plain cover + centre crop.
    """
    if plan is not None and plan.mode == "track" and plan.commands_path is not None:
        script = _rel_for_filter(plan.commands_path, work_dir)
        return (
            f"[0:v]scale={plan.scaled_w}:{height},sendcmd=f={script},"
            f"crop={width}:{height}:{plan.x0}:0,setsar=1[v0]"
        )
    if plan is not None and plan.mode == "blur":
        small_w, small_h = max(2, width // 8), max(2, height // 8)
        return (
            "[0:v]split[bgs][fgs];"
            f"[bgs]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},scale={small_w}:{small_h},boxblur=4:2,"
            f"scale={width}:{height},eq=brightness=-0.06[bg];"
            f"[fgs]scale={width}:-2[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
        )
    return (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}[v0]"
    )


def _build_crop_filter_complex(width: int, height: int, opts: ClipOptions, work_dir: Path) -> str:
    """-filter_complex graph for crop mode: reframe, cinematic FX, then captions."""
    stages = [_reframe_stage(width, height, opts.reframe, work_dir)]
    stages += _finish_stages("v0", width, height, opts, work_dir)
    return ";".join(stages)


def _build_square_filter_complex(opts: ClipOptions, work_dir: Path) -> str:
    """-filter_complex graph for square mode.

    Input 0 = source video, input 1 = the rounded mask. We split the source: one
    branch becomes a 9:16 black canvas (so the canvas shares the video's fps and
    timing), the other is cropped to a square and rounded via ``alphamerge``. The
    rounded square is then ``overlay``-composited onto the black canvas — doing
    the compositing explicitly (rather than ``pad`` + dropping the alpha at encode)
    is what makes the soft corners actually survive into the rendered pixels.
    """
    s = _SQUARE_INNER
    w, h = _SQUARE_CANVAS
    mx, my = (w - s) // 2, (h - s) // 2

    stages = [
        "[0:v]split[base][fg]",
        f"[base]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},drawbox=0:0:iw:ih:black:t=fill[bg]",
        f"[fg]scale={s}:{s}:force_original_aspect_ratio=increase,"
        f"crop={s}:{s}[fgsq]",
    ]

    # Cinematic FX go on the SQUARE itself (vw=vh=s) — exactly the region the live
    # preview grades — so gradients/vignette/grade land on the footage, not on the
    # black canvas around it. (Crop mode applies them to the full frame instead.)
    cine_stages, sq_cine = effects.cinematic_stages(opts.cinematic, "fgsq", s, s)
    stages += cine_stages

    stages += [
        f"[{sq_cine}]format=yuva420p[sq]",
        f"[1:v]format=gray,scale={s}:{s}[m]",
        "[sq][m]alphamerge[r]",
        f"[bg][r]overlay={mx}:{my}[ov]",
    ]
    last = "ov"

    if opts.bar_text and opts.bar_text.strip():
        # Title can be MULTI-LINE (the UI sends \n for Shift+Enter). Each line is a
        # separate drawtext, stacked, so the whole block sits in the top black band
        # just above the square.
        lines = [ln.strip() for ln in opts.bar_text.split("\n") if ln.strip()][:3]
        font_size = max(28, int(round(h * 0.040)))
        line_h = int(round(font_size * 1.3))
        block_h = line_h * len(lines)
        start_y = max(16, my - block_h - 22)  # bottom of block ~22px above the square
        col = (opts.bar_text_color or "#FFFFFF").replace("#", "0x")
        anim = (opts.bar_text_anim or "none").lower()
        # Entrance animation (commas escaped for the filtergraph expression parser):
        #   fade  → alpha ramps 0→1 over 0.5s; slide → drops in from ~40px below.
        alpha_expr = ":alpha='if(lt(t\\,0.5)\\,t/0.5\\,1)'" if anim == "fade" else ""
        for li, ln in enumerate(lines):
            base_y = start_y + li * line_h
            y = (f"'{base_y}+40*(1-min(t/0.45\\,1))'" if anim == "slide" else str(base_y))
            stages.append(
                f"[{last}]drawtext=fontfile={_rel_for_filter(_BAR_FONT, work_dir)}:"
                f"text='{_escape_drawtext(ln)}':"
                f"fontcolor={col}:fontsize={font_size}:x=(w-text_w)/2:y={y}{alpha_expr}:"
                f"borderw=3:bordercolor=black@0.85[ttl{li}]"
            )
            last = f"ttl{li}"

    # Signature/watermark, then captions, on the composited canvas.
    sig_stages, last = _signature_stages(last, opts.signature, w, h, work_dir)
    stages += sig_stages
    ov_stages, last = _overlay_stages(last, opts, w, h, work_dir)
    stages += ov_stages
    stages.append(_caption_stage(last, opts, work_dir))
    return ";".join(stages)


def _music_audio_graph(mus_idx: int, volume: float, duck: float = 70.0) -> str:
    """Filtergraph that mixes a background track UNDER the original audio, reels-style.

    The voice is split: one copy is the sidechain key for ``sidechaincompress``,
    which automatically ducks the music whenever the speaker is talking, and one
    copy is mixed back at full level. So you hear the voice clearly with music
    filling the gaps — never a wall of loud music. ``normalize=0`` keeps the voice
    at unity gain instead of amix halving everything.

    ``volume`` (0-100) sets the music's resting loudness; ``duck`` (0-100) sets how
    hard the music dips while someone is talking, by scaling the sidechain
    compression ratio: 0 leaves the music steady (ratio ≈ 1), 100 pulls it down
    aggressively (ratio ≈ 20) so the voice always cuts through.

    Input 0's audio is the source voice; ``mus_idx`` is the music input's index.
    Produces the ``[aout]`` label the caller maps as the output audio.
    """
    base = max(0.0, min(1.0, (volume if volume is not None else 35) / 100.0 * 0.7))
    d = max(0.0, min(1.0, (duck if duck is not None else 70) / 100.0))
    ratio = 1.0 + d * 19.0  # 0 → 1 (no ducking), 100 → 20 (hard ducking)
    return (
        f"[0:a]asplit=2[__v1][__v2];"
        f"[{mus_idx}:a]volume={base:.3f},aresample=async=1[__m];"
        f"[__m][__v2]sidechaincompress=threshold=0.02:ratio={ratio:.2f}:attack=15:release=300[__md];"
        f"[__v1][__md]amix=inputs=2:duration=first:normalize=0[aout]"
    )


# Shared output-encoding args (everything after the filter graph).
_ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
]


def generate_clip(
    source_mp4: Path,
    start: float,
    end: float,
    opts: ClipOptions,
    out_path: Optional[Path] = None,
) -> Path:
    """Cut [start, end] from `source_mp4`, reframe + caption it, return the mp4.

    Raises:
        ClipGenerationError: if ffmpeg is missing or fails.
    """
    duration = max(0.1, end - start)

    if out_path is None:
        out_dir = (CLIPS_DIR / opts.clip_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{opts.index}.mp4"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg runs here, next to the .ass file and on the same drive as the fonts,
    # so every in-filtergraph path stays relative. The finished mp4 is written to
    # out_dir, which may be anywhere (a synced folder, another disk).
    work_dir = Path(opts.ass_path).resolve().parent
    work_dir.mkdir(parents=True, exist_ok=True)

    src = str(Path(source_mp4).resolve())

    has_music = opts.music_path is not None and Path(opts.music_path).is_file()

    # Logos are extra ffmpeg inputs after the source, the square mask and the
    # music; a logo that vanished from the library is skipped, not fatal.
    items, logo_paths = [], []
    for item in opts.overlays or []:
        if item["type"] == "logo":
            try:
                logo_paths.append(overlays.resolve_logo(item["file"]))
            except overlays.OverlayError as exc:
                logger.warning("Skipping overlay: %s", exc)
                continue
        items.append(item)
    first_free = (2 if opts.fit_mode == FitMode.SQUARE else 1) + (1 if has_music else 0)
    opts = dataclasses.replace(opts, overlays=items, logo_base=first_free)

    # In-filtergraph paths are relative to work_dir (no Windows drive colon /
    # spaces). Inputs/outputs are absolute argv, which is fine.
    if opts.fit_mode == FitMode.SQUARE:
        mask = ensure_rounded_mask()
        fc = _build_square_filter_complex(opts, work_dir)
        inputs = ["-ss", f"{start:.3f}", "-i", src, "-loop", "1", "-i", str(mask.resolve())]
        music_idx = 2  # 0 = source, 1 = mask
        tail = ["-shortest"]
    else:
        width, height = target_size(opts.aspect_ratio, opts.fit_mode)
        fc = _build_crop_filter_complex(width, height, opts, work_dir)
        inputs = ["-ss", f"{start:.3f}", "-i", src]
        music_idx = 1  # 0 = source
        tail = []

    # Background music: loop the track, mix it under the audio, duck it under speech.
    # ``music_start`` seeks into the track first (so a chosen beat lands at the clip
    # start); ``-ss`` before ``-i`` applies to the looped input's first pass.
    if has_music:
        seek = ["-ss", f"{max(0.0, opts.music_start):.2f}"] if opts.music_start and opts.music_start > 0 else []
        inputs += ["-stream_loop", "-1", *seek, "-i", str(Path(opts.music_path).resolve())]
        fc = fc + ";" + _music_audio_graph(music_idx, opts.music_volume, opts.music_duck)
        audio_map = ["-map", "[aout]"]
    else:
        audio_map = ["-map", "0:a?"]
    for path in logo_paths:
        inputs += ["-loop", "1", "-i", str(path.resolve())]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-t", f"{duration:.3f}",
        "-filter_complex", fc,
        "-map", "[outv]", *audio_map,
        *_ENCODE_ARGS,
        *tail,
        str(out_path),
    ]

    logger.info("Rendering clip %d (cwd=%s): %s", opts.index, work_dir, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ClipGenerationError(
            "ffmpeg was not found on PATH. Install it (winget/brew/apt) — it "
            "does the cutting, reframing, and caption burning."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ClipGenerationError(f"Failed to run ffmpeg: {exc}") from exc

    if proc.returncode != 0 or not out_path.exists():
        # Surface the tail of ffmpeg's stderr — it usually pinpoints the problem.
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise ClipGenerationError(
            "ffmpeg failed to render the clip:\n" + "\n".join(tail)
        )

    return out_path
