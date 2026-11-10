"""Beat-synced character flash outro generator.

Creates high-retention "phonk / beat drop edit" endings where rapid cuts of the
main characters flash with white exposure bursts on each music beat (e.g. ~130 BPM).
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from .paths import CHARACTERS_DIR, EDITS_MUSIC_DIR, MUSIC_DIR

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_character_images(characters: Optional[List[str]] = None) -> List[Path]:
    """Search CHARACTERS_DIR for images of the specified characters.

    Looks in subdirectories (e.g. assets/characters/Tim_Bradford/) and for files
    whose names contain the character's name.
    """
    found: List[Path] = []
    if not CHARACTERS_DIR.is_dir():
        return found

    char_names = [c.strip().lower() for c in (characters or []) if c.strip()]

    # 1. Search subdirectories
    for item in CHARACTERS_DIR.iterdir():
        if item.is_dir():
            dir_name = item.name.lower().replace("_", " ").replace("-", " ")
            match = any(c in dir_name or dir_name in c for c in char_names) if char_names else True
            if match:
                for img in item.iterdir():
                    if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                        found.append(img)
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            f_name = item.stem.lower().replace("_", " ").replace("-", " ")
            if not char_names or any(c in f_name for c in char_names):
                found.append(item)

    return sorted(found, key=lambda p: p.name)


def extract_fallback_keyframes(
    source_path: Path,
    start: float,
    end: float,
    count: int,
    work_dir: Path,
    width: int = 1080,
    height: int = 1920,
) -> List[Path]:
    """Extract representative frames from the source video if no photos are provided."""
    work_dir.mkdir(parents=True, exist_ok=True)
    frames: List[Path] = []
    duration = max(1.0, end - start)

    # Sample evenly across the clip duration
    step = duration / (count + 1)
    for i in range(count):
        t = start + (i + 1) * step
        out_img = work_dir / f"extracted_flash_{i:02d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(source_path.resolve()),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-frames:v", "1", "-q:v", "2", str(out_img),
        ]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            if proc.returncode == 0 and out_img.exists() and out_img.stat().st_size > 0:
                frames.append(out_img)
        except Exception:
            logger.exception("Failed to extract frame at %s", t)

    return frames


def resolve_edit_music(track_name: Optional[str] = None) -> Optional[Path]:
    """Find a beat-drop music track from EDITS_MUSIC_DIR or MUSIC_DIR."""
    if track_name:
        cand = EDITS_MUSIC_DIR / track_name
        if cand.is_file():
            return cand
        cand2 = MUSIC_DIR / track_name
        if cand2.is_file():
            return cand2

    # Fallback: check if anything exists in EDITS_MUSIC_DIR
    if EDITS_MUSIC_DIR.is_dir():
        for f in EDITS_MUSIC_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}:
                return f

    # Fallback to general music
    if MUSIC_DIR.is_dir():
        for f in MUSIC_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}:
                return f
    return None


def render_flash_outro(
    images: List[Path],
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    beat_duration: float = 0.35,
    music_track: Optional[Path] = None,
) -> Path:
    """Render a sequence of character flash images with white flash transitions.

    Each cut lasts `beat_duration` (default 0.35s ~ 130 BPM).
    """
    if not images:
        raise ValueError("No images provided for flash outro.")

    # Limit to 4 to 8 images
    images = images[:8] if len(images) >= 4 else (images * 4)[:6]
    n = len(images)
    total_dur = n * beat_duration

    inputs: List[str] = []
    filter_chains: List[str] = []

    for i, img in enumerate(images):
        inputs.extend(["-loop", "1", "-t", f"{beat_duration:.3f}", "-i", str(img.resolve())])
        # Scale to fill frame, center crop, and apply white flash fade-out at t=0
        fc = (
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fade=t=out:st=0:d=0.08:color=white[v{i}]"
        )
        filter_chains.append(fc)

    # Concat video streams
    v_labels = "".join(f"[v{i}]" for i in range(n))
    filter_chains.append(f"{v_labels}concat=n={n}:v=1:a=0[outv]")

    fc_str = ";".join(filter_chains)

    cmd = ["ffmpeg", "-y"] + inputs
    if music_track and music_track.is_file():
        cmd.extend(["-i", str(music_track.resolve())])
        # Mix music with fade out at end
        aud_fc = (
            f"{fc_str};"
            f"[{n}:a]atrim=0:{total_dur:.3f},afade=t=out:st={max(0.1, total_dur - 0.25):.3f}:d=0.25[outa]"
        )
        cmd.extend([
            "-filter_complex", aud_fc,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(output_path.resolve()),
        ])
    else:
        # Generate silent audio so concat doesn't fail on missing audio stream
        cmd.extend([
            "-f", "lavfi", "-t", f"{total_dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", fc_str,
            "-map", "[outv]", "-map", f"{n}:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", str(output_path.resolve()),
        ])

    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0 or not output_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError(f"Could not render flash outro:\n" + "\n".join(tail))

    return output_path


def concat_main_and_outro(main_clip: Path, outro_clip: Path, final_path: Path) -> Path:
    """Concatenate dialogue clip and flash outro into a single video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(main_clip.resolve()),
        "-i", str(outro_clip.resolve()),
        "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(final_path.resolve()),
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0 or not final_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError(f"Could not concatenate main clip and outro:\n" + "\n".join(tail))

    return final_path
