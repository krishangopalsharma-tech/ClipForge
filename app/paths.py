"""Central filesystem layout for the project.

All other modules import these so directory locations are defined exactly once.
The directories are created on import (and again at startup) so the very first
run works without any manual setup.

The clips directory is the one location a user can move. Pointing it at a synced
folder (Google Drive, OneDrive, Dropbox) means finished clips land where they are
wanted without a manual copy — and without a second copy eating disk here. It is
read once at import from ``config.json``, so a change takes effect on restart;
everything downstream keeps importing a single ``CLIPS_DIR`` and needs no changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root = the directory that contains the `app/` package.
ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_FILE = ROOT_DIR / "config.json"

DOWNLOADS_DIR = ROOT_DIR / "downloads"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
# Where clips live unless the user points somewhere else. Kept as its own name
# because history entries written before the setting existed refer to it.
DEFAULT_CLIPS_DIR = ROOT_DIR / "clips"
# Small app state that must survive the clips folder moving (the clip history).
DATA_DIR = ROOT_DIR / "data"
# Scratch space for the .ass caption files ffmpeg reads. These MUST stay on the
# project drive: ffmpeg's filtergraph gets them as paths relative to its working
# directory (to dodge Windows drive-colon escaping), and a relative path cannot
# cross drives — which it would if they sat in a clips folder on another disk.
WORK_DIR = DATA_DIR / "work"
ASSETS_DIR = ROOT_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
MASKS_DIR = ASSETS_DIR / "masks"
# Background-music library. Users can drop .mp3/.m4a/.wav files straight in here,
# or upload them through the UI — both show up as track options.
MUSIC_DIR = ASSETS_DIR / "music"
EDITS_MUSIC_DIR = MUSIC_DIR / "edits"
CHARACTERS_DIR = ASSETS_DIR / "characters"
STATIC_DIR = ROOT_DIR / "static"
# Built React dashboard (web/dist) — served as the primary frontend when present.
WEB_DIST_DIR = ROOT_DIR / "web" / "dist"


def load_config() -> dict:
    """Read config.json, or an empty config if it is missing/unreadable."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("config.json is unreadable; using defaults.", exc_info=True)
        return {}


def save_config(config: dict) -> None:
    """Write config.json (pretty-printed, so it stays hand-editable)."""
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def configured_clips_dir() -> Optional[Path]:
    """The user's chosen clips directory, if one is set and still usable.

    A missing or unwritable folder (an external drive that isn't plugged in, a
    sync client that hasn't started) falls back to the default rather than
    failing a render halfway through.
    """
    raw = (load_config().get("clips_dir") or "").strip()
    if not raw:
        return None

    path = Path(raw)
    if not path.is_absolute():
        logger.warning("clips_dir %r is not an absolute path; ignoring it.", raw)
        return None

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "clips_dir %s is not usable (%s); falling back to %s.",
            path, exc, DEFAULT_CLIPS_DIR,
        )
        return None
    return path


CLIPS_DIR = configured_clips_dir() or DEFAULT_CLIPS_DIR


def ensure_dirs() -> None:
    """Create every runtime directory if it does not already exist."""
    for directory in (
        DOWNLOADS_DIR,
        TRANSCRIPTS_DIR,
        CLIPS_DIR,
        DEFAULT_CLIPS_DIR,
        DATA_DIR,
        WORK_DIR,
        ASSETS_DIR,
        FONTS_DIR,
        MASKS_DIR,
        MUSIC_DIR,
        EDITS_MUSIC_DIR,
        CHARACTERS_DIR,
        STATIC_DIR,
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create %s", directory, exc_info=True)


# Create directories eagerly so importing any module is enough to get a working
# tree. Startup also calls ensure_dirs() defensively.
ensure_dirs()
