"""Logos and text blocks placed on a finished clip.

The manual replacement for a burned-in title: an item is either a logo (an image
from the library in ``assets/overlays``) or a line of text, placed by fractions
of the frame, so one list lands in the same relative spot on 9:16, 16:9 and the
square reel, and "apply to every clip" needs no per-clip fixing.

This module only validates, stores and looks up. The drawing is ffmpeg's
(``clipper._overlay_stages``): logos through ``overlay``, text through
``drawtext``, under the captions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, BinaryIO, List

from .models import InvalidVideoURLError
from .paths import ASSETS_DIR

OVERLAYS_DIR = ASSETS_DIR / "overlays"
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_LOGO_BYTES = 5 * 1024 * 1024
MAX_ITEMS = 6
MAX_TEXT_CHARS = 120

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# First bytes of the formats above, so a renamed file is not accepted as a logo.
_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",   # png
    b"\xff\xd8\xff",        # jpeg
)


class OverlayError(ValueError):
    """An overlay item that cannot be used; the message says which and why."""


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(_SIGNATURES):
        return True
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def list_logos() -> List[dict]:
    """The logo library as ``{name, file}``, sorted by name."""
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(OVERLAYS_DIR.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            items.append({"name": re.sub(r"[_\-]+", " ", p.stem).strip() or p.stem, "file": p.name})
    return items


def _unique_name(name: str) -> str:
    path = OVERLAYS_DIR / name
    if not path.exists():
        return name
    stem, ext = path.stem, path.suffix
    n = 2
    while (OVERLAYS_DIR / f"{stem}-{n}{ext}").exists():
        n += 1
    return f"{stem}-{n}{ext}"


def save_logo(filename: str, fileobj: BinaryIO) -> dict:
    """Store an uploaded logo. PNG with transparency is the useful case.

    Raises:
        InvalidVideoURLError: wrong type, too big, empty, or not really an image.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise InvalidVideoURLError("A logo must be a PNG, JPG or WebP image.")
    data = fileobj.read(MAX_LOGO_BYTES + 1)
    if not data:
        raise InvalidVideoURLError("The uploaded logo was empty.")
    if len(data) > MAX_LOGO_BYTES:
        raise InvalidVideoURLError("That logo is larger than 5 MB.")
    if not _looks_like_image(data):
        raise InvalidVideoURLError("That file is not a valid image.")
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name) or f"logo{ext}"
    name = _unique_name(safe)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    (OVERLAYS_DIR / name).write_bytes(data)
    return {"name": re.sub(r"[_\-]+", " ", Path(name).stem).strip(), "file": name}


def resolve_logo(name: str) -> Path:
    """The library file for ``name`` (a bare filename), or raise OverlayError."""
    base = Path(str(name or "")).name
    if not base or base != str(name) or Path(base).suffix.lower() not in ALLOWED_EXTS:
        raise OverlayError(f"'{name}' is not a logo in the library.")
    path = (OVERLAYS_DIR / base).resolve()
    if OVERLAYS_DIR.resolve() not in path.parents or not path.is_file():
        raise OverlayError(f"The logo '{base}' was not found. Upload it again.")
    return path


def _num(item: dict, key: str, default: float, lo: float, hi: float) -> float:
    value = item.get(key)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise OverlayError(f"'{key}' must be a number.")
    if number != number:
        raise OverlayError(f"'{key}' must be a number.")
    return max(lo, min(hi, number))


def normalize(items: Any) -> List[dict]:
    """Validate a list of overlay items and fill in defaults.

    Positions and sizes are clamped into range rather than rejected, because a
    slider overshooting by a hair is not an error. Wrong types, an unknown logo,
    empty text or too many items are.
    """
    if items is None:
        return []
    if not isinstance(items, list):
        raise OverlayError("Overlays must be a list.")
    if len(items) > MAX_ITEMS:
        raise OverlayError(f"At most {MAX_ITEMS} overlays per clip.")

    out: List[dict] = []
    for n, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise OverlayError(f"Overlay {n} must be an object.")
        kind = str(raw.get("type") or "").lower()
        item: dict = {"type": kind}
        if kind == "logo":
            item["file"] = resolve_logo(raw.get("file")).name
            item["x"] = _num(raw, "x", 0.85, 0.0, 1.0)
            item["y"] = _num(raw, "y", 0.10, 0.0, 1.0)
            item["w"] = _num(raw, "w", 0.18, 0.02, 1.0)
        elif kind == "text":
            text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip()
            if not text:
                raise OverlayError(f"Overlay {n}: the text is empty.")
            item["text"] = text[:MAX_TEXT_CHARS]
            item["x"] = _num(raw, "x", 0.5, 0.0, 1.0)
            item["y"] = _num(raw, "y", 0.12, 0.0, 1.0)
            item["size"] = _num(raw, "size", 0.05, 0.015, 0.15)
            color = str(raw.get("color") or "#FFFFFF")
            if not _HEX.match(color):
                raise OverlayError(f"Overlay {n}: the colour must look like #RRGGBB.")
            item["color"] = color.upper()
        else:
            raise OverlayError(f"Overlay {n}: type must be 'logo' or 'text'.")

        item["opacity"] = _num(raw, "opacity", 1.0, 0.05, 1.0)
        start = raw.get("start")
        end = raw.get("end")
        if start is not None or end is not None:
            s = _num(raw, "start", 0.0, 0.0, 36000.0)
            item["start"] = s
            if end is not None:
                e = _num(raw, "end", s, 0.0, 36000.0)
                if e <= s:
                    raise OverlayError(f"Overlay {n}: it must end after it starts.")
                item["end"] = e
        out.append(item)
    return out
