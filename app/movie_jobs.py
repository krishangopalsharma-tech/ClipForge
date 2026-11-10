"""Background rendering for Movie Shorts.

Each short is built in two steps, and the split is the point of this module.

First the timeline is ASSEMBLED — the chosen slices of film, in order, with
the narration mixed over them — at the film's own size, with no framing, no
grade and no captions. That file becomes a source video in its own right, with
the narration and dialogue written out as its transcript.

Then it is rendered from that source like any other clip, and recorded in the
history the rest of the app reads. So a movie short arrives in the main app as
an ordinary clip: its framing, caption style, logo and music can all be changed
and seen there, instead of being decided blind before a render that takes
minutes to find out you were wrong.

A short that fails does not take the others down with it — a two-hour film
usually yields four or five shorts, and losing all of them because one segment
id pointed at a gap would be a poor trade.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import history, movie_renderer, rendering
from .kokoro_tts import KokoroError
from .models import ClipGenerationError
from .paths import CLIPS_DIR, DOWNLOADS_DIR, TRANSCRIPTS_DIR, WORK_DIR

logger = logging.getLogger(__name__)

_JOBS: Dict[str, "MovieJob"] = {}
_JOBS_LOCK = threading.Lock()


class MovieJob:
    """One render of a whole movie-shorts plan."""

    def __init__(self, source_id: str, plan: dict, source: Path) -> None:
        self.id = uuid.uuid4().hex
        self.clip_id = uuid.uuid4().hex
        self.source_id = source_id
        self.source = source
        self.plan = plan
        self.status = "running"          # running | done | error
        self.progress = 0.0
        self.message = "Preparing..."
        self.error: Optional[str] = None
        self.clips: List[dict] = []
        self.failed: List[str] = []
        self._lock = threading.Lock()

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "clip_id": self.clip_id,
                "status": self.status,
                "progress": round(self.progress, 3),
                "message": self.message,
                "error": self.error,
                "clips": list(self.clips),
                "failed": list(self.failed),
            }


def publish_as_source(assembled: Path, report: dict, language: str = "en") -> str:
    """Move an assembled short into the video library and write its transcript.

    Returns the new source id. After this the short is, as far as the rest of
    the app is concerned, a video someone uploaded and transcribed — which is
    exactly what makes it editable.
    """
    source_id = uuid.uuid4().hex
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    destination = DOWNLOADS_DIR / f"{source_id}.mp4"
    shutil.move(str(assembled), str(destination))

    words = report.get("words") or []
    segments = report.get("segments") or []
    transcript = {
        "language": language,
        "duration": float(report.get("duration") or 0.0),
        "text": " ".join(s.get("text", "") for s in segments).strip(),
        "words": words,
        "segments": segments,
    }
    # Written under the "auto" key, which is what pretranscribe.load falls back
    # to when nothing exists for a specific language.
    path = TRANSCRIPTS_DIR / f"{source_id}__auto.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return source_id


def _recipe(source_id: str, duration: float, options: dict) -> dict:
    """A render recipe for the whole of a freshly published short.

    Built by hand rather than from a GenerateRequest because there is no
    request here — but the KEYS are the ones rendering.make_recipe writes, so
    restyling works on these clips like any other.
    """
    return {
        "version": rendering.RECIPE_VERSION,
        "source_id": source_id,
        "transcript_language": None,
        "start": 0.0,
        "end": round(float(duration), 3),
        "aspect_ratio": options.get("aspect_ratio", "9:16"),
        "fit_mode": options.get("fit_mode", "letterbox"),
        "caption_style": options.get("caption_style", "alex_caps"),
        "caption_overrides": options.get("caption_overrides") or None,
        "cinematic": options.get("cinematic"),
        "music_track": None,
        "music_volume": 35.0,
        "music_duck": 70.0,
        "music_start": 0.0,
        "signature": None,
        "bar_text": options.get("bar_text"),
        "bar_text_color": "#FFFFFF",
        "bar_text_anim": "none",
        "overlays": [],
        "flash_outro": None,
        "flash_music_track": None,
        "window": {
            "characters": [],
            "punchline_words": [],
            "speaker_segments": {},
            "climax_time": None,
        },
    }


def get_job(job_id: str) -> Optional[MovieJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def start(
    source_id: str,
    plan: dict,
    source: Path,
    caption_style: str = "alex_caps",
    caption_overrides: Optional[dict] = None,
    fit_mode: str = "letterbox",
    aspect_ratio: str = "9:16",
) -> MovieJob:
    """Render every short in ``plan`` on a daemon thread."""
    job = MovieJob(source_id, plan, Path(source))
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(
        target=_run,
        args=(job, caption_style, caption_overrides, fit_mode, aspect_ratio),
        daemon=True,
    ).start()
    logger.info(
        "[%s] movie render started: source=%s clips=%d mode=%s",
        job.id, source_id, len(plan.get("clips") or []), plan.get("mode"),
    )
    return job


def _run(
    job: MovieJob,
    caption_style: str,
    caption_overrides: Optional[dict],
    fit_mode: str,
    aspect_ratio: str,
) -> None:
    from . import pretranscribe
    from .models import AspectRatio, FitMode

    fit = FitMode(fit_mode)
    aspect = AspectRatio(aspect_ratio)
    recipes: List[tuple] = []

    try:
        transcript = pretranscribe.load(job.source_id)
        if transcript is None:
            job.update(
                status="error",
                error="No transcript for this video any more.",
                message="Could not render.",
            )
            return

        out_dir = CLIPS_DIR / job.clip_id
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = WORK_DIR / job.clip_id
        work_dir.mkdir(parents=True, exist_ok=True)

        clips = job.plan.get("clips") or []
        voice = job.plan.get("voice") or "am_michael"
        speed = float(job.plan.get("speed") or 1.15)

        for index, clip in enumerate(clips):
            base = index / max(1, len(clips))
            title = clip.get("title") or f"Short {index + 1}"
            job.update(
                progress=base,
                message=f"Short {index + 1} of {len(clips)}: {title}",
            )

            def status(text: str, _base: float = base) -> None:
                job.update(message=f"Short {index + 1}: {text}", progress=_base)

            out_path = out_dir / f"{index}.mp4"
            try:
                # 1. Assemble the timeline at the film's own size, bare.
                assembled = work_dir / f"assembled_{index}.mp4"
                report = movie_renderer.build_short(
                    source=job.source,
                    clip=clip,
                    transcript=transcript,
                    out_path=assembled,
                    work_dir=work_dir,
                    voice=voice,
                    speed=speed,
                    bake_look=False,
                    status=status,
                )

                # 2. It becomes a source video with its own transcript.
                status("Adding it to your videos...")
                new_source_id = publish_as_source(assembled, report)

                # 3. Render it like any other clip, from that source.
                status("Rendering...")
                recipe = _recipe(new_source_id, report["duration"], {
                    "aspect_ratio": aspect.value,
                    "fit_mode": fit.value,
                    "caption_style": caption_style,
                    "caption_overrides": caption_overrides,
                    "bar_text": clip.get("hook_banner"),
                })
                short_transcript = pretranscribe.load(new_source_id)
                rendering.render_clip(
                    recipe=recipe,
                    source_mp4=DOWNLOADS_DIR / f"{new_source_id}.mp4",
                    transcript=short_transcript,
                    clip_id=job.clip_id,
                    index=index,
                    clip_dir=out_dir,
                    work_dir=work_dir,
                    status=status,
                )
                recipes.append((index, recipe, new_source_id))
            except (ClipGenerationError, KokoroError) as exc:
                logger.warning("[%s] short %d failed: %s", job.id, index, exc)
                job.failed.append(f"{title}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad short, not the batch
                logger.exception("[%s] short %d crashed", job.id, index)
                job.failed.append(f"{title}: unexpected error ({exc})")
                continue

            job.clips.append({
                "index": index,
                "title": title,
                "hook_banner": clip.get("hook_banner"),
                "description": clip.get("description"),
                "duration": report["duration"],
                "blocks": report["blocks"],
                "narration_blocks": report["narration_blocks"],
                "source_id": new_source_id,
                "url": f"/clips/{job.clip_id}/{index}.mp4",
            })

        if not job.clips:
            job.update(
                status="error",
                progress=1.0,
                error="; ".join(job.failed) or "Nothing rendered.",
                message="Could not render any short.",
            )
            return

        # Recorded as one generation so the shorts show up in the main app's
        # clip list, each with the recipe that lets it be restyled there.
        by_index = {index: (recipe, src) for index, recipe, src in recipes}
        history.add_entry(
            clip_id=job.clip_id,
            source=f"Movie Shorts — {len(job.clips)} from one film",
            settings={
                "aspect_ratio": aspect.value,
                "fit_mode": fit.value,
                "caption_style": caption_style,
                "movie_shorts": True,
            },
            clips=[{
                "index": c["index"],
                "title": c["title"],
                "hook": c.get("hook_banner"),
                "description": c.get("description"),
                "start": 0.0,
                "end": c["duration"],
                "filename": f"{c['index']}.mp4",
                "recipe": by_index.get(c["index"], (None, None))[0],
            } for c in job.clips],
            source_type="upload",
            source_id=job.source_id,
        )

        done = len(job.clips)
        note = f"{done} short{'s' if done != 1 else ''} ready — edit them in the main app"
        if job.failed:
            note += f" ({len(job.failed)} failed)"
        job.update(status="done", progress=1.0, message=note + ".")
        logger.info("[%s] movie render done: %d ok, %d failed", job.id, done, len(job.failed))

    except Exception as exc:  # noqa: BLE001 - never let the thread die silently
        logger.exception("[%s] movie render failed", job.id)
        job.update(
            status="error",
            error=f"Unexpected error: {exc}",
            message="Could not render.",
        )
