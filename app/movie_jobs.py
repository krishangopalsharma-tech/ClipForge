"""Background rendering for Movie Shorts.

One job renders every short in a resolved plan, one after another, on a daemon
thread while the page polls for progress. Same shape as :mod:`app.verify`'s
job: a dict of live jobs behind a lock, each with thread-safe progress.

A short that fails does not take the others down with it — a two-hour film
usually yields four or five shorts, and losing all of them because one segment
id pointed at a gap would be a poor trade.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import movie_renderer
from .kokoro_tts import KokoroError
from .models import ClipGenerationError
from .paths import CLIPS_DIR, WORK_DIR

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


def get_job(job_id: str) -> Optional[MovieJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def start(source_id: str, plan: dict, source: Path, caption_style: str = "alex_caps") -> MovieJob:
    """Render every short in ``plan`` on a daemon thread."""
    job = MovieJob(source_id, plan, Path(source))
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(
        target=_run, args=(job, caption_style), daemon=True
    ).start()
    logger.info(
        "[%s] movie render started: source=%s clips=%d mode=%s",
        job.id, source_id, len(plan.get("clips") or []), plan.get("mode"),
    )
    return job


def _run(job: MovieJob, caption_style: str) -> None:
    from . import pretranscribe

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
                report = movie_renderer.build_short(
                    source=job.source,
                    clip=clip,
                    transcript=transcript,
                    out_path=out_path,
                    work_dir=work_dir,
                    voice=voice,
                    speed=speed,
                    caption_style=caption_style,
                    status=status,
                )
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

        done = len(job.clips)
        note = f"{done} short{'s' if done != 1 else ''} ready"
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
