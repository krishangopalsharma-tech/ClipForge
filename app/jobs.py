"""Background job model + in-memory registry for the clip pipeline.

The full pipeline (download -> transcribe -> select -> render) can take minutes,
so running it inside the HTTP request would make the browser appear frozen. Each
request instead creates a :class:`Job`, runs the pipeline on a daemon thread, and
reports fine-grained progress that the frontend streams live (see the SSE
endpoint in ``main.py``).

Everything is in-memory and single-process - perfect for this local, single-user
tool. Restarting the server clears jobs (the rendered clips on disk survive).
"""

from __future__ import annotations

import logging
import random
import shutil
import re
import string
import threading
import time
import uuid
from typing import Dict, List, Optional

from . import (
    downloader, history, manual, pretranscribe, rendering, selector, speech,
    transcriber, uploads,
)
from .models import (
    LANGUAGE_NAMES,
    ClipGenerationError,
    GenerateRequest,
    BadInputError,
    TranscriptionError,
)
from .paths import CLIPS_DIR, WORK_DIR

logger = logging.getLogger("ai_video_clipper.jobs")


# Overall-progress span (0..1) owned by each stage. The per-stage fraction is
# mapped into these bands so the single top progress bar advances smoothly across
# the whole pipeline rather than jumping per stage.
_STAGE_SPANS = {
    "queued": (0.0, 0.0),
    "downloading": (0.02, 0.25),
    "transcribing": (0.25, 0.70),
    "selecting": (0.70, 0.74),
    "rendering": (0.74, 0.99),
    "done": (1.0, 1.0),
    "error": (0.0, 0.0),
}

# Human labels for the stepper in the UI.
STAGE_LABELS = {
    "downloading": "Download",
    "transcribing": "Transcribe",
    "selecting": "Analyze",
    "rendering": "Render",
}

# Characters not allowed in filenames on Windows (and best avoided elsewhere).
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def _shuffle_prefix(length: int = 10) -> str:
    """A random all-letter prefix used to scramble how clips sort in a folder.

    Clips from one video would otherwise sit together in upload dialogs, which
    makes a batch obviously sequential when posted. Letters only (no digits) so
    the prefix never reads as an episode or part number.
    """
    return "".join(random.choices(string.ascii_lowercase, k=length))


def download_filename(title: str, language: Optional[str], index: int) -> str:
    """Suggested download name for a clip: ``<random letters><2 spaces><title>``.

    The random prefix is what mixes clips from the same video into a different
    order every time, so a batch can be uploaded alongside other content without
    looking like one video sliced end to end. Falls back to a numbered name when
    the title is empty. Always ends in ``.mp4``.
    """
    clean_title = _BAD_FILENAME_CHARS.sub(" ", (title or "").strip()).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)
    if not clean_title:
        lang = (language or "").strip().lower()
        native = LANGUAGE_NAMES.get(lang, lang.upper() if lang else "Clip")
        clean_title = f"{native} clip {index + 1}"

    # Trim the title, not the prefix — the prefix is the part doing the shuffling.
    name = f"{_shuffle_prefix()}  {clean_title[:100].strip()}"
    return name + ".mp4"


class Job:
    """A single pipeline run with thread-safe progress + result state."""

    def __init__(self, req: GenerateRequest) -> None:
        self.id = uuid.uuid4().hex
        self.req = req
        self.status = "queued"  # queued | running | done | error
        self.stage = "queued"
        self.stage_progress = 0.0  # 0..1 within the current stage
        self.progress = 0.0  # 0..1 overall (derived from stage span)
        self.message = "Queued..."
        self.clips: List[dict] = []
        self.error: Optional[str] = None
        self.clip_id: Optional[str] = None
        self.cancelled = False
        self._lock = threading.Lock()
        self._rev = 0  # bumped on every change so the SSE stream only sends diffs

    def cancel(self) -> None:
        """Flag the run for cancellation; the pipeline stops at the next checkpoint."""
        with self._lock:
            if self.status in ("done", "error"):
                return
            self.cancelled = True
            self.status = "cancelled"
            self.stage = "cancelled"
            self.message = "Cancelled."
            self._rev += 1

    # -- mutation ---------------------------------------------------------- #
    def set_stage(self, stage: str, frac: float, message: str) -> None:
        """Move to/within a stage and recompute the overall progress band."""
        frac = min(1.0, max(0.0, frac))
        lo, hi = _STAGE_SPANS.get(stage, (0.0, 0.0))
        with self._lock:
            self.status = "running"
            self.stage = stage
            self.stage_progress = frac
            self.progress = lo + frac * (hi - lo)
            self.message = message
            self._rev += 1

    def add_clip(self, clip: dict) -> None:
        """Publish a finished clip so the UI can show it before the run ends."""
        with self._lock:
            self.clips.append(clip)
            self._rev += 1

    def finish(self, clips: List[dict]) -> None:
        with self._lock:
            self.status = "done"
            self.stage = "done"
            self.stage_progress = 1.0
            self.progress = 1.0
            self.message = f"Done - {len(clips)} clip(s) ready."
            self.clips = clips
            self._rev += 1

    def fail(self, message: str) -> None:
        with self._lock:
            self.status = "error"
            self.stage = "error"
            self.error = message
            self.message = message
            self._rev += 1

    # -- read -------------------------------------------------------------- #
    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of the current state."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "stage_progress": round(self.stage_progress, 4),
                "progress": round(self.progress, 4),
                "message": self.message,
                "clips": list(self.clips),
                "error": self.error,
                "clip_id": self.clip_id,
                "rev": self._rev,
            }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def create_job(req: GenerateRequest) -> Job:
    job = Job(req)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def start_job(job: Job) -> None:
    """Run the pipeline for `job` on a background daemon thread."""
    thread = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    thread.start()


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #
def _run_pipeline(job: Job) -> None:
    """download -> transcribe -> select -> render, updating `job` throughout."""
    req = job.req
    logger.info("[%s] starting pipeline: %s", job.id, req.model_dump())

    try:
        # 1) Get the source video. Order of preference:
        #    a) a file already fetched in the background (prefetch) — instant reuse,
        #    b) a previously uploaded file,
        #    c) a fresh URL download.
        source_mp4 = None

        if req.download_id:
            try:
                source_mp4 = uploads.resolve_upload(req.download_id)
                job.set_stage("downloading", 1.0, "Video already downloaded. Preparing...")
            except BadInputError:
                # Prefetched file vanished — fall back to a normal download below.
                logger.warning(
                    "[%s] prefetched file %s missing; re-downloading.",
                    job.id,
                    req.download_id,
                )

        if source_mp4 is None and req.upload_id:
            job.set_stage("downloading", 0.2, "Loading uploaded video...")
            source_mp4 = uploads.resolve_upload(req.upload_id)
            job.set_stage("downloading", 1.0, "Uploaded video ready. Preparing...")

        if source_mp4 is None:
            job.set_stage("downloading", 0.0, "Starting download...")

            def on_download(d: dict) -> None:
                status = d.get("status")
                if status == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate")
                    done = d.get("downloaded_bytes") or 0
                    if total:
                        pct = int(done * 100 / total)
                        job.set_stage("downloading", done / total, f"Downloading video... {pct}%")
                    else:
                        mb = done / 1_048_576
                        job.set_stage("downloading", 0.1, f"Downloading video... {mb:.1f} MB")
                elif status == "finished":
                    job.set_stage("downloading", 1.0, "Download complete. Preparing...")

            source_mp4 = downloader.download_video(req.video_url, progress_hook=on_download)

        # 2) Transcribe locally (word timestamps) on the requested device. The
        # transcript is keyed by the source file id (its stem), so a transcript
        # prepared in the background (pre-transcription, while the user was still
        # adjusting settings) is reused here and this stage finishes instantly.
        clip_id = uuid.uuid4().hex
        job.clip_id = clip_id
        source_id = source_mp4.stem
        device_label = {"auto": "Auto", "cuda": "GPU", "cpu": "CPU"}.get(
            req.device.value, "Auto"
        )
        def on_transcribe(frac: float, msg: str) -> None:
            job.set_stage("transcribing", frac, msg)

        if pretranscribe.cached(source_id, req.language) is not None:
            job.set_stage(
                "transcribing", 1.0, "Using transcript prepared while you set things up..."
            )
        else:
            # A background pre-transcription (started on Step 2) may already be
            # running and holding the shared Whisper lock. Rather than freeze the
            # bar while we block on that lock, mirror the background job's live
            # progress here so the user sees real movement; once it finishes,
            # get_or_transcribe returns its cached result instantly.
            running = pretranscribe.find_running(source_id)
            if running is not None:
                while True:
                    snap = running.snapshot()
                    if snap.get("status") != "running":
                        break
                    job.set_stage(
                        "transcribing", snap.get("progress", 0.0),
                        f"Transcribing audio with local Whisper ({device_label})... "
                        f"{int((snap.get('progress') or 0) * 100)}%",
                    )
                    time.sleep(0.4)
            else:
                job.set_stage(
                    "transcribing", 0.0,
                    f"Transcribing audio with local Whisper ({device_label})...",
                )

        transcript = pretranscribe.get_or_transcribe(
            source_mp4, source_id, req.device.value,
            progress=on_transcribe, language=req.language,
        )
        # The language the captions are actually in: what the user forced, or
        # what Whisper detected. Drives the per-clip download filename.
        forced = (req.language or "").strip().lower()
        caption_language = (
            forced if forced and forced != "auto" else transcript.get("language")
        )

        # A video with next to no speech (music only, a muted mic) transcribes to
        # a few stray words. The selector would score those and cut around them,
        # which is worse than saying so. Clips the caller chose by hand are fine.
        report = speech.speech_report(transcript)
        if report["sparse"]:
            if not req.manual_windows:
                raise ClipGenerationError(
                    f"There is almost no speech in this video ({report['words']} "
                    f"words in {report['minutes']} min), so there is nothing for the "
                    "selector to read. Pick the clips yourself in Studio, or use a "
                    "video with talking in it."
                )
            logger.warning(
                "[%s] sparse speech (%s words); rendering the supplied clips anyway.",
                job.id, report["words"],
            )

        # 3) Choose the clip windows. Either the caller supplied them (picked
        # outside ClipForge — see app/manual.py) or the local selector finds them.
        if req.manual_windows:
            job.set_stage("selecting", 0.3, "Using the clips you supplied...")
            try:
                windows = manual.resolve_windows(
                    transcript,
                    [w.model_dump(exclude_none=True) for w in req.manual_windows],
                )
            except manual.ManualWindowError as exc:
                raise ClipGenerationError(str(exc)) from exc
            job.set_stage(
                "selecting", 1.0,
                f"Using {len(windows)} clip(s) from your JSON.",
            )
        else:
            job.set_stage(
                "selecting", 0.3, "Analyzing transcript for the best moments..."
            )
            windows = selector.select_clips(transcript, req.num_clips, req.clip_length)
            if not windows:
                raise ClipGenerationError(
                    "Could not find any suitable clip windows in this video. "
                    "Try a longer video or fewer clips."
                )
            job.set_stage("selecting", 1.0, f"Found {len(windows)} clip(s) to render.")

        # 4) Per clip: render it from its recipe (app/rendering.py). The same code
        # re-renders the clip later when the user edits it, so a clip looks the
        # same whichever way it was made.
        clip_dir = CLIPS_DIR / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Caption files ffmpeg reads live here, not beside the finished clips —
        # see WORK_DIR in app/paths.py for why.
        work_dir = WORK_DIR / clip_id
        work_dir.mkdir(parents=True, exist_ok=True)

        results: List[dict] = []
        total = len(windows)
        for index, win in enumerate(windows):
            if job.cancelled:
                logger.info("[%s] cancelled before clip %d", job.id, index)
                return
            start, end = float(win["start"]), float(win["end"])
            job.set_stage(
                "rendering",
                index / total,
                f"Rendering clip {index + 1} of {total}...",
            )

            recipe = rendering.make_recipe(req, win, source_id, start, end)
            plan = rendering.render_clip(
                recipe, source_mp4, transcript, clip_id, index, clip_dir, work_dir,
                status=lambda msg, i=index: job.set_stage(
                    "rendering", i / total, f"Clip {i + 1} of {total}: {msg}"
                ),
            )

            title = win.get("title") or f"Clip {index + 1}"
            clip = {
                "index": index,
                "title": title,
                "start": round(start, 2),
                "end": round(end, 2),
                "url": f"/clips/{clip_id}/{index}.mp4",
                "subtitle_url": f"/clips/{clip_id}/{index}.ass",
                "language": caption_language,
                "filename": download_filename(title, caption_language, index),
                # Everything needed to render this clip again with changes.
                "recipe": recipe,
            }
            # Post-ready copy and provenance from a chat-picked clip. Absent on
            # clips the local selector found, so older history entries and the
            # UI both keep working without them.
            for key in ("hook", "description", "tags", "score", "picked_by"):
                if win.get(key) not in (None, ""):
                    clip[key] = win[key]
            if plan is not None:
                clip["reframe"] = plan.summary()
            results.append(clip)
            job.add_clip(clip)  # stream the finished clip to the UI immediately

        job.finish(results)
        logger.info("[%s] pipeline complete: %d clips", job.id, len(results))

        # Persist to history so the History panel survives restarts. Use a
        # meaningful label: the uploaded file's name, or the source URL.
        if req.upload_id:
            source_label = req.upload_name or "Uploaded file"
            source_type = "upload"
        else:
            source_label = req.video_url or "Unknown source"
            source_type = "url"
        try:
            history.add_entry(
                clip_id=clip_id,
                source=source_label,
                source_type=source_type,
                source_id=source_id,
                settings={
                    "aspect_ratio": req.aspect_ratio.value,
                    "fit_mode": req.fit_mode.value,
                    "caption_style": req.caption_style,
                    "num_clips": req.num_clips,
                    "language": caption_language,
                },
                clips=results,
            )
        except Exception:  # noqa: BLE001 - history is best-effort, never fail the job
            logger.warning("[%s] could not write history entry", job.id, exc_info=True)

    except BadInputError as exc:
        logger.warning("[%s] download error: %s", job.id, exc)
        job.fail(str(exc))
    except TranscriptionError as exc:
        logger.warning("[%s] transcription error: %s", job.id, exc)
        job.fail(str(exc))
    except ClipGenerationError as exc:
        logger.warning("[%s] render error: %s", job.id, exc)
        job.fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - never let a thread die silently
        logger.exception("[%s] unexpected pipeline failure", job.id)
        job.fail(f"Unexpected error: {exc}")
    finally:
        # The .ass caption files were only ever input to ffmpeg. Drop them however
        # the job ended, so the work folder doesn't grow with every render.
        if job.clip_id:
            shutil.rmtree(WORK_DIR / job.clip_id, ignore_errors=True)
