"""FastAPI app wiring the local pipeline together.

Startup loads the whisper model once and ensures the bundled font exists. The
heavy pipeline (download -> transcribe -> select -> render) runs on a background
thread as a :class:`~app.jobs.Job` so the request returns instantly; the
frontend then streams live progress over Server-Sent Events. Every domain error
is captured on the job (and surfaced in the stream) so the server never crashes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import (
    captions, export, fonts, history, jobs, kokoro_tts, manual, mood, movie,
    movie_jobs, music, overlays, prefetch, pretranscribe, profiles, rendering,
    sources, speech, storage, transcriber, uploads, verify,
)
from .models import (
    AspectRatio, ClipChanges, Device, FitMode, GenerateRequest, BadInputError,
    TranscriptionError,
)
from .paths import (
    CLIPS_DIR, DEFAULT_CLIPS_DIR, FONTS_DIR, MUSIC_DIR, STATIC_DIR, WEB_DIST_DIR,
    WORK_DIR, ensure_dirs, load_config, save_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_video_clipper")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create dirs, ensure the font, and load whisper ONCE before serving."""
    ensure_dirs()
    fonts.ensure_fonts()
    transcriber.load_model()
    logger.info(
        "Startup complete. Whisper '%s' on %s. No external AI APIs are used.",
        transcriber.MODEL_SIZE,
        transcriber.get_device(),
    )
    yield


app = FastAPI(title="Local AI Video Clipper", version="0.1.0", lifespan=lifespan)

# Permissive CORS for local development (frontend served from the same origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generated clips are served by a route (see `clip_file` below) rather than a
# static mount: the clips folder is user-configurable, and older generations may
# still sit in a previous one.
# Serve caption fonts so the UI can @font-face them for an accurate live preview.
app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")
# Serve the music library so the UI can preview tracks with an <audio> element.
app.mount("/music", StaticFiles(directory=str(MUSIC_DIR)), name="music")
# Serve the logo library so the editor can show thumbnails of the logos.
app.mount("/overlays", StaticFiles(directory=str(overlays.OVERLAYS_DIR)), name="overlays")
# Serve the built React dashboard's hashed JS/CSS bundles (web/dist/assets) when a
# production build exists, so the backend can serve the React app at "/" directly.
if (WEB_DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST_DIR / "assets")), name="assets")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": transcriber.get_device()}


@app.get("/api/caption-styles")
def caption_styles() -> list[dict]:
    """Return the caption style presets for the UI (chips + live preview)."""
    return captions.get_presets_for_api()


@app.get("/api/devices")
def devices() -> dict:
    """Report compute devices the UI may offer for transcription.

    Lets the frontend enable/disable the GPU option and show the active device.
    """
    return {
        "devices": transcriber.available_devices(),
        "default": transcriber.get_device(),
        "cuda_available": transcriber.cuda_available(),
        "gpu_name": transcriber.gpu_name(),
    }


@app.post("/api/warmup")
def warmup(device: Device = Device.AUTO) -> dict:
    """Load the Whisper model on the chosen device and report readiness.

    The frontend calls this when the user changes the Compute dropdown so it can
    show a live "loading / ready / failed" status. Loads are cached per device,
    so re-selecting a warm device returns instantly.
    """
    already = device.value != "auto" and transcriber.is_loaded(device.value)
    try:
        transcriber.load_model(device.value)
        return {
            "status": "ready",
            "device": transcriber.get_device(),
            "cached": already,
        }
    except TranscriptionError as exc:
        return {"status": "error", "device": device.value, "message": str(exc)}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    """Accept a video file from the user's machine and return an upload reference.

    The returned ``upload_id`` is then passed to ``POST /api/generate`` instead of
    a ``video_url``. Runs in the threadpool (sync def) so streaming a large file
    to disk doesn't block the event loop.
    """
    try:
        info = uploads.save_upload(file.filename, file.file)
    except BadInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    return {"status": "ok", **info}


@app.post("/api/upload-srt")
def upload_srt(
    source_id: str = Form(...),
    file: UploadFile = File(...),
    verify_timing: bool = Form(False),
    device: str = Form("auto"),
    language: Optional[str] = Form(None),
) -> dict:
    """Accept a subtitle or timestamped-text file for an uploaded video.

    Reads SubRip (``.srt``), WebVTT (``.vtt``) and plain text with timestamps,
    including Premiere/Resolve timecode (``00:00:20:03``). The file's text is used
    straight away. With ``verify_timing`` Whisper also listens to the video and
    every caption is checked against what is said (see ``GET /api/verify/{id}``
    and ``/api/verify-report/{source_id}``); without it Whisper is skipped, as
    before. ``language`` names the spoken language for that check.
    """
    try:
        fps = None
        try:
            fps = uploads.probe_fps(uploads.resolve_upload(source_id))
        except BadInputError:
            pass
        content = file.file.read()
        res = uploads.save_srt_for_source(source_id, content, file.filename or "", fps)
    except BadInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    out = {"status": "ok", **res, "fps": fps}
    if verify_timing:
        job = verify.start(source_id, device, (language or "").strip() or None)
        out["verify_id"] = job.id
    return out


def _source_id_or_400(source_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", source_id or ""):
        raise HTTPException(status_code=400, detail="Invalid source id.")
    return source_id


class VerifyRequest(BaseModel):
    """Body for POST /api/verify-start."""

    source_id: str
    device: Device = Device.AUTO
    language: Optional[str] = None


@app.post("/api/verify-start")
def verify_start(req: VerifyRequest) -> dict:
    """Check the imported subtitle file's timing against the audio (again).

    Runs Whisper on the video (reused if already done for this language) and
    compares. Poll ``GET /api/verify/{verify_id}``; read the result from
    ``GET /api/verify-report/{source_id}``.
    """
    _source_id_or_400(req.source_id)
    if not verify.path(req.source_id, "srt").is_file():
        raise HTTPException(status_code=409, detail="Upload the subtitle file first.")
    job = verify.start(req.source_id, req.device.value, (req.language or "").strip() or None)
    return {"verify_id": job.id, **job.snapshot()}


@app.get("/api/verify/{job_id}")
def verify_status(job_id: str) -> dict:
    """Progress of a timing check."""
    job = verify.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown verification id.")
    return job.snapshot()


@app.get("/api/verify-report/{source_id}")
def verify_report(source_id: str) -> dict:
    """The saved report: a summary and, for every caption, where it is vs. the audio."""
    report = verify.load_report(_source_id_or_400(source_id))
    if report is None:
        raise HTTPException(status_code=404, detail="This video has no timing report yet.")
    return report


@app.get("/api/verify-report/{source_id}/corrected.srt")
def verify_corrected_srt(source_id: str) -> Response:
    """The corrected captions as an .srt, to take back into Premiere or anywhere else."""
    corrected = verify._read(verify.path(_source_id_or_400(source_id), "verified"))
    if corrected is None:
        raise HTTPException(status_code=404, detail="This video has no corrected captions yet.")
    return Response(
        content=("﻿" + verify.to_srt(corrected["segments"])).encode("utf-8"),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="corrected-{source_id[:8]}.srt"'},
    )


class VerifyApplyRequest(BaseModel):
    """Body for POST /api/verify-apply."""

    source_id: str
    use: str = Field(description="'corrected' (measured word times) or 'original' (the file as imported).")


@app.post("/api/verify-apply")
def verify_apply(req: VerifyApplyRequest) -> dict:
    """Choose which version of the captions the app uses for this video."""
    try:
        return {"status": "ok", **verify.activate(_source_id_or_400(req.source_id), req.use)}
    except verify.VerifyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))



@app.get("/api/fonts")
def get_fonts() -> dict:
    """List caption fonts the UI can offer (bundled trending set + user uploads)."""
    return fonts.list_fonts()


@app.post("/api/fonts/upload")
def upload_font(file: UploadFile = File(...)) -> dict:
    """Accept a user .ttf/.otf font and register it for use in captions."""
    try:
        info = fonts.save_user_font(file.filename, file.file)
    except BadInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    return {"status": "ok", **info}


@app.get("/api/music")
def get_music() -> dict:
    """List background-music tracks (files in assets/music + uploads)."""
    return {"tracks": music.list_tracks()}


@app.post("/api/music/upload")
def upload_music(file: UploadFile = File(...)) -> dict:
    """Accept a user audio file and add it to the background-music library."""
    try:
        info = music.save_track(file.filename, file.file)
    except BadInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    return {"status": "ok", **info}


class PrefetchRequest(BaseModel):
    """Body for POST /api/prefetch — start fetching a URL video in the background."""

    video_url: str


@app.post("/api/prefetch")
def prefetch_start(req: PrefetchRequest) -> dict:
    """Begin downloading a URL video ahead of time and return a prefetch id.

    The frontend calls this when the user reaches Step 2 with a pasted link, then
    polls ``GET /api/prefetch/{id}`` for progress. When done, the returned
    ``download_id`` is passed to ``/api/generate`` so the pipeline reuses the file.
    """
    url = (req.video_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="No video URL was provided.")
    pf = prefetch.start(url)
    return {"prefetch_id": pf.id, **pf.snapshot()}


@app.get("/api/prefetch/{prefetch_id}")
def prefetch_status(prefetch_id: str) -> dict:
    """Return the current progress/result of a background prefetch."""
    pf = prefetch.get(prefetch_id)
    if pf is None:
        raise HTTPException(status_code=404, detail="Unknown prefetch id.")
    return pf.snapshot()


class PretranscribeRequest(BaseModel):
    """Body for POST /api/pretranscribe — transcribe a ready video in the background."""

    source_id: str
    device: Device = Device.AUTO
    language: Optional[str] = None


@app.post("/api/pretranscribe")
def pretranscribe_start(req: PretranscribeRequest) -> dict:
    """Begin transcribing an already-downloaded/uploaded video ahead of Generate.

    Called once the video is on disk (download finished, or an upload). The
    result is cached by source id, so the Generate pipeline reuses it and skips
    the transcribe stage. The frontend polls ``GET /api/pretranscribe/{id}``.
    """
    source_id = (req.source_id or "").strip()
    if not source_id:
        raise HTTPException(status_code=400, detail="No source id was provided.")
    job = pretranscribe.start(source_id, req.device.value, req.language)
    return {"pretranscribe_id": job.id, **job.snapshot()}


@app.get("/api/pretranscribe/{job_id}")
def pretranscribe_status(job_id: str) -> dict:
    """Return the current progress/result of a background pre-transcription."""
    job = pretranscribe.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown pretranscribe id.")
    return job.snapshot()


@app.get("/api/transcript/{source_id}")
def get_transcript(
    source_id: str, language: Optional[str] = None, format: str = "txt"
):
    """Export a prepared transcript for an external model to read.

    ``format=txt`` (the default) returns numbered, timestamped lines and drops
    word-level timings — those are needed to burn captions but are useless for
    judging a moment, and they are most of the file's bulk. ``format=json``
    returns the raw stored transcript.

    Read-only: if the video has not been transcribed yet this 404s rather than
    starting a Whisper run. Use ``POST /api/pretranscribe`` for that.
    """
    transcript = pretranscribe.load(source_id, language)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail="No transcript for this video yet. Transcribe it first.",
        )

    if format == "json":
        # `speech` / `model_hint` are derived, not stored: the Studio page uses
        # them to warn about a video with no speech and about a Hindi/Urdu
        # source decoded by a model that is not fit for it.
        return JSONResponse({
            **transcript,
            "speech": speech.speech_report(transcript),
            "model_hint": speech.model_hint(transcript),
        })

    text = manual.format_transcript_text(transcript, source_id)
    return PlainTextResponse(
        text,
        headers={
            "Content-Disposition": (
                f'attachment; filename="transcript-{source_id[:8]}.txt"'
            )
        },
    )


@app.get("/api/clip-prompt")
def clip_prompt(
    num_clips: Optional[int] = None,
    clip_length: Optional[float] = None,
    mode: str = "general",
    audience: Optional[str] = None,
    source_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """Return the paste-ready instruction block to send with the transcript.

    Served from the backend so the prompt and the JSON validator that has to
    accept the reply can never drift apart.

    With a ``source_id`` whose transcript exists, the audience profile follows
    the spoken language (``audience`` = ``auto``) and an unset ``num_clips`` is
    replaced by a count band sized to the video's length. ``clip_length`` unset
    uses the audience's own length band. Without a transcript, the old fixed
    defaults apply (5 clips) and the prompt is generic unless ``audience`` says
    otherwise.
    """
    transcript = pretranscribe.load(source_id, language) if source_id else None
    spoken = (transcript or {}).get("language") if transcript else None
    key = profiles.pick(audience, spoken)

    count_band = None
    if not num_clips:
        if transcript:
            _, hi = profiles.length_band(key, clip_length)
            count_band = profiles.clip_count_band(transcript.get("duration") or 0.0, hi)
        else:
            num_clips = 5

    prompt = manual.build_prompt(
        num_clips, clip_length, mode=mode, audience=key, count_band=count_band
    )
    return {
        "prompt": prompt,
        "audience": key,
        "audience_label": profiles.label(key),
        "clip_count": num_clips,
        "clip_count_band": list(count_band) if count_band else None,
        "length_band": list(profiles.length_band(key, clip_length)),
    }


class ManualWindowsRequest(BaseModel):
    """Body for POST /api/manual-windows — check clip JSON before rendering."""

    source_id: str
    raw: str = Field(description="Whatever the model replied with (JSON, possibly "
                                 "wrapped in prose or a markdown fence).")
    language: Optional[str] = None
    audience: Optional[str] = Field(
        default=None,
        description="Audience profile used for the prompt (in / us / none / auto); "
        "sets the length band clips are checked against.",
    )
    clip_length: Optional[float] = Field(
        default=None, description="Target length the prompt asked for, if any."
    )
    picked_by: Optional[str] = Field(
        default=None, description="Which chat answered: claude, chatgpt, gemini, other."
    )


@app.post("/api/manual-windows")
def manual_windows(req: ManualWindowsRequest) -> dict:
    """Validate model-supplied clip JSON and resolve it to real timings.

    Lets the UI show exactly what will be rendered — times, durations, and the
    words in each clip — before committing the GPU to it. Segment ids are
    resolved against the transcript, so the timings come from Whisper rather
    than from the model.
    """
    transcript = pretranscribe.load(req.source_id, req.language)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail="No transcript for this video yet. Transcribe it first.",
        )
    key = profiles.pick(req.audience, transcript.get("language"))
    band = profiles.length_band(key, req.clip_length)
    try:
        entries = manual.parse_windows_json(req.raw)
        windows, skipped = manual.resolve_with_report(
            transcript, entries, length_band=band, picked_by=req.picked_by
        )
    except manual.ManualWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "ok",
        "windows": windows,
        "length_band": list(band),
        **manual.summarise(windows, skipped),
    }


# --------------------------------------------------------------------------- #
# Movie Shorts
# --------------------------------------------------------------------------- #
class MoviePlanRequest(BaseModel):
    source_id: str
    raw: str
    language: Optional[str] = None


class MovieRenderRequest(BaseModel):
    source_id: str
    raw: str
    language: Optional[str] = None
    caption_style: str = "alex_caps"
    # Same framing choices as the rest of the app. Letterbox shows the whole
    # frame, which also means it keeps any damaged edge the source has; the
    # cropping modes remove it.
    fit_mode: FitMode = FitMode.LETTERBOX
    aspect_ratio: AspectRatio = AspectRatio.NINE_16
    caption_scale: float = Field(default=1.0, ge=0.3, le=2.0)
    caption_position: Optional[str] = None  # top | center | bottom


@app.get("/api/movie/prompt")
def movie_prompt(
    mode: str = movie.VOICEOVER_HYBRID,
    num_clips: int = 4,
    clip_length: float = 90.0,
    source_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """The paste-ready instruction block, and the transcript to send with it.

    Served from the backend for the same reason as ``/api/clip-prompt``: the
    prompt and the validator that has to accept the reply live together, so
    they cannot drift apart.
    """
    try:
        prompt = movie.build_movie_prompt(mode, num_clips, clip_length)
    except movie.MovieJsonError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    transcript_text = ""
    segments = 0
    if source_id:
        transcript = pretranscribe.load(source_id, language)
        if transcript:
            transcript_text = manual.format_transcript_text(transcript, source_id)
            segments = len(transcript.get("segments") or [])

    return {
        "status": "ok",
        "mode": mode,
        "prompt": prompt,
        "transcript": transcript_text,
        "segments": segments,
        "full": prompt + transcript_text if transcript_text else prompt,
    }


@app.get("/api/movie/voices")
def movie_voices() -> dict:
    """Kokoro's voices, or why it cannot be reached."""
    state = kokoro_tts.status()
    return {
        "status": "ok" if state["ok"] else "unavailable",
        "running": state["ok"],
        "voices": state.get("voices") or [],
        "error": state.get("error"),
        "url": kokoro_tts.KOKORO_URL,
    }


@app.post("/api/movie/plan")
def movie_plan(req: MoviePlanRequest) -> dict:
    """Validate pasted movie JSON and resolve it against the transcript.

    Shows exactly what would be rendered — every block's real times, the
    estimated length of each short and anything questionable about it — before
    the GPU is committed to it.
    """
    transcript = pretranscribe.load(req.source_id, req.language)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail="No transcript for this video yet. Transcribe it first.",
        )
    try:
        payload = movie.parse_movie_json(req.raw)
        resolved = movie.resolve_payload(payload, transcript)
    except movie.MovieJsonError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", **resolved}


@app.post("/api/movie/render")
def movie_render(req: MovieRenderRequest) -> dict:
    """Resolve the JSON again, then render every short on a background thread."""
    transcript = pretranscribe.load(req.source_id, req.language)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail="No transcript for this video yet. Transcribe it first.",
        )
    try:
        payload = movie.parse_movie_json(req.raw)
        resolved = movie.resolve_payload(payload, transcript)
    except movie.MovieJsonError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    needs_voice = any(
        block["type"] == "voiceover"
        for clip in resolved["clips"] for block in clip["timeline"]
    )
    if needs_voice and not kokoro_tts.status()["ok"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "These shorts need narration, but Kokoro is not answering. "
                + kokoro_tts.START_HINT
            ),
        )

    try:
        source = uploads.resolve_upload(req.source_id)
    except BadInputError:
        raise HTTPException(status_code=404, detail="The source video is missing.")

    overrides = {"font_scale": req.caption_scale, "position": req.caption_position}
    job = movie_jobs.start(
        req.source_id, resolved, source,
        caption_style=req.caption_style,
        caption_overrides={k: v for k, v in overrides.items() if v is not None},
        fit_mode=req.fit_mode.value,
        aspect_ratio=req.aspect_ratio.value,
    )
    return {"status": "ok", "job_id": job.id, "clips": len(resolved["clips"])}


@app.get("/api/movie/job/{job_id}")
def movie_job(job_id: str) -> dict:
    """Progress for one movie render."""
    job = movie_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such render.")
    return job.snapshot()


@app.get("/movie")
def movie_page() -> FileResponse:
    """The Movie Shorts page."""
    page = STATIC_DIR / "movie.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="movie.html is missing.")
    return FileResponse(str(page))


@app.get("/clips/{clip_id}/{filename}")
def clip_file(clip_id: str, filename: str) -> FileResponse:
    """Serve one rendered clip or its .ass subtitle file for preview/download.

    Resolves against the folder that generation was actually written to.
    """
    match = re.fullmatch(r"(\d+)\.(mp4|ass|srt)", filename or "")
    if not match:
        raise HTTPException(status_code=404, detail="File not found.")
    idx = int(match.group(1))
    ext = match.group(2)
    # find_entry_dir returns the clips ROOT the generation was written to; the
    # clip's own folder is <root>/<clip_id>. (This used to look in the root
    # itself, which never held a file, and then crashed on an undefined WORK_DIR,
    # so no clip could be previewed or downloaded.)
    path = history.find_entry_dir(clip_id) / clip_id / f"{idx}.{ext}"
    if not path.is_file():
        path = WORK_DIR / clip_id / f"{idx}.{ext}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "video/mp4" if ext == "mp4" else "text/plain; charset=utf-8"
    headers = {}
    if ext in ("ass", "srt"):
        headers["Content-Disposition"] = f'attachment; filename="clip-{idx}.{ext}"'
    return FileResponse(str(path), media_type=media_type, headers=headers)


class SettingsRequest(BaseModel):
    """Body for POST /api/settings — where finished clips should be written."""

    clips_dir: str = Field(
        default="",
        description="Absolute folder for rendered clips. Empty restores the default.",
    )


@app.get("/api/settings")
def get_settings() -> dict:
    """Report the clips folder in use, and the one that will be used next start."""
    configured = (load_config().get("clips_dir") or "").strip()
    return {
        "clips_dir": str(CLIPS_DIR),
        "configured": configured,
        "default_clips_dir": str(DEFAULT_CLIPS_DIR),
        "is_default": CLIPS_DIR.resolve() == DEFAULT_CLIPS_DIR.resolve(),
        "pending_restart": bool(configured)
        and Path(configured).resolve() != CLIPS_DIR.resolve(),
    }


@app.post("/api/settings")
def set_settings(req: SettingsRequest) -> dict:
    """Point rendered clips at another folder (e.g. a synced Drive folder).

    Validated by actually writing a probe file — a path can exist and still be
    unwritable, and finding that out mid-render would waste the whole job. Takes
    effect on restart, because the folder is resolved once at import.
    """
    raw = (req.clips_dir or "").strip().strip('"')
    config = load_config()

    if not raw:
        config.pop("clips_dir", None)
        save_config(config)
        return {
            "status": "ok",
            "clips_dir": str(DEFAULT_CLIPS_DIR),
            "restart_required": CLIPS_DIR.resolve() != DEFAULT_CLIPS_DIR.resolve(),
        }

    target = Path(raw)
    if not target.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="Use a full path, e.g. G:\My Drive\ClipForge.",
        )
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".clipforge-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"That folder can't be written to: {exc}"
        )

    config["clips_dir"] = str(target)
    save_config(config)
    logger.info("Clips folder set to %s (effective after restart)", target)
    return {
        "status": "ok",
        "clips_dir": str(target),
        "restart_required": target.resolve() != CLIPS_DIR.resolve(),
    }


@app.get("/api/storage")
def storage_report() -> dict:
    """Report disk usage: rendered clips, source videos, and cached transcripts."""
    return storage.scan()


@app.delete("/api/storage/generation/{clip_id}")
def storage_delete_generation(clip_id: str) -> dict:
    """Delete every clip from one generation, plus its history entry."""
    try:
        return storage.delete_generation(clip_id)
    except storage.StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/storage/source/{source_id}")
def storage_delete_source(source_id: str, keep_transcript: bool = False) -> dict:
    """Delete a source video (the big files), and its transcript unless kept."""
    try:
        return storage.delete_source(source_id, keep_transcript=keep_transcript)
    except storage.StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/storage/transcript/{source_id}")
def storage_delete_transcript(source_id: str) -> dict:
    """Delete only the cached transcript(s) for a source, keeping the video."""
    try:
        return storage.delete_transcripts(source_id)
    except storage.StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/storage/purge/{target}")
def storage_purge(target: str) -> dict:
    """Bulk delete one category: 'clips', 'sources', or 'orphan_transcripts'."""
    try:
        return storage.purge(target)
    except storage.StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/storage")
def storage_page() -> FileResponse:
    """Serve the disk-cleanup page."""
    page = STATIC_DIR / "storage.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="storage.html is missing.")
    return FileResponse(str(page), headers={"Cache-Control": "no-store"})


@app.get("/studio")
def studio() -> FileResponse:
    """Serve the manual-selection workflow page (transcript out, clip JSON in)."""
    page = STATIC_DIR / "studio.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="studio.html is missing.")
    return FileResponse(str(page), headers={"Cache-Control": "no-store"})


@app.get("/api/download/{download_id}/video")
def download_file(download_id: str) -> FileResponse:
    """Stream a prefetched video file so the Step-2 preview can swap from the
    embeddable player to the real downloaded file once it's ready.

    The id is resolved via the same ``downloads/<id>.*`` glob used for uploads,
    so it can never reach a file outside the downloads directory. FileResponse
    serves Range requests, so the ``<video>`` element can seek/stream normally.
    """
    try:
        path = uploads.resolve_upload(download_id)
    except BadInputError:
        raise HTTPException(status_code=404, detail="Downloaded video not found.")
    return FileResponse(str(path))


class ClipRef(BaseModel):
    """Reference to one generated clip."""

    clip_id: str
    index: int


@app.get("/api/history")
def get_history() -> list:
    """All past generations and their clips, newest first (for the Clips panel)."""
    return history.list_entries()


@app.delete("/api/clip/{clip_id}/{index}")
def delete_clip(clip_id: str, index: int) -> dict:
    """Remove one generated clip (deletes the file and drops it from history)."""
    if history.clip_path(clip_id, index) is None:
        raise HTTPException(status_code=400, detail="Invalid clip reference.")
    deleted = history.remove_clip(clip_id, index)
    return {"status": "ok", "deleted": deleted}


class RestyleRequest(BaseModel):
    """Body for POST /api/clip/restyle."""

    clip_id: str
    index: int
    changes: ClipChanges


def _entry_or_404(clip_id: str) -> dict:
    if history.clip_path(clip_id, 0) is None:
        raise HTTPException(status_code=400, detail="Invalid clip reference.")
    for entry in history.list_entries():
        if entry.get("clip_id") == clip_id:
            return entry
    raise HTTPException(status_code=404, detail="That generation is not in your history.")


def _csv_response(data: bytes, filename: str) -> Response:
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{clip_id}")
def export_generation(clip_id: str) -> FileResponse:
    """Every clip of one generation in a ZIP, plus ``clips.csv``, a tracking sheet.

    The CSV has each clip's title, hook, description, tags and score, how it was
    cut, and blank platform / posted_at / post_url / status columns to fill in as
    clips are uploaded. Its ``filename`` column matches the file's name in the ZIP.
    """
    entry = _entry_or_404(clip_id)
    items = export.collect([entry])
    if not items:
        raise HTTPException(status_code=404, detail="None of this generation's clip files exist any more.")
    fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="clipforge-export-")
    os.close(fd)
    try:
        export.build_zip(items, Path(tmp))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    stamp = (entry.get("created") or "")[:10] or "clips"
    return FileResponse(
        tmp,
        media_type="application/zip",
        filename=f"clipforge-{stamp}-{clip_id[:8]}.zip",
        background=BackgroundTask(lambda: Path(tmp).unlink(missing_ok=True)),
    )


@app.get("/api/export/{clip_id}/clips.csv")
def export_generation_csv(clip_id: str) -> Response:
    """Just the tracking sheet for one generation."""
    entry = _entry_or_404(clip_id)
    return _csv_response(export.build_csv(export.collect([entry])), f"clipforge-{clip_id[:8]}.csv")


@app.get("/api/export.csv")
def export_all_csv() -> Response:
    """One tracking sheet for every clip in your history, newest generation first."""
    return _csv_response(export.build_csv(export.collect(history.list_entries())), "clipforge-all-clips.csv")


@app.post("/api/clip/restyle")
def restyle_clip(req: RestyleRequest) -> dict:
    """Change a finished clip and render it again from its source.

    Only the fields present in ``changes`` are applied (see ``ClipChanges``). The
    source video and its transcript must still be on disk; nothing is
    re-downloaded or re-transcribed and no AI is called. The old file is
    replaced only when the new render succeeds. Takes as long as a normal render
    of one clip.
    """
    if history.clip_path(req.clip_id, req.index) is None:
        raise HTTPException(status_code=400, detail="Invalid clip reference.")
    try:
        clip = rendering.restyle_clip(req.clip_id, req.index, req.changes)
    except rendering.RestyleError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message)
    return {"status": "ok", "clip": clip}


@app.get("/api/overlays")
def get_overlays() -> dict:
    """The logo library (files in assets/overlays + uploads)."""
    return {"logos": overlays.list_logos()}


@app.post("/api/overlays/upload")
def upload_overlay(file: UploadFile = File(...)) -> dict:
    """Add a logo (PNG with transparency is best; JPG and WebP work too)."""
    try:
        info = overlays.save_logo(file.filename, file.file)
    except BadInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    return {"status": "ok", **info}


@app.post("/api/reveal")
def reveal_clip(ref: ClipRef) -> dict:
    """Open the clip's folder in the OS file manager with the file selected.

    Local-only convenience (this app runs on the user's own machine). The path is
    validated to live under the clips directory before anything is launched.
    """
    path = history.clip_path(ref.clip_id, ref.index)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found on disk.")
    try:
        if sys.platform.startswith("win"):
            # explorer returns a non-zero exit code even on success — ignore it.
            subprocess.run(["explorer", "/select,", str(path)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not open the folder: {exc}")
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend.

    Prefer the built React dashboard (web/dist) so the backend and the dev server
    show the SAME app; fall back to the legacy vanilla page only if no build exists.
    ``no-store`` keeps the browser from pinning a stale index that points at old,
    since-rebuilt asset hashes (the "old UI until hard refresh" bug).
    """
    react_index = WEB_DIST_DIR / "index.html"
    target = react_index if react_index.is_file() else (STATIC_DIR / "index.html")
    return FileResponse(str(target), headers={"Cache-Control": "no-store"})


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    """Start the pipeline on a background thread and return a job id.

    The actual work (download/transcribe/select/render) runs asynchronously; the
    client subscribes to ``/api/progress/{job_id}`` for live progress and the
    final clips.
    """
    # Advisory: has this video been cut before? Never blocks the job.
    duplicates = sources.check(
        source_id=req.upload_id, video_url=req.video_url, filename=req.upload_name,
    )
    job = jobs.create_job(req)
    jobs.start_job(job)
    logger.info("[%s] job accepted", job.id)
    return {"job_id": job.id, "duplicates": duplicates}


class SourceCheckRequest(BaseModel):
    """Body for POST /api/source/check."""

    source_id: Optional[str] = Field(
        default=None, description="An uploaded video already on disk (its size and "
        "duration are read from the file).")
    video_url: Optional[str] = None
    filename: Optional[str] = Field(default=None, description="The file's name, for the title match.")
    size: Optional[int] = Field(default=None, ge=0, description="Bytes, if the file is not on disk yet.")
    duration: Optional[float] = Field(default=None, ge=0, description="Seconds, if known.")


@app.post("/api/source/check")
def source_check(req: SourceCheckRequest) -> dict:
    """Has this source been cut before? Advisory: the answer never blocks a job.

    Says which signal matched (same file, same YouTube id, same title, same size
    and duration) so the caller can word the warning. The file itself is never
    sent: a name and a byte size are enough.
    """
    duplicates = sources.check(
        source_id=req.source_id, video_url=req.video_url, filename=req.filename,
        size=req.size, duration=req.duration,
    )
    return {"duplicates": duplicates}


@app.get("/api/music-suggest/{source_id}")
def music_suggest(source_id: str, language: Optional[str] = None) -> dict:
    """Suggest a music mood (sad/happy/romantic/…) from the prepared transcript."""
    tr = pretranscribe.cached(source_id, language) or pretranscribe.cached(source_id, None)
    if not tr:
        return {"ready": False}
    segs = tr.get("segments") or []
    text = " ".join((s.get("text") or "") for s in segs)
    return {"ready": True, **mood.suggest_mood(text)}


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str) -> dict:
    """Flag a running job for cancellation; it stops at the next clip boundary."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    job.cancel()
    logger.info("[%s] cancel requested", job_id)
    return {"status": "cancelled"}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str) -> StreamingResponse:
    """Stream a job's progress as Server-Sent Events until it finishes."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    async def event_stream():
        last_rev = -1
        while True:
            snap = job.snapshot()
            if snap["rev"] != last_rev:
                last_rev = snap["rev"]
                yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/result/{job_id}")
def result(job_id: str) -> dict:
    """Return the current snapshot of a job (useful after a stream reconnect)."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job.snapshot()


# Catch-all guard: turn any unexpected error into a clean 500 (no crash).
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {exc}"},
    )
