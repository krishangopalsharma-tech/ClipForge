"""Face-aware vertical reframing.

A centre crop of a 16:9 talking-head video often cuts the speaker in half.
This module looks at the clip first, then decides how to fit it into 9:16:

* ``track``  one person on screen: a crop window that follows their face.
* ``blur``   no face (a slide, a screen recording, a landscape) or a group:
             the whole picture at full width over a blurred copy of itself, so
             nothing is cut off.
* ``center`` the plain centre crop, used when the source is already vertical,
             the output is not vertical, or detection is not available.

Detection is a ~1 MB face detector (UltraFace RFB-320) run with the onnxruntime
that faster-whisper already installs, on CPU, 5 frames a second. No OpenCV, no
MediaPipe, no GPU memory: the GPU belongs to Whisper and ffmpeg.

Everything that decides something (subject choice, track/blur choice, camera
path) is a pure function over plain data, so it is tested without a model or a
video. Reframing never fails a render: any problem degrades to ``center`` with
the reason recorded on the plan.
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .paths import ROOT_DIR

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The detector model
# --------------------------------------------------------------------------- #
MODEL_NAME = "version-RFB-320.onnx"
MODEL_URL = (
    "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB"
    "/master/models/onnx/" + MODEL_NAME
)
MODEL_MIN_BYTES = 1_000_000          # the real file is ~1.27 MB; less means a bad download
_IN_W, _IN_H = 320, 240              # the network's fixed input size

# Sampling and decision thresholds. Each is a trade that was chosen, not measured
# on your footage; they are module constants so they can be tuned in one place.
SAMPLE_FPS = 5.0
SCORE_THRESHOLD = 0.7                # detector confidence to count as a face
MIN_FACE_HEIGHT = 0.04               # of frame height; smaller is background noise
MIN_FACE_FRACTION = 0.35             # fewer sampled frames with a face -> no subject
GROUP_RATIO = 0.35                   # a 2nd face this big (by area) vs the largest = a group
MAX_GROUP_FRACTION = 0.5             # more group frames than this -> blur layout
STICKY_RADIUS = 0.15                 # keep the same person if they stay within this
STICKY_AREA = 0.6                    # ...and are at least this big vs the largest face
CUT_JUMP = 0.20                      # subject moved this far (of width) between samples = a cut
DEAD_ZONE = 0.12                     # of the crop width: no camera move inside it
SMOOTHING = 0.5                      # fraction of the remaining gap closed per sample
MAX_SPEED = 1.0                      # crop widths per second, so a pan is never a whip
OUTPUT_FPS = 30.0                    # camera-path commands per second of video


def model_path() -> Path:
    """Where the face model lives (``CLIPFORGE_FACE_MODEL`` overrides)."""
    env = os.environ.get("CLIPFORGE_FACE_MODEL")
    return Path(env) if env else ROOT_DIR / "models" / "face" / MODEL_NAME


def model_available() -> bool:
    path = model_path()
    return path.is_file() and path.stat().st_size >= MODEL_MIN_BYTES


def download_model(dest: Optional[Path] = None) -> Path:
    """Fetch the detector model (about 1.3 MB). Run on request, never implicitly:

        python\\python.exe -m app.reframe --download
    """
    dest = Path(dest) if dest else model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(tmp, "wb") as out:
        out.write(resp.read())
    if tmp.stat().st_size < MODEL_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The downloaded face model is too small; try again.")
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# Faces and the detector
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Face:
    """A face box in fractions of the frame (0..1)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Sample:
    """The faces seen at one moment of the clip (``t`` seconds from its start)."""

    t: float
    faces: List[Face] = field(default_factory=list)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Indices of the boxes kept by greedy non-maximum suppression."""
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_detections(
    scores: np.ndarray,
    boxes: np.ndarray,
    score_threshold: float = SCORE_THRESHOLD,
    iou_threshold: float = 0.3,
) -> List[Face]:
    """Turn the network's raw outputs (N x 2 scores, N x 4 boxes) into faces."""
    confidence = scores[:, 1]
    mask = confidence > score_threshold
    if not mask.any():
        return []
    picked_boxes = np.clip(boxes[mask], 0.0, 1.0)
    picked_scores = confidence[mask]
    keep = _nms(picked_boxes, picked_scores, iou_threshold)
    return [
        Face(*(float(v) for v in picked_boxes[i]), float(picked_scores[i])) for i in keep
    ]


class FaceDetector:
    """UltraFace RFB-320 on onnxruntime's CPU provider."""

    def __init__(self, path: Optional[Path] = None) -> None:
        import onnxruntime as ort  # already installed for faster-whisper's VAD

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(path or model_path()), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def detect(self, rgb: np.ndarray) -> List[Face]:
        """Faces in a (240, 320, 3) uint8 RGB frame."""
        x = (rgb.astype(np.float32) - 127.0) / 128.0
        x = np.transpose(x, (2, 0, 1))[None]
        scores, boxes = self._session.run(None, {self._input: x})
        return decode_detections(scores[0], boxes[0])


_detector: Optional[FaceDetector] = None
_detector_failed = False


def get_detector() -> Optional[FaceDetector]:
    """The shared detector, or None when the model or onnxruntime is unusable."""
    global _detector, _detector_failed
    if _detector is not None:
        return _detector
    if _detector_failed or not model_available():
        return None
    try:
        _detector = FaceDetector()
    except Exception:  # noqa: BLE001 - never let detection break a render
        logger.warning("Face detector could not be loaded; centre crop will be used.", exc_info=True)
        _detector_failed = True
        return None
    return _detector


# --------------------------------------------------------------------------- #
# Reading the video
# --------------------------------------------------------------------------- #
def probe_size(video: Path) -> Tuple[int, int]:
    """(width, height) of the first video stream, via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0 or "x" not in out.stdout:
        raise RuntimeError(f"ffprobe could not read {video}: {out.stderr.strip()[:200]}")
    w, h = out.stdout.strip().splitlines()[0].split("x")[:2]
    return int(w), int(h)


def _read_exact(stream, n: int) -> bytes:
    chunks: List[bytes] = []
    got = 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def sample_frames(video: Path, start: float, duration: float, fps: float = SAMPLE_FPS) -> Iterator[np.ndarray]:
    """Yield the clip's frames at ``fps``, each scaled to the detector's 320x240.

    ffmpeg does the decoding and the scaling, so no image library is needed.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(video), "-an", "-vf", f"fps={fps},scale={_IN_W}:{_IN_H}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    size = _IN_W * _IN_H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = _read_exact(proc.stdout, size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(_IN_H, _IN_W, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------- #
# Decisions (pure)
# --------------------------------------------------------------------------- #
def usable_faces(faces: Sequence[Face]) -> List[Face]:
    """Faces big enough to be the subject rather than background."""
    return [f for f in faces if f.height >= MIN_FACE_HEIGHT]


def pick_subject(faces: Sequence[Face], previous_cx: Optional[float] = None) -> Optional[Face]:
    """The face to follow: the largest, but sticky to the one followed last time.

    Two similar-sized faces would otherwise swap places every few frames as
    their areas wobble, and the camera would flick between them.
    """
    candidates = usable_faces(faces)
    if not candidates:
        return None
    largest = max(candidates, key=lambda f: f.area)
    if previous_cx is not None:
        near = [
            f for f in candidates
            if abs(f.cx - previous_cx) <= STICKY_RADIUS and f.area >= STICKY_AREA * largest.area
        ]
        if near:
            return min(near, key=lambda f: abs(f.cx - previous_cx))
    return largest


@dataclass
class Decision:
    mode: str                  # "track" or "blur"
    reason: str
    face_fraction: float
    group_fraction: float


def decide(samples: Sequence[Sample]) -> Decision:
    """Track one person, or show the whole picture over a blur."""
    n = len(samples)
    if n == 0:
        return Decision("blur", "no frames could be read", 0.0, 0.0)
    with_face = 0
    group = 0
    for s in samples:
        faces = sorted(usable_faces(s.faces), key=lambda f: f.area, reverse=True)
        if faces:
            with_face += 1
        if len(faces) >= 2 and faces[1].area >= GROUP_RATIO * faces[0].area:
            group += 1
    face_fraction = with_face / n
    group_fraction = group / n
    if face_fraction < MIN_FACE_FRACTION:
        return Decision("blur", "no face on screen", face_fraction, group_fraction)
    if group_fraction > MAX_GROUP_FRACTION:
        return Decision("blur", "several people on screen", face_fraction, group_fraction)
    return Decision("track", "one person on screen", face_fraction, group_fraction)


@dataclass
class Key:
    t: float
    left: float      # left edge of the crop window, as a fraction of the source width
    cut: bool = False


def camera_keyframes(samples: Sequence[Sample], crop_frac: float) -> List[Key]:
    """Where the crop window sits at each sample, smoothed.

    The window holds still while the subject stays inside a dead zone, then
    eases towards them at a capped speed, so the picture drifts rather than
    jitters. A large jump between samples is a camera cut in the source, and the
    window jumps with it instead of panning across the gap.
    """
    crop_frac = max(0.01, min(1.0, crop_frac))
    half = crop_frac / 2.0
    lo, hi = half, 1.0 - half

    # 1. Who is being followed at each sample (None when nobody usable).
    targets: List[Optional[float]] = []
    previous: Optional[float] = None
    for s in samples:
        subject = pick_subject(s.faces, previous)
        if subject is None:
            targets.append(None)
        else:
            previous = subject.cx
            targets.append(subject.cx)

    # 2. Fill gaps by holding the last known position; before the first sighting
    #    use the first one; with nobody at all, the middle.
    known = [t for t in targets if t is not None]
    if not known:
        return [Key(s.t, 0.5 - half) for s in samples]
    filled: List[float] = []
    last = known[0]
    for target in targets:
        if target is not None:
            last = target
        filled.append(last)

    # 3. Ease the window after the subject.
    dead = DEAD_ZONE * crop_frac
    keys: List[Key] = []
    pos = min(hi, max(lo, filled[0]))
    for i, s in enumerate(samples):
        target = filled[i]
        cut = False
        if i > 0:
            step = max(1e-3, s.t - samples[i - 1].t)
            if targets[i] is not None and targets[i - 1] is not None and abs(target - filled[i - 1]) > CUT_JUMP:
                pos = target
                cut = True
            else:
                err = target - pos
                if abs(err) > dead:
                    wanted = target - dead * (1 if err > 0 else -1)
                    move = (wanted - pos) * SMOOTHING
                    limit = MAX_SPEED * crop_frac * step
                    pos += max(-limit, min(limit, move))
        pos = min(hi, max(lo, pos))
        keys.append(Key(s.t, pos - half, cut))
    return keys


def path_at_fps(keys: Sequence[Key], duration: float, fps: float = OUTPUT_FPS) -> List[Tuple[float, float]]:
    """(time, left) for every output frame: linear between keys, a step at a cut."""
    if not keys:
        return []
    out: List[Tuple[float, float]] = []
    frames = max(1, int(duration * fps) + 1)
    j = 0
    for n in range(frames):
        t = n / fps
        while j + 1 < len(keys) and keys[j + 1].t <= t:
            j += 1
        a = keys[j]
        if j + 1 >= len(keys) or keys[j + 1].cut:
            out.append((t, a.left))
            continue
        b = keys[j + 1]
        span = max(1e-6, b.t - a.t)
        frac = min(1.0, max(0.0, (t - a.t) / span))
        out.append((t, a.left + (b.left - a.left) * frac))
    return out


def sendcmd_script(path: Sequence[Tuple[float, float]], scaled_w: int, crop_w: int) -> str:
    """An ffmpeg ``sendcmd`` script moving the crop's ``x`` along ``path``.

    Only frames where the integer position changes get a line, so a still camera
    produces a nearly empty file.
    """
    max_x = max(0, scaled_w - crop_w)
    lines: List[str] = []
    last: Optional[int] = None
    for t, left in path:
        x = int(round(min(max_x, max(0, left * scaled_w))))
        if x != last:
            lines.append(f"{t:.3f} crop x {x};")
            last = x
    return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------------- #
# The plan the renderer consumes
# --------------------------------------------------------------------------- #
@dataclass
class ReframePlan:
    mode: str                              # "center" | "track" | "blur" | "letterbox"
    reason: str
    scaled_w: int = 0                      # width after scaling to the output height (track)
    x0: int = 0                            # the crop's starting x (track)
    commands_path: Optional[Path] = None   # sendcmd script (track)
    face_fraction: Optional[float] = None
    group_fraction: Optional[float] = None

    def summary(self) -> dict:
        """What the UI shows next to the clip."""
        out = {"mode": self.mode, "reason": self.reason}
        if self.face_fraction is not None:
            out["face_fraction"] = round(self.face_fraction, 2)
        return out


def _center(reason: str) -> ReframePlan:
    return ReframePlan("center", reason)


def plan_reframe(
    source: Path,
    start: float,
    end: float,
    target_w: int,
    target_h: int,
    requested: str,
    work_dir: Path,
    name: str,
    detector: Optional[FaceDetector] = None,
) -> ReframePlan:
    """Decide how to fit ``source[start:end]`` into ``target_w x target_h``.

    ``requested`` is ``"auto"`` (decide from the footage), ``"track"`` (always
    follow a face), ``"blur"`` (always the blurred layout) or ``"letterbox"``
    (full width on black). Never raises.
    """
    try:
        # Letterbox crops nothing, so it needs no probe, no model and no
        # sampling — and it is just as right for a source that is already
        # vertical, where it simply adds no bars.
        if requested == "letterbox":
            return ReframePlan("letterbox", "chosen by you")
        if target_w >= target_h:
            return _center("output is not vertical")
        src_w, src_h = probe_size(source)
        if src_w / src_h <= (target_w / target_h) * 1.05:
            return _center("source is already vertical")
        if requested == "blur":
            return ReframePlan("blur", "chosen by you")

        det = detector or get_detector()
        if det is None:
            reason = "face model not installed; see README (Phase 3)"
            return ReframePlan("blur", reason + ", showing the full picture") if requested == "auto" else _center(reason)

        duration = max(0.1, end - start)
        samples: List[Sample] = []
        for i, frame in enumerate(sample_frames(source, start, duration)):
            samples.append(Sample(i / SAMPLE_FPS, det.detect(frame)))

        decision = decide(samples)
        if requested == "auto" and decision.mode == "blur":
            return ReframePlan(
                "blur", decision.reason, face_fraction=decision.face_fraction,
                group_fraction=decision.group_fraction,
            )
        if decision.face_fraction == 0.0:
            return _center("no face found to follow")

        scaled_w = int(round(src_w * target_h / src_h / 2.0)) * 2
        crop_frac = target_w / scaled_w
        keys = camera_keyframes(samples, crop_frac)
        script = sendcmd_script(path_at_fps(keys, duration), scaled_w, target_w)
        cmd_path = Path(work_dir) / f"{name}.cmd"
        cmd_path.write_text(script, encoding="utf-8")
        x0 = int(round(min(scaled_w - target_w, max(0, keys[0].left * scaled_w))))
        reason = decision.reason if decision.mode == "track" else "you asked to follow a face"
        return ReframePlan(
            "track", reason, scaled_w=scaled_w, x0=x0, commands_path=cmd_path,
            face_fraction=decision.face_fraction, group_fraction=decision.group_fraction,
        )
    except Exception as exc:  # noqa: BLE001 - a reframe problem must not cost the clip
        logger.warning("Reframe planning failed (%s); using a centre crop.", exc, exc_info=True)
        return _center(f"reframe unavailable ({type(exc).__name__})")


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        print("Saved", download_model())
    else:
        print("model:", model_path(), "present" if model_available() else "MISSING")
