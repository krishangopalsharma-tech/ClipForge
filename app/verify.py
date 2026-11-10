"""Check an imported subtitle file's timing against what is actually said.

A subtitle or transcript exported from an editor (Premiere Pro, Resolve) is
usually right, but "usually" is not what a caption burned onto a clip can rest
on. This module runs Whisper over the audio and lines every caption cue up with
the words that were really spoken, then reports, cue by cue, whether the caption
sits where the speech is.

The file's TEXT is trusted (an editor's spelling beats an ASR's), the audio's
TIMING is trusted (Whisper measures it, the file only states it). So the corrected
transcript keeps the file's words and gives them measured times; a cue whose
timing is already right keeps its own start and end.

How it matches, because the details are what make it hold up:

* **Time is not assumed.** The file's clock may start at 01:00:00:00 (a Premiere
  sequence default) or be a second off. A constant offset is found first, from
  3-word phrases that occur exactly once in both texts, so the answer is a vote
  of many unrelated places, not one guess.
* **Then chunk by chunk.** Each ~90 s of cues is matched only against the audio
  around where it should be, which keeps repeated phrases ("you know") from
  matching the wrong minute, and follows slow drift because every chunk is
  centred on where the previous one landed.
* **Spelling differences are tolerated.** Hindi in Devanagari on one side and
  Roman letters on the other are compared after romanising; a second, fuzzy pass
  pairs "okay" with "ok" in the gaps the exact pass left.

Pure functions over plain data. Running Whisper is ``run_verification``'s job.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import statistics
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import translit
from .paths import TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

# A caption that starts within this of the speech is "on time". Cue starts are
# what a viewer sees; ends carry the editor's padding, so they get more room.
START_TOLERANCE = 0.35
END_TOLERANCE = 1.0
# Corrections: how far before the first spoken word a cue starts, and after the last it ends.
LEAD = 0.05
TAIL = 0.10

MIN_VOTES = 3                  # unrelated phrases that must agree on the offset
SEARCH_MARGIN = 20.0           # seconds of audio either side of where a chunk should be
OUTLIER_SECONDS = 6.0          # a match this far from its chunk's offset is a coincidence
CHUNK_TOKENS = 150
CHUNK_SECONDS = 90.0
FUZZY_RATIO = 0.78
FUZZY_GAP = 40                 # largest gap (tokens) the fuzzy pass will search
CONFIDENT_MATCHES = 5          # matches a chunk needs for its offset to be believed
MAX_INTERPOLATION_SECONDS = 12.0


class VerifyError(ValueError):
    """The two texts could not be lined up; the message says why."""


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def norm_key(word: str) -> str:
    """A word reduced to what is worth comparing: letters, digits and marks only.

    Devanagari is romanised first so a Hindi transcript in either script compares
    equal to the other.
    """
    text = unicodedata.normalize("NFKC", str(word or "")).lower()
    if _DEVANAGARI.search(text):
        text = translit.to_roman(text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM")


@dataclass
class CueTok:
    key: str
    cue: int            # which cue this word belongs to
    pos: int            # index within the cue
    count: int          # words in the cue
    est: float          # where the file says it is (spread across the cue)


@dataclass
class HeardTok:
    key: str
    start: float
    end: float


def cue_tokens(segments: List[dict]) -> Tuple[List[CueTok], List[List[int]]]:
    """Every word of every cue, and for each cue the indexes of its words."""
    toks: List[CueTok] = []
    by_cue: List[List[int]] = []
    for ci, seg in enumerate(segments):
        words = str(seg.get("text") or "").split()
        start, end = float(seg["start"]), float(seg["end"])
        span = max(0.2, end - start)
        idxs = []
        for wi, word in enumerate(words):
            idxs.append(len(toks))
            toks.append(CueTok(norm_key(word), ci, wi, len(words), start + (wi + 0.5) / len(words) * span))
        by_cue.append(idxs)
    return toks, by_cue


def heard_tokens(words: List[dict]) -> List[HeardTok]:
    out = [
        HeardTok(norm_key(w.get("word", "")), float(w["start"]), float(w.get("end", w["start"])))
        for w in words or []
        if w.get("start") is not None
    ]
    out.sort(key=lambda t: t.start)
    return out


# --------------------------------------------------------------------------- #
# A constant offset, from phrases unique to both texts
# --------------------------------------------------------------------------- #
def _unique_ngrams(keys: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    seen: Dict[Tuple[str, ...], List[int]] = {}
    for i in range(len(keys) - n + 1):
        gram = tuple(keys[i:i + n])
        if all(gram):
            seen.setdefault(gram, []).append(i)
    return {g: pos[0] for g, pos in seen.items() if len(pos) == 1}


def estimate_offset(ctoks: List[CueTok], htoks: List[HeardTok]) -> Optional[Tuple[float, int]]:
    """``(offset, votes)``: add ``offset`` to the file's times to get the audio's.

    None when no phrase is shared (a different video, or a language mismatch).
    """
    ckeys = [t.key for t in ctoks]
    hkeys = [t.key for t in htoks]
    for n, minimum in ((3, MIN_VOTES), (2, MIN_VOTES * 2), (1, MIN_VOTES * 4)):
        cu, hu = _unique_ngrams(ckeys, n), _unique_ngrams(hkeys, n)
        votes = []
        for gram, ci in cu.items():
            hi = hu.get(gram)
            if hi is None or (n == 1 and len(gram[0]) < 5):
                continue
            votes.append(htoks[hi].start - ctoks[ci].est)
        if len(votes) < minimum:
            continue
        bins: Dict[int, int] = {}
        for v in votes:
            bins[round(v)] = bins.get(round(v), 0) + 1
        best = max(bins, key=lambda b: bins[b] + 0.5 * (bins.get(b - 1, 0) + bins.get(b + 1, 0)))
        near = [v for v in votes if abs(v - best) <= 1.5]
        if len(near) >= minimum:
            return statistics.median(near), len(near)
    return None


# --------------------------------------------------------------------------- #
# Matching, chunk by chunk
# --------------------------------------------------------------------------- #
def _fuzzy_fill(
    ckeys: List[str], hkeys: List[str], c_range: range, h_range: range,
    ctime: List[float], htime: List[float], center: float, pairs: Dict[int, int],
) -> None:
    """Pair leftover words in a gap, in order, when they are nearly the same word."""
    if not c_range or not h_range or len(c_range) > FUZZY_GAP or len(h_range) > FUZZY_GAP:
        return
    h_next = h_range.start
    for ci in c_range:
        key = ckeys[ci]
        if len(key) < 3:
            continue
        best, best_ratio = None, FUZZY_RATIO
        for hi in range(h_next, h_range.stop):
            hk = hkeys[hi]
            if len(hk) < 3 or abs((htime[hi] - ctime[ci]) - center) > OUTLIER_SECONDS:
                continue
            ratio = difflib.SequenceMatcher(None, key, hk).ratio()
            if ratio >= best_ratio:
                best, best_ratio = hi, ratio
        if best is not None:
            pairs[ci] = best
            h_next = best + 1


def match_chunk(
    ckeys: List[str], hkeys: List[str], ctime: List[float], htime: List[float],
) -> Dict[int, int]:
    """``{cue word index: heard word index}`` for one chunk, exact then fuzzy."""
    # Empty keys (pure punctuation) must never match each other.
    a = [k or f"\x00c{i}" for i, k in enumerate(ckeys)]
    b = [k or f"\x00h{i}" for i, k in enumerate(hkeys)]
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    pairs: Dict[int, int] = {}
    for blk in blocks:
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    if not pairs:
        return {}

    deltas = [htime[h] - ctime[c] for c, h in pairs.items()]
    center = statistics.median(deltas)
    pairs = {c: h for c, h in pairs.items() if abs((htime[h] - ctime[c]) - center) <= OUTLIER_SECONDS}

    # Fuzzy pass in the gaps the exact pass left between (and around) its pairs.
    anchors = sorted(pairs.items())
    edges = [(-1, -1)] + anchors + [(len(ckeys), len(hkeys))]
    for (c0, h0), (c1, h1) in zip(edges, edges[1:]):
        _fuzzy_fill(ckeys, hkeys, range(c0 + 1, c1), range(h0 + 1, h1), ctime, htime, center, pairs)
    return pairs


def align(
    ctoks: List[CueTok], htoks: List[HeardTok], by_cue: List[List[int]], offset: float,
) -> Tuple[Dict[int, int], Dict[int, Tuple[float, bool]]]:
    """Match every cue word to a heard word.

    Returns ``(pairs, cue_offsets)``: ``pairs`` maps a cue-word index to a
    heard-word index; ``cue_offsets`` maps each cue to its chunk's offset and
    whether that chunk matched enough words to be believed.
    """
    htimes = [t.start for t in htoks]
    pairs: Dict[int, int] = {}
    cue_offsets: Dict[int, Tuple[float, bool]] = {}
    off = offset

    ci = 0
    while ci < len(by_cue):
        # One chunk: consecutive cues until it holds enough words or time.
        first = ci
        tokens = 0
        t0 = ctoks[by_cue[ci][0]].est if by_cue[ci] else 0.0
        while ci < len(by_cue):
            tokens += len(by_cue[ci])
            ci += 1
            last_est = ctoks[by_cue[ci - 1][-1]].est if by_cue[ci - 1] else t0
            if tokens >= CHUNK_TOKENS or last_est - t0 >= CHUNK_SECONDS:
                break
        idx = [i for c in range(first, ci) for i in by_cue[c]]
        if not idx:
            continue
        lo = bisect_left(htimes, ctoks[idx[0]].est + off - SEARCH_MARGIN)
        hi = bisect_right(htimes, ctoks[idx[-1]].est + off + SEARCH_MARGIN)
        local = match_chunk(
            [ctoks[i].key for i in idx], [htoks[j].key for j in range(lo, hi)],
            [ctoks[i].est for i in idx], [htoks[j].start for j in range(lo, hi)],
        )
        found = {idx[c]: lo + h for c, h in local.items()}
        pairs.update(found)
        deltas = [htoks[h].start - ctoks[c].est for c, h in found.items()]
        confident = len(deltas) >= CONFIDENT_MATCHES
        if confident:
            off = statistics.median(deltas)
        for c in range(first, ci):
            cue_offsets[c] = (off, confident)
    return pairs, cue_offsets


# --------------------------------------------------------------------------- #
# Times for every word, matched or not
# --------------------------------------------------------------------------- #
def _pace(ctoks: List[CueTok], htoks: List[HeardTok], pairs: Dict[int, int]) -> float:
    """Seconds from one spoken word to the next, from words matched side by side."""
    steps = []
    for c, h in pairs.items():
        nxt = pairs.get(c + 1)
        if nxt is not None and c + 1 < len(ctoks) and ctoks[c + 1].cue == ctoks[c].cue:
            d = htoks[nxt].start - htoks[h].start
            if 0 < d < 2.0:
                steps.append(d)
    return min(0.8, max(0.15, statistics.median(steps))) if steps else 0.4


def word_times(
    segments: List[dict], ctoks: List[CueTok], htoks: List[HeardTok], pairs: Dict[int, int],
    cue_offsets: Dict[int, Tuple[float, bool]], fallback_offset: float,
) -> List[Tuple[float, float]]:
    """A (start, end) for each cue word: measured where matched, else interpolated.

    An unmatched run between two matched words is spread across the gap between
    them. A run with no neighbour on one side, or a gap too wide to be one
    sentence, falls back to the file's own timing plus its chunk's offset.
    """
    n = len(ctoks)
    times: List[Optional[Tuple[float, float]]] = [None] * n
    for c, h in pairs.items():
        times[c] = (htoks[h].start, max(htoks[h].end, htoks[h].start))

    def fallback(i: int) -> Tuple[float, float]:
        seg = segments[ctoks[i].cue]
        span = max(0.2, float(seg["end"]) - float(seg["start"]))
        width = span / ctoks[i].count
        off = cue_offsets.get(ctoks[i].cue, (fallback_offset, False))[0]
        start = float(seg["start"]) + ctoks[i].pos * width + off
        return start, start + width

    step = _pace(ctoks, htoks, pairs)

    def spread(ks: List[int], lo: float, hi: float) -> None:
        weights = [max(1, len(ctoks[k].key) or 1) for k in ks]
        total, cursor = float(sum(weights)), lo
        for k, w in zip(ks, weights):
            width = (hi - lo) * w / total
            times[k] = (cursor, cursor + width)
            cursor += width

    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left_i = i - 1 if i > 0 and times[i - 1] is not None else None
        right_i = j if j < n and times[j] is not None else None

        # Words missed at the START of a caption hang off the first matched word
        # at the speaking pace, and words missed at its END continue from the last
        # one. Spreading them across the gap to the neighbouring caption instead
        # would stretch them over the silence between sentences and make the
        # caption look late or early by up to that silence.
        fwd: List[int] = []
        if left_i is not None:
            k = i
            while k < j and ctoks[k].cue == ctoks[left_i].cue:
                fwd.append(k)
                k += 1
        bwd: List[int] = []
        if right_i is not None:
            k = j - 1
            while k >= i + len(fwd) and ctoks[k].cue == ctoks[right_i].cue:
                bwd.append(k)
                k -= 1
            bwd.reverse()
        mid = list(range(i + len(fwd), j - len(bwd)))

        both = left_i is not None and right_i is not None
        gap = (times[right_i][0] - times[left_i][1]) if both else None
        if both and not mid and gap is not None and 0 <= gap < 0.6 * (len(fwd) + len(bwd)) * step:
            spread(list(range(i, j)), times[left_i][1], times[right_i][0])   # too tight for the pace
        else:
            for m, k in enumerate(fwd, 1):
                t0 = times[left_i][0] + m * step
                times[k] = (t0, t0 + 0.9 * step)
            for m, k in enumerate(reversed(bwd), 1):
                t0 = times[right_i][0] - m * step
                times[k] = (t0, t0 + 0.9 * step)
            if mid:
                lo = times[fwd[-1]][1] if fwd else (times[left_i][1] if left_i is not None else None)
                hi = times[bwd[0]][0] if bwd else (times[right_i][0] if right_i is not None else None)
                if lo is not None and hi is not None and 0 <= hi - lo <= MAX_INTERPOLATION_SECONDS:
                    spread(mid, lo, hi)
                else:
                    for k in mid:
                        times[k] = fallback(k)
        # Anything still unset (no neighbour on either side) uses the file's own timing.
        for k in range(i, j):
            if times[k] is None:
                times[k] = fallback(k)
        i = j

    # Keep every word's start non-decreasing so a caption never plays backwards.
    out: List[Tuple[float, float]] = []
    floor = 0.0
    for t in times:
        start = max(floor, t[0])
        out.append((start, max(start + 0.02, t[1])))
        floor = start
    return out


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #
@dataclass
class Verification:
    summary: dict
    cues: List[dict]
    corrected: dict


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def verify(imported: dict, heard: dict) -> Verification:
    """Check ``imported`` (the file's cues) against ``heard`` (Whisper's words).

    Raises:
        VerifyError: if the two texts share nothing to line up on.
    """
    segments = [s for s in imported.get("segments") or [] if str(s.get("text") or "").strip()]
    if not segments:
        raise VerifyError("The subtitle file has no text to check.")
    ctoks, by_cue = cue_tokens(segments)
    htoks = heard_tokens(heard.get("words") or [])
    if not htoks:
        raise VerifyError("Whisper heard no speech in this video, so there is nothing to check against.")

    found = estimate_offset(ctoks, htoks)
    if found is None:
        raise VerifyError(
            "The subtitle text could not be matched to the audio: no phrase appears in both. "
            "Is this the right video, and is the language set correctly (Hindi in Roman letters "
            "needs the Hinglish language option)?"
        )
    offset, votes = found

    pairs, cue_offsets = align(ctoks, htoks, by_cue, offset)
    times = word_times(segments, ctoks, htoks, pairs, cue_offsets, offset)

    # First pass: where does each cue's speech really start, and which cues can
    # be trusted to say so?
    facts: List[dict] = []
    for ci, seg in enumerate(segments):
        idxs = by_cue[ci]
        start, end = float(seg["start"]), float(seg["end"])
        keyed = [i for i in idxs if ctoks[i].key]
        matched = [i for i in keyed if i in pairs]
        frac = len(matched) / len(keyed) if keyed else 0.0
        chunk_off, confident = cue_offsets.get(ci, (offset, False))
        if idxs:
            true_start, true_end = times[idxs[0]][0], times[idxs[-1]][1]
        else:
            true_start, true_end = start + chunk_off, end + chunk_off
        facts.append({
            "idxs": idxs, "start": start, "end": end, "keyed": keyed, "matched": matched,
            "verified": bool(
                (len(matched) >= 2 and frac >= 0.4) or (len(keyed) <= 3 and matched and frac >= 0.5)
            ),
            "chunk_off": chunk_off, "confident": confident,
            "true_start": true_start, "true_end": true_end,
            "start_off": true_start - start, "end_off": true_end - end,
        })

    # The constant shift is what the typical verified cue is off by. Measured
    # on the cues (not on single words) it keeps the editor's own padding: a
    # caption that opens a tenth of a second early still does after the shift.
    offsets = [f["start_off"] for f in facts if f["verified"]]
    shift_raw = statistics.median(offsets) if offsets else 0.0
    shift = shift_raw if abs(shift_raw) > START_TOLERANCE else 0.0

    cues: List[dict] = []
    corrected_segments: List[dict] = []
    corrected_words: List[dict] = []
    prev_end_orig, prev_new_end = None, None

    for ci, seg in enumerate(segments):
        f = facts[ci]
        idxs, start, end = f["idxs"], f["start"], f["end"]
        keyed, matched, verified = f["keyed"], f["matched"], f["verified"]
        chunk_off, confident = f["chunk_off"], f["confident"]
        true_start, true_end = f["true_start"], f["true_end"]
        start_off, end_off = f["start_off"], f["end_off"]

        if not verified:
            status = "unverified"
        elif abs(start_off - shift) <= START_TOLERANCE and abs(end_off - shift) <= END_TOLERANCE:
            status = "ok"
        else:
            status = "off"

        if status == "off":
            new_start, new_end = max(0.0, true_start - LEAD), true_end + TAIL
        else:
            local = chunk_off if (confident and status == "unverified") else shift
            applied = local if abs(local) > START_TOLERANCE else 0.0
            new_start, new_end = max(0.0, start + applied), end + applied
        if new_end <= new_start:
            new_end = new_start + 0.2
        # A corrected cue may not run into the one before it, unless the file's did.
        if prev_new_end is not None and new_start < prev_new_end and (prev_end_orig or 0) <= start + 0.01:
            corrected_segments[-1]["end"] = round(max(corrected_segments[-1]["start"] + 0.1, new_start), 3)
            prev_new_end = corrected_segments[-1]["end"]

        corrected_segments.append({
            "id": ci, "start": round(new_start, 3), "end": round(new_end, 3),
            "text": seg["text"], "speaker": seg.get("speaker"),
        })
        for i, token in zip(idxs, str(seg["text"]).split()):
            ws, we = times[i]
            ws = min(max(ws, new_start), max(new_start, new_end - 0.02))
            corrected_words.append({
                "word": token, "start": round(ws, 3), "end": round(min(max(we, ws + 0.02), new_end), 3),
                "speaker": seg.get("speaker"),
            })
        prev_end_orig, prev_new_end = end, new_end

        cues.append({
            "index": ci, "start": round(start, 3), "end": round(end, 3),
            "new_start": round(new_start, 3), "new_end": round(new_end, 3),
            "text": str(seg["text"])[:140], "words": len(keyed), "matched": len(matched),
            "status": status, "start_off": round(start_off, 2), "end_off": round(end_off, 2),
        })

    verified_cues = [c for c in cues if c["status"] != "unverified"]
    devs = [abs(c["start_off"] - shift) for c in verified_cues]
    n_ok = sum(1 for c in cues if c["status"] == "ok")
    n_off = sum(1 for c in cues if c["status"] == "off")
    n_unv = len(cues) - len(verified_cues)

    # Drift: the slope of "how far off" against "how far into the file", over the
    # cues that are not wild outliers, and only when the file spans enough time.
    drift = None
    trend = [(c["start"], c["start_off"]) for c in verified_cues if abs(c["start_off"] - shift) < 5]
    if len(trend) >= 10 and max(x for x, _ in trend) - min(x for x, _ in trend) >= 300:
        mean_x = statistics.fmean(x for x, _ in trend)
        mean_y = statistics.fmean(y for _, y in trend)
        spread = sum((x - mean_x) ** 2 for x, _ in trend)
        if spread > 0:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in trend) / spread
            drift = round(slope * 3600.0, 2)

    summary = {
        "cues": len(cues), "ok": n_ok, "off": n_off, "unverified": n_unv,
        "ok_percent": round(100.0 * n_ok / len(cues), 1) if cues else 0.0,
        "coverage_percent": round(100.0 * len(verified_cues) / len(cues), 1) if cues else 0.0,
        "global_offset_s": round(offset, 2), "offset_votes": votes,
        "shift_applied_s": round(shift, 2),
        "median_abs_dev_s": round(statistics.median(devs), 2) if devs else None,
        "p95_abs_dev_s": round(_percentile(devs, 0.95), 2) if devs else None,
        "max_abs_dev_s": round(max(devs), 2) if devs else None,
        "drift_s_per_hour": drift,
        "start_tolerance_s": START_TOLERANCE,
    }
    summary["verdict"] = _verdict(summary)

    language = heard.get("language") or imported.get("language") or "en"
    corrected = {
        "language": language,
        "duration": float(max((s["end"] for s in corrected_segments), default=0.0)),
        "text": " ".join(s["text"] for s in corrected_segments),
        "words": corrected_words,
        "segments": corrected_segments,
        "source": "srt_verified",
    }
    return Verification(summary, cues, corrected)


def _verdict(s: dict) -> str:
    """One plain sentence about the file, for the top of the report."""
    parts = []
    shift = s["shift_applied_s"]
    if shift:
        # The shift is what must be ADDED to the file's times, so a positive one
        # means the captions appear before the speech.
        way = "early" if shift > 0 else "late"
        hint = (
            "; a Premiere sequence that starts at 01:00:00:00 gives exactly 3600 s"
            if abs(shift) >= 60 else ""
        )
        parts.append(f"Every caption is {abs(shift):.1f} s {way} (a constant shift{hint}).")
    if s["off"] == 0 and not parts:
        parts.append(f"Timing looks right: {s['ok']} of {s['cues']} captions sit on the speech.")
    elif s["off"]:
        parts.append(f"{s['off']} of {s['cues']} captions are more than {s['start_tolerance_s']:.2f} s from the speech.")
    if s["unverified"]:
        parts.append(f"{s['unverified']} could not be checked (the words were not heard clearly enough).")
    if s["drift_s_per_hour"] is not None and abs(s["drift_s_per_hour"]) >= 0.5:
        parts.append(f"The timing drifts by about {s['drift_s_per_hour']:+.1f} s per hour.")
    return " ".join(parts)


def to_srt(segments: List[dict]) -> str:
    """A SubRip file for the given segments (e.g. the corrected ones)."""

    def stamp(t: float) -> str:
        ms = int(round(max(0.0, t) * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for n, seg in enumerate(segments, start=1):
        blocks.append(f"{n}\n{stamp(seg['start'])} --> {stamp(seg['end'])}\n{seg['text']}\n")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Files and the background run
# --------------------------------------------------------------------------- #
def path(source_id: str, kind: str) -> Path:
    """``kind``: ``srt`` (the file as imported), ``verified`` (corrected) or ``verify`` (report)."""
    return TRANSCRIPTS_DIR / f"{source_id}__{kind}.json"


def _read(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_report(source_id: str) -> Optional[dict]:
    return _read(path(source_id, "verify"))


def activate(source_id: str, which: str) -> dict:
    """Make ``corrected`` or ``original`` the transcript the app uses for this source.

    Raises:
        VerifyError: if there is nothing to switch to.
    """
    if which not in ("corrected", "original"):
        raise VerifyError("Choose 'corrected' or 'original'.")
    source = _read(path(source_id, "verified" if which == "corrected" else "srt"))
    if source is None:
        raise VerifyError("There is no verified transcript for this video yet.")
    from . import pretranscribe

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS_DIR / f"{source_id}__auto.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with pretranscribe._CACHE_LOCK:
        pretranscribe._CACHE[f"{source_id}__auto"] = source
        pretranscribe._CACHE[f"{source_id}__en"] = source
    report = load_report(source_id)
    if report is not None:
        report["applied"] = which
        path(source_id, "verify").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return {"applied": which, "segments": len(source.get("segments") or [])}


def run_verification(
    source_id: str, device: str, language: Optional[str],
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Transcribe the video, verify the imported file against it, save and apply.

    Whisper's result is cached beside the transcript, so verifying again (after
    fixing the file, say) does not transcribe again.

    Raises:
        VerifyError / BadInputError / TranscriptionError.
    """
    from . import pretranscribe, uploads

    imported = _read(path(source_id, "srt"))
    if imported is None:
        raise VerifyError("Upload the subtitle file first.")
    video = uploads.resolve_upload(source_id)

    def inner(frac: float, msg: str) -> None:
        if progress:
            progress(min(0.9, frac * 0.9), msg)

    heard = pretranscribe.get_or_transcribe(
        video, source_id, device, progress=inner, language=language, variant="heard"
    )
    if progress:
        progress(0.92, "Checking every caption against the audio...")
    result = verify(imported, heard)

    report = {
        "summary": result.summary, "cues": result.cues, "applied": "corrected",
        "language": language or "auto", "whisper_model": heard.get("whisper_model"),
    }
    path(source_id, "verified").write_text(json.dumps(result.corrected, ensure_ascii=False), encoding="utf-8")
    path(source_id, "verify").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    activate(source_id, "corrected")
    if progress:
        progress(1.0, "Verified.")
    return report


# --------------------------------------------------------------------------- #
# Background job (same progress shape as a pre-transcription)
# --------------------------------------------------------------------------- #
_JOBS: Dict[str, object] = {}
_JOBS_LOCK = __import__("threading").Lock()


def get_job(job_id: str):
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def start(source_id: str, device: str = "auto", language: Optional[str] = None):
    """Verify ``source_id``'s imported subtitle file on a daemon thread."""
    import threading

    from . import pretranscribe

    job = pretranscribe.TranscriptJob(source_id, device, language)
    job.message = "Starting..."
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    logger.info("[%s] verification started: source=%s language=%s", job.id, source_id, language or "auto")
    return job


def _run_job(job) -> None:
    from .models import BadInputError, TranscriptionError

    def on_progress(frac: float, msg: str) -> None:
        job.update(progress=min(0.99, frac), message=msg)

    try:
        run_verification(job.source_id, job.device, job.language, progress=on_progress)
        job.update(status="done", progress=1.0, message="Verified.")
    except (VerifyError, BadInputError, TranscriptionError) as exc:
        logger.warning("[%s] verification failed: %s", job.id, exc)
        job.update(status="error", error=str(exc), message="Could not verify.")
    except Exception as exc:  # noqa: BLE001 - never let the thread die silently
        logger.exception("[%s] unexpected verification failure", job.id)
        job.update(status="error", error=f"Unexpected error: {exc}", message="Could not verify.")
