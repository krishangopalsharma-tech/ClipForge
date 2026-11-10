"""Subtitle timing verification: parsing editor files, then checking cues against
synthetic "speech" whose true timing is known."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cues, verify  # noqa: E402

VOCAB = [
    "market", "river", "window", "yellow", "silent", "garden", "number", "pocket", "winter", "rocket",
    "button", "candle", "forest", "mirror", "planet", "bridge", "castle", "dragon", "engine", "farmer",
    "guitar", "harbor", "island", "jungle", "kitten", "ladder", "magnet", "nephew", "orange", "pepper",
    "quiet", "rabbit", "sunset", "tunnel", "umpire", "violin", "wizard", "yogurt", "zipper", "anchor",
    "basket", "cactus", "diamond", "eagle", "feather", "galaxy", "hammer", "iceberg", "jacket", "kernel",
    "lantern", "meadow", "napkin", "oyster", "puzzle", "quartz", "ribbon", "saddle", "thunder", "uniform",
    "velvet", "walnut", "yachts", "zebra", "apple", "bread", "cloud", "dance", "earth", "flame",
    "grape", "house", "ivory", "joker", "knife", "lemon", "mouse", "night", "ocean", "piano",
    "queen", "robot", "stone", "tiger", "union", "voice", "water", "xenon", "young", "zesty",
]


def speech(n_sentences=300, seed=7):
    """(heard words, sentences) with exact true times. Words last 0.35 s; sentences are 0.5 s apart."""
    rng = random.Random(seed)
    t, heard, sentences = 5.0, [], []
    for _ in range(n_sentences):
        words = [rng.choice(VOCAB) for _ in range(rng.randint(6, 10))]
        start = t
        for w in words:
            jitter = rng.uniform(-0.03, 0.03)
            heard.append({"word": " " + w, "start": round(t + jitter, 3), "end": round(t + 0.33 + jitter, 3)})
            t += 0.38
        sentences.append({"words": words, "start": start, "end": t - 0.05})
        t += 0.5
    return heard, sentences


def imported(sentences, shift=0.0, bump=None, drift_per_s=0.0, typo=0.0, seed=3):
    """The file's cues: Premiere-style padding, then a deliberate fault."""
    rng = random.Random(seed)
    segs = []
    for i, s in enumerate(sentences):
        words = list(s["words"])
        if typo:
            words = [w[:-1] + "x" if rng.random() < typo and len(w) > 4 else w for w in words]
        extra = (bump or {}).get(i, 0.0)
        start = s["start"] - 0.1 + shift + extra + drift_per_s * s["start"]
        end = s["end"] + 0.2 + shift + extra + drift_per_s * s["start"]
        segs.append({"id": i, "start": round(start, 3), "end": round(end, 3), "text": " ".join(words), "speaker": None})
    return {"language": "en", "segments": segs}


class CueParserTests(unittest.TestCase):
    def test_premiere_timecode_ranges_use_the_real_frame_rate(self):
        text = "00:00:20:03 - 00:00:21:12\nhello there\n\n00:00:22:00 - 00:00:24:10\nsecond line\n"
        out = cues.parse_cues(text, "x.txt", fps=24.0)
        self.assertAlmostEqual(out[0].start, 20 + 3 / 24, places=3)
        self.assertAlmostEqual(out[0].end, 21 + 12 / 24, places=3)
        self.assertEqual(out[0].text, "hello there")
        self.assertAlmostEqual(out[1].end, 24 + 10 / 24, places=3)

    def test_frame_rate_is_guessed_from_the_highest_frame_when_unknown(self):
        out = cues.parse_cues("00:00:01:29 - 00:00:03:00 a b\n", "x.txt")
        self.assertAlmostEqual(out[0].start, 1 + 29 / 30, places=3)   # 29 proves at least 30 fps

    def test_the_timecode_is_not_left_in_the_text(self):
        text = "00:00:20:03 - 00:00:21:12 baat shuru\n"
        self.assertEqual(cues.parse_cues(text, "x.txt")[0].text, "baat shuru")

    def test_timestamp_on_its_own_line_and_open_ended_cues(self):
        out = cues.parse_cues("0:14\nfirst line\n0:20\nsecond line\n", "x.txt")
        self.assertEqual((out[0].start, out[0].end, out[1].start), (14.0, 20.0, 20.0))
        self.assertEqual(out[1].end, 20.0 + cues.DEFAULT_TAIL_SECONDS)

    def test_a_time_inside_a_sentence_is_not_a_cue(self):
        out = cues.parse_cues("0:10 we did it at 3:30 sharp\n", "x.txt")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "we did it at 3:30 sharp")

    def test_headers_before_the_first_time_are_dropped(self):
        out = cues.parse_cues("Transcript\nEnglish\n0:05 hi\n", "x.txt")
        self.assertEqual([c.text for c in out], ["hi"])

    def test_webvtt(self):
        vtt = "WEBVTT\n\nNOTE x\n\n1\n00:00:01.000 --> 00:00:03.500 align:start\n<c>Hello</c> there\n\n00:04.000 --> 00:05.000\nBye\n"
        out = cues.parse_cues(vtt, "x.vtt")
        self.assertEqual([(c.start, c.end, c.text) for c in out], [(1.0, 3.5, "Hello there"), (4.0, 5.0, "Bye")])

    def test_format_sniffing(self):
        self.assertTrue(cues.looks_like_srt("1\n00:00:01,000 --> 00:00:02,000\nhi\n"))
        self.assertFalse(cues.looks_like_srt("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"))
        self.assertFalse(cues.looks_like_srt("00:00:20:03 hello"))

    def test_errors_are_sentences(self):
        for text in ("", "   \n", "no times here at all"):
            with self.assertRaises(cues.CueError):
                cues.parse_cues(text, "x.txt")

    def test_cues_become_a_transcript_with_weighted_words(self):
        t = cues.cues_to_transcript([cues.Cue(0.0, 3.0, "a hippopotamus")])
        a, hippo = t["words"]
        self.assertLess(a["end"] - a["start"], hippo["end"] - hippo["start"])
        self.assertAlmostEqual(hippo["end"], 3.0, places=2)
        self.assertEqual(t["segments"][0]["text"], "a hippopotamus")


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heard_words, cls.sentences = speech()
        cls.heard = {"language": "en", "words": cls.heard_words, "whisper_model": "medium"}

    def run_verify(self, **kw):
        return verify.verify(imported(self.sentences, **kw), self.heard)

    def test_accurate_file_is_confirmed_and_left_alone(self):
        r = self.run_verify()
        s = r.summary
        self.assertEqual(s["off"], 0)
        self.assertGreaterEqual(s["ok_percent"], 98)
        self.assertEqual(s["shift_applied_s"], 0.0)
        self.assertIn("Timing looks right", s["verdict"])
        # The file's own cue timing is kept where it was right.
        orig = imported(self.sentences)["segments"]
        moved = [abs(a["start"] - b["start"]) for a, b in zip(r.corrected["segments"], orig)]
        self.assertLess(max(moved), 0.01)

    def test_word_times_are_measured_not_spread(self):
        r = self.run_verify()
        first = r.corrected["words"][0]
        self.assertAlmostEqual(first["start"], self.heard_words[0]["start"], places=2)
        self.assertEqual(first["word"], self.sentences[0]["words"][0])        # the file's spelling
        self.assertEqual(len(r.corrected["words"]), len(self.heard_words))

    def test_a_timecode_that_starts_at_one_hour_is_found_and_fixed(self):
        r = self.run_verify(shift=3600.0)
        self.assertAlmostEqual(r.summary["shift_applied_s"], -3600.0, delta=1.0)
        self.assertAlmostEqual(r.summary["global_offset_s"], -3600.0, delta=1.0)
        self.assertEqual(r.summary["off"], 0)
        self.assertIn("constant shift", r.summary["verdict"])
        first = r.corrected["segments"][0]
        self.assertAlmostEqual(first["start"], self.sentences[0]["start"] - 0.1, delta=0.3)

    def test_a_small_constant_shift(self):
        r = self.run_verify(shift=1.2)
        # The file is 1.2 s late, and its captions open 0.1 s before the speech as
        # an editor's do, so the typical cue is 1.1 s off: that is what is applied.
        self.assertAlmostEqual(r.summary["shift_applied_s"], -1.1, delta=0.15)
        self.assertEqual(r.summary["off"], 0)
        self.assertIn("late", r.summary["verdict"])
        self.assertAlmostEqual(r.corrected["segments"][10]["start"],
                               self.sentences[10]["start"] - 0.1, delta=0.3)

    def test_individual_late_captions_are_flagged_and_corrected(self):
        bad = {20: 2.5, 21: -2.0, 150: 3.0, 280: 1.6}
        r = self.run_verify(bump=bad)
        flagged = {c["index"] for c in r.cues if c["status"] == "off"}
        self.assertEqual(flagged, set(bad))
        self.assertEqual(r.summary["off"], 4)
        for i in bad:
            fixed = r.corrected["segments"][i]
            self.assertAlmostEqual(fixed["start"], self.sentences[i]["start"] - verify.LEAD, delta=0.15)
        untouched = r.corrected["segments"][22]
        self.assertAlmostEqual(untouched["start"], imported(self.sentences)["segments"][22]["start"], places=2)

    def test_start_offsets_in_the_report_say_which_way(self):
        r = self.run_verify(bump={30: 2.5})
        cue = r.cues[30]
        self.assertAlmostEqual(cue["start_off"], -2.5 - 0.0 + 0.1, delta=0.2)   # audio is 2.5 s earlier than the file says
        self.assertLess(cue["new_start"], cue["start"])

    def test_spelling_differences_still_match(self):
        r = verify.verify(imported(self.sentences, typo=0.2), self.heard)
        self.assertGreaterEqual(r.summary["coverage_percent"], 95)
        self.assertEqual(r.summary["off"], 0)

    def test_slow_drift_is_followed_and_reported(self):
        long_heard, long_sent = speech(n_sentences=900, seed=11)     # ~50 minutes
        heard = {"language": "en", "words": long_heard}
        r = verify.verify(imported(long_sent, drift_per_s=0.001), heard)   # 3.6 s per hour... per 1000 s: 1 s
        self.assertGreaterEqual(r.summary["coverage_percent"], 95)
        self.assertIsNotNone(r.summary["drift_s_per_hour"])
        self.assertAlmostEqual(r.summary["drift_s_per_hour"], -3.6, delta=1.5)

    def test_text_that_was_never_spoken_is_unverified_not_moved(self):
        file = imported(self.sentences)
        file["segments"][40]["text"] = "completely different invented words nobody said here"
        r = verify.verify(file, self.heard)
        self.assertEqual(r.cues[40]["status"], "unverified")
        self.assertEqual(r.corrected["segments"][40]["start"], file["segments"][40]["start"])
        self.assertEqual(r.summary["unverified"], 1)
        self.assertIn("could not be checked", r.summary["verdict"])

    def test_a_different_video_is_refused_with_a_reason(self):
        other, _ = speech(n_sentences=50, seed=999)
        for w in other:
            w["word"] = " zzq" + w["word"].strip()[::-1]
        with self.assertRaises(verify.VerifyError) as cm:
            verify.verify(imported(self.sentences), {"language": "en", "words": other})
        self.assertIn("could not be matched", str(cm.exception))

    def test_no_speech_and_no_text(self):
        with self.assertRaises(verify.VerifyError):
            verify.verify(imported(self.sentences), {"words": []})
        with self.assertRaises(verify.VerifyError):
            verify.verify({"segments": []}, self.heard)

    def test_corrected_transcript_is_well_formed(self):
        r = self.run_verify(bump={5: 2.5})
        segs, words = r.corrected["segments"], r.corrected["words"]
        self.assertEqual(r.corrected["source"], "srt_verified")
        self.assertTrue(all(s["end"] > s["start"] for s in segs))
        self.assertTrue(all(a["end"] <= b["start"] + 1e-6 for a, b in zip(segs, segs[1:])))
        self.assertTrue(all(w["end"] > w["start"] for w in words))
        self.assertTrue(all(a["start"] <= b["start"] + 1e-9 for a, b in zip(words, words[1:])))
        json.dumps(r.corrected)

    def test_hindi_in_two_scripts_matches_through_romanisation(self):
        hindi = ["आप", "कैसे", "हैं", "यह", "बहुत", "अच्छा", "है", "दोस्त", "घर", "मतलब"]
        roman = ["aap", "kaise", "hain", "yah", "bahut", "achcha", "hai", "dost", "ghar", "matlab"]
        rng = random.Random(5)
        heard, segs, t = [], [], 5.0
        for i in range(60):
            idx = [rng.randrange(len(hindi)) for _ in range(8)]
            start = t
            for k in idx:
                heard.append({"word": " " + hindi[k], "start": round(t, 3), "end": round(t + 0.3, 3)})
                t += 0.36
            segs.append({"id": i, "start": start - 0.1, "end": t, "text": " ".join(roman[k] for k in idx), "speaker": None})
            t += 0.6
        r = verify.verify({"language": "hinglish", "segments": segs}, {"language": "hi", "words": heard})
        self.assertGreaterEqual(r.summary["coverage_percent"], 90)

    def test_norm_key(self):
        self.assertEqual(verify.norm_key("Don't,"), "dont")
        self.assertEqual(verify.norm_key("—"), "")
        self.assertEqual(verify.norm_key("आप"), verify.norm_key("aap"))

    def test_srt_writer(self):
        out = verify.to_srt([{"start": 3661.5, "end": 3662.25, "text": "hi"}, {"start": 0, "end": 1, "text": "x"}])
        self.assertTrue(out.startswith("1\n01:01:01,500 --> 01:01:02,250\nhi\n"))
        self.assertIn("2\n00:00:00,000 --> 00:00:01,000\nx", out)


class FilesAndImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target in (verify, ):
            p = mock.patch.object(target, "TRANSCRIPTS_DIR", Path(self.tmp.name))
            p.start()
            self.addCleanup(p.stop)
        from app import uploads

        p = mock.patch.object(uploads, "TRANSCRIPTS_DIR", Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)

    SID = "a" * 32

    def test_importing_keeps_the_file_as_given_and_reports_its_format(self):
        from app import uploads

        text = "00:00:20:03 - 00:00:21:12\nhello there\n"
        res = uploads.save_srt_for_source(self.SID, text.encode("utf-8"), "x.txt", fps=25.0)
        self.assertEqual((res["format"], res["segments"]), ("text", 1))
        saved = json.loads((Path(self.tmp.name) / f"{self.SID}__srt.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(saved["segments"][0]["start"], 20.12, places=2)
        res = uploads.save_srt_for_source(self.SID, b"1\n00:00:01,000 --> 00:00:02,000\nhi\n", "x.srt")
        self.assertEqual(res["format"], "srt")

    def test_activate_switches_between_original_and_corrected(self):
        from app import pretranscribe

        verify.path(self.SID, "srt").write_text(json.dumps({"segments": [{"text": "orig"}]}))
        verify.path(self.SID, "verified").write_text(json.dumps({"segments": [{"text": "fixed"}]}))
        verify.path(self.SID, "verify").write_text(json.dumps({"summary": {}, "applied": "corrected"}))
        verify.activate(self.SID, "original")
        auto = json.loads((Path(self.tmp.name) / f"{self.SID}__auto.json").read_text(encoding="utf-8"))
        self.assertEqual(auto["segments"][0]["text"], "orig")
        self.assertEqual(pretranscribe._CACHE[f"{self.SID}__auto"]["segments"][0]["text"], "orig")
        self.assertEqual(verify.load_report(self.SID)["applied"], "original")
        verify.activate(self.SID, "corrected")
        self.assertEqual(pretranscribe._CACHE[f"{self.SID}__auto"]["segments"][0]["text"], "fixed")
        with self.assertRaises(verify.VerifyError):
            verify.activate(self.SID, "bogus")
        with self.assertRaises(verify.VerifyError):
            verify.activate("b" * 32, "corrected")


class RunVerificationTests(unittest.TestCase):
    """The orchestration, with Whisper replaced by synthetic speech."""

    SID = "c" * 32

    def test_end_to_end_saves_the_report_and_activates_the_corrected_captions(self):
        heard_words, sentences = speech(n_sentences=60, seed=21)
        file = imported(sentences, bump={10: 2.5})
        with tempfile.TemporaryDirectory() as tmp:
            from app import pretranscribe, uploads

            patches = [
                mock.patch.object(verify, "TRANSCRIPTS_DIR", Path(tmp)),
                mock.patch.object(uploads, "resolve_upload", return_value=Path("v.mp4")),
                mock.patch.object(pretranscribe, "get_or_transcribe",
                                  return_value={"language": "en", "words": heard_words, "whisper_model": "medium"}),
            ]
            for p in patches:
                p.start()
            try:
                verify.path(self.SID, "srt").write_text(json.dumps(file))
                steps = []
                report = verify.run_verification(self.SID, "cpu", "en", progress=lambda f, m: steps.append(f))
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(report["summary"]["off"], 1)
            self.assertEqual(report["applied"], "corrected")
            self.assertEqual(report["whisper_model"], "medium")
            self.assertTrue((Path(tmp) / f"{self.SID}__verified.json").is_file())
            self.assertEqual(steps[-1], 1.0)
            self.assertEqual(steps, sorted(steps))
            active = json.loads((Path(tmp) / f"{self.SID}__auto.json").read_text(encoding="utf-8"))
            self.assertEqual(active["source"], "srt_verified")

    def test_requires_an_uploaded_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(verify, "TRANSCRIPTS_DIR", Path(tmp)):
            with self.assertRaises(verify.VerifyError):
                verify.run_verification("d" * 32, "cpu", None)


class ApiTests(unittest.TestCase):
    SID = "e" * 32

    def setUp(self):
        from fastapi.testclient import TestClient
        from app import main, uploads

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target in (verify, uploads):
            p = mock.patch.object(target, "TRANSCRIPTS_DIR", Path(self.tmp.name))
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(uploads, "resolve_upload", return_value=Path("v.mp4"))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(uploads, "probe_fps", return_value=25.0)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_upload_without_verification_behaves_as_before(self):
        res = self.client.post("/api/upload-srt", data={"source_id": self.SID},
                               files={"file": ("a.srt", b"1\n00:00:01,000 --> 00:00:02,000\nhi there\n")})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("verify_id", res.json())
        self.assertEqual(res.json()["format"], "srt")

    def test_upload_with_verification_starts_a_job(self):
        with mock.patch.object(verify, "start") as start:
            start.return_value = mock.Mock(id="job1")
            res = self.client.post(
                "/api/upload-srt", data={"source_id": self.SID, "verify_timing": "true", "language": "hi"},
                files={"file": ("a.txt", b"00:00:01:00 - 00:00:02:00\nnamaste dost\n")})
        self.assertEqual(res.json()["verify_id"], "job1")
        self.assertEqual(res.json()["fps"], 25.0)
        start.assert_called_once_with(self.SID, "auto", "hi")

    def test_a_bad_file_is_a_400_with_a_sentence(self):
        res = self.client.post("/api/upload-srt", data={"source_id": self.SID},
                               files={"file": ("a.txt", b"nothing with times")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("No timestamps", res.json()["detail"])

    def test_report_corrected_srt_and_apply(self):
        verify.path(self.SID, "verify").write_text(json.dumps({"summary": {"cues": 1}, "cues": [], "applied": "corrected"}))
        verify.path(self.SID, "verified").write_text(json.dumps(
            {"segments": [{"start": 1.0, "end": 2.0, "text": "hello"}]}))
        verify.path(self.SID, "srt").write_text(json.dumps({"segments": [{"start": 9.0, "end": 9.5, "text": "orig"}]}))
        self.assertEqual(self.client.get(f"/api/verify-report/{self.SID}").json()["summary"]["cues"], 1)
        srt = self.client.get(f"/api/verify-report/{self.SID}/corrected.srt")
        self.assertEqual(srt.status_code, 200)
        self.assertIn("00:00:01,000 --> 00:00:02,000\nhello", srt.content.decode("utf-8-sig"))
        ok = self.client.post("/api/verify-apply", json={"source_id": self.SID, "use": "original"})
        self.assertEqual(ok.json()["applied"], "original")
        self.assertEqual(self.client.post("/api/verify-apply", json={"source_id": self.SID, "use": "x"}).status_code, 409)

    def test_ids_and_missing_things(self):
        self.assertEqual(self.client.get("/api/verify-report/nope").status_code, 400)
        self.assertEqual(self.client.get("/api/verify-report/" + "f" * 32).status_code, 404)
        self.assertEqual(self.client.get("/api/verify/unknown").status_code, 404)
        res = self.client.post("/api/verify-start", json={"source_id": "f" * 32})
        self.assertEqual(res.status_code, 409)


if __name__ == "__main__":
    unittest.main()
