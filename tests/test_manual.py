"""Chat-route tests: prompt, profiles, word snapping and clip resolution.

Run from the ClipForge folder:  python\\python.exe -m unittest discover -s tests -v
"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import manual, profiles  # noqa: E402


def make_transcript():
    """Five 10 s segments, two words each, with 1 s of silence between segments.

    Segment i spans [i*11, i*11+10]; its words fill it (0-5 s and 5-10 s).
    """
    segments, words = [], []
    for i in range(5):
        s = i * 11.0
        segments.append({"id": i, "start": s, "end": s + 10.0, "text": f"line {i} here"})
        words.append({"word": "line", "start": s, "end": s + 5.0})
        words.append({"word": f"{i}", "start": s + 5.0, "end": s + 10.0})
    return {"language": "en", "duration": 54.0, "segments": segments, "words": words}


class ProfileTests(unittest.TestCase):
    def test_explicit_choice_wins(self):
        self.assertEqual(profiles.pick("us", "hi"), "us")
        self.assertEqual(profiles.pick("in", "en"), "in")

    def test_none_means_generic_even_for_known_language(self):
        self.assertIsNone(profiles.pick("none", "hi"))

    def test_auto_follows_language(self):
        self.assertEqual(profiles.pick("auto", "hinglish"), "in")
        self.assertEqual(profiles.pick(None, "hi"), "in")
        self.assertEqual(profiles.pick("", "en"), "us")
        self.assertIsNone(profiles.pick("auto", "fr"))
        self.assertIsNone(profiles.pick(None, None))

    def test_length_band(self):
        self.assertEqual(profiles.length_band("in"), (20, 40))
        self.assertEqual(profiles.length_band("us"), (25, 50))
        self.assertEqual(profiles.length_band(None), profiles.GENERIC_CLIP_SECONDS)
        self.assertEqual(profiles.length_band("us", 40), (28, 52))

    def test_count_band_grows_with_material(self):
        short = profiles.clip_count_band(240)
        long_ = profiles.clip_count_band(1800)
        self.assertEqual(short, (4, 8))
        self.assertEqual(long_, (6, 12))
        self.assertLess(short[1], long_[1])
        lo, hi = profiles.clip_count_band(0)
        self.assertLessEqual(lo, hi)


class PromptTests(unittest.TestCase):
    def test_fixed_count_and_copy_fields(self):
        p = manual.build_prompt(5, 45, "general")
        self.assertIn("Pick the 5 moments", p)
        self.assertIn("Return exactly 5 clips", p)
        for field in ("hook", "description", "tags", "score"):
            self.assertIn(field, p)
        self.assertIn("Do NOT write timestamps", p)

    def test_band_when_count_unset(self):
        p = manual.build_prompt(None, None, "general", audience="us", count_band=(4, 8))
        self.assertIn("Pick 4 to 8 moments", p)
        self.assertIn("Return between 4 and 8 clips", p)
        self.assertIn("25 to 50 seconds each", p)
        self.assertIn("English speakers", p)

    def test_india_profile_asks_for_roman_script(self):
        p = manual.build_prompt(None, None, "series", audience="in", count_band=(3, 6))
        self.assertIn("ROMAN script", p)
        self.assertIn("climax_seg", p)

    def test_generic_prompt_has_no_audience_block(self):
        p = manual.build_prompt(5, 45, "general")
        self.assertNotIn("AUDIENCE:", p)

    def test_json_example_is_valid_json(self):
        for mode in ("general", "series"):
            p = manual.build_prompt(5, 45, mode)
            start = p.index('{\n  "clips"')
            end = p.index("\n}\n", start) + 2
            example = json.loads(p[start:end])
            self.assertIn("start_seg", example["clips"][0])
            self.assertIn("hook", example["clips"][0])


class SnapTests(unittest.TestCase):
    def setUp(self):
        self.t = make_transcript()
        self.words = self.t["words"]

    def test_edges_land_on_words_with_silence_lead_and_tail(self):
        # Segment 1 is [11, 21]; the silence before it is 1 s, so lead = 0.35.
        start, end = manual.snap_to_words(11.2, 20.8, self.words, 54.0)
        self.assertAlmostEqual(start, 11.0 - 0.35, places=3)
        self.assertAlmostEqual(end, 21.0 + 0.45, places=3)

    def test_idempotent(self):
        once = manual.snap_to_words(11.2, 20.8, self.words, 54.0)
        twice = manual.snap_to_words(*once, self.words, 54.0)
        self.assertEqual(once, twice)

    def test_no_words_leaves_times_alone(self):
        self.assertEqual(manual.snap_to_words(3.1416, 9.0, [], 54.0), (3.142, 9.0))

    def test_far_from_any_word_is_left_alone(self):
        words = [{"word": "x", "start": 100.0, "end": 101.0}]
        self.assertEqual(manual.snap_to_words(10.0, 20.0, words, 200.0), (10.0, 20.0))

    def test_end_never_past_duration(self):
        words = [{"word": "x", "start": 0.0, "end": 10.0}]
        _, end = manual.snap_to_words(0.0, 9.9, words, 10.1)
        self.assertLessEqual(end, 10.1)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.t = make_transcript()

    def test_segment_ids_resolve_and_snap(self):
        w = manual.resolve_windows(self.t, [{"start_seg": 1, "end_seg": 3, "title": "T"}])[0]
        self.assertAlmostEqual(w["start"], 10.65, places=2)
        self.assertAlmostEqual(w["end"], 43.45, places=2)  # segment 3 ends at 43 s
        self.assertTrue(w["snapped"])
        self.assertEqual(w["title"], "T")
        self.assertIn("line 1 here", w["text"])

    def test_copy_fields_are_cleaned(self):
        w = manual.resolve_windows(self.t, [{
            "start_seg": 0, "end_seg": 2,
            "hook": "  Big   news ", "description": "d",
            "tags": ["#a", "b", "A", " "], "score": 140,
        }])[0]
        self.assertEqual(w["hook"], "Big news")
        self.assertEqual(w["tags"], "a, b")
        self.assertEqual(w["score"], 100)

    def test_tags_from_string_and_bad_score(self):
        w = manual.resolve_windows(self.t, [{
            "start_seg": 0, "end_seg": 2, "tags": "#x, y\n#x", "score": "n/a",
        }])[0]
        self.assertEqual(w["tags"], "x, y")
        self.assertIsNone(w["score"])

    def test_picked_by(self):
        entry = {"start_seg": 0, "end_seg": 2}
        self.assertEqual(manual.resolve_windows(self.t, [entry])[0]["picked_by"], "chat")
        self.assertEqual(
            manual.resolve_windows(self.t, [entry], picked_by="Gemini")[0]["picked_by"], "gemini")
        self.assertEqual(
            manual.resolve_windows(self.t, [entry], picked_by="Llama")[0]["picked_by"], "other")
        self.assertEqual(
            manual.resolve_windows(
                self.t, [dict(entry, picked_by="chatgpt")], picked_by="claude")[0]["picked_by"],
            "chatgpt")

    def test_seconds_path_accepts_clock_strings_and_snaps(self):
        w = manual.resolve_windows(self.t, [{"start": "0:11.2", "end": "0:20.8"}])[0]
        self.assertAlmostEqual(w["start"], 10.65, places=2)

    def test_already_snapped_entry_is_not_moved(self):
        w = manual.resolve_windows(
            self.t, [{"start": 11.2, "end": 20.8, "snapped": True}])[0]
        self.assertEqual((w["start"], w["end"]), (11.2, 20.8))

    def test_preview_then_render_does_not_creep(self):
        first = manual.resolve_windows(self.t, [{"start_seg": 1, "end_seg": 3}])[0]
        again = manual.resolve_windows(
            self.t, [{"start": first["start"], "end": first["end"]}])[0]
        self.assertEqual((first["start"], first["end"]), (again["start"], again["end"]))

    def test_bad_clips_are_reported_not_dropped_silently(self):
        windows, skipped = manual.resolve_with_report(self.t, [
            {"start_seg": 0, "end_seg": 2},
            {"start_seg": 9, "end_seg": 12},
            {"start_seg": 3, "end_seg": 1},
        ])
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(skipped), 2)
        self.assertIn("out of range", skipped[0])
        self.assertEqual(manual.summarise(windows, skipped)["skipped"], skipped)

    def test_nothing_usable_raises(self):
        with self.assertRaises(manual.ManualWindowError):
            manual.resolve_windows(self.t, [{"start_seg": 40, "end_seg": 41}])

    def test_length_warning_only_outside_band(self):
        entries = [{"start_seg": 0, "end_seg": 4}, {"start_seg": 0, "end_seg": 1}]
        long_w, ok_w = manual.resolve_windows(self.t, entries, length_band=(15, 25))
        self.assertTrue(long_w["warnings"])
        self.assertEqual(ok_w["warnings"], [])

    def test_shorts_key_is_accepted(self):
        raw = 'Sure!\n```json\n{"shorts": [{"start_seg": 0, "end_seg": 2}]}\n```'
        self.assertEqual(len(manual.parse_windows_json(raw)), 1)

    def test_window_survives_the_request_model(self):
        """The preview's output must go through ManualWindow and resolve to the same cut."""
        from app.models import ManualWindow

        first = manual.resolve_windows(self.t, [{
            "start_seg": 1, "end_seg": 3, "hook": "h", "tags": "a, b", "score": 80,
            "picked_by": "claude",
        }])[0]
        payload = {k: first[k] for k in (
            "start", "end", "snapped", "title", "hook", "description", "tags", "score",
            "picked_by", "reason", "climax_seg", "climax_time", "characters",
            "speaker_segments", "punchline_words")}
        model = ManualWindow(**payload).model_dump(exclude_none=True)
        again = manual.resolve_windows(self.t, [model])[0]
        self.assertEqual((first["start"], first["end"]), (again["start"], again["end"]))
        self.assertEqual(again["hook"], "h")
        self.assertEqual(again["score"], 80)
        self.assertEqual(again["picked_by"], "claude")


if __name__ == "__main__":
    unittest.main()
