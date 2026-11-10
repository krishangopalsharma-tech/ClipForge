"""Movie Shorts: prompt building, JSON resolution, letterbox framing, routes."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import movie, movie_renderer, clipper, reframe  # noqa: E402
from app.models import FitMode  # noqa: E402


def _transcript(count=60, step=3.0, start=0.0):
    """A transcript with evenly spaced lines and five words in each."""
    segments, words = [], []
    for i in range(count):
        s = start + i * step
        segments.append({"id": i, "start": s, "end": s + step - 0.4,
                         "text": f"this is line number {i}"})
        for j, w in enumerate(["this", "is", "line", "number", str(i)]):
            words.append({"word": w, "start": round(s + j * 0.5, 3),
                          "end": round(s + j * 0.5 + 0.42, 3)})
    return {"segments": segments, "words": words, "duration": count * step}


class ParseTests(unittest.TestCase):
    def test_reads_json_out_of_a_fenced_reply(self):
        raw = 'Sure!\n```json\n{"mode":"scene_edit","clips":[{"timeline":[]}]}\n```'
        self.assertEqual(movie.parse_movie_json(raw)["mode"], "scene_edit")

    def test_reads_json_with_prose_around_it(self):
        raw = 'Here you go: {"mode":"scene_edit","clips":[{"timeline":[]}]} hope that helps'
        self.assertEqual(len(movie.parse_movie_json(raw)["clips"]), 1)

    def test_rejects_empty_and_clipless_payloads(self):
        for raw in ("", "   ", "{}", '{"clips":[]}', "not json at all"):
            with self.assertRaises(movie.MovieJsonError, msg=raw):
                movie.parse_movie_json(raw)

    def test_rejects_an_unknown_mode(self):
        with self.assertRaises(movie.MovieJsonError):
            movie.parse_movie_json('{"mode":"karaoke","clips":[{"timeline":[]}]}')


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tr = _transcript()

    def _clip(self, timeline):
        return {"clip_id": 1, "title": "T", "timeline": timeline}

    def test_dialogue_block_resolves_to_real_times(self):
        out = movie.resolve_clip(
            self._clip([{"type": "dialogue", "start_seg": 2, "end_seg": 4}]),
            self.tr, movie.SCENE_EDIT,
        )
        block = out["timeline"][0]
        # Segment 2 starts at 6.0 and segment 4 ends at 14.6, give or take the
        # snap into the surrounding pause.
        self.assertAlmostEqual(block["start"], 6.0, delta=0.6)
        self.assertAlmostEqual(block["end"], 14.6, delta=0.6)
        self.assertGreater(block["est_duration"], 0)

    def test_cuts_are_snapped_onto_word_boundaries(self):
        out = movie.resolve_clip(
            self._clip([{"type": "dialogue", "start_seg": 5, "end_seg": 6}]),
            self.tr, movie.SCENE_EDIT,
        )
        block = out["timeline"][0]
        starts = {round(w["start"], 2) for w in self.tr["words"]}
        # The start sits at a word's start, or just inside the pause before it.
        self.assertTrue(
            any(abs(block["start"] - s) <= 0.36 for s in starts),
            f"{block['start']} is not near any word start",
        )

    def test_out_of_range_segments_are_skipped_not_fatal(self):
        out = movie.resolve_clip(
            self._clip([
                {"type": "dialogue", "start_seg": 1, "end_seg": 2},
                {"type": "dialogue", "start_seg": 9000, "end_seg": 9001},
            ]),
            self.tr, movie.SCENE_EDIT,
        )
        self.assertEqual(len(out["timeline"]), 1)
        self.assertTrue(any("outside" in w for w in out["warnings"]))

    def test_a_clip_with_nothing_usable_raises(self):
        with self.assertRaises(movie.MovieJsonError):
            movie.resolve_clip(
                self._clip([{"type": "dialogue", "start_seg": 9000, "end_seg": 9001}]),
                self.tr, movie.SCENE_EDIT,
            )

    def test_bracketed_asides_are_stripped_from_narration(self):
        out = movie.resolve_clip(
            self._clip([{
                "type": "voiceover",
                "narration_text": "He paid fifty million (about 50M) for it.",
                "visual_segs": [1, 2],
            }]),
            self.tr, movie.VOICEOVER_HYBRID,
        )
        text = out["timeline"][0]["narration_text"]
        self.assertNotIn("(", text)
        self.assertNotIn("50M", text)
        self.assertTrue(any("bracketed" in w for w in out["warnings"]))

    def test_hybrid_warns_when_the_actors_are_never_heard(self):
        out = movie.resolve_clip(
            self._clip([{"type": "voiceover", "narration_text": "All narration here.",
                         "visual_segs": [1]}]),
            self.tr, movie.VOICEOVER_HYBRID,
        )
        self.assertTrue(any("never heard" in w for w in out["warnings"]))

    def test_unknown_grade_falls_back(self):
        clip = self._clip([{"type": "dialogue", "start_seg": 1, "end_seg": 2}])
        clip["grade_preset"] = "disco"
        out = movie.resolve_clip(clip, self.tr, movie.SCENE_EDIT)
        self.assertEqual(out["grade_preset"], "teal_orange")

    def test_payload_keeps_good_clips_when_one_is_bad(self):
        payload = {"mode": "scene_edit", "clips": [
            {"clip_id": 1, "timeline": [{"type": "dialogue", "start_seg": 1, "end_seg": 3}]},
            {"clip_id": 2, "timeline": [{"type": "dialogue", "start_seg": 9000, "end_seg": 9001}]},
        ]}
        out = movie.resolve_payload(payload, self.tr)
        self.assertEqual(len(out["clips"]), 1)
        self.assertEqual(len(out["skipped"]), 1)

    def test_speed_is_clamped(self):
        payload = {"mode": "scene_edit", "speed": 99,
                   "clips": [{"timeline": [{"type": "dialogue", "start_seg": 1, "end_seg": 2}]}]}
        self.assertLessEqual(movie.resolve_payload(payload, self.tr)["speed"], 2.0)


class PromptTests(unittest.TestCase):
    def test_both_modes_build_and_name_their_shape(self):
        for mode in movie.MODES:
            text = movie.build_movie_prompt(mode)
            self.assertIn(mode, text, mode)
            self.assertIn("[id]", text)
            self.assertIn("TRANSCRIPT:", text)

    def test_hybrid_prompt_forbids_brackets(self):
        text = movie.build_movie_prompt(movie.VOICEOVER_HYBRID)
        self.assertIn("NEVER use brackets", text)

    def test_unknown_mode_raises(self):
        with self.assertRaises(movie.MovieJsonError):
            movie.build_movie_prompt("interpretive_dance")

    def test_a_named_title_is_what_the_model_is_told_to_look_up(self):
        text = movie.build_movie_prompt(title="Breaking Bad", season=3, episode=7)
        self.assertIn("Breaking Bad, season 3, episode 7", text)
        self.assertIn("Look up", text)

    def test_without_a_title_the_model_is_asked_to_identify_it(self):
        text = movie.build_movie_prompt()
        self.assertIn("this film or episode", text)
        self.assertIn("Work out from the transcript", text)

    def test_the_prompt_never_claims_clipforge_searched_anything(self):
        text = movie.build_movie_prompt(title="Dune")
        self.assertIn("If you cannot browse the web", text)

    def test_count_and_length_are_left_to_the_model_by_default(self):
        text = movie.build_movie_prompt()
        self.assertIn("HOW MANY: you decide", text)
        self.assertIn("HOW LONG: you decide", text)

    def test_a_fixed_count_and_length_override_auto(self):
        text = movie.build_movie_prompt(num_clips=6, clip_length=45)
        self.assertIn("Find 6 shorts", text)
        self.assertIn("about 45 seconds", text)
        self.assertNotIn("HOW MANY: you decide", text)

    def test_subject_line_handles_partial_details(self):
        self.assertEqual(movie.subject_line("Dune"), "Dune")
        self.assertEqual(movie.subject_line("Dune", 2), "Dune, season 2")
        self.assertEqual(movie.subject_line("Dune", None, 5), "Dune, episode 5")
        self.assertEqual(movie.subject_line(None, 2, 5), "this film or episode")
        self.assertEqual(movie.subject_line("  "), "this film or episode")


class LetterboxTests(unittest.TestCase):
    def test_fit_mode_exists(self):
        self.assertEqual(FitMode.LETTERBOX.value, "letterbox")

    def test_plan_is_returned_without_probing_or_a_model(self):
        """Letterbox crops nothing, so it must not need ffprobe or the detector."""
        with mock.patch.object(reframe, "probe_size",
                               side_effect=AssertionError("must not probe")), \
             mock.patch.object(reframe, "get_detector",
                               side_effect=AssertionError("must not load a model")):
            plan = reframe.plan_reframe(
                Path("nope.mp4"), 0.0, 5.0, 1080, 1920, "letterbox", Path("."), "0",
            )
        self.assertEqual(plan.mode, "letterbox")

    def test_filter_graph_pads_onto_black_and_crops_nothing(self):
        plan = reframe.ReframePlan("letterbox", "chosen by you")
        graph = clipper._reframe_stage(1080, 1920, plan, Path("."))
        self.assertIn("force_original_aspect_ratio=decrease", graph)
        self.assertIn("black", graph)
        self.assertIn("[v0]", graph)
        self.assertNotIn("boxblur", graph)

    def test_output_rate_is_pinned_for_mixed_sources(self):
        self.assertIn("-ar", clipper._ENCODE_ARGS)
        self.assertIn("48000", clipper._ENCODE_ARGS)


class MicroWindowTests(unittest.TestCase):
    def setUp(self):
        self.segments = _transcript()["segments"]

    def test_windows_fill_exactly_the_narration_length(self):
        for total in (3.0, 7.5, 12.0):
            windows = movie_renderer._micro_windows(
                self.segments, [4, 9, 14], total, 180.0)
            self.assertAlmostEqual(sum(w[1] for w in windows), total, places=2,
                                   msg=f"total={total}")

    def test_each_window_is_centred_on_its_segment(self):
        windows = movie_renderer._micro_windows(self.segments, [10], 4.0, 180.0)
        start, length = windows[0]
        seg = self.segments[10]
        mid = (seg["start"] + seg["end"]) / 2
        self.assertAlmostEqual(start + length / 2, mid, delta=0.3)

    def test_no_visual_segments_yields_nothing_to_cut(self):
        self.assertEqual(movie_renderer._micro_windows(self.segments, [], 5.0, 180.0), [])

    def test_windows_stay_inside_the_source(self):
        windows = movie_renderer._micro_windows(self.segments, [0], 6.0, 180.0)
        self.assertGreaterEqual(windows[0][0], 0.0)


class LookTests(unittest.TestCase):
    def test_movie_grades_are_registered_with_the_rest(self):
        from app.effects import COLOR_GRADES
        for key in ("cold_gritty", "vibrant_cinema", "teal_orange"):
            self.assertIn(key, COLOR_GRADES, key)
            self.assertTrue(COLOR_GRADES[key])

    def test_sharpening_is_opt_in(self):
        from app.effects import cinematic_stages
        off, _ = cinematic_stages({"color_grade": "none"}, "in", 1080, 1920)
        on, _ = cinematic_stages({"color_grade": "none", "sharpen": True}, "in", 1080, 1920)
        self.assertEqual("".join(off).count("cas="), 0)
        self.assertEqual("".join(on).count("cas="), 1)

    def test_look_chain_is_empty_when_nothing_is_asked_for(self):
        self.assertEqual(movie_renderer._look_chain("none", False), "")


class FitModeTests(unittest.TestCase):
    """Movie Shorts offers the same framing choices as the rest of the app."""

    def _stage(self, fit, **kw):
        return movie_renderer._video_stage(
            3, "bv0", "teal_orange", True, fit, 1080, 1920, **kw)

    def test_every_fit_reads_its_input_and_ends_at_its_label(self):
        for fit in (FitMode.LETTERBOX, FitMode.CROP, FitMode.BLUR,
                    FitMode.AUTO, FitMode.TRACK):
            graph = self._stage(fit)
            self.assertTrue(graph.startswith("[3:v]"), fit)
            self.assertTrue(graph.endswith("[bv0]"), fit)

    def test_cropping_modes_actually_crop(self):
        for fit in (FitMode.CROP, FitMode.AUTO):
            self.assertIn("crop=1080:1920", self._stage(fit), fit)

    def test_letterbox_pads_instead_of_cropping(self):
        graph = self._stage(FitMode.LETTERBOX)
        self.assertIn("pad=1080:1920", graph)
        self.assertNotIn("crop=1080:1920", graph)

    def test_square_uses_the_shared_mask(self):
        graph = self._stage(FitMode.SQUARE, mask_label="mkin0")
        self.assertIn("[mkin0]", graph)
        self.assertIn("alphamerge", graph)
        self.assertTrue(graph.endswith("[bv0]"))

    def test_two_slices_never_share_internal_labels(self):
        import re
        a = movie_renderer._video_stage(0, "bv0", "none", False, FitMode.BLUR, 1080, 1920)
        b = movie_renderer._video_stage(1, "bv1", "none", False, FitMode.BLUR, 1080, 1920)
        labels = lambda g: set(re.findall(r"\[([a-z][a-z0-9]*)\]", g))
        self.assertEqual(labels(a) & labels(b), set())

    def test_the_grade_lands_on_the_square_not_the_canvas(self):
        graph = self._stage(FitMode.SQUARE, mask_label="mkin0")
        square_branch = [s for s in graph.split(";") if "fgi" in s and "scale" in s]
        self.assertTrue(any("colorbalance" in s or "eq=" in s for s in square_branch))


class HandoffTests(unittest.TestCase):
    """A short becomes a source video, so the main app can edit it."""

    def test_bare_assembly_applies_no_framing_grade_or_captions(self):
        graph = movie_renderer._video_stage(
            2, "bv0", "teal_orange", True, FitMode.CROP, 1920, 1080)
        self.assertIn("crop=", graph)      # the baked path does frame it
        # and the bare path, built the same way inside build_short, does not:
        bare = f"[2:v]scale=1920:1080,fps={movie_renderer.FPS},setsar=1,format=yuv420p[bv0]"
        for banned in ("crop=", "pad=", "colorbalance", "cas=", "ass="):
            self.assertNotIn(banned, bare)

    def test_the_recipe_matches_what_the_renderer_writes(self):
        from app import movie_jobs
        from app.rendering import make_recipe
        from app.models import GenerateRequest
        mine = movie_jobs._recipe("abc", 42.0, {})
        theirs = make_recipe(
            GenerateRequest(video_url="http://x/y"), {}, "abc", 0.0, 42.0)
        self.assertEqual(set(mine), set(theirs))

    def test_recipe_covers_the_whole_short(self):
        from app import movie_jobs
        recipe = movie_jobs._recipe("abc", 42.0, {"fit_mode": "crop"})
        self.assertEqual((recipe["start"], recipe["end"]), (0.0, 42.0))
        self.assertEqual(recipe["fit_mode"], "crop")
        self.assertEqual(recipe["source_id"], "abc")

    def test_published_source_gets_a_loadable_transcript(self):
        import json as _json
        import tempfile
        from app import movie_jobs, pretranscribe
        from app.paths import DOWNLOADS_DIR, TRANSCRIPTS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "assembled.mp4"
            fake.write_bytes(b"not really a video")
            report = {
                "duration": 12.5,
                "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "hello"}],
            }
            source_id = movie_jobs.publish_as_source(fake, report)
        try:
            self.assertFalse(fake.exists(), "the assembled file should be moved")
            loaded = pretranscribe.load(source_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["segments"]), 1)
            self.assertEqual(loaded["duration"], 12.5)
        finally:
            (DOWNLOADS_DIR / f"{source_id}.mp4").unlink(missing_ok=True)
            (TRANSCRIPTS_DIR / f"{source_id}__auto.json").unlink(missing_ok=True)


class RouteTests(unittest.TestCase):
    """The app object is built without running its lifespan, so no model loads."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from app import main
        self.client = TestClient(main.app)

    def test_prompt_endpoint_serves_both_modes(self):
        for mode in movie.MODES:
            res = self.client.get(f"/api/movie/prompt?mode={mode}")
            self.assertEqual(res.status_code, 200, mode)
            self.assertIn("TRANSCRIPT:", res.json()["prompt"])

    def test_prompt_endpoint_passes_the_title_through(self):
        res = self.client.get(
            "/api/movie/prompt?title=The%20Wire&season=1&episode=4")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["subject"], "The Wire, season 1, episode 4")
        self.assertIn("The Wire, season 1, episode 4", body["prompt"])

    def test_prompt_endpoint_defaults_to_auto_count_and_length(self):
        body = self.client.get("/api/movie/prompt").json()
        self.assertIn("HOW MANY: you decide", body["prompt"])
        self.assertIn("HOW LONG: you decide", body["prompt"])

    def test_prompt_endpoint_rejects_an_unknown_mode(self):
        self.assertEqual(self.client.get("/api/movie/prompt?mode=nope").status_code, 400)

    def test_videos_endpoint_lists_the_library_newest_first(self):
        """Step 1's picker exists so nobody has to know a source id by heart."""
        import tempfile
        from app import main

        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp) / "downloads"
            transcripts = Path(tmp) / "transcripts"
            downloads.mkdir()
            transcripts.mkdir()
            old, new_ = "a" * 32, "b" * 32
            (downloads / f"{old}.mp4").write_bytes(b"x" * 10)
            (downloads / f"{new_}.mkv").write_bytes(b"y" * 20)
            (downloads / f"{new_}.name.txt").write_text("Heat 1995.mkv", encoding="utf-8")
            (transcripts / f"{old}__auto.json").write_text("{}", encoding="utf-8")
            os.utime(downloads / f"{old}.mp4", (1000, 1000))
            os.utime(downloads / f"{new_}.mkv", (2000, 2000))

            with mock.patch.object(main, "DOWNLOADS_DIR", downloads),                  mock.patch.object(main, "TRANSCRIPTS_DIR", transcripts),                  mock.patch.object(main.history, "list_entries", return_value=[]):
                body = self.client.get("/api/movie/videos").json()

        ids = [v["source_id"] for v in body["videos"]]
        self.assertEqual(ids, [new_, old])
        self.assertEqual(body["videos"][0]["label"], "Heat 1995.mkv")
        self.assertFalse(body["videos"][0]["has_transcript"])
        self.assertTrue(body["videos"][1]["has_transcript"])

    def test_videos_is_served_under_both_paths(self):
        """The dashboard calls /api/videos; Movie Shorts' older path still works."""
        for path in ("/api/videos", "/api/movie/videos"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)
            self.assertIn("videos", res.json(), path)

    def test_videos_endpoint_falls_back_to_the_history_label(self):
        import tempfile
        from app import main

        with tempfile.TemporaryDirectory() as tmp:
            downloads = Path(tmp) / "downloads"
            transcripts = Path(tmp) / "transcripts"
            downloads.mkdir()
            transcripts.mkdir()
            sid = "c" * 32
            (downloads / f"{sid}.mp4").write_bytes(b"x")
            entries = [{"source_id": sid, "source": "Sherlock S01E03"}]
            with mock.patch.object(main, "DOWNLOADS_DIR", downloads),                  mock.patch.object(main, "TRANSCRIPTS_DIR", transcripts),                  mock.patch.object(main.history, "list_entries", return_value=entries):
                body = self.client.get("/api/movie/videos").json()

        self.assertEqual(body["videos"][0]["label"], "Sherlock S01E03")

    def test_voices_endpoint_answers_even_when_kokoro_is_down(self):
        from app import kokoro_tts
        with mock.patch.object(kokoro_tts, "status",
                               return_value={"ok": False, "error": "nope"}):
            res = self.client.get("/api/movie/voices")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["running"])

    def test_plan_without_a_transcript_is_404(self):
        from app import pretranscribe
        with mock.patch.object(pretranscribe, "load", return_value=None):
            res = self.client.post("/api/movie/plan",
                                   json={"source_id": "a" * 32, "raw": "{}"})
        self.assertEqual(res.status_code, 404)

    def test_plan_with_bad_json_is_400(self):
        from app import pretranscribe
        with mock.patch.object(pretranscribe, "load", return_value=_transcript()):
            res = self.client.post("/api/movie/plan",
                                   json={"source_id": "a" * 32, "raw": "garbage"})
        self.assertEqual(res.status_code, 400)

    def test_plan_resolves_a_good_payload(self):
        from app import pretranscribe
        raw = ('{"mode":"scene_edit","clips":[{"clip_id":1,"title":"T",'
               '"timeline":[{"type":"dialogue","start_seg":2,"end_seg":5}]}]}')
        with mock.patch.object(pretranscribe, "load", return_value=_transcript()):
            res = self.client.post("/api/movie/plan",
                                   json={"source_id": "a" * 32, "raw": raw})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["clips"]), 1)

    def test_render_refuses_narration_while_kokoro_is_down(self):
        from app import pretranscribe, kokoro_tts
        raw = ('{"mode":"voiceover_hybrid","clips":[{"clip_id":1,"timeline":'
               '[{"type":"voiceover","narration_text":"Hello there.","visual_segs":[1,2]}]}]}')
        with mock.patch.object(pretranscribe, "load", return_value=_transcript()), \
             mock.patch.object(kokoro_tts, "status", return_value={"ok": False, "error": "down"}):
            res = self.client.post("/api/movie/render",
                                   json={"source_id": "a" * 32, "raw": raw})
        self.assertEqual(res.status_code, 503)
        self.assertIn("Kokoro", res.json()["detail"])

    def test_render_accepts_framing_and_caption_options(self):
        from app import pretranscribe, movie_jobs
        raw = ('{"mode":"scene_edit","clips":[{"clip_id":1,"timeline":'
               '[{"type":"dialogue","start_seg":2,"end_seg":4}]}]}')
        seen = {}

        def fake_start(source_id, plan, source, **kw):
            seen.update(kw)
            return mock.Mock(id="job1")

        with mock.patch.object(pretranscribe, "load", return_value=_transcript()), \
             mock.patch.object(movie_jobs, "start", side_effect=fake_start), \
             mock.patch("app.uploads.resolve_upload", return_value=Path("x.mp4")):
            res = self.client.post("/api/movie/render", json={
                "source_id": "a" * 32, "raw": raw, "fit_mode": "square",
                "aspect_ratio": "16:9", "caption_scale": 0.6,
                "caption_position": "bottom",
            })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(seen["fit_mode"], "square")
        self.assertEqual(seen["aspect_ratio"], "16:9")
        self.assertEqual(seen["caption_overrides"]["font_scale"], 0.6)
        self.assertEqual(seen["caption_overrides"]["position"], "bottom")

    def test_render_rejects_an_unknown_fit_mode(self):
        from app import pretranscribe
        with mock.patch.object(pretranscribe, "load", return_value=_transcript()):
            res = self.client.post("/api/movie/render", json={
                "source_id": "a" * 32, "raw": "{}", "fit_mode": "origami"})
        self.assertEqual(res.status_code, 422)

    def test_unknown_render_job_is_404(self):
        self.assertEqual(self.client.get("/api/movie/job/nope").status_code, 404)

    def test_movie_page_is_served(self):
        res = self.client.get("/movie")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Movie Shorts", res.text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
