"""Editing finished clips: overlays, the ffmpeg overlay stages, recipes, applying
changes, and the restyle flow (render mocked, history on a temp file)."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import clipper, history, overlays, rendering  # noqa: E402
from app.models import (  # noqa: E402
    AspectRatio, ClipChanges, ClipGenerationError, FitMode, GenerateRequest,
    BadInputError,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def tearDownModule():
    """The graph tests need a work folder on the project drive; remove it."""
    import shutil

    shutil.rmtree(Path.cwd() / "data" / "work" / "t", ignore_errors=True)


def transcript():
    words = [{"word": "w", "start": float(i), "end": i + 0.8} for i in range(0, 100)]
    return {"language": "en", "duration": 100.0, "words": words,
            "segments": [{"id": 0, "start": 0.0, "end": 100.0, "text": "w " * 100}]}


class LogoLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(overlays, "OVERLAYS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_list_and_resolve(self):
        info = overlays.save_logo("My Logo.png", io.BytesIO(PNG))
        self.assertEqual(info["file"], "My Logo.png")
        self.assertEqual([x["file"] for x in overlays.list_logos()], ["My Logo.png"])
        self.assertTrue(overlays.resolve_logo("My Logo.png").is_file())

    def test_duplicate_names_get_a_suffix(self):
        overlays.save_logo("a.png", io.BytesIO(PNG))
        self.assertEqual(overlays.save_logo("a.png", io.BytesIO(PNG))["file"], "a-2.png")

    def test_rejects_wrong_type_empty_big_and_fake_images(self):
        for name, data in (("a.gif", PNG), ("a.png", b""), ("a.png", b"not an image"),
                           ("a.png", PNG + b"\x00" * overlays.MAX_LOGO_BYTES)):
            with self.assertRaises(BadInputError, msg=name):
                overlays.save_logo(name, io.BytesIO(data))

    def test_resolve_refuses_paths_and_missing_files(self):
        overlays.save_logo("a.png", io.BytesIO(PNG))
        for bad in ("../a.png", "sub/a.png", "nope.png", "a.txt", "", None):
            with self.assertRaises(overlays.OverlayError, msg=str(bad)):
                overlays.resolve_logo(bad)


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(overlays, "OVERLAYS_DIR", Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)
        overlays.save_logo("l.png", io.BytesIO(PNG))

    def test_defaults_and_clamping(self):
        out = overlays.normalize([
            {"type": "logo", "file": "l.png", "x": 1.7, "w": 0.0},
            {"type": "text", "text": "  hi   there ", "size": 9},
        ])
        self.assertEqual(out[0]["x"], 1.0)
        self.assertEqual(out[0]["w"], 0.02)
        self.assertEqual(out[0]["y"], 0.10)
        self.assertEqual(out[1]["text"], "hi there")
        self.assertEqual(out[1]["size"], 0.15)
        self.assertEqual(out[1]["color"], "#FFFFFF")

    def test_timing(self):
        item = overlays.normalize([{"type": "text", "text": "x", "start": 1, "end": 3}])[0]
        self.assertEqual((item["start"], item["end"]), (1.0, 3.0))
        with self.assertRaises(overlays.OverlayError):
            overlays.normalize([{"type": "text", "text": "x", "start": 3, "end": 2}])

    def test_errors(self):
        bad = [
            [{"type": "banner"}], [{"type": "text", "text": "  "}],
            [{"type": "text", "text": "x", "color": "red"}],
            [{"type": "logo", "file": "gone.png"}], ["nope"],
            [{"type": "text", "text": "x"}] * (overlays.MAX_ITEMS + 1),
            [{"type": "text", "text": "x", "x": "left"}],
        ]
        for items in bad:
            with self.assertRaises(overlays.OverlayError, msg=str(items)[:50]):
                overlays.normalize(items)
        with self.assertRaises(overlays.OverlayError):
            overlays.normalize("x")

    def test_none_and_empty(self):
        self.assertEqual(overlays.normalize(None), [])
        self.assertEqual(overlays.normalize([]), [])


class OverlayGraphTests(unittest.TestCase):
    def opts(self, items, **kw):
        return clipper.ClipOptions(
            aspect_ratio=AspectRatio.NINE_16, fit_mode=FitMode.CROP,
            ass_path=Path.cwd() / "data" / "work" / "t" / "0.ass", clip_id="c", index=0,
            overlays=items, logo_base=1, **kw)

    def graph(self, items):
        work = Path.cwd() / "data" / "work" / "t"
        return clipper._build_crop_filter_complex(1080, 1920, self.opts(items), work)

    def test_logo_and_text_sit_under_the_captions(self):
        g = self.graph([
            {"type": "logo", "file": "l.png", "x": 0.85, "y": 0.1, "w": 0.2, "opacity": 0.5},
            {"type": "text", "text": "Hi: you", "x": 0.5, "y": 0.9, "size": 0.05, "color": "#FFD400", "opacity": 1.0},
        ])
        self.assertIn("[1:v]scale=216:-2,format=rgba,colorchannelmixer=aa=0.500[lg0]", g)
        self.assertIn("overlay=x=W*0.8500-w/2:y=H*0.1000-h/2[ovl0]", g)
        self.assertIn("drawtext=", g)
        self.assertIn("fontsize=96", g)
        self.assertIn("Hi\\: you", g)
        self.assertLess(g.index("[ovl1]"), g.index("ass="))   # overlays first, captions last
        self.assertTrue(g.endswith("[outv]"))

    def test_timed_overlay_gets_an_enable_window(self):
        g = self.graph([{"type": "text", "text": "x", "x": .5, "y": .5, "size": .05,
                         "color": "#FFFFFF", "opacity": 1.0, "start": 1.0, "end": 3.0}])
        self.assertIn("enable='between(t\\,1.000\\,3.000)'", g)

    def test_two_logos_use_consecutive_inputs(self):
        item = {"type": "logo", "file": "l.png", "x": .5, "y": .5, "w": .1, "opacity": 1.0}
        g = self.graph([item, dict(item)])
        self.assertIn("[1:v]scale", g)
        self.assertIn("[2:v]scale", g)

    def test_no_overlays_changes_nothing(self):
        self.assertNotIn("ovl", self.graph([]))


class InputOrderTests(unittest.TestCase):
    """generate_clip must give logos the input indexes the graph refers to."""

    def run_generate(self, fit, music, items):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path.cwd() / "data" / "work" / "t"
        work.mkdir(parents=True, exist_ok=True)
        (work / "0.ass").write_text("x")
        out = Path(tmp.name) / "o.mp4"
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            out.write_bytes(b"x")
            return mock.Mock(returncode=0, stderr="")

        opts = clipper.ClipOptions(
            aspect_ratio=AspectRatio.NINE_16, fit_mode=fit, ass_path=work / "0.ass",
            clip_id="c", index=0, overlays=items,
            music_path=Path(tmp.name) / "m.mp3" if music else None)
        if music:
            (Path(tmp.name) / "m.mp3").write_bytes(b"x")
        with mock.patch.object(clipper.subprocess, "run", fake_run), \
                mock.patch.object(clipper.overlays, "resolve_logo",
                                  side_effect=lambda n: Path(tmp.name) / n), \
                mock.patch.object(clipper, "ensure_rounded_mask", return_value=Path(tmp.name) / "mask.png"):
            clipper.generate_clip(Path("v.mp4"), 0, 4, opts, out_path=out)
        return seen["cmd"]

    def test_logo_input_follows_the_source(self):
        cmd = self.run_generate(FitMode.CROP, False, [{"type": "logo", "file": "l.png", "x": .5, "y": .5, "w": .1, "opacity": 1}])
        self.assertEqual(cmd[cmd.index("-loop") + 2], "-i")
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[1:v]scale", fc)

    def test_music_shifts_the_logo_index(self):
        cmd = self.run_generate(FitMode.CROP, True, [{"type": "logo", "file": "l.png", "x": .5, "y": .5, "w": .1, "opacity": 1}])
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[2:v]scale", fc)          # 0 = source, 1 = music, 2 = logo
        self.assertIn("[1:a]volume", fc)

    def test_square_mode_shifts_it_for_the_mask(self):
        cmd = self.run_generate(FitMode.SQUARE, True, [{"type": "logo", "file": "l.png", "x": .5, "y": .5, "w": .1, "opacity": 1}])
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[3:v]scale", fc)          # source, mask, music, logo

    def test_a_missing_logo_is_skipped_not_fatal(self):
        items = [{"type": "logo", "file": "gone.png", "x": .5, "y": .5, "w": .1, "opacity": 1},
                 {"type": "text", "text": "kept", "x": .5, "y": .5, "size": .05, "color": "#FFFFFF", "opacity": 1}]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path.cwd() / "data" / "work" / "t"
        work.mkdir(parents=True, exist_ok=True)
        (work / "0.ass").write_text("x")
        out = Path(tmp.name) / "o.mp4"
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            out.write_bytes(b"x")
            return mock.Mock(returncode=0, stderr="")

        opts = clipper.ClipOptions(aspect_ratio=AspectRatio.NINE_16, fit_mode=FitMode.CROP,
                                   ass_path=work / "0.ass", clip_id="c", index=0, overlays=items)
        with mock.patch.object(clipper.subprocess, "run", fake_run):
            clipper.generate_clip(Path("v.mp4"), 0, 4, opts, out_path=out)
        self.assertNotIn("-loop", seen["cmd"])
        self.assertIn("kept", seen["cmd"][seen["cmd"].index("-filter_complex") + 1])


class RecipeTests(unittest.TestCase):
    def test_recipe_is_json_and_complete(self):
        req = GenerateRequest(
            video_url="http://x", fit_mode="auto", caption_style="bold_white",
            cinematic={"color_grade": "warm"}, music_track="a.mp3", language="hi",
            signature={"enabled": True, "text": "@me"})
        r = rendering.make_recipe(req, {"characters": ["A"], "climax_time": 9.0}, "a" * 32, 10.0, 40.0)
        json.dumps(r)
        self.assertEqual((r["source_id"], r["start"], r["end"]), ("a" * 32, 10.0, 40.0))
        self.assertEqual((r["fit_mode"], r["aspect_ratio"]), ("auto", "9:16"))
        self.assertEqual(r["transcript_language"], "hi")
        self.assertEqual(r["cinematic"]["color_grade"], "warm")
        self.assertEqual(r["window"]["characters"], ["A"])
        self.assertEqual(r["overlays"], [])


def base_recipe(**kw):
    r = rendering.make_recipe(
        GenerateRequest(video_url="http://x", caption_style="bold_white", music_track=None),
        {}, "b" * 32, 10.0, 30.0)
    r.update(kw)
    return r


class ApplyChangesTests(unittest.TestCase):
    def test_only_the_fields_present_change(self):
        r = base_recipe()
        new = rendering.apply_changes(r, ClipChanges(caption_style="minimal", fit_mode="blur"), transcript())
        self.assertEqual((new["caption_style"], new["fit_mode"]), ("minimal", "blur"))
        self.assertEqual((new["start"], new["end"], new["aspect_ratio"]), (10.0, 30.0, "9:16"))
        self.assertEqual(r["caption_style"], "bold_white")      # input untouched

    def test_explicit_null_clears(self):
        r = base_recipe(cinematic={"color_grade": "warm"}, signature={"enabled": True, "text": "x"})
        new = rendering.apply_changes(r, ClipChanges(cinematic=None, signature=None), transcript())
        self.assertIsNone(new["cinematic"])
        self.assertIsNone(new["signature"])

    def test_trim_is_snapped_and_checked(self):
        new = rendering.apply_changes(base_recipe(), ClipChanges(start=11.2, end=30.1), transcript())
        self.assertAlmostEqual(new["start"], 10.9, places=2)      # 11.0 word start minus half the 0.2 s gap
        with self.assertRaises(rendering.RestyleError) as cm:
            rendering.apply_changes(base_recipe(), ClipChanges(start=20, end=21), transcript())
        self.assertEqual(cm.exception.status, 400)
        # A long clip is the user's call (the cap is 15 minutes), not an error.
        long_clip = rendering.apply_changes(base_recipe(), ClipChanges(start=0, end=99), transcript())
        self.assertGreater(long_clip["end"] - long_clip["start"], 90)

    def test_end_is_capped_at_the_video_length(self):
        new = rendering.apply_changes(base_recipe(), ClipChanges(start=90, end=500), transcript())
        self.assertLessEqual(new["end"], 100.0)

    def test_unknown_music_is_a_400_and_blank_clears(self):
        with mock.patch.object(rendering.music, "resolve_track", side_effect=BadInputError("no")):
            with self.assertRaises(rendering.RestyleError) as cm:
                rendering.apply_changes(base_recipe(), ClipChanges(music_track="x.mp3"), transcript())
        self.assertEqual(cm.exception.status, 400)
        new = rendering.apply_changes(base_recipe(music_track="a.mp3"), ClipChanges(music_track=""), transcript())
        self.assertIsNone(new["music_track"])

    def test_bad_overlays_are_a_400(self):
        with self.assertRaises(rendering.RestyleError) as cm:
            rendering.apply_changes(base_recipe(), ClipChanges(overlays=[{"type": "nope"}]), transcript())
        self.assertEqual(cm.exception.status, 400)

    def test_values_are_range_checked_by_the_model(self):
        with self.assertRaises(ValueError):
            ClipChanges(music_volume=500)


class RestyleFlowTests(unittest.TestCase):
    CID = "c" * 32

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.clips_root = root / "clips"
        (self.clips_root / self.CID).mkdir(parents=True)
        self.mp4 = self.clips_root / self.CID / "0.mp4"
        self.mp4.write_bytes(b"old")
        self.hist = root / "history.json"
        clip = {"index": 0, "title": "T", "start": 10.0, "end": 30.0,
                "url": f"/clips/{self.CID}/0.mp4", "recipe": base_recipe()}
        self.hist.write_text(json.dumps([{
            "clip_id": self.CID, "clips_dir": str(self.clips_root), "clips": [clip]}]))
        self.rendered = []

        def fake_render(recipe, source, tr, clip_id, index, clip_dir, work_dir, out_name=None, status=None):
            self.rendered.append((recipe, out_name))
            (clip_dir / out_name).write_bytes(b"new")
            return None

        self.fake_render = fake_render
        for p in (
            mock.patch.object(history, "HISTORY_FILE", self.hist),
            mock.patch.object(rendering.uploads, "resolve_upload", return_value=root / "src.mp4"),
            mock.patch.object(rendering.pretranscribe, "load", return_value=transcript()),
            mock.patch.object(rendering, "render_clip", fake_render),
            mock.patch.object(rendering, "WORK_DIR", root / "work"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_success_replaces_the_file_and_updates_history(self):
        clip = rendering.restyle_clip(self.CID, 0, ClipChanges(caption_style="minimal"))
        self.assertEqual(self.mp4.read_bytes(), b"new")
        self.assertEqual(clip["version"], 1)
        self.assertTrue(clip["url"].endswith("0.mp4?v=1"))
        self.assertEqual(clip["recipe"]["caption_style"], "minimal")
        _, saved = history.get_clip(self.CID, 0)
        self.assertEqual(saved["recipe"]["caption_style"], "minimal")
        self.assertEqual(list((self.clips_root / self.CID).glob("*.edit-*")), [])   # temp file gone
        again = rendering.restyle_clip(self.CID, 0, ClipChanges(fit_mode="blur"))
        self.assertEqual(again["version"], 2)
        self.assertEqual(again["recipe"]["caption_style"], "minimal")                # earlier edit kept

    def test_a_failed_render_leaves_the_clip_alone(self):
        def boom(*a, **k):
            raise ClipGenerationError("ffmpeg exploded")

        with mock.patch.object(rendering, "render_clip", boom):
            with self.assertRaises(rendering.RestyleError) as cm:
                rendering.restyle_clip(self.CID, 0, ClipChanges(caption_style="minimal"))
        self.assertEqual(cm.exception.status, 500)
        self.assertEqual(self.mp4.read_bytes(), b"old")
        _, saved = history.get_clip(self.CID, 0)
        self.assertEqual(saved["recipe"]["caption_style"], "bold_white")
        # and the clip is not stuck "busy"
        rendering.restyle_clip(self.CID, 0, ClipChanges(caption_style="minimal"))

    def test_clips_without_a_recipe_get_a_clear_409(self):
        data = json.loads(self.hist.read_text())
        del data[0]["clips"][0]["recipe"]
        self.hist.write_text(json.dumps(data))
        with self.assertRaises(rendering.RestyleError) as cm:
            rendering.restyle_clip(self.CID, 0, ClipChanges())
        self.assertEqual(cm.exception.status, 409)

    def test_missing_clip_source_and_transcript(self):
        with self.assertRaises(rendering.RestyleError) as cm:
            rendering.restyle_clip(self.CID, 5, ClipChanges())
        self.assertEqual(cm.exception.status, 404)
        with mock.patch.object(rendering.uploads, "resolve_upload", side_effect=BadInputError("gone")):
            with self.assertRaises(rendering.RestyleError) as cm:
                rendering.restyle_clip(self.CID, 0, ClipChanges())
        self.assertEqual(cm.exception.status, 409)
        with mock.patch.object(rendering.pretranscribe, "load", return_value=None):
            with self.assertRaises(rendering.RestyleError) as cm:
                rendering.restyle_clip(self.CID, 0, ClipChanges())
        self.assertEqual(cm.exception.status, 409)

    def test_a_second_edit_while_one_is_rendering_is_refused(self):
        def reenter(*a, **k):
            with self.assertRaises(rendering.RestyleError) as cm:
                rendering.restyle_clip(self.CID, 0, ClipChanges())
            self.assertEqual(cm.exception.status, 409)
            return self.fake_render(*a, **k)

        with mock.patch.object(rendering, "render_clip", reenter):
            rendering.restyle_clip(self.CID, 0, ClipChanges(caption_style="minimal"))

    def test_clip_file_route_serves_from_the_generation_folder(self):
        """Regression: /clips/<id>/<n>.mp4 looked in the clips root and then hit an
        undefined WORK_DIR, so no clip could be previewed or downloaded."""
        from fastapi.testclient import TestClient
        from app import main

        client = TestClient(main.app)
        res = client.get(f"/clips/{self.CID}/0.mp4")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"old")
        self.assertEqual(client.get(f"/clips/{self.CID}/9.mp4").status_code, 404)


class RestyleEndpointTests(unittest.TestCase):
    def test_errors_map_to_http_status(self):
        from fastapi.testclient import TestClient
        from app import main

        client = TestClient(main.app)
        body = {"clip_id": "d" * 32, "index": 0, "changes": {}}
        with mock.patch.object(rendering, "restyle_clip",
                               side_effect=rendering.RestyleError(409, "no source")):
            res = client.post("/api/clip/restyle", json=body)
        self.assertEqual((res.status_code, res.json()["detail"]), (409, "no source"))
        res = client.post("/api/clip/restyle", json={**body, "clip_id": "../x"})
        self.assertEqual(res.status_code, 400)
        res = client.post("/api/clip/restyle", json={**body, "changes": {"music_volume": 900}})
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
