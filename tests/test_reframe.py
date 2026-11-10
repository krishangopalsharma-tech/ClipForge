"""Face-aware reframing: detection decoding, subject and layout decisions, the camera
path, the ffmpeg graphs, and the planner's fallbacks. No model or video needed."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import clipper, reframe as rf  # noqa: E402
from app.models import AspectRatio, FitMode, GenerateRequest  # noqa: E402


def face(cx, height=0.2, width=0.1, cy=0.4, score=0.95):
    return rf.Face(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2, score)


def samples(*per_sample, step=0.2):
    return [rf.Sample(i * step, list(faces)) for i, faces in enumerate(per_sample)]


class DecodeTests(unittest.TestCase):
    def test_threshold_and_overlap_suppression(self):
        scores = np.array([[0.1, 0.95], [0.2, 0.90], [0.9, 0.10], [0.1, 0.80]], dtype=np.float32)
        boxes = np.array([
            [0.10, 0.10, 0.30, 0.40],   # face
            [0.11, 0.11, 0.31, 0.41],   # the same face again -> suppressed
            [0.50, 0.50, 0.60, 0.60],   # background, low face score
            [0.60, 0.10, 0.80, 0.40],   # a second, separate face
        ], dtype=np.float32)
        faces = rf.decode_detections(scores, boxes)
        self.assertEqual(len(faces), 2)
        self.assertAlmostEqual(faces[0].score, 0.95, places=2)

    def test_no_confident_box_means_no_faces(self):
        scores = np.zeros((5, 2), dtype=np.float32)
        self.assertEqual(rf.decode_detections(scores, np.zeros((5, 4), dtype=np.float32)), [])

    def test_boxes_are_clamped_into_the_frame(self):
        scores = np.array([[0.0, 0.99]], dtype=np.float32)
        boxes = np.array([[-0.2, 0.1, 0.3, 1.4]], dtype=np.float32)
        f = rf.decode_detections(scores, boxes)[0]
        self.assertEqual((f.x1, f.y2), (0.0, 1.0))


class SubjectTests(unittest.TestCase):
    def test_largest_face_wins(self):
        big, small = face(0.3, height=0.3, width=0.15), face(0.7, height=0.1, width=0.05)
        self.assertIs(rf.pick_subject([small, big]), big)

    def test_background_faces_are_ignored(self):
        self.assertIsNone(rf.pick_subject([face(0.5, height=0.02, width=0.01)]))
        self.assertIsNone(rf.pick_subject([]))

    def test_sticks_to_the_person_already_followed(self):
        a = face(0.30, height=0.20, width=0.10)
        b = face(0.70, height=0.22, width=0.11)   # slightly larger, far away
        self.assertIs(rf.pick_subject([a, b], previous_cx=0.31), a)

    def test_leaves_a_person_who_became_much_smaller(self):
        a = face(0.30, height=0.08, width=0.04)
        b = face(0.70, height=0.30, width=0.15)
        self.assertIs(rf.pick_subject([a, b], previous_cx=0.30), b)


class DecisionTests(unittest.TestCase):
    def test_one_person_is_tracked(self):
        d = rf.decide(samples(*[[face(0.5)]] * 10))
        self.assertEqual(d.mode, "track")
        self.assertEqual(d.face_fraction, 1.0)

    def test_no_face_is_blur(self):
        d = rf.decide(samples(*[[]] * 10))
        self.assertEqual((d.mode, d.reason), ("blur", "no face on screen"))

    def test_a_few_stray_detections_are_not_a_subject(self):
        d = rf.decide(samples([face(0.5)], [], [], [], [], [], [], [], [], []))
        self.assertEqual(d.mode, "blur")

    def test_a_group_is_blur(self):
        two = [face(0.3), face(0.7)]
        d = rf.decide(samples(*[two] * 10))
        self.assertEqual((d.mode, d.reason), ("blur", "several people on screen"))

    def test_a_small_second_face_does_not_make_a_group(self):
        d = rf.decide(samples(*[[face(0.4, height=0.3, width=0.15), face(0.8, height=0.06, width=0.03)]] * 10))
        self.assertEqual(d.mode, "track")

    def test_two_people_taking_turns_is_still_tracked(self):
        turns = [[face(0.25)] if i % 4 < 2 else [face(0.75)] for i in range(12)]
        self.assertEqual(rf.decide(samples(*turns)).mode, "track")

    def test_no_frames_is_blur(self):
        self.assertEqual(rf.decide([]).mode, "blur")


class CameraTests(unittest.TestCase):
    CROP = 0.316   # a 9:16 window on a 16:9 frame

    def test_stays_inside_the_frame(self):
        keys = rf.camera_keyframes(samples(*[[face(0.02)]] * 6 + [[face(0.99)]] * 6), self.CROP)
        for k in keys:
            self.assertGreaterEqual(k.left, 0.0)
            self.assertLessEqual(k.left + self.CROP, 1.0 + 1e-9)

    def test_holds_still_while_the_subject_stays_in_the_dead_zone(self):
        wobble = [[face(0.50 + (0.005 if i % 2 else -0.005))] for i in range(12)]
        lefts = {round(k.left, 4) for k in rf.camera_keyframes(samples(*wobble), self.CROP)}
        self.assertEqual(len(lefts), 1)

    def test_follows_a_moving_subject_and_catches_up(self):
        walk = [[face(0.35 + 0.03 * i)] for i in range(12)]
        keys = rf.camera_keyframes(samples(*walk), self.CROP)
        lefts = [k.left for k in keys]
        self.assertEqual(lefts, sorted(lefts))               # only ever moves right
        self.assertGreater(lefts[-1], lefts[0] + 0.1)
        centre = keys[-1].left + self.CROP / 2
        self.assertLess(abs(centre - walk[-1][0].cx), 0.12)  # roughly centred on them

    def test_speed_is_capped(self):
        jump = [[face(0.30)]] * 3 + [[face(0.45)]] * 3        # a quick but not cut-sized move
        keys = rf.camera_keyframes(samples(*jump), self.CROP)
        for a, b in zip(keys, keys[1:]):
            self.assertLessEqual(abs(b.left - a.left), rf.MAX_SPEED * self.CROP * 0.2 + 1e-9)

    def test_a_big_jump_is_a_cut_and_snaps(self):
        keys = rf.camera_keyframes(samples(*[[face(0.2)]] * 4 + [[face(0.8)]] * 4), self.CROP)
        self.assertTrue(keys[4].cut)
        self.assertAlmostEqual(keys[4].left + self.CROP / 2, 0.8, places=2)

    def test_holds_position_when_the_face_is_lost(self):
        keys = rf.camera_keyframes(samples([face(0.3)], [face(0.3)], [], [], []), self.CROP)
        self.assertEqual(len({round(k.left, 4) for k in keys}), 1)

    def test_before_the_first_sighting_uses_the_first_one(self):
        keys = rf.camera_keyframes(samples([], [], [face(0.7)], [face(0.7)]), self.CROP)
        self.assertAlmostEqual(keys[0].left, keys[-1].left, places=3)

    def test_nobody_at_all_is_the_middle(self):
        keys = rf.camera_keyframes(samples([], [], []), self.CROP)
        self.assertAlmostEqual(keys[0].left + self.CROP / 2, 0.5, places=3)


class PathTests(unittest.TestCase):
    def test_interpolates_between_keys(self):
        path = rf.path_at_fps([rf.Key(0.0, 0.0), rf.Key(1.0, 0.5)], 1.0, fps=4)
        self.assertEqual([round(left, 3) for _, left in path], [0.0, 0.125, 0.25, 0.375, 0.5])

    def test_a_cut_is_a_step_not_a_pan(self):
        path = rf.path_at_fps([rf.Key(0.0, 0.1), rf.Key(1.0, 0.6, cut=True)], 1.0, fps=4)
        lefts = [round(left, 3) for _, left in path]
        self.assertEqual(lefts[:4], [0.1, 0.1, 0.1, 0.1])
        self.assertEqual(lefts[4], 0.6)

    def test_empty(self):
        self.assertEqual(rf.path_at_fps([], 5.0), [])

    def test_sendcmd_only_writes_changes_and_clamps(self):
        path = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.1), (0.3, 5.0), (0.4, -1.0)]
        lines = rf.sendcmd_script(path, scaled_w=3000, crop_w=1000).splitlines()
        self.assertEqual(lines, [
            "0.000 crop x 0;", "0.200 crop x 300;", "0.300 crop x 2000;", "0.400 crop x 0;"])


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.work = Path.cwd() / "data" / "work" / "t"
        self.opts = clipper.ClipOptions(
            aspect_ratio=AspectRatio.NINE_16, fit_mode=FitMode.AUTO,
            ass_path=self.work / "0.ass", clip_id="c", index=0)

    def test_no_plan_is_the_plain_cover_crop(self):
        g = clipper._build_crop_filter_complex(1080, 1920, self.opts, self.work)
        self.assertIn("force_original_aspect_ratio=increase,crop=1080:1920[v0]", g)
        self.assertTrue(g.endswith("[outv]"))

    def test_track_uses_a_relative_sendcmd_script(self):
        plan = rf.ReframePlan("track", "x", scaled_w=3414, x0=700, commands_path=self.work / "0.cmd")
        self.opts.reframe = plan
        g = clipper._build_crop_filter_complex(1080, 1920, self.opts, self.work)
        self.assertIn("scale=3414:1920,sendcmd=f=0.cmd,crop=1080:1920:700:0", g)
        self.assertNotIn(":\\", g)                            # no Windows drive path in the graph
        self.assertTrue(g.endswith("[outv]"))

    def test_blur_layout(self):
        self.opts.reframe = rf.ReframePlan("blur", "x")
        g = clipper._build_crop_filter_complex(1080, 1920, self.opts, self.work)
        self.assertIn("boxblur", g)
        self.assertIn("[fgs]scale=1080:-2[fg]", g)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", g)
        self.assertTrue(g.endswith("[outv]"))

    def test_a_center_plan_changes_nothing(self):
        self.opts.reframe = rf.ReframePlan("center", "already vertical")
        plain = clipper._build_crop_filter_complex(
            1080, 1920, clipper.ClipOptions(
                aspect_ratio=AspectRatio.NINE_16, fit_mode=FitMode.CROP,
                ass_path=self.work / "0.ass", clip_id="c", index=0), self.work)
        self.assertEqual(clipper._build_crop_filter_complex(1080, 1920, self.opts, self.work), plain)


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return list(self.faces)


class PlannerTests(unittest.TestCase):
    def plan(self, requested="auto", size=(1920, 1080), target=(1080, 1920), detector="none", frames=20):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patches = [
            mock.patch.object(rf, "probe_size", return_value=size),
            mock.patch.object(
                rf, "sample_frames",
                side_effect=lambda *a, **k: iter([np.zeros((240, 320, 3), np.uint8)] * frames)),
        ]
        if detector == "none":
            patches.append(mock.patch.object(rf, "get_detector", return_value=None))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        det = None if detector == "none" else detector
        return rf.plan_reframe(Path("v.mp4"), 0.0, 4.0, *target, requested, Path(tmp.name), "0", det), Path(tmp.name)

    def test_vertical_output_only(self):
        plan, _ = self.plan(target=(1920, 1080), detector=FakeDetector([face(0.5)]))
        self.assertEqual((plan.mode, plan.reason), ("center", "output is not vertical"))

    def test_a_vertical_source_is_left_alone(self):
        plan, _ = self.plan(size=(1080, 1920), detector=FakeDetector([face(0.5)]))
        self.assertEqual((plan.mode, plan.reason), ("center", "source is already vertical"))

    def test_blur_needs_no_detector(self):
        plan, _ = self.plan("blur")
        self.assertEqual(plan.mode, "blur")

    def test_missing_model_degrades_visibly(self):
        auto, _ = self.plan("auto")
        self.assertEqual(auto.mode, "blur")
        self.assertIn("face model not installed", auto.reason)
        track, _ = self.plan("track")
        self.assertEqual(track.mode, "center")
        self.assertIn("face model not installed", track.reason)

    def test_one_face_is_tracked_and_the_script_is_written(self):
        plan, tmp = self.plan("auto", detector=FakeDetector([face(0.7)]))
        self.assertEqual(plan.mode, "track")
        self.assertEqual(plan.scaled_w, 3414)
        self.assertTrue((tmp / "0.cmd").is_file())
        self.assertGreater(plan.x0, 0)
        self.assertEqual(plan.summary()["mode"], "track")

    def test_no_face_in_auto_is_blur_but_in_track_is_center(self):
        auto, _ = self.plan("auto", detector=FakeDetector([]))
        self.assertEqual(auto.mode, "blur")
        track, _ = self.plan("track", detector=FakeDetector([]))
        self.assertEqual((track.mode, track.reason), ("center", "no face found to follow"))

    def test_a_group_in_auto_is_blur_but_track_still_follows(self):
        two = FakeDetector([face(0.3), face(0.7)])
        auto, _ = self.plan("auto", detector=two)
        self.assertEqual(auto.mode, "blur")
        track, _ = self.plan("track", detector=two)
        self.assertEqual(track.mode, "track")

    def test_any_failure_is_a_center_crop_not_an_error(self):
        with mock.patch.object(rf, "probe_size", side_effect=RuntimeError("boom")):
            plan = rf.plan_reframe(Path("v.mp4"), 0, 4, 1080, 1920, "auto", Path("."), "0")
        self.assertEqual(plan.mode, "center")
        self.assertIn("RuntimeError", plan.reason)


class RequestTests(unittest.TestCase):
    def test_new_fit_modes_are_accepted(self):
        for mode in ("auto", "track", "blur", "crop", "square"):
            req = GenerateRequest(video_url="http://x", fit_mode=mode)
            self.assertEqual(req.fit_mode.value, mode)

    def test_default_is_unchanged(self):
        self.assertEqual(GenerateRequest(video_url="http://x").fit_mode, FitMode.CROP)


if __name__ == "__main__":
    unittest.main()
