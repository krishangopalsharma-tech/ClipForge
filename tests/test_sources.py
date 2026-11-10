"""Duplicate-source detection: the four signals, what must NOT match, and the API."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import sources  # noqa: E402

A, B, C = "a" * 32, "b" * 32, "c" * 32


def entry(source_id, source, source_type="url", created="2026-09-01T10:00:00", clips=3):
    return {"clip_id": source_id[::-1], "created": created, "source": source,
            "source_type": source_type, "source_id": source_id,
            "clips": [{"index": i} for i in range(clips)]}


def info_table(table):
    return lambda sid: table.get(sid, (None, None))


class YoutubeIdTests(unittest.TestCase):
    def test_every_common_link_shape(self):
        vid = "dQw4w9WgXcQ"
        for url in (
            f"https://www.youtube.com/watch?v={vid}", f"https://youtube.com/watch?v={vid}&t=5s",
            f"https://www.youtube.com/watch?feature=share&v={vid}", f"https://m.youtube.com/watch?v={vid}",
            f"https://youtu.be/{vid}", f"https://youtu.be/{vid}?si=abc", f"https://www.youtube.com/shorts/{vid}",
            f"https://www.youtube.com/embed/{vid}", f"https://www.youtube.com/live/{vid}?feature=share",
            f"https://music.youtube.com/watch?v={vid}",
        ):
            self.assertEqual(sources.youtube_id(url), vid, url)

    def test_not_youtube(self):
        for url in ("https://studio.local/2e90e8a3e027a0e44f02f68b485f07cb", "https://vimeo.com/123456789",
                    "https://www.youtube.com/channel/UC1234567890", "", None):
            self.assertIsNone(sources.youtube_id(url), url)


class TitleTests(unittest.TestCase):
    def test_case_punctuation_spacing_and_extension_are_ignored(self):
        self.assertEqual(sources.normalize_title("My  Video - Part_1!.mp4"), "my video part 1")
        self.assertEqual(sources.normalize_title("my video part 1.MOV"), "my video part 1")

    def test_quality_tags_are_kept(self):
        self.assertNotEqual(sources.normalize_title("Show (HD).mp4"), sources.normalize_title("Show.mp4"))
        self.assertNotEqual(sources.normalize_title("Show 1080p.mp4"), sources.normalize_title("Show 720p.mp4"))

    def test_non_latin_titles_survive(self):
        self.assertEqual(sources.normalize_title("कपिल शर्मा शो.mp4"), "कपिल शर्मा शो")

    def test_empty(self):
        self.assertEqual(sources.normalize_title(None), "")


class MatchTests(unittest.TestCase):
    def find(self, entries, **kw):
        kw.setdefault("info", info_table({}))
        return sources.find_duplicates(entries, **kw)

    def test_same_file(self):
        out = self.find([entry(A, "x.mp4", "upload")], source_id=A)
        self.assertEqual(out[0]["signals"], ["same_file"])
        self.assertEqual(out[0]["clip_count"], 3)

    def test_same_file_even_when_the_source_is_deleted(self):
        self.assertEqual(self.find([entry(A, "gone.mp4", "upload")], source_id=A)[0]["signals"], ["same_file"])

    def test_youtube_id_matches_across_link_shapes(self):
        e = entry(A, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(self.find([e], video_url="https://youtu.be/dQw4w9WgXcQ?si=z")[0]["signals"], ["youtube"])
        self.assertEqual(self.find([e], video_url="https://youtu.be/AAAAAAAAAAA"), [])

    def test_title_matches_uploads_only(self):
        up = entry(A, "Episode 12 Final.mp4", "upload")
        self.assertEqual(self.find([up], filename="episode-12_final.MP4")[0]["signals"], ["title"])
        url = entry(B, "episode 12 final", "url")
        self.assertEqual(self.find([url], filename="Episode 12 Final.mp4"), [])

    def test_a_quality_variant_is_not_a_title_match(self):
        self.assertEqual(self.find([entry(A, "Show.mp4", "upload")], filename="Show (HD).mp4"), [])

    def test_very_short_titles_never_match(self):
        self.assertEqual(self.find([entry(A, "v1.mp4", "upload")], filename="v1.mp4"), [])

    def test_size_and_duration(self):
        table = info_table({A: (1000, 60.0), B: (1000, 90.0), C: (999, 60.2)})
        e = [entry(A, "old name.mp4", "upload")]
        self.assertEqual(self.find(e, size=1000, duration=60.5, info=table)[0]["signals"], ["size_duration"])
        self.assertEqual(self.find(e, size=1000, duration=75.0, info=table), [])   # same size, other length
        self.assertEqual(self.find([entry(C, "x.mp4", "upload")], size=1000, duration=60.0, info=table), [])

    def test_size_without_a_known_duration_is_not_enough(self):
        table = info_table({A: (1000, 60.0)})
        self.assertEqual(self.find([entry(A, "x.mp4", "upload")], size=1000, info=table), [])

    def test_query_facts_come_from_the_source_on_disk(self):
        table = info_table({A: (500, 30.0), B: (500, 30.4)})
        out = self.find([entry(A, "old.mp4", "upload")], source_id=B, info=table)
        self.assertEqual(out[0]["signals"], ["size_duration"])

    def test_several_signals_are_all_reported(self):
        e = entry(A, "Episode 12 Final.mp4", "upload")
        out = self.find([e], source_id=A, filename="Episode 12 Final.mp4")
        self.assertEqual(out[0]["signals"], ["same_file", "title"])

    def test_newest_first_and_capped(self):
        entries = [entry(A, "https://youtu.be/dQw4w9WgXcQ", created=f"2026-09-0{i}T10:00:00") for i in range(1, 6)]
        out = self.find(entries, video_url="https://youtu.be/dQw4w9WgXcQ")
        self.assertEqual(len(out), sources.MAX_RESULTS)
        self.assertEqual(out[0]["created"], "2026-09-05T10:00:00")

    def test_nothing_known_means_nothing_found(self):
        self.assertEqual(self.find([entry(A, "x.mp4", "upload")]), [])
        self.assertEqual(self.find([]), [])


class CheckSafetyTests(unittest.TestCase):
    def test_check_never_raises(self):
        with mock.patch("app.history.list_entries", side_effect=RuntimeError("disk on fire")):
            self.assertEqual(sources.check(source_id=A), [])

    def test_file_info_for_a_bad_or_missing_id(self):
        self.assertEqual(sources.file_info("not-an-id"), (None, None))
        self.assertEqual(sources.file_info("f" * 32), (None, None))


class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app import main

        self.client = TestClient(main.app)
        self.entries = [entry(A, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", clips=7)]
        p = mock.patch("app.history.list_entries", return_value=self.entries)
        p.start()
        self.addCleanup(p.stop)

    def test_check_endpoint(self):
        res = self.client.post("/api/source/check", json={"video_url": "https://youtu.be/dQw4w9WgXcQ"})
        self.assertEqual(res.status_code, 200)
        dup = res.json()["duplicates"][0]
        self.assertEqual((dup["signals"], dup["clip_count"]), (["youtube"], 7))
        self.assertEqual(self.client.post("/api/source/check", json={}).json(), {"duplicates": []})

    def test_check_rejects_nonsense(self):
        self.assertEqual(self.client.post("/api/source/check", json={"size": -5}).status_code, 422)

    def test_generate_reports_duplicates_but_still_starts_the_job(self):
        from app import jobs

        with mock.patch.object(jobs, "start_job") as start:
            res = self.client.post("/api/generate", json={"video_url": "https://youtu.be/dQw4w9WgXcQ"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["job_id"])
        self.assertEqual(body["duplicates"][0]["signals"], ["youtube"])
        start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
