"""ZIP + tracking-CSV export of a generation."""
import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import export, history  # noqa: E402

CID = "a" * 32


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "clips"
        (self.root / CID).mkdir(parents=True)
        self.hist = Path(self.tmp.name) / "history.json"
        p = mock.patch.object(history, "HISTORY_FILE", self.hist)
        p.start()
        self.addCleanup(p.stop)

    def write_history(self, clips, **entry_kw):
        entry = {
            "clip_id": CID, "created": "2026-10-15T09:00:00", "source": "My Podcast.mp4",
            "source_type": "upload", "source_id": "b" * 32, "clips_dir": str(self.root),
            "settings": {"aspect_ratio": "9:16", "fit_mode": "crop", "caption_style": "minimal", "language": "en"},
            "clips": clips,
        }
        entry.update(entry_kw)
        self.hist.write_text(json.dumps([entry]), encoding="utf-8")
        for c in clips:
            if c.get("_write", True):
                (self.root / CID / f"{c['index']}.mp4").write_bytes(b"video-%d" % c["index"])

    def clip(self, i, **kw):
        base = {"index": i, "title": f"Title {i}", "start": 10.0 + i, "end": 40.0 + i,
                "filename": f"abcd  Title {i}.mp4", "language": "en"}
        base.update(kw)
        return base

    def export(self):
        items = export.collect(history.list_entries())
        dest = Path(self.tmp.name) / "out.zip"
        export.build_zip(items, dest)
        z = zipfile.ZipFile(dest)
        rows = list(csv.DictReader(io.StringIO(z.read("clips.csv").decode("utf-8-sig"))))
        return z, rows

    def test_zip_holds_every_clip_and_the_csv(self):
        self.write_history([self.clip(0), self.clip(1)])
        z, rows = self.export()
        self.assertEqual(sorted(z.namelist()), ["abcd  Title 0.mp4", "abcd  Title 1.mp4", "clips.csv"])
        self.assertEqual(z.read("abcd  Title 1.mp4"), b"video-1")
        self.assertEqual(len(rows), 2)

    def test_csv_rows_match_the_files_and_carry_the_copy(self):
        self.write_history([self.clip(0, hook="Big news", description="Teaser #a", tags="a, b",
                                      score=88, picked_by="gemini",
                                      reframe={"mode": "track"},
                                      recipe={"start": 12.5, "end": 40.0, "caption_style": "boxed_tiktok",
                                              "aspect_ratio": "9:16", "fit_mode": "auto"})])
        z, rows = self.export()
        row = rows[0]
        self.assertIn(row["filename"], z.namelist())
        self.assertEqual((row["hook"], row["description"], row["tags"], row["score"]),
                         ("Big news", "Teaser #a", "a, b", "88"))
        self.assertEqual((row["start_s"], row["end_s"], row["duration_s"]), ("12.5", "40", "27.5"))
        self.assertEqual((row["framing"], row["caption_style"], row["picked_by"]), ("track", "boxed_tiktok", "gemini"))
        self.assertEqual((row["source"], row["generated_at"], row["clip_id"], row["index"]),
                         ("My Podcast.mp4", "2026-10-15T09:00:00", CID, "0"))

    def test_tracking_columns_start_empty_with_a_status(self):
        self.write_history([self.clip(0)])
        _, rows = self.export()
        self.assertEqual([rows[0][k] for k in ("platform", "posted_at", "post_url")], ["", "", ""])
        self.assertEqual(rows[0]["status"], "not_posted")
        self.assertEqual(list(rows[0].keys()), export.CSV_COLUMNS)

    def test_spreadsheet_formulas_are_defused(self):
        evil = '=HYPERLINK("http://x","click")'
        self.write_history([self.clip(0, title=evil, description="+1 deal", tags="@everyone", hook="-5%")])
        _, rows = self.export()
        for key in ("title", "description", "tags", "hook"):
            self.assertTrue(rows[0][key].startswith("'"), key)
        self.assertEqual(rows[0]["title"], "'" + evil)

    def test_numbers_are_not_mangled_by_the_defusing(self):
        self.write_history([self.clip(0, score=0)])
        _, rows = self.export()
        self.assertEqual(rows[0]["score"], "0")

    def test_non_latin_text_survives_and_the_file_starts_with_a_bom(self):
        self.write_history([self.clip(0, title="कपिल शर्मा का शो", hook="मज़ाक")])
        items = export.collect(history.list_entries())
        raw = export.build_csv(items)
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn("कपिल शर्मा का शो", raw.decode("utf-8-sig"))

    def test_commas_quotes_and_newlines_stay_in_their_cell(self):
        self.write_history([self.clip(0, title='He said "no", then\nleft', description="a,b")])
        _, rows = self.export()
        self.assertEqual(rows[0]["title"], 'He said "no", then\nleft')
        self.assertEqual(rows[0]["description"], "a,b")
        self.assertEqual(len(rows), 1)

    def test_clips_whose_file_is_gone_are_left_out(self):
        self.write_history([self.clip(0), self.clip(1, _write=False)])
        z, rows = self.export()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len([n for n in z.namelist() if n.endswith(".mp4")]), 1)

    def test_hashtags_in_the_description_fill_an_empty_tags_column(self):
        self.write_history([
            self.clip(0, description="Fast bike. #SpeedRecord #BurtMunro #speedrecord #shorts"),
            self.clip(1, description="No tags here."),
        ])
        _z, rows = self.export()
        by_index = {r["index"]: r for r in rows}
        self.assertEqual(by_index["0"]["tags"], "SpeedRecord, BurtMunro, shorts")
        self.assertEqual(by_index["1"]["tags"], "")

    def test_duplicate_and_unsafe_names_are_made_safe(self):
        self.write_history([
            self.clip(0, filename="same.mp4"), self.clip(1, filename="same.mp4"),
            self.clip(2, filename='../../evil:name?.mp4'), self.clip(3, filename=""),
            self.clip(4, filename="noext"),
        ])
        z, rows = self.export()
        names = [n for n in z.namelist() if n != "clips.csv"]
        self.assertEqual(len(set(n.lower() for n in names)), 5)
        for n in names:
            self.assertNotIn("/", n)
            self.assertNotIn("\\", n)
            self.assertNotIn(":", n)
            self.assertTrue(n.endswith(".mp4"))
        self.assertIn("same.mp4", names)
        self.assertIn("same (2).mp4", names)
        self.assertEqual({r["filename"] for r in rows}, set(names))

    def test_old_clips_without_a_recipe_or_copy_still_export(self):
        self.write_history([{"index": 0, "title": "Old", "start": 1.0, "end": 31.0}])
        _, rows = self.export()
        self.assertEqual((rows[0]["title"], rows[0]["duration_s"], rows[0]["hook"]), ("Old", "30", ""))
        self.assertTrue(rows[0]["filename"].endswith(".mp4"))

    def test_stored_not_deflated(self):
        self.write_history([self.clip(0)])
        z, _ = self.export()
        self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))


class ExportApiTests(ExportTests):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from app import main

        self.client = TestClient(main.app)

    def test_zip_endpoint(self):
        self.write_history([self.clip(0), self.clip(1)])
        res = self.client.get(f"/api/export/{CID}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/zip")
        self.assertIn("clipforge-2026-10-15-aaaaaaaa.zip", res.headers["content-disposition"])
        z = zipfile.ZipFile(io.BytesIO(res.content))
        self.assertIn("clips.csv", z.namelist())
        self.assertEqual(len(z.namelist()), 3)

    def test_temp_zip_is_removed_after_sending(self):
        self.write_history([self.clip(0)])
        made = []
        real = tempfile.mkstemp

        def spy(*a, **k):
            fd, path = real(*a, **k)
            made.append(path)
            return fd, path

        with mock.patch("app.main.tempfile.mkstemp", spy):
            self.client.get(f"/api/export/{CID}")
        self.assertEqual(len(made), 1)
        self.assertFalse(Path(made[0]).exists())

    def test_csv_endpoints(self):
        self.write_history([self.clip(0)])
        one = self.client.get(f"/api/export/{CID}/clips.csv")
        self.assertEqual(one.status_code, 200)
        self.assertTrue(one.headers["content-type"].startswith("text/csv"))
        self.assertTrue(one.content.startswith(b"\xef\xbb\xbf"))
        allrows = self.client.get("/api/export.csv")
        self.assertEqual(allrows.status_code, 200)
        self.assertIn(b"Title 0", allrows.content)

    def test_errors(self):
        self.write_history([self.clip(0)])
        self.assertEqual(self.client.get("/api/export/not-an-id").status_code, 400)
        self.assertEqual(self.client.get("/api/export/" + "f" * 32).status_code, 404)
        (self.root / CID / "0.mp4").unlink()
        res = self.client.get(f"/api/export/{CID}")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
