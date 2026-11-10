"""Background transcription: one run per video, and no silent waiting."""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import pretranscribe  # noqa: E402


class _NoThread:
    """Stands in for threading.Thread so start() records a job without running it."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class StartTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(pretranscribe, "_JOBS", {})
        p.start()
        self.addCleanup(p.stop)
        t = mock.patch.object(pretranscribe.threading, "Thread", _NoThread)
        t.start()
        self.addCleanup(t.stop)

    def test_asking_again_while_running_returns_the_same_job(self):
        first = pretranscribe.start("a" * 32, "auto")
        again = pretranscribe.start("a" * 32, "cuda")
        self.assertIs(first, again)
        self.assertEqual(len(pretranscribe._JOBS), 1)

    def test_another_video_or_language_gets_its_own_job(self):
        a = pretranscribe.start("a" * 32, "auto")
        b = pretranscribe.start("b" * 32, "auto")
        hi = pretranscribe.start("a" * 32, "auto", "hi")
        self.assertEqual(len({a.id, b.id, hi.id}), 3)

    def test_a_finished_job_is_not_reused(self):
        first = pretranscribe.start("a" * 32, "auto")
        first.update(status="done")
        self.assertIsNot(pretranscribe.start("a" * 32, "auto"), first)


class WaitingTests(unittest.TestCase):
    def test_a_queued_run_says_it_is_waiting(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patches = [
            mock.patch.object(pretranscribe, "TRANSCRIPTS_DIR", Path(tmp.name)),
            mock.patch.object(pretranscribe, "_CACHE", {}),
            mock.patch.object(pretranscribe.transcriber, "transcribe_video",
                              return_value={"segments": [], "words": []}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        messages = []
        pretranscribe._TRANSCRIBE_LOCK.acquire()     # another video is mid-run
        try:
            worker = threading.Thread(target=pretranscribe.get_or_transcribe, args=(
                Path(tmp.name) / "v.mp4", "c" * 32, "auto"),
                kwargs={"progress": lambda f, m: messages.append(m)})
            worker.start()
            for _ in range(50):
                if messages:
                    break
                time.sleep(0.02)
        finally:
            pretranscribe._TRANSCRIBE_LOCK.release()
        worker.join(5)
        self.assertTrue(messages and messages[0].startswith("Waiting for another transcription"))


if __name__ == "__main__":
    unittest.main()
