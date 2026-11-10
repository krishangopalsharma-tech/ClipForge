"""Model choice per language, sparse-speech detection, and the model cache."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import speech  # noqa: E402


def transcript(words, minutes, **extra):
    return {
        "duration": minutes * 60.0,
        "words": [{"word": "w", "start": i, "end": i + 0.1} for i in range(words)],
        "segments": [],
        **extra,
    }


class ModelChoiceTests(unittest.TestCase):
    def test_hindustani_gets_the_larger_model(self):
        with mock.patch.object(speech, "DEFAULT_MODEL", "medium"), \
                mock.patch.object(speech, "HINDUSTANI_MODEL", "large-v3"):
            for lang in ("hi", "ur", "hinglish", "HI", " hinglish "):
                self.assertEqual(speech.model_for(lang), "large-v3", lang)

    def test_everything_else_keeps_the_default(self):
        with mock.patch.object(speech, "DEFAULT_MODEL", "medium"):
            for lang in ("en", "es", "ar", "", None, "auto"):
                self.assertEqual(speech.model_for(lang), "medium", lang)

    def test_a_default_that_is_already_large_is_not_replaced(self):
        for default in ("large-v3", "large-v2"):
            with mock.patch.object(speech, "DEFAULT_MODEL", default):
                self.assertEqual(speech.model_for("hi"), default)

    def test_turbo_and_distilled_defaults_are_replaced(self):
        for default in ("large-v3-turbo", "turbo", "distil-large-v3", "small"):
            with mock.patch.object(speech, "DEFAULT_MODEL", default), \
                    mock.patch.object(speech, "HINDUSTANI_MODEL", "large-v3"):
                self.assertEqual(speech.model_for("hi"), "large-v3", default)

    def test_a_path_is_left_alone(self):
        with mock.patch.object(speech, "DEFAULT_MODEL", "F:/models/my-whisper"):
            self.assertEqual(speech.model_for("hi"), "F:/models/my-whisper")


class SparseSpeechTests(unittest.TestCase):
    def test_normal_speech_is_not_sparse(self):
        r = speech.speech_report(transcript(1300, 10))
        self.assertFalse(r["sparse"])
        self.assertEqual(r["words"], 1300)

    def test_a_handful_of_words_is_sparse(self):
        self.assertTrue(speech.speech_is_sparse(transcript(5, 1)))

    def test_few_words_over_a_long_video_is_sparse(self):
        # 30 words in 20 minutes: more than the absolute floor, far below 5/min.
        self.assertTrue(speech.speech_is_sparse(transcript(30, 20)))

    def test_boundaries(self):
        self.assertTrue(speech.speech_is_sparse(transcript(7, 1)))
        self.assertFalse(speech.speech_is_sparse(transcript(8, 1)))
        self.assertTrue(speech.speech_is_sparse(transcript(99, 20)))   # 4.95/min
        self.assertFalse(speech.speech_is_sparse(transcript(100, 20)))  # 5.0/min

    def test_segment_text_is_used_when_there_is_no_word_list(self):
        t = {"duration": 60, "segments": [{"text": "one two three four five six seven eight nine"}]}
        r = speech.speech_report(t)
        self.assertEqual(r["words"], 9)
        self.assertFalse(r["sparse"])

    def test_empty_and_missing_transcripts(self):
        self.assertTrue(speech.speech_is_sparse({}))
        self.assertTrue(speech.speech_is_sparse(None))

    def test_duration_falls_back_to_the_last_segment_end(self):
        t = {"segments": [{"text": "a b c d e f g h i j", "end": 120}]}
        self.assertEqual(speech.speech_report(t)["minutes"], 2.0)


class ModelHintTests(unittest.TestCase):
    def test_hindi_decoded_by_medium_gets_a_tip(self):
        with mock.patch.object(speech, "HINDUSTANI_MODEL", "large-v3"):
            hint = speech.model_hint({"language": "hi", "whisper_model": "medium"})
        self.assertIn("large-v3", hint)

    def test_no_tip_when_the_right_model_was_used(self):
        self.assertIsNone(speech.model_hint({"language": "hi", "whisper_model": "large-v3"}))

    def test_no_tip_for_english(self):
        self.assertIsNone(speech.model_hint({"language": "en", "whisper_model": "medium"}))

    def test_no_tip_when_the_swap_is_switched_off(self):
        with mock.patch.object(speech, "HINDUSTANI_MODEL", "medium"):
            self.assertIsNone(speech.model_hint({"language": "ur", "whisper_model": "medium"}))

    def test_old_transcripts_without_a_recorded_model_still_get_one(self):
        with mock.patch.object(speech, "HINDUSTANI_MODEL", "large-v3"):
            self.assertIsNotNone(speech.model_hint({"language": "hi"}))


class FakeModel:
    """Stands in for faster_whisper.WhisperModel; records how it was built."""

    built = []

    def __init__(self, size, device, compute_type):
        self.size, self.device, self.compute_type = size, device, compute_type
        FakeModel.built.append((size, device))

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        seg = SimpleNamespace(
            id=0, start=0.0, end=2.0, text=" hello there",
            words=[SimpleNamespace(word=" hello", start=0.0, end=1.0),
                   SimpleNamespace(word=" there", start=1.0, end=2.0)],
        )
        return iter([seg]), SimpleNamespace(duration=2.0, language="en")


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        from app import transcriber

        self.t = transcriber
        FakeModel.built = []
        self.patches = [
            mock.patch.object(transcriber, "WhisperModel", FakeModel),
            mock.patch.object(transcriber, "_models", {}),
            mock.patch.object(transcriber, "_model_names", {}),
            mock.patch.object(transcriber, "cuda_available", lambda: False),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def test_same_size_is_reused(self):
        a = self.t.load_model("cpu", "medium")
        b = self.t.load_model("cpu", "medium")
        self.assertIs(a, b)
        self.assertEqual(FakeModel.built, [("medium", "cpu")])

    def test_another_size_replaces_the_cached_one(self):
        self.t.load_model("cpu", "medium")
        big = self.t.load_model("cpu", "large-v3")
        self.assertEqual(big.size, "large-v3")
        self.assertEqual(FakeModel.built, [("medium", "cpu"), ("large-v3", "cpu")])
        self.assertTrue(self.t.has_model("large-v3"))
        self.assertFalse(self.t.has_model("medium"))
        self.assertEqual(len(self.t._models), 1)

    def test_is_loaded_still_works_with_just_a_device(self):
        self.assertFalse(self.t.is_loaded("cpu"))
        self.t.load_model("cpu", "medium")
        self.assertTrue(self.t.is_loaded("cpu"))
        self.assertTrue(self.t.is_loaded("cpu", "medium"))
        self.assertFalse(self.t.is_loaded("cpu", "large-v3"))

    def test_transcribe_uses_the_language_model_and_records_it(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.t, "TRANSCRIPTS_DIR", Path(tmp)), \
                mock.patch.object(speech, "DEFAULT_MODEL", "medium"), \
                mock.patch.object(speech, "HINDUSTANI_MODEL", "large-v3"):
            hindi = self.t.transcribe_video(Path("x.mp4"), "k1", device="cpu", language="hi")
            english = self.t.transcribe_video(Path("x.mp4"), "k2", device="cpu", language="en")
        self.assertEqual(hindi["whisper_model"], "large-v3")
        self.assertEqual(english["whisper_model"], "medium")
        self.assertEqual([s for s, _ in FakeModel.built], ["large-v3", "medium"])


if __name__ == "__main__":
    unittest.main()
