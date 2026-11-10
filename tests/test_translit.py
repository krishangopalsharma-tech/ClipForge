# -*- coding: utf-8 -*-
"""Hinglish romanisation: Devanagari speech written in Latin letters."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import translit  # noqa: E402


def test_common_words_read_the_way_people_type_them():
    cases = {
        "आप": "aap",            # long aa doubles inside a word
        "कैसे": "kaise",
        "हैं": "hain",          # anusvara -> n
        "क्या": "kya",          # final long aa does NOT double
        "नहीं": "nahin",
        "बहुत": "bahut",
        "दिल": "dil",
        "प्यार": "pyaar",
        "यार": "yaar",
        "सच": "sach",
        "बोलो": "bolo",
    }
    for source, expected in cases.items():
        assert translit.to_roman(source) == expected


def test_trailing_schwa_is_deleted():
    # The #1 thing that makes naive transliteration look wrong: "ghara".
    assert translit.to_roman("घर") == "ghar"
    assert translit.to_roman("दोस्त") == "dost"
    assert translit.to_roman("कपिल") == "kapil"


def test_medial_schwa_deletion_and_its_two_blockers():
    assert translit.to_roman("लड़की") == "ladki"      # VC_CV -> dropped
    assert translit.to_roman("मतलब") == "matlab"
    assert translit.to_roman("नमकीन") == "namkin"
    # No vowel after it (the final schwa went first), so it stays.
    assert translit.to_roman("समझ") == "samajh"
    # Closed syllable before it: zindgi is not a word anyone types.
    assert translit.to_roman("ज़िंदगी") == "zindagi"


def test_nukta_forms_are_the_urdu_sounds():
    assert translit.to_roman("मज़ाक") == "mazaak"
    assert translit.to_roman("बड़ा") == "bada"
    assert translit.to_roman("पढ़ाई") == "padhaai"
    # Decomposed nukta (क + ़) must romanise like the precomposed codepoint.
    assert translit.to_roman("क़") == translit.to_roman("क़")


def test_non_devanagari_text_is_returned_unchanged():
    assert translit.to_roman("OpenShorts 2026!") == "OpenShorts 2026!"
    assert translit.to_roman("") == ""
    assert translit.to_roman(None) is None


def test_leading_space_word_boundary_survives():
    # The whole pipeline marks a word start with a leading space; losing it
    # glues caption words together.
    assert translit.to_roman(" आप") == " aap"
    assert translit.to_roman(" आप कैसे हैं") == " aap kaise hain"


def test_romanize_transcript_keeps_timings_and_retags_the_language():
    transcript = {
        "language": "hi",
        "text": "आप कैसे हैं",
        "segments": [{
            "start": 1.0, "end": 2.5, "text": "आप कैसे हैं",
            "words": [
                {"word": " आप", "start": 1.0, "end": 1.4},
                {"word": " कैसे", "start": 1.4, "end": 1.9},
                {"word": " हैं", "start": 1.9, "end": 2.5},
            ],
        }],
    }
    out = translit.romanize_transcript(transcript)

    assert out["language"] == "hinglish"
    assert out["text"] == "aap kaise hain"
    assert out["segments"][0]["text"] == "aap kaise hain"
    assert [w["word"] for w in out["segments"][0]["words"]] == [
        " aap", " kaise", " hain"]
    assert [(w["start"], w["end"]) for w in out["segments"][0]["words"]] == [
        (1.0, 1.4), (1.4, 1.9), (1.9, 2.5)]
    # The input is not mutated — callers keep the native-script transcript.
    assert transcript["segments"][0]["text"] == "आप कैसे हैं"
    assert transcript["language"] == "hi"


def test_romanize_transcript_handles_a_flat_word_list():
    out = translit.romanize_transcript({
        "language": "hi", "text": "घर",
        "words": [{"word": " घर", "start": 0.0, "end": 0.5}],
        "segments": [],
    })
    assert out["words"] == [{"word": " ghar", "start": 0.0, "end": 0.5}]


def test_attached_punctuation_does_not_defeat_the_last_unit_rules():
    """A comma used to become the last "unit", so the trailing-schwa deletion
    and the aa/a choice — both of which look at the last unit — misfired:
    "कपिल," romanised as "kapila," and "क्या," as "kyaa,". Caption text is full
    of sentence-final words, so this was visible on almost every line."""
    assert translit.to_roman("कपिल") == "kapil"
    for token, expected in {
        "कपिल,": "kapil,",
        "कपिल।": "kapil.",   # danda still becomes a full stop
        "वह?": "vah?",
        "क्या,": "kya,",     # long final aa still does not double
        "अच्छा!": "achcha!",
        "शर्मा।": "sharma.",
        "हुए?": "hue?",
        '"घर"': '"ghar"',
    }.items():
        assert translit.to_roman(token) == expected


def test_punctuation_only_and_numeric_tokens_survive():
    assert translit.to_roman("।") == "."
    assert translit.to_roman("१२३") == "123"
    assert translit.to_roman("...") == "..."


def test_a_single_syllable_long_aa_does_not_double_either():
    """का/था/ना are among the most frequent words in the language and nobody
    types "kaa". The rule used to require more than one syllable."""
    for token, expected in {"का": "ka", "था": "tha", "ना": "na",
                            "जा": "ja", "या": "ya"}.items():
        assert translit.to_roman(token) == expected
    # But a bare vowel IS the word, and a nasalised ending keeps both letters.
    assert translit.to_roman("आ") == "aa"
    assert translit.to_roman("हाँ") == "haan"
    assert translit.to_roman("कहाँ") == "kahaan"
    # A final independent vowel after another syllable still shortens.
    assert translit.to_roman("हुआ") == "hua"
    assert translit.to_roman("दुआ") == "dua"
    # And nothing about the multi-syllable cases moved.
    assert translit.to_roman("शर्मा") == "sharma"
    assert translit.to_roman("आप") == "aap"
    assert translit.to_roman("प्यार") == "pyaar"


def test_english_loanwords_get_their_english_spelling_back():
    """Phonetic romanisation can only spell what it heard — नेल पेंट as
    "nel pent", हेलो as "helo", स्टाइल as "staail". Hinglish speakers type the
    English spelling, and these words are a large share of media speech."""
    for token, expected in {
        "हेलो": "hello", "नेल": "nail", "पेंट": "paint", "स्टाइल": "style",
        "मोबाइल": "mobile", "थैंक्यू": "thank you", "आइसक्रीम": "ice cream",
        "स्कूल": "school", "डॉक्टर": "doctor",
    }.items():
        assert translit.to_roman(token) == expected


def test_a_loanword_is_found_whichever_way_the_nukta_was_written():
    """Whisper writes both फ़ोन and फोन for the same word."""
    assert translit.to_roman("फ़ोन") == "phone"
    assert translit.to_roman("फोन") == "phone"
    assert translit.to_roman("कॉफ़ी") == "coffee"
    assert translit.to_roman("कॉफी") == "coffee"


def test_a_loanword_keeps_its_punctuation():
    assert translit.to_roman("हेलो,") == "hello,"
    assert translit.to_roman("सर।") == "sir."


def test_words_that_are_also_common_hindi_words_stay_phonetic():
    """A lookup cannot tell "cheese" from चीज़ ("thing"), and over one real
    episode every occurrence was the Hindi word — 4 of चीज़, 3 of बस, none of
    them cheese or a bus. Guessing English there would be a downgrade."""
    assert translit.to_roman("चीज़") == "chiz"
    assert translit.to_roman("चीज") == "chij"
    assert translit.to_roman("बस") == "bas"
    assert translit.to_roman("पास") == "paas"
    assert translit.to_roman("हाय") == "haay"


def test_a_full_line_romanises_end_to_end():
    assert (translit.to_roman("हेलो अजय सर, हेलो कपिल सर।")
            == "hello ajay sir, hello kapil sir.")
    # Long uu folds to u by design ("people type 'ki', not 'kii'").
    assert (translit.to_roman("अंगूठे पे नेल पेंट लगा हुआ है?")
            == "anguthe pe nail paint laga hua hai?")


def test_a_loanword_is_found_whichever_matra_whisper_chose():
    """Devanagari has no settled spelling for an English vowel and whisper
    alternates: मैसेज was in the table and मेसेज was not, so one real episode
    captioned the same word "message" on one line and "mesej" on another."""
    assert translit.to_roman("मैसेज") == "message"
    assert translit.to_roman("मेसेज") == "message"
    assert translit.to_roman("फोन") == "phone"
    assert translit.to_roman("फॉन") == "phone"
    assert translit.to_roman("कोलेज") == "college"
    assert translit.to_roman("कॉलेज") == "college"


def test_the_matra_fold_does_not_invent_words_it_cannot_tell_apart():
    """A variant spelling can be a different word: रोड is "road" but रॉड is
    "rod", रॉल a roll rather than a role, फेन Hindi for foam, टोप a cap."""
    assert translit.to_roman("रोड") == "road"
    assert translit.to_roman("रॉड") != "road"
    assert translit.to_roman("रॉल") != "role"
    assert translit.to_roman("फेन") != "fan"
    for blocked in ("पैंट", "रॉड", "फेन", "टोप", "रॉल"):
        assert blocked not in translit._LOANWORDS


def test_the_loanwords_the_code_switch_measurement_exposed():
    assert translit.to_roman("हेयरस्टाइल") == "hairstyle"
    assert translit.to_roman("सेक्सी") == "sexy"
    assert translit.to_roman("सेंचरी") == "century"


class TranslitTests(unittest.TestCase):
    """unittest discovery only collects TestCase methods; these tests are plain
    functions (pytest style), so expose each one as a method."""


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(TranslitTests, _name, staticmethod(_fn))


if __name__ == "__main__":
    unittest.main()
